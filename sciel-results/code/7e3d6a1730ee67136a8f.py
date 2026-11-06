import time
import random
from collections import defaultdict

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s buffer for safety
    
    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d) with 1 <= a, a+3d <= N
    # For each element x, store the set of pairs (y, z) such that {x, y, z} is a 3-term subset
    # of a 4-AP. Adding x to S creates a 4-AP iff there exist y, z in S such that (y, z) is in conflicts[x].
    # More precisely, we need to check if adding x completes a 4-AP.
    # A 4-AP is (a, a+d, a+2d, a+3d).
    # If we add x, it could be any of the 4 terms.
    # Instead, let's precompute for each x, the list of "witness" pairs.
    # If x is added, it forms a 4-AP if there are y, z in S such that {x, y, z, w} is a 4-AP for some w.
    # This is equivalent to: for every 4-AP containing x, the other 3 elements must not all be in S.
    # So, for each x, let's find all 4-APs containing x.
    # For each such 4-AP, let O be the set of the other 3 elements.
    # If O is a subset of S, then adding x creates a 4-AP.
    # This is a 3-way check, which is slow.
    
    # Better approach:
    # For each x, precompute a list of pairs (y, z) such that there exists a 4-AP containing x, y, z.
    # Actually, if x, y, z are in a 4-AP, then the 4th element w is determined.
    # So, if we add x, and y, z are in S, we must check if the 4th element w is NOT in S.
    # Wait, if x, y, z are in S, and w is not in S, then adding x doesn't create a 4-AP *yet*?
    # No, the condition is: S ∪ {x} contains a 4-AP.
    # This happens if there exists a 4-AP {a, b, c, d} such that {a, b, c, d} ⊆ S ∪ {x}.
    # Since S is 4-AP free, the only way this happens is if x is one of the elements, and the other 3 are in S.
    # So, for each x, we need to check if there exists a 4-AP containing x where the other 3 elements are in S.
    
    # Let's precompute for each x, the list of "triples" (y, z, w) such that {x, y, z, w} is a 4-AP.
    # Then can_add(x, S) is: for all (y, z, w) in triples[x], NOT (y in S and z in S and w in S).
    # This is still O(k) checks per x, where k is the number of 4-APs containing x.
    # k is roughly N/d for each d, summed over d.
    # Sum_{d=1}^{N/3} N/d ≈ N log N.
    # So total triples is N^2 log N. For N=512, this is ~512^2 * 9 ≈ 2.3 million.
    # This is feasible to precompute once.
    
    # However, checking "y in S and z in S and w in S" for each triple is 3 set lookups.
    # If we have ~100 triples per x, that's 300 lookups per insertion.
    # 512 insertions * 300 lookups = 150k lookups per greedy run.
    # This is fast.
    
    # Let's optimize the precomputation.
    # For each x, we store a list of tuples (y, z, w).
    # To speed up the check, we can store the triples as a list of sets? No, tuples are fine.
    
    # Even better:
    # For each x, group the triples by the "missing" element? No.
    # Just iterate.
    
    # Let's try to reduce the check cost.
    # Instead of storing triples, can we store pairs?
    # If x is added, it forms a 4-AP if there exist y, z in S such that the 4th element w is NOT in S?
    # No, the 4th element w must be in S ∪ {x}. Since x is the new one, w must be in S.
    # So, if x, y, z, w is a 4-AP, and y, z, w are in S, then adding x creates a 4-AP.
    # So we need to check if there exists a triple (y, z, w) from the precomputed list for x such that y, z, w are all in S.
    
    # Is there a way to make this O(1) or O(1) amortized?
    # We can maintain a counter for each x: how many of its "forbidden triples" are fully contained in S?
    # When we add an element y to S, we need to update the counters for all x such that y is part of a forbidden triple for x.
    # This is the "incremental" approach.
    
    # Let's build the graph of dependencies.
    # For each 4-AP (a, b, c, d):
    # If we add a, we need to check if b, c, d are in S.
    # If we add b, we need to check if a, c, d are in S.
    # Etc.
    
    # Let's precompute `neighbors[x]` = list of (y, z, w) such that {x, y, z, w} is a 4-AP.
    # And `reverse_neighbors[y]` = list of (x, z, w) such that {x, y, z, w} is a 4-AP.
    # Actually, just `neighbors` is enough if we recompute from scratch, but incremental is faster.
    
    # Let's stick to the simple "check on add" first, but with precomputed triples.
    # It's simpler and likely fast enough for N=512.
    
    # Precomputation
    neighbors = defaultdict(list)
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            dap = a + 3 * d
            # 4-AP: a, b, c, dap
            neighbors[a].append((b, c, dap))
            neighbors[b].append((a, c, dap))
            neighbors[c].append((a, b, dap))
            neighbors[dap].append((a, b, c))
    
    # Convert to list of lists for faster access
    neighbors_list = [neighbors[i] for i in range(N + 1)]
    
    def can_add(x, S):
        for (y, z, w) in neighbors_list[x]:
            if y in S and z in S and w in S:
                return False
        return True
    
    current_best = {1}
    best_size = 1
    
    # Run random greedy
    while time.time() < deadline:
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        current_set = set()
        
        for x in candidates:
            if can_add(x, current_set):
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            current_best = current_set.copy()
            
    return sorted(current_best)
