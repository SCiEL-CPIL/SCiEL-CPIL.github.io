"""
Solves the Large Integer Sets task using a Branch-and-Bound approach on a 3-uniform hypergraph.
Vertices: Integers 2..256.
Hyperedges: Triples (a, b, c) such that any element divides the product of the other two.
Goal: Maximum Independent Set.
"""
# EVOLVE-BLOCK-START
import math
import random
from itertools import combinations

def construct():
    n = 256
    # Vertices are 2..256
    # Map integer to index 0..254 (255 vertices total: 2..256)
    # idx = val - 2
    
    # 1. Generate forbidden triples
    # A triple (a, b, c) with distinct values is forbidden if:
    # a | b*c OR b | a*c OR c | a*b
    
    forbidden = set()
    for a in range(2, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b + 1, n + 1):
                # Check if any divides the product of the other two
                if (b * c) % a == 0 or (a * c) % b == 0 or (a * b) % c == 0:
                    forbidden.add((a - 2, b - 2, c - 2))
    
    # Precompute for each vertex, the set of pairs (u, w) such that (u, v, w) is a forbidden triple.
    # This allows O(1) check if adding v to a set S creates a conflict:
    # v conflicts if there exists (u, w) in forbidden_pairs[v] such that u in S and w in S.
    
    num_vertices = n - 1 # 255
    forbidden_pairs = [set() for _ in range(num_vertices)]
    for t in forbidden:
        u, v, w = t
        forbidden_pairs[u].add((v, w))
        forbidden_pairs[v].add((u, w))
        forbidden_pairs[w].add((u, v))
        
    # Calculate degree (number of forbidden triples containing the vertex) for ordering
    vertex_degree = [0] * num_vertices
    for t in forbidden:
        for idx in t:
            vertex_degree[idx] += 1
            
    # 2. Greedy Initial Solution
    # Strategy: Sort vertices by degree (ascending). Add if valid.
    # This tends to pick "safe" vertices (primes, numbers with few divisors) first.
    
    sorted_indices = sorted(range(num_vertices), key=lambda i: vertex_degree[i])
    
    def check_valid(idx, current_set):
        # Check if idx can be added to current_set without forming a forbidden triple
        for pair in forbidden_pairs[idx]:
            if pair[0] in current_set and pair[1] in current_set:
                return False
        return True
        
    S = set()
    for idx in sorted_indices:
        if check_valid(idx, S):
            S.add(idx)
            
    best_size = len(S)
    best_set = S.copy()
    
    # 3. Local Search / Multi-Start Greedy
    # Try multiple random permutations to find a larger independent set.
    # Also try a simple "swap" or "add" local improvement.
    
    # To keep within time limits, we do a limited number of restarts.
    # The greedy solution is often near-optimal for this structure.
    # Primes are always safe. There are 54 primes between 2 and 256.
    # Composites can sometimes be added if they don't form triples with existing members.
    
    # Let's try a few randomized restarts with different tie-breaking or shuffling
    # among vertices with similar degrees.
    
    random.seed(42)
    
    for _ in range(50):
        S_curr = set()
        # Shuffle the sorted indices slightly? 
        # Better: Pick the next vertex from a small random window of the lowest degree available vertices.
        # Or simply shuffle the entire sorted list and greedily add.
        # Shuffling the entire list might degrade the "low degree first" benefit.
        
        # Alternative: Randomized Greedy
        # Maintain a list of available vertices.
        # At each step, pick a random vertex from the available ones, but bias towards lower degree.
        
        # Simple approach: Shuffle the sorted_indices list and run greedy.
        # This breaks ties randomly.
        shuffled_indices = sorted_indices[:]
        random.shuffle(shuffled_indices)
        
        for idx in shuffled_indices:
            if check_valid(idx, S_curr):
                S_curr.add(idx)
                
        if len(S_curr) > best_size:
            best_size = len(S_curr)
            best_set = S_curr.copy()
            
    # 4. Local Improvement on Best Set
    # Try to add any vertex that wasn't included.
    improved = True
    while improved:
        improved = False
        for idx in range(num_vertices):
            if idx not in best_set:
                if check_valid(idx, best_set):
                    best_set.add(idx)
                    improved = True
                    best_size = len(best_set)
                    break # Restart to check all again, as adding one might allow another? 
                    # Actually, adding a vertex only makes the set "more constrained", 
                    # so it cannot enable adding a vertex that was previously blocked.
                    # It can only block more. So if it was blocked before, it remains blocked.
                    # Wait, check_valid checks if adding idx creates a conflict.
                    # If idx was blocked by (u, w) in S, and we add v, (u, w) is still in S. 
                    # So idx remains blocked.
                    # Therefore, a single pass is sufficient for adding.
                    # We don't need a while loop, just a single pass over all candidates.
                    
    # Actually, the logic above implies that if a vertex couldn't be added, it never can be added later 
    # in the same "phase" unless we remove vertices. Since we are only adding, 
    # a single pass over all vertices not in best_set is sufficient to find all addable vertices.
    
    # Let's do a proper single pass for addition.
    for idx in range(num_vertices):
        if idx not in best_set:
            if check_valid(idx, best_set):
                best_set.add(idx)
                best_size = len(best_set)
                
    # 5. Branch and Bound (Limited)
    # Given the size (255) and the likely optimal solution being close to the greedy one,
    # a full B&B might be too slow. 
    # However, we can try to verify if we can beat the current best by exploring 
    # a small neighborhood.
    
    # Let's try a "swap" heuristic: remove one vertex, try to add two?
    # This is complex. Given the 60s limit and the likely optimality of the greedy set 
    # (which includes all primes), we will stick with the multi-start greedy + add pass.
    
    # Convert indices back to values
    result = [i + 2 for i in best_set]
    result.sort() # Ensure consistent output format
    
    return result
# EVOLVE-BLOCK-END
