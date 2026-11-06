import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin
    
    # Generate all 4-subsets of [1, 16] as bitmasks
    candidates = []
    for comb in combinations(range(1, 17), 4):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        candidates.append(mask)
    
    N = len(candidates)  # 1820
    
    # Precompute which pairs of sets can form a sunflower with a third set
    # For each pair (i, j), compute the "forbidden region" for a third set k
    # A sunflower with petals A, B, C has core K = A∩B∩C
    # and each petal has 4-|K| unique elements
    # 
    # For two sets A and B, a third set C forms a sunflower if:
    # - There exists a core K such that K ⊆ A, K ⊆ B, K ⊆ C
    # - C \ K is disjoint from (A\K) and (B\K)
    # - |A\K| = |B\K| = |C\K| (since all are 4-sets)
    #
    # Actually, let's think differently. Precompute for each pair (i,j)
    # the set of k that would form a sunflower.
    
    # For a pair (A, B), the possible cores are subsets of A∩B.
    # If core = K, then C must contain K, and C\K must be disjoint from (A\K) and (B\K),
    # and |C\K| = 4 - |K|.
    
    # Let's precompute conflicts[i] = set of j > i such that there exists k forming sunflower
    # Actually, for branch-and-bound, we need: for each vertex v, which other vertices conflict with it
    # in the sense that {v, u, w} forms a sunflower for some w.
    
    # Better approach: Precompute for each pair (i, j), the list of k that form a sunflower with i and j.
    # Then, in the independent set problem, vertices i, j, k form a hyperedge if they're a sunflower.
    
    # For branch and bound on maximum independent set in a hypergraph:
    # We want the largest set of vertices with no hyperedge fully contained in it.
    
    # Let's precompute: for each i, conflicts_with[i] = set of j such that 
    # there exists some k where {i, j, k} is a sunflower.
    # But that's not quite right either. We need to track which triples are sunflowers.
    
    # Actually, for a cleaner approach: 
    # Two vertices i and j are "incompatible" if there exists a k such that {i,j,k} is a sunflower.
    # But that's too strong - i and j could be compatible if the only k that completes the sunflower
    # is not in our solution.
    
    # Let me think about this more carefully. The constraint is: no three distinct sets form a sunflower.
    # So we have a 3-uniform hypergraph where hyperedges are sunflower triples.
    # We want the maximum set of vertices containing no hyperedge.
    
    # For branch and bound:
    # State: (included, excluded, undecided)
    # Upper bound: |included| + |undecided|
    # But we can do better: if two undecided vertices are in a hyperedge with an included vertex,
    # we can't include both.
    
    # Let me use a different approach: precompute for each pair (i,j), whether there exists k
    # such that {i,j,k} is a sunflower. If not, then i and j can always be together.
    # If yes, then including both i and j forces us to exclude all k that complete the sunflower.
    
    # Precompute: for each pair (i, j), the set of k such that {i,j,k} is a sunflower.
    # sunflower_triples[i][j] = list of k values
    
    # This is O(N^3) which is 1820^3 ≈ 6 billion - too slow.
    
    # Better: For each pair (i, j), compute the possible cores and check if any 4-set k works.
    # For pair (A, B) with intersection I = A∩B:
    #   For each subset K of I (the core):
    #     C must contain K, and C\K must be a (4-|K|)-subset of ([16]\(A∪B)) ∪ (K)
    #     Wait, C\K must be disjoint from A\K and B\K.
    #     A\K and B\K are the parts of A and B outside the core.
    #     So C\K must be chosen from [16] \ (A∪B) ∪ K... no.
    #     C\K must be disjoint from (A\K) ∪ (B\K).
    #     (A\K) ∪ (B\K) = (A∪B) \ K.
    #     So C\K must be a subset of [16] \ ((A∪B)\K) = ([16]\(A∪B)) ∪ K.
    #     But C\K is disjoint from K by definition (K is the core, C\K are the unique elements).
    #     So C\K must be a subset of [16]\(A∪B).
    #     And |C\K| = 4 - |K|.
    #     So we need |[16]\(A∪B)| ≥ 4 - |K|, and then C = K ∪ S where S is a (4-|K|)-subset of [16]\(A∪B).
    
    # So for pair (A, B) with |A∩B| = r:
    #   For each K ⊆ A∩B with |K| = k (k from 0 to r):
    #     Need 4-k ≤ |[16]\(A∪B)| = 16 - |A∪B| = 16 - (8-k) = 8-k
    #     So need 4-k ≤ 8-k, which is always true.
    #     Number of choices for S: C(8-k, 4-k)
    #     Each such S gives a valid C = K ∪ S.
    
    # But we need to check if C is one of our candidates (it always is, since it's a 4-subset of [16]).
    # And we need to track which specific C values form sunflowers with A and B.
    
    # This is still potentially expensive. Let me estimate:
    # For each pair (A, B), we iterate over subsets K of A∩B.
    # |A∩B| can be 0, 1, 2, or 3 (since A and B are 4-sets).
    # If |A∩B| = 0: K = ∅, need C disjoint from A∪B, |C|=4. 
    #   [16]\(A∪B) has 16-8=8 elements. C(8,4)=70 choices.
    # If |A∩B| = 1: K can be ∅ or {x} where x is the intersection.
    #   K=∅: same as above, 70 choices.
    #   K={x}: need C = {x} ∪ S where S is 3-subset of [16]\(A∪B). |[16]\(A∪B)| = 16-7=9. C(9,3)=84.
    # If |A∩B| = 2: K can be ∅, {x}, {y}, {x,y}.
    #   K=∅: 70
    #   K={x}: C(9,3)=84 (since |A∪B|=6, [16]\(A∪B) has 10... wait)
    #   Let me recompute. |A|=|B|=4, |A∩B|=r, |A∪B|=8-r.
    #   |[16]\(A∪B)| = 16-(8-r) = 8+r.
    #   K=∅: need 4-subset of 8+r elements: C(8+r, 4)
    #   K of size 1: need 3-subset of 8+r elements: C(8+r, 3)
    #   K of size 2: need 2-subset of 8+r elements: C(8+r, 2)
    #   K of size 3: need 1-subset of 8+r elements: C(8+r, 1)
    
    # For r=0: C(8,4)=70
    # For r=1: C(9,4)+C(9,3) = 126+84=210
    # For r=2: C(10,4)+2*C(10,3)+C(10,2) = 210+2*120+45=495
    # For r=3: C(11,4)+3*C(11,3)+3*C(11,2)+C(11,1) = 330+3*165+3*55+11=951
    
    # So for each pair, we might need to check up to ~1000 candidates.
    # With 1820*1819/2 ≈ 1.65 million pairs, this is ~1.65 billion operations. Too slow.
    
    # Alternative: Use a different strategy. Let's use a randomized local search with 
    # precomputed conflict information.
    
    # For each vertex i, precompute conflicts[i] = set of j such that there exists k
    # with {i,j,k} forming a sunflower. This tells us that if we include both i and j,
    # we must exclude all k in sunflower_k_for(i,j).
    
    # Actually, let's just use a smarter greedy + local search with better conflict tracking.
    
    max_size = 0
    best_family = []
    
    # Precompute for each pair (i, j), whether they can coexist (i.e., no k forms sunflower with them)
    # And if they can't, what k's would be excluded.
    
    # For efficiency, let's precompute a conflict matrix:
    # conflict[i][j] = True if there exists k such that {i,j,k} is a sunflower
    # This is symmetric.
    
    # But we also need to know which k's are the problem, for local search.
    
    # Let me try a different approach: use simulated annealing or tabu search.
    
    # Actually, let's try a more structured approach:
    # 1. Greedy initialization with random ordering
    # 2. Local search: try to add a set, and if blocked, try to remove one of the blockers
    # 3. Use precomputed data to speed up validity checks
    
    # Precompute: for each mask, store its index
    mask_to_idx = {m: i for i, m in enumerate(candidates)}
    
    # Precompute pairwise intersections
    # inter[i][j] = candidates[i] & candidates[j]
    # This is a 1820x1820 matrix of integers. Memory: 1820*1820*8 bytes ≈ 26 MB. OK.
    
    # Actually, let's be smarter. For validity checking, we need to check if adding mask c
    # creates a sunflower with any pair (a, b) in the family.
    # A sunflower with a, b, c means:
    #   core = a & b & c
    #   (a & ~core) & c == 0  (c's unique part disjoint from a's unique part)
    #   (b & ~core) & c == 0
    #   |a & ~core| == |b & ~core| == |c & ~core| == 4 - |core|
    
    # Equivalently:
    #   Let K = a & b & c
    #   a\K, b\K, c\K are pairwise disjoint and each has size 4-|K|
    
    # For a given a, b, the possible cores are subsets of a&b.
    # For each such core K, c must satisfy:
    #   K ⊆ c
    #   c\K is disjoint from (a\K) and (b\K)
    #   |c\K| = 4 - |K|
    
    # So c\K must be a (4-|K|)-subset of [16]\((a\K) ∪ (b\K)) = [16]\((a∪b)\K)
    # = ([16]\(a∪b)) ∪ K
    # But c\K is disjoint from K, so c\K ⊆ [16]\(a∪b).
    
    # So for given a, b, and core K ⊆ a∩b:
    #   c = K ∪ S where S is a (4-|K|)-subset of [16]\(a∪b)
    
    # Let me precompute for each pair (i, j), the list of k that form sunflowers.
    # But this might be too large. Let me estimate the total number of sunflower triples.
    
    # Total 4-subsets: C(16,4) = 1820
    # For each triple of 4-subsets, check if it's a sunflower.
    # C(1820,3) ≈ 1 billion. Too many.
    
    # Better: for each pair (i, j), enumerate possible cores and count how many k work.
    # Then sum over all pairs.
    
    # Let me just try the original approach but with better optimization.
    # The key bottleneck is the O(|F|²) validity check.
    
    # Optimization: maintain a set of "active pairs" that could potentially block new additions.
    # When we add a new set c, we only need to check pairs (a, b) where a and b are in the family.
    # But we can prune: if a & b has size < 0, skip (always true).
    # If a & b has size 0, then core must be empty, so c must be disjoint from a and b.
    #   a and b are disjoint 4-sets, so a∪b has 8 elements. c must be a 4-subset of the other 8.
    #   There are C(8,4)=70 such c's.
    
    # Let me try a different local search: instead of random swaps, use a more systematic approach.
    
    # Run multiple random restarts with greedy + local search.
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        random.shuffle(candidates)
        family = []
        family_set = set()
        
        # Greedy construction
        for mask in candidates:
            if is_valid_fast(family, mask):
                family.append(mask)
                family_set.add(mask)
        
        if len(family) > max_size:
            max_size = len(family)
            best_family = family.copy()
        
        # Local search: try to improve
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add each candidate not in family
            for c in candidates:
                if c in family_set:
                    continue
                valid, blockers = is_valid_with_blockers(family, c)
                if valid:
                    family.append(c)
                    family_set.add(c)
                    improved = True
                    if len(family) > max_size:
                        max_size = len(family)
                        best_family = family.copy()
                else:
                    # Try to swap: remove one blocker and add c
                    a, b = blockers
                    for victim in [a, b]:
                        family.remove(victim)
                        family_set.remove(victim)
                        if is_valid_fast(family, c):
                            family.append(c)
                            family_set.add(c)
                            improved = True
                            if len(family) > max_size:
                                max_size = len(family)
                                best_family = family.copy()
                            break
                        else:
                            family.append(victim)
                            family_set.add(victim)
    
    # Convert best_family masks back to sorted lists of integers
    result = []
    for m in best_family:
        s = [i for i in range(1, 17) if (m >> i) & 1]
        result.append(sorted(s))
        
    return result

def is_valid_fast(family, mask):
    """
    Check if mask can be added to family without forming a sunflower.
    Returns True if valid, False otherwise.
    """
    n = len(family)
    for i in range(n):
        a = family[i]
        for j in range(i + 1, n):
            b = family[j]
            core = a & b
            if (core & mask) == core:
                if (mask & (a ^ core)) == 0 and (mask & (b ^ core)) == 0:
                    return False
    return True

def is_valid_with_blockers(family, mask):
    """
    Check if mask can be added to family.
    Returns (True, None) if valid.
    Returns (False, (a, b)) if invalid, where a, b are a blocking pair.
    """
    n = len(family)
    for i in range(n):
        a = family[i]
        for j in range(i + 1, n):
            b = family[j]
            core = a & b
            if (core & mask) == core:
                if (mask & (a ^ core)) == 0 and (mask & (b ^ core)) == 0:
                    return False, (a, b)
    return True, None
