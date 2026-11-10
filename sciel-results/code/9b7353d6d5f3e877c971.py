"""Rewritten program: construct largest pairwise non-coprime subset of [1,255] containing 255."""
import time
import math
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    """
    Find the maximum subset S ⊆ [1, 255] containing 255 such that gcd(a,b) > 1 
    for all distinct pairs a,b in S.
    
    Since 255 ∈ S, for any x ∈ S, gcd(x, 255) > 1.
    Let C = {x ∈ [1, 255] | gcd(x, 255) > 1}.
    Then S ⊆ C.
    The condition "pairwise gcd > 1" means S is a clique in the graph G_C where 
    vertices are C and edges connect u,v if gcd(u,v) > 1.
    Since 255 is connected to all vertices in C (by definition of C), 
    finding the max S containing 255 is equivalent to finding the max clique 
    in the subgraph induced by C \ {255}, and adding 255 to it.
    """
    
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # 1. Generate candidates C
    C = [x for x in range(1, 256) if gcd(x, 255) > 1]
    
    # 2. We need max clique in G[C \ {255}]
    # Let V = C \ {255}
    V = [x for x in C if x != 255]
    n = len(V)
    
    # Map values to indices
    val_to_idx = {v: i for i, v in enumerate(V)}
    
    # Build adjacency bitmasks
    # adj[i] is an integer bitmask where bit j is 1 if gcd(V[i], V[j]) > 1
    adj = [0] * n
    for i in range(n):
        vi = V[i]
        mask = 0
        for j in range(i + 1, n):
            vj = V[j]
            if gcd(vi, vj) > 1:
                mask |= (1 << j)
        adj[i] = mask
        # Symmetric part: for j > i, i is a neighbor of j
        # We need to set bit i in adj[j]
        # It's easier to just compute full adj for all i
    # Recompute to ensure full symmetry and correctness for all bits
    # The loop above only sets bits j > i in adj[i]. 
    # Let's do it properly for all pairs.
    adj = [0] * n
    for i in range(n):
        vi = V[i]
        for j in range(n):
            if i == j: continue
            if gcd(vi, V[j]) > 1:
                adj[i] |= (1 << j)
                
    # 3. Maximum Clique Algorithm (Branch and Bound) using Tomita's algorithm
    # We use a list of bitmasks to represent sets.
    
    best_clique = []
    best_size = 0
    
    def count_bits(x):
        return x.bit_count()
        
    def expand(clique, candidates):
        nonlocal best_clique, best_size
        
        # Deadline check
        if time.time() > deadline:
            return
            
        if not candidates:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
            return
        
        # Pruning: if current clique size + max possible size <= best, prune
        if len(clique) + count_bits(candidates) <= best_size:
            return
            
        # Pivot selection: choose v in candidates with max degree in candidates
        # This reduces the number of branches
        pivot_v = -1
        max_common_neighbors = -1
        
        # Iterate through bits in candidates
        c = candidates
        while c:
            # Get least significant set bit
            lsb = c & -c
            v = lsb.bit_length() - 1
            common = count_bits(adj[v] & candidates)
            if common > max_common_neighbors:
                max_common_neighbors = common
                pivot_v = v
            c ^= lsb
            
        # Branch on candidates \ N(pivot_v)
        # We iterate through candidates that are NOT neighbors of pivot_v
        # This ensures we cover all maximal cliques
        candidates_to_branch = candidates & ~adj[pivot_v]
        
        # Iterate through bits in candidates_to_branch
        c = candidates_to_branch
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            
            new_clique = clique + [v]
            new_candidates = candidates & adj[v]
            expand(new_clique, new_candidates)
            
            # Remove v from candidates
            candidates ^= lsb
            
            # Pruning check after removing v
            if len(clique) + count_bits(candidates) <= best_size:
                break
            c ^= lsb

    # Start with empty clique and all V as candidates
    # All V indices are 0 to n-1
    initial_candidates = (1 << n) - 1
    
    expand([], initial_candidates)
    
    # 4. Construct final set
    # best_clique contains indices into V
    result_indices = best_clique
    result_values = [V[i] for i in result_indices]
    result_values.append(255)
    
    # 5. Verification
    s = set(result_values)
    assert 255 in s
    assert len(s) == len(result_values)
    for a in s:
        assert 1 <= a <= 255
    for i in range(len(result_values)):
        for j in range(i + 1, len(result_values)):
            assert gcd(result_values[i], result_values[j]) > 1
            
    return result_values
# EVOLVE-BLOCK-END
