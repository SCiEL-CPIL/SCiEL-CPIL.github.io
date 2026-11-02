"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Generate candidate vectors
    # We focus on weight 1, 2, and 3 vectors.
    # Weight 1: Norm^2 = 1.
    # Weight 2: Norm^2 = 2.
    # Weight 3: Norm^2 = 3.
    
    candidates = []
    
    # Weight 1
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            
    # Weight 2
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Weight 3
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)
                    
    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    # 2. Compute conflict graph
    # Conflict if dist^2 < max(norm_i^2, norm_j^2)
    # dist^2 = ||v_i||^2 + ||v_j||^2 - 2 <v_i, v_j>
    
    norms_sq = np.sum(candidates * candidates, axis=1)
    G = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * G
    max_norms = np.maximum(norms_sq[:, None], norms_sq[None, :])
    
    # Conflict mask: True if conflict
    # Exclude self (i==j) from conflicts for graph building, but the condition holds for i!=j.
    # For i==j, dist_sq=0, max_norms=norms_sq[i]. 0 < norms_sq[i] is True. 
    # We don't want self-loops in the adjacency list.
    
    conflict_mask = dist_sq < max_norms
    np.fill_diagonal(conflict_mask, False) # No self conflict in graph
    
    # 3. Solve Maximum Independent Set (MIS)
    # We use a branch and bound approach with a simple greedy coloring bound.
    # To handle n_cand ~ 1540+ efficiently within 145s, we need a robust implementation.
    
    # Convert conflict mask to adjacency lists for faster neighbor lookup in Python?
    # Or stick with bitsets?
    # n_cand is ~ 22 + 220 + 1320 = 1562.
    # Bitsets are efficient for intersection.
    
    adj_bits = np.zeros(n_cand, dtype=np.uint64) # Wait, n_cand > 64. Need Python ints or arrays of uint64.
    # Let's use Python integers for bitsets as they are arbitrary precision.
    
    adj_bits = [0] * n_cand
    # Build adj_bits
    # Iterate over rows
    # Using numpy to find indices is faster than pure python loops for building.
    i_idx, j_idx = np.where(conflict_mask)
    for i, j in zip(i_idx, j_idx):
        adj_bits[i] |= (1 << j)
        # adj_bits[j] |= (1 << i) # Symmetric, but we can just do it once if we iterate upper triangle?
        # np.where returns all pairs. So i,j and j,i are both there.
        # To avoid double work, we can just build from the full matrix.
        # It's O(N^2) memory/time to build, which is fine for 1500.
        
    # Optimization: The graph is dense?
    # Let's check density roughly.
    # Weight 1 vs Weight 1:
    # v=(1,0..), w=(1,0..). Same? No, distinct.
    # v=(1,0..), w=(-1,0..). dot=-1. dist^2 = 1+1-2(-1)=4. max=1. 4>=1. OK.
    # v=(1,0..), w=(0,1..). dot=0. dist^2=2. max=1. 2>=1. OK.
    # So all weight 1 are compatible with each other?
    # Wait. v=(1,0..), w=(1,0..) is same vector. We don't have duplicates.
    # v=(1,0..), w=(1,0..) is not in list twice.
    # So weight 1 set is an independent set in the conflict graph?
    # Yes.
    
    # Weight 1 vs Weight 2:
    # v=(1,0..). w=(1,1,0..). dot=1. dist^2 = 1+2-2=1. max=2. 1<2. CONFLICT.
    # So (1,0..) conflicts with (1,1,0..) and (-1,1,0..) etc where index 0 is 1.
    # If w=(1,-1,0..), dot=1. dist^2=1. Conflict.
    # If w=(-1,1,0..), dot=-1. dist^2 = 1+2-2(-1)=5. max=2. 5>=2. OK.
    # So v=(1,0..) conflicts with w if w[0]=1.
    
    # This graph structure is specific.
    
    # MIS Solver
    best_set = []
    
    def color_bound(P):
        """
        Greedy coloring bound for MIS.
        P: bitset of candidates.
        Returns: max color count.
        """
        if P == 0:
            return 0
        colors = 0
        remaining = P
        while remaining:
            colors += 1
            # Pick a vertex to start the color class
            # Greedy: pick vertex with smallest index?
            # To maximize independent set size in color class (minimize colors), 
            # we should pick vertices that have few neighbors in 'remaining'?
            # Or just pick first one for speed.
            lsb = remaining & -remaining
            v = lsb.bit_length() - 1
            
            # Remove v and its neighbors from remaining
            # Neighbors of v in remaining: remaining & adj_bits[v]
            # v itself: 1 << v
            remaining = remaining & ~(adj_bits[v] | (1 << v))
            
        return colors

    def expand(P, X, depth):
        nonlocal best_set
        
        if time.time() > deadline:
            return
            
        if P == 0:
            if X.bit_count() > len(best_set):
                # Extract indices
                curr_set = []
                temp = X
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    curr_set.append(v)
                    temp = temp & (temp - 1)
                best_set = curr_set
            return
            
        # Bound
        # Simple bound: |P|
        # Color bound:
        cb = color_bound(P)
        if X.bit_count() + cb <= len(best_set):
            return
            
        # Branch
        # Pick a vertex from P.
        # Heuristic: Pick vertex with highest degree in P?
        # Or just first one?
        # Let's try picking the vertex with the most neighbors in P to prune faster.
        # Calculating degree in P for all v in P is expensive.
        # Let's just pick the first one (LSB).
        
        lsb = P & -P
        v = lsb.bit_length() - 1
        
        # Branch 1: Include v
        # New P = P \ (N(v) U {v})
        new_P = P & ~(adj_bits[v] | (1 << v))
        new_X = X | (1 << v)
        expand(new_P, new_X, depth + 1)
        
        # Branch 2: Exclude v
        new_P2 = P & ~(1 << v)
        expand(new_P2, X, depth + 1)
        
    initial_P = (1 << n_cand) - 1
    expand(initial_P, 0, 0)
    
    if not best_set:
        # Fallback: Greedy
        selected = []
        remaining = set(range(n_cand))
        # Sort by degree?
        # Just pick arbitrarily or by norm?
        # Let's pick weight 1 vectors first, they are very compatible.
        # Actually, let's just do a simple greedy on the conflict graph.
        # Pick vertex, remove neighbors.
        
        # To get a good starting point, let's try to include all weight 1 vectors.
        # They are mutually compatible.
        # Then add compatible weight 2 and 3.
        
        w1_indices = [i for i in range(n_cand) if norms_sq[i] == 1]
        w2_indices = [i for i in range(n_cand) if norms_sq[i] == 2]
        w3_indices = [i for i in range(n_cand) if norms_sq[i] == 3]
        
        selected = list(w1_indices)
        selected_set = set(selected)
        
        # Add weight 2
        for i in w2_indices:
            compatible = True
            for j in selected:
                if conflict_mask[i, j]:
                    compatible = False
                    break
            if compatible:
                selected.append(i)
                selected_set.add(i)
                
        # Add weight 3
        for i in w3_indices:
            compatible = True
            for j in selected:
                if conflict_mask[i, j]:
                    compatible = False
                    break
            if compatible:
                selected.append(i)
                selected_set.add(i)
                
        best_set = selected

    if not best_set:
        # Absolute fallback
        return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)

    result = candidates[best_set]
    return result
# EVOLVE-BLOCK-END
