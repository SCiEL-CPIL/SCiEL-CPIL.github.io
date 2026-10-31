import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    D = 11
    
    # Generate supports (4-subsets)
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports)  # 330
    
    # Build support adjacency: supports sharing exactly 3 elements
    tri_to_tet = defaultdict(list)
    for i, s in enumerate(supports):
        for tri in itertools.combinations(s, 3):
            tri_to_tet[tri].append(i)
    
    # Generate all 5280 weight-4 vectors
    all_vectors = []
    vec_support = []
    for s_idx, s in enumerate(supports):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            all_vectors.append(v)
            vec_support.append(s_idx)
            
    all_vectors_np = np.array(all_vectors, dtype=np.int8)
    n_total = len(all_vectors_np)  # 5280
    
    # Build vector-level conflict graph
    # Two vectors conflict if their dot product > 2
    # We only need to check pairs of vectors whose supports share >= 3 elements
    
    # Create a set of support pairs that share 3 elements
    support_pairs = set()
    for tri, tets in tri_to_tet.items():
        for u in range(len(tets)):
            for v in range(u + 1, len(tets)):
                i, j = tets[u], tets[v]
                support_pairs.add((min(i, j), max(i, j)))
    
    # Build bitset adjacency list
    # adj_bitset[i] is an integer where bit j is 1 if i and j conflict
    adj_bitset = [0] * n_total
    
    for s1, s2 in support_pairs:
        base1 = s1 * 16
        base2 = s2 * 16
        
        v1 = all_vectors_np[base1:base1+16]
        v2 = all_vectors_np[base2:base2+16]
        
        # Dot products
        d = v1 @ v2.T
        
        # Find conflicts (dot > 2)
        indices = np.argwhere(d > 2)
        for i, j in indices:
            idx1 = base1 + i
            idx2 = base2 + j
            # Set bit idx2 in adj_bitset[idx1] and vice versa
            adj_bitset[idx1] |= (1 << idx2)
            adj_bitset[idx2] |= (1 << idx1)
            
    # Precompute a list of all nodes for iteration
    all_nodes = list(range(n_total))
    
    def can_add(node, selected_bits):
        """Check if node can be added to the set represented by selected_bits."""
        return (selected_bits & adj_bitset[node]) == 0
        
    def popcount(x):
        """Count bits set in integer x."""
        return bin(x).count('1')
        
    rng = np.random.default_rng(42)
    
    best_k = 0
    best_selected_bits = 0
    
    # Phase 1: Iterated Local Search
    # 1. Start with a greedy solution
    # 2. Local optimize (add all possible)
    # 3. Perturb (remove random subset)
    # 4. Local optimize again
    # 5. Repeat
    
    max_iter = 2000
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        # Initialize with a greedy minimum-degree-like selection
        # For speed, we'll start with a random subset or empty and fill greedily
        # A better start: Randomly select 50% of nodes? No, that causes many conflicts.
        # Let's start with a deterministic greedy on a random permutation of nodes.
        
        perm = rng.permutation(n_total)
        selected_bits = 0
        # Greedy fill
        for u in perm:
            if can_add(u, selected_bits):
                selected_bits |= (1 << u)
                
        # Local Optimization: Add any node that fits
        improved = True
        while improved:
            improved = False
            # We can check all nodes, but that's O(N^2) if done naively.
            # With bitsets, checking is O(1) per node. N=5280. 5280 checks is fast.
            for u in all_nodes:
                if (selected_bits & (1 << u)) == 0: # If not selected
                    if can_add(u, selected_bits):
                        selected_bits |= (1 << u)
                        improved = True
                        
        k = popcount(selected_bits)
        if k > best_k:
            best_k = k
            best_selected_bits = selected_bits
            
        # Perturbation: Remove a random percentage of selected nodes
        # This helps escape local optima where a few "bad" blocks prevent many "good" adds
        if k > 0:
            # Convert selected bits to list
            sel_list = [u for u in all_nodes if (selected_bits & (1 << u)) != 0]
            # Remove 10% of them
            num_remove = max(1, int(len(sel_list) * 0.1))
            to_remove = rng.choice(len(sel_list), size=num_remove, replace=False)
            for idx in to_remove:
                u = sel_list[idx]
                selected_bits &= ~(1 << u)
                
            # Re-optimize
            improved = True
            while improved:
                improved = False
                for u in all_nodes:
                    if (selected_bits & (1 << u)) == 0:
                        if can_add(u, selected_bits):
                            selected_bits |= (1 << u)
                            improved = True
                            
            k = popcount(selected_bits)
            if k > best_k:
                best_k = k
                best_selected_bits = selected_bits

    # Phase 2: Refinement with Swap Moves (2-opt)
    # If we have time, try swapping 1 out, 1 in, or 2 in
    if time.time() < deadline - 10:
        selected_bits = best_selected_bits
        # Extract list
        sel_list = [u for u in all_nodes if (selected_bits & (1 << u)) != 0]
        
        for _ in range(50): # Limit refinement loops
            if time.time() > deadline - 5:
                break
            if not sel_list:
                break
            # Try removing a random node and adding others
            rem_idx = rng.integers(len(sel_list))
            rem_node = sel_list[rem_idx]
            
            # Remove
            selected_bits &= ~(1 << rem_node)
            
            # Greedy add phase
            improved = True
            while improved:
                improved = False
                for u in all_nodes:
                    if (selected_bits & (1 << u)) == 0:
                        if can_add(u, selected_bits):
                            selected_bits |= (1 << u)
                            improved = True
                            
            k = popcount(selected_bits)
            if k > best_k:
                best_k = k
                best_selected_bits = selected_bits
                sel_list = [u for u in all_nodes if (selected_bits & (1 << u)) != 0]
            else:
                # Revert if no improvement
                selected_bits |= (1 << rem_node)
                # Break if we keep failing? Or just continue.
                
    if best_selected_bits == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    # Extract result
    result_mask = np.zeros(n_total, dtype=bool)
    for u in all_nodes:
        if (best_selected_bits & (1 << u)) != 0:
            result_mask[u] = True
            
    result = all_vectors_np[result_mask]
    return result.astype(np.int64)
