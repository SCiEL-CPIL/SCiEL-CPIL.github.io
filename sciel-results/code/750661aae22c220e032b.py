import numpy as np
import time
import random
import sys
import heapq

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    # 630s limit, use 600s to be safe
    deadline = start_time + 600
    
    best_set_global = []
    best_size_global = 0
    
    # Generate candidates for R=3 and R=4
    # We will process R=3 and R=4 separately
    R_values = [3, 4]
    
    for R in R_values:
        if time.time() > deadline - 10:
            break
            
        # Generate candidates with norm_sq <= R
        candidates = []
        
        # Norm 1: e_i, -e_i
        for i in range(d):
            v = np.zeros(d, dtype=np.int64)
            v[i] = 1
            candidates.append(v)
            candidates.append(-v)
            
        # Norm 2: e_i +/- e_j
        for i in range(d):
            for j in range(i + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = s1
                        v[j] = s2
                        candidates.append(v)
                        
        # Norm 3: e_i +/- e_j +/- e_k
        if R >= 3:
            for i in range(d):
                for j in range(i + 1, d):
                    for k in range(j + 1, d):
                        for s1 in [1, -1]:
                            for s2 in [1, -1]:
                                for s3 in [1, -1]:
                                    v = np.zeros(d, dtype=np.int64)
                                    v[i] = s1
                                    v[j] = s2
                                    v[k] = s3
                                    candidates.append(v)
                                    
        # Norm 4
        if R >= 4:
            # Type 1: 2e_i
            for i in range(d):
                v = np.zeros(d, dtype=np.int64)
                v[i] = 2
                candidates.append(v)
                candidates.append(-v)
            # Type 2: e_i +/- e_j +/- e_k +/- e_l
            for i in range(d):
                for j in range(i + 1, d):
                    for k in range(j + 1, d):
                        for l in range(k + 1, d):
                            for s1 in [1, -1]:
                                for s2 in [1, -1]:
                                    for s3 in [1, -1]:
                                        for s4 in [1, -1]:
                                            v = np.zeros(d, dtype=np.int64)
                                            v[i] = s1
                                            v[j] = s2
                                            v[k] = s3
                                            v[l] = s4
                                            candidates.append(v)
                                            
        # Unique candidates
        unique_candidates = []
        seen = set()
        for v in candidates:
            if np.all(v == 0):
                continue
            key = tuple(v)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(v)
                
        cand_array = np.array(unique_candidates, dtype=np.int64)
        norms_sq = np.sum(cand_array**2, axis=1)
        
        # Filter for current R
        valid_mask = norms_sq <= R
        sub_cands = cand_array[valid_mask]
        sub_norms = norms_sq[valid_mask]
        n_sub = len(sub_cands)
        
        if n_sub < 2:
            continue
            
        # Compute distance matrix
        # dist_sq[i,j] = ||p_i - p_j||^2 = ||p_i||^2 + ||p_j||^2 - 2 p_i.p_j
        dots = sub_cands @ sub_cands.T
        dist_sq = sub_norms[:, None] + sub_norms[None, :] - 2 * dots
        
        # Conflict if dist_sq < R
        # Note: dist_sq[i,i] = 0 < R. We ignore self.
        conflict_mask = dist_sq < R
        np.fill_diagonal(conflict_mask, False)
        
        # Convert conflict_mask to adjacency list for faster traversal in B&B
        adj_list = [np.where(conflict_mask[i])[0] for i in range(n_sub)]
        deg = np.array([len(neigh) for neigh in adj_list])
        
        # --- Phase 1: Greedy Initial Solution ---
        # Run a few greedy trials to get a good initial lower bound
        best_greedy_set = set()
        best_greedy_size = 0
        
        # Sort indices by degree for static greedy
        sorted_indices = np.argsort(deg)
        
        for trial in range(20):
            if time.time() > deadline - 5:
                break
                
            available = np.ones(n_sub, dtype=bool)
            selected_indices = []
            
            # Simple greedy: pick min degree available
            # Using sorted_indices, we scan for first available
            # This is O(N) per pick, but we can optimize by maintaining a heap or just scanning
            # For N=1500, scanning is fine.
            
            # To randomize, we can shuffle the order of ties or add noise
            # Let's just use the sorted order but maybe skip some? 
            # Actually, the inspiration code used a while loop with linear scan.
            # Let's stick to a fast greedy.
            
            # Optimization: Instead of linear scan every time, we can use a heap of (degree, index)
            # But degrees are static. So the first available in sorted_indices is min degree.
            # Linear scan on sorted_indices is O(N) worst case per pick.
            # Total O(N^2).
            
            # Let's try a slightly different greedy: 
            # Pick random available node with lowest degree?
            # Or just standard greedy.
            
            # Standard Greedy
            for idx in sorted_indices:
                if not available[idx]:
                    continue
                selected_indices.append(idx)
                available[idx] = False
                # Mark neighbors
                neigh = adj_list[idx]
                available[neigh] = False
            
            current_size = len(selected_indices)
            if current_size > best_greedy_size:
                best_greedy_size = current_size
                best_greedy_set = set(selected_indices)
                
        # --- Phase 2: Branch and Bound (Maximum Independent Set) ---
        # We use the greedy set as the initial lower bound.
        # We will implement a recursive B&B with:
        # 1. Reduction: If a node has degree 0, it must be in the MIS.
        # 2. Reduction: If a node has degree 1, we can safely add it and remove its neighbor?
        #    Actually, if u is connected only to v, either u or v must be in the set.
        #    If we pick u, v is out. If we pick v, u is out.
        #    Standard reduction: If deg(u) <= deg(v) and N(u) \ {v} is subset of N(v), then there is a max independent set containing u.
        #    This is "domination" reduction.
        
        # For simplicity and speed in Python, let's implement a basic B&B with:
        # - Bitsets for available nodes? N=1500 fits in 24 64-bit integers.
        # - Or just lists/sets. Sets are slower.
        # - Let's use integer bitmasks if possible? Python ints are arbitrary precision.
        #   N=1500 is a 1500-bit integer. Operations are fast in C under the hood.
        
        # Let's try bitmask approach for the B&B core.
        
        # Initialize bitmask for all nodes
        # We need to map node indices to bits.
        # Since n_sub can be ~1500-7000, Python int bit ops are efficient.
        
        # However, constructing the adjacency bitmasks might be heavy.
        # adj_mask[i] is an int with bits set for neighbors of i.
        
        adj_masks = []
        for i in range(n_sub):
            mask = 0
            for j in adj_list[i]:
                mask |= (1 << j)
            adj_masks.append(mask)
            
        all_mask = (1 << n_sub) - 1
        
        # Current best solution size
        best_size = best_greedy_size
        best_set_bits = 0
        # Reconstruct best_set_bits from best_greedy_set
        for i in best_greedy_set:
            best_set_bits |= (1 << i)
            
        # Recursive B&B
        # State: available_mask, current_set_size, current_set_bits
        # We want to maximize current_set_size.
        
        # Pruning:
        # 1. If current_set_size + popcount(available_mask) <= best_size, return.
        # 2. If available_mask == 0, update best if current_set_size > best_size.
        
        # Heuristic for branching:
        # Pick a node v from available_mask.
        # Branch 1: Include v. New available = available & ~adj_masks[v] & ~(1<<v). Size + 1.
        # Branch 2: Exclude v. New available = available & ~(1<<v). Size + 0.
        
        # To improve pruning, we can use a greedy upper bound on the available set.
        # Greedy UB: While available: pick min degree node (in available subgraph), remove it and its neighbors. Count picks.
        # This is expensive to compute at every node.
        # Alternative: Simple bound is popcount.
        # Let's stick to popcount for now, but maybe add a simple greedy bound if time permits.
        
        # To make B&B effective, we need good node ordering.
        # We should branch on nodes with high degree? Or low degree?
        # Usually, high degree nodes are more restrictive, so branching on them might prune more.
        # But for MIS, we often branch on a node, include it (removes neighbors), or exclude it.
        
        # Let's implement a simple recursive B&B.
        # Python recursion depth might be an issue for N=1500.
        # We need to set recursion limit or use iterative approach.
        # sys.setrecursionlimit(10000)
        
        sys.setrecursionlimit(10000)
        
        # Helper to count bits
        def popcount(x):
            return bin(x).count('1')
            
        # We will use a global variable for best_size to avoid passing it around, 
        # but we need to be careful with thread safety (not an issue here) and clarity.
        # Better to pass it or use nonlocal.
        
        best_size_nonlocal = best_size
        best_set_bits_nonlocal = best_set_bits
        
        def bb(available_mask, current_size, current_bits):
            nonlocal best_size_nonlocal, best_set_bits_nonlocal
            
            # Pruning
            # Upper bound: current_size + number of nodes in available
            # A better upper bound is greedy independent set size on available subgraph.
            # Let's compute a simple greedy bound.
            # This is O(N^2) worst case per call, which is too slow.
            # Let's stick to popcount for now, but it's weak.
            
            # If we can't beat best, return
            if current_size + popcount(available_mask) <= best_size_nonlocal:
                return
                
            if available_mask == 0:
                if current_size > best_size_nonlocal:
                    best_size_nonlocal = current_size
                    best_set_bits_nonlocal = current_bits
                return
                
            # Reduction: If there is a node with degree 0 in the available subgraph, it must be in the set.
            # We need to check degrees in the *available* subgraph.
            # This is expensive to compute dynamically.
            # Let's skip dynamic reduction for now and rely on branching order and pruning.
            
            # Pick a node to branch on.
            # Heuristic: Pick the node with the highest degree in the available subgraph?
            # Or lowest?
            # Let's pick the first available node for simplicity, but ordered by global degree.
            
            # Get list of available nodes
            # available_nodes = [i for i in range(n_sub) if (available_mask >> i) & 1]
            # This is O(N).
            
            # To speed up, we can maintain a list of available nodes?
            # Bitmask iteration:
            # while m:
            #   lsb = m & -m
            #   idx = lsb.bit_length() - 1
            #   m ^= lsb
            
            # Let's find the "best" node to branch on.
            # We'll pick the one with the highest number of neighbors in the available set.
            # This requires iterating all available nodes and counting neighbors.
            
            best_node = -1
            max_neigh_count = -1
            
            m = available_mask
            # Limit iteration to first 50 nodes for speed? No, we need the best one.
            # But N=1500. Iterating 1500 nodes and doing bit ops is fast.
            
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                m ^= lsb
                
                # Count neighbors in available
                # neigh_in_avail = adj_masks[idx] & available_mask
                # count = popcount(neigh_in_avail)
                # We can optimize popcount? bin(x).count('1') is C implementation, fast.
                
                neigh_count = popcount(adj_masks[idx] & available_mask)
                if neigh_count > max_neigh_count:
                    max_neigh_count = neigh_count
                    best_node = idx
                    
            if best_node == -1:
                return
                
            bit = 1 << best_node
            
            # Branch 1: Include best_node
            # New available: remove best_node and its neighbors
            new_avail_1 = available_mask & ~(adj_masks[best_node] | bit)
            bb(new_avail_1, current_size + 1, current_bits | bit)
            
            # Check time
            if time.time() > deadline - 5:
                return
                
            # Branch 2: Exclude best_node
            # New available: remove best_node
            new_avail_2 = available_mask & ~bit
            bb(new_avail_2, current_size, current_bits)
            
        # Run B&B
        # We start with all nodes available
        bb(all_mask, 0, 0)
        
        if best_size_nonlocal > best_size_global:
            best_size_global = best_size_nonlocal
            # Extract points from best_set_bits
            selected_indices = []
            m = best_set_bits_nonlocal
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                m ^= lsb
                selected_indices.append(idx)
                
            best_set_global = [sub_cands[i] for i in selected_indices]
            
    # Verify and return
    if best_size_global >= 2:
        result = np.array(best_set_global, dtype=np.int64)
        max_norm = np.max(np.sum(result**2, axis=1))
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                d2 = np.dot(result[i] - result[j], result[i] - result[j])
                if d2 < max_norm:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            return result
        
    # Fallback
    return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

if __name__ == "__main__":
    res = kissing_number11()
    print(f"Size: {len(res)}")

