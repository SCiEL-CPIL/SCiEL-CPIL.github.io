import random
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer

    # Helper to calculate penalty (number of regular induced subgraphs of size 6)
    # Using bitmasks for speed
    def calculate_penalty_size6(adj, subsets):
        count = 0
        for subset in subsets:
            # Check if induced subgraph on 'subset' is regular
            # Degrees of vertices in 'subset' within the induced subgraph
            degs = []
            for u in subset:
                # Mask of neighbors of u
                # adj[u] is the bitmask of all neighbors
                # We need neighbors that are in 'subset'
                # We precomputed a bitmask for each subset? 
                # It's faster to compute mask on the fly if subsets are small, 
                # but precomputing is better.
                # Let's precompute subset_masks.
                pass
        return count

    # Precomputation for a given n
    def get_subsets_and_masks(n, k):
        subsets = []
        masks = []
        for comb in combinations(range(n), k):
            subsets.append(comb)
            mask = 0
            for v in comb:
                mask |= (1 << v)
            masks.append(mask)
        return subsets, masks

    # Full validity check for a graph (adj list of ints)
    # Checks all subsets of size >= 6
    def is_valid_full(adj, n):
        if n < 6:
            return True
        
        # Generate all subsets of size k from 6 to n
        # This is expensive, so we do it once per n
        # We can cache this
        all_subsets = []
        all_masks = []
        for k in range(6, n + 1):
            subs, masks = get_subsets_and_masks(n, k)
            all_subsets.extend(subs)
            all_masks.extend(masks)
        
        for i, subset in enumerate(all_subsets):
            mask = all_masks[i]
            degs = []
            for u in subset:
                # Count neighbors of u that are in the subset
                # adj[u] & mask gives the neighbors of u that are in the subset
                deg = (adj[u] & mask).bit_count()
                degs.append(deg)
            if len(set(degs)) == 1:
                return False
        return True

    # Fast penalty for search (size 6 only)
    def calculate_penalty_fast(adj, subsets_6, masks_6):
        count = 0
        for i in range(len(subsets_6)):
            subset = subsets_6[i]
            mask = masks_6[i]
            # Optimization: Early exit if not regular?
            # Degrees must be equal.
            # First vertex degree
            d0 = (adj[subset[0]] & mask).bit_count()
            is_reg = True
            for u in subset[1:]:
                if (adj[u] & mask).bit_count() != d0:
                    is_reg = False
                    break
            if is_reg:
                count += 1
        return count

    # Local Search
    def local_search(n, max_iters):
        # Precompute subsets of size 6 for penalty
        subsets_6, masks_6 = get_subsets_and_masks(n, 6)
        
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        current_penalty = calculate_penalty_fast(adj, subsets_6, masks_6)
        best_penalty = current_penalty
        best_adj = adj[:]
        
        T = 1.0
        cooling = 0.99
        
        for t in range(max_iters):
            if time.time() - start_time > time_limit - 10:
                break
                
            # Pick random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Flip edge (u, v)
            if adj[u] & (1 << v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            new_penalty = calculate_penalty_fast(adj, subsets_6, masks_6)
            
            delta = new_penalty - current_penalty
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_penalty = new_penalty
                if new_penalty < best_penalty:
                    best_penalty = new_penalty
                    best_adj = adj[:]
            else:
                # Revert
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            T *= cooling
            if T < 0.01:
                T = 0.01
        
        return best_adj, best_penalty

    # Try n=21, then 20, etc.
    for n in [21, 20, 19]:
        if time.time() - start_time > time_limit - 20:
            break
            
        # Run local search multiple times or one long search
        # Let's run a search for ~30s
        search_start = time.time()
        max_iters = 5000 # Estimate time per iter
        
        # We need to estimate time per iteration to set max_iters
        # Let's just run until time runs out
        # But we need to return something.
        
        # Strategy: Run local search, keep track of best graph.
        # Periodically check if best graph is fully valid.
        
        subsets_6, masks_6 = get_subsets_and_masks(n, 6)
        
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        current_penalty = calculate_penalty_fast(adj, subsets_6, masks_6)
        best_penalty = current_penalty
        best_adj = adj[:]
        
        T = 1.0
        cooling = 0.999
        last_full_check = 0
        
        while time.time() - start_time < time_limit - 15:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Flip
            if adj[u] & (1 << v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            new_penalty = calculate_penalty_fast(adj, subsets_6, masks_6)
            delta = new_penalty - current_penalty
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_penalty = new_penalty
                if new_penalty < best_penalty:
                    best_penalty = new_penalty
                    best_adj = adj[:]
                    # Check full validity if penalty is 0
                    if best_penalty == 0:
                        if is_valid_full(best_adj, n):
                            # Found valid graph!
                            edges = []
                            for i in range(n):
                                for j in range(i + 1, n):
                                    if best_adj[i] & (1 << j):
                                        edges.append([i, j])
                            return {'n': n, 'edges': edges}
            else:
                # Revert
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            T *= cooling
            if T < 0.01:
                T = 0.01
                
            # Periodic full check on best_adj if penalty is low
            if best_penalty <= 1 and time.time() - last_full_check > 5:
                if is_valid_full(best_adj, n):
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if best_adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
                last_full_check = time.time()

    # Fallback: Return a known valid graph or small graph
    # K6 minus an edge is valid for n=6
    n = 6
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (i == 0 and j == 1):
                edges.append([i, j])
    return {'n': n, 'edges': edges}
