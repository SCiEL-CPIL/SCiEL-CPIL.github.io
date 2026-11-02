import numpy as np
import itertools
import time

def construct():
        start_time = time.time()
        deadline = start_time + 115

        D = 11
        
        # 1. Generate Vectors
        vectors = []
        norms_sq = []
        
        # Type A: Weight 1, val +/-2. Norm^2 = 4
        for i in range(D):
            for s in [1, -1]:
                v = np.zeros(D, dtype=np.int64)
                v[i] = 2 * s
                vectors.append(v)
                norms_sq.append(4)
        
        # Type B: Weight 2, val +/-1. Norm^2 = 2
        for i in range(D):
            for j in range(i+1, D):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = s1
                        v[j] = s2
                        vectors.append(v)
                        norms_sq.append(2)
                        
        # Type C: Weight 4, val +/-1. Norm^2 = 4
        for positions in itertools.combinations(range(D), 4):
            for signs in itertools.product([1, -1], repeat=4):
                v = np.zeros(D, dtype=np.int64)
                for pos, s in zip(positions, signs):
                    v[pos] = s
                vectors.append(v)
                norms_sq.append(4)
                
        V = np.array(vectors)
        N = len(V)
        norms = np.array(norms_sq)
        
        # 2. Compute Conflict Graph
        # Dot products
        dots = V @ V.T
        
        # Thresholds
        # Conflict if dot > (n_i + n_j - 4) / 2
        # n_i + n_j - 4 can be:
        # 2+2-4 = 0 -> dot > 0
        # 2+4-4 = 2 -> dot > 1
        # 4+4-4 = 4 -> dot > 2
        
        # Create threshold matrix
        # T[i,j] = (norms[i] + norms[j] - 4) / 2
        # We only care if dots[i,j] > T[i,j]
        
        # To avoid huge memory for T, we can compute conflicts directly.
        # But N ~ 5500. N^2 is 30M. Float64 is 240MB. Might be too high?
        # Let's check memory. 5500*5500*8 bytes = 242 MB.
        # This might be close to limits.
        # We can compute in chunks or use int16 for dots?
        # Max dot product:
        # Type C dot Type C: max 4.
        # Type C dot Type A: max 2.
        # Type B dot Type B: max 2.
        # So dots are small integers.
        
        # Let's use int8 or int16 for dots to save memory.
        # V is int64. V @ V.T will be int64.
        # We can cast to int16 after computation?
        # Or compute block by block.
        
        # Let's try to compute the full matrix if possible, else block.
        # 242MB might be OK.
        
        dots = (V @ V.T).astype(np.int16)
        
        # Compute conflicts
        # T = (norms[:, None] + norms[None, :] - 4) // 2
        # Conflict = dots > T
        
        T = (norms[:, None] + norms[None, :] - 4) // 2
        conflicts = dots > T
        np.fill_diagonal(conflicts, False)
        
        # Convert to neighbor lists
        neighbors = [np.where(conflicts[i])[0] for i in range(N)]
        
        rng = np.random.default_rng(42)
        
        best_size = 0
        best_in_set = np.zeros(N, dtype=bool)
        
        def run_mis_search(max_time):
            in_set = np.zeros(N, dtype=bool)
            conf_counts = np.zeros(N, dtype=np.int32)
            
            # Greedy Init
            order = rng.permutation(N)
            for v in order:
                if conf_counts[v] == 0:
                    in_set[v] = True
                    conf_counts[neighbors[v]] += 1
                    
            current_size = in_set.sum()
            
            sa_deadline = time.time() + max_time
            T_temp = 1.0
            cooling = 0.9995
            
            while time.time() < sa_deadline:
                # 1. Try add free node
                free_nodes = np.where((~in_set) & (conf_counts == 0))[0]
                if len(free_nodes) > 0:
                    v = free_nodes[rng.integers(len(free_nodes))]
                    in_set[v] = True
                    conf_counts[neighbors[v]] += 1
                    current_size += 1
                    continue # restart loop to check for more free nodes
                
                # 2. Try swap
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    break
                    
                v = not_in_set[rng.integers(len(not_in_set))]
                nbrs_in_set = np.where(conflicts[v] & in_set)[0]
                
                if len(nbrs_in_set) == 0:
                    # Should be free, but conf_counts might be stale?
                    # Actually if conf_counts[v]==0 it would have been added above.
                    # So conf_counts[v] > 0.
                    # This means nbrs_in_set should not be empty.
                    continue
                
                # Pick a neighbor to remove
                u = nbrs_in_set[rng.integers(len(nbrs_in_set))]
                
                # Remove u
                in_set[u] = False
                conf_counts[neighbors[u]] -= 1
                current_size -= 1
                
                # Try to add v
                if conf_counts[v] == 0:
                    in_set[v] = True
                    conf_counts[neighbors[v]] += 1
                    current_size += 1
                else:
                    # Try to add another node that became free?
                    # Or revert?
                    # Reverting is safer to maintain size.
                    # But we might want to accept size decrease to escape?
                    # Let's try to find a free node now.
                    free_nodes = np.where((~in_set) & (conf_counts == 0))[0]
                    if len(free_nodes) > 0:
                        w = free_nodes[rng.integers(len(free_nodes))]
                        in_set[w] = True
                        conf_counts[neighbors[w]] += 1
                        current_size += 1
                    else:
                        # Revert
                        in_set[u] = True
                        conf_counts[neighbors[u]] += 1
                        current_size += 1
                        
                T_temp *= cooling
                if T_temp < 0.1:
                    T_temp = 0.1
                    
            return in_set, current_size

        n_restarts = 0
        while time.time() < deadline - 5:
            n_restarts += 1
            remaining = deadline - 5 - time.time()
            if remaining < 2:
                break
            t_alloc = min(remaining * 0.8, 15.0)
            
            in_set, size = run_mis_search(t_alloc)
            
            if size > best_size:
                best_size = size
                best_in_set = in_set.copy()
                
        final_indices = np.where(best_in_set)[0]
        return V[final_indices]
