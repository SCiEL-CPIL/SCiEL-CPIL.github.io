"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import itertools
from collections import deque

D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
        
    # Compute conflict matrix: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists as arrays
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def spectral_init():
        """Spectral initialization using principal eigenvector of conflict graph."""
        # Build sparse adjacency matrix
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        
        # Create edge list
        rows = []
        cols = []
        for i in range(N4):
            nbrs = neighbors[i]
            if len(nbrs) > 0:
                rows.extend([i] * len(nbrs))
                cols.extend(nbrs.tolist())
        
        if len(rows) == 0:
            return np.ones(N4, dtype=bool)
        
        adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N4, N4))
        
        # Get principal eigenvector
        try:
            # Use k=1 to get the largest eigenvalue/eigenvector
            # sigma=0.5 for shift-invert to speed up
            eigvals, eigvecs = eigsh(adj, k=1, sigma=0.5, which='LM', v0=np.ones(N4))
            principal_evec = np.abs(eigvecs[:, 0])
        except:
            # Fallback to greedy if spectral fails
            return greedy_min_degree()
        
        # Select vertices in order of decreasing principal eigenvector component
        order = np.argsort(-principal_evec)
        
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        
        for v in order:
            if alive[v]:
                in_set[v] = True
                to_remove = np.unique(np.concatenate(([v], neighbors[v])))
                alive[to_remove] = False
                if time.time() > deadline - 5:
                    break
        
        return in_set
    
    def sa_local_search(in_set_init, max_time=10.0):
        """Min-conflict simulated annealing with 1-for-0, 2-for-1, and 2-for-2 swaps."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        T = 2.0
        cooling = 0.999
        min_T = 0.01
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            move_type = rng.random()
            
            if move_type < 0.6:
                # Single vertex addition (1-for-0)
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    stuck_counter += 1
                    T *= cooling
                    if T < min_T: T = min_T
                    continue
                    
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                
                k = min_count
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
                else:
                    accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    count_conflicts[neighbors[v]] += 1
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
                    
            elif move_type < 0.85:
                # 2-for-1 swap: remove one selected, add two non-selected
                selected = np.where(in_set)[0]
                if len(selected) == 0:
                    stuck_counter += 1
                    T *= cooling
                    if T < min_T: T = min_T
                    continue
                    
                remove_v = selected[rng.integers(len(selected))]
                
                in_set[remove_v] = False
                count_conflicts[neighbors[remove_v]] -= 1
                current_size -= 1
                
                added = []
                for _ in range(2):
                    not_in_set_now = np.where(~in_set)[0]
                    if len(not_in_set_now) == 0:
                        break
                    counts_now = count_conflicts[not_in_set_now]
                    min_count_now = np.min(counts_now)
                    if min_count_now > 0:
                        break
                    min_cands_now = not_in_set_now[counts_now == min_count_now]
                    add_v = min_cands_now[rng.integers(len(min_cands_now))]
                    in_set[add_v] = True
                    count_conflicts[neighbors[add_v]] += 1
                    current_size += 1
                    added.append(add_v)
                
                if len(added) < 2:
                    in_set[remove_v] = True
                    count_conflicts[neighbors[remove_v]] += 1
                    current_size += 1
                    for u in added:
                        in_set[u] = False
                        count_conflicts[neighbors[u]] -= 1
                        current_size -= 1
                    stuck_counter += 1
                else:
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                        
            else:
                # 2-for-2 swap: remove two selected, add two non-selected
                selected = np.where(in_set)[0]
                if len(selected) < 2:
                    stuck_counter += 1
                    T *= cooling
                    if T < min_T: T = min_T
                    continue
                    
                idx1, idx2 = rng.choice(len(selected), size=2, replace=False)
                rem1, rem2 = selected[idx1], selected[idx2]
                
                in_set[rem1] = False
                in_set[rem2] = False
                count_conflicts[neighbors[rem1]] -= 1
                count_conflicts[neighbors[rem2]] -= 1
                current_size -= 2
                
                added = []
                for _ in range(2):
                    not_in_set_now = np.where(~in_set)[0]
                    if len(not_in_set_now) == 0:
                        break
                    counts_now = count_conflicts[not_in_set_now]
                    min_count_now = np.min(counts_now)
                    if min_count_now > 0:
                        break
                    min_cands_now = not_in_set_now[counts_now == min_count_now]
                    add_v = min_cands_now[rng.integers(len(min_cands_now))]
                    in_set[add_v] = True
                    count_conflicts[neighbors[add_v]] += 1
                    current_size += 1
                    added.append(add_v)
                
                if len(added) < 2:
                    in_set[rem1] = True
                    in_set[rem2] = True
                    count_conflicts[neighbors[rem1]] += 1
                    count_conflicts[neighbors[rem2]] += 1
                    current_size += 2
                    for u in added:
                        in_set[u] = False
                        count_conflicts[neighbors[u]] -= 1
                        current_size -= 1
                    stuck_counter += 1
                else:
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
            
            if stuck_counter > 1000:
                T = max(T, 1.5)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    # Multiple restarts
    n_restarts = 0
    max_restarts = 50
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        # Alternate between greedy and spectral initialization
        if n_restarts % 2 == 0:
            in_set = greedy_min_degree()
        else:
            in_set = spectral_init()
        
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # SA Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.5, 12.0)
        if sa_time < 2:
            sa_time = 2.0
            
        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        # Second SA pass on best found so far (polish)
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 5 and sa_size > current_size:
            polish_time = min(remaining_time * 0.4, 8.0)
            if polish_time > 2:
                polish_result, polish_size = sa_local_search(sa_result, max_time=polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()
    
    # Final deterministic polish: try to add any vertex with zero conflicts
    final_in_set = best_in_set.copy()
    polished = True
    while polished and time.time() < deadline - 2:
        polished = False
        not_in_set = np.where(~final_in_set)[0]
        if len(not_in_set) == 0:
            break
        for v in not_in_set:
            if not np.any(conflicts[v] & final_in_set):
                final_in_set[v] = True
                polished = True
                break
    
    final_v4_indices = np.where(final_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
