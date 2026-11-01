"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # 1. Generate candidates
    # Type 1: Weight 1, norm^2 = 4. Entries +/-2.
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    # Type 2: Weight 4, norm^2 = 4. Entries +/-1.
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    
    N1 = len(vectors_v1)
    N4 = len(vectors_v4)
    
    # 2. Precompute conflicts for v4
    # Conflict if dot > 2
    # dots = vectors_v4 @ vectors_v4.T
    # This is 5280x5280 matrix. ~28M entries. Fits in memory.
    dots = vectors_v4 @ vectors_v4.T
    conflicts_v4 = dots > 2
    np.fill_diagonal(conflicts_v4, False)
    
    # Neighbor lists for v4
    neighbors_v4 = [np.where(conflicts_v4[i])[0] for i in range(N4)]
    
    # v1 vectors do not conflict with v4 vectors (verified in thought process)
    # v1 vectors do not conflict with each other (verified in thought process)
    # So we can always include all v1 vectors.
    
    # 3. Find max independent set in v4 graph
    # Use greedy with random restarts and local search
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_independent_set(order):
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        for v in order:
            if not alive[v]:
                continue
            # Check if v conflicts with any in_set
            # neighbors[v] are the conflicts
            # We need to check if any neighbor is in in_set
            # This is slow if done naively.
            # Better: maintain a set of "blocked" nodes.
            # But for simplicity and speed in Python, we might need numpy tricks.
            # Let's stick to the previous approach but optimized.
            pass
        
        # The previous program's greedy was:
        # pick min degree, remove neighbors.
        # I will use a randomized greedy: pick random node, add, remove neighbors.
        # Repeat until no nodes left.
        
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        
        # Precompute degrees
        degrees = np.array([len(n) for n in neighbors_v4], dtype=np.int32)
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            # Pick node with min degree among alive
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors_v4[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            # Update degrees
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # degrees[rem] -= number of neighbors in to_remove
                # conflicts[rem][:, to_remove].sum(axis=1)
                degrees[rem] -= conflicts_v4[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def local_search(in_set_init, max_time=5.0):
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        
        # count_conflicts[v] = number of neighbors of v in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors_v4[i]] += 1
            
        sa_deadline = time.time() + max_time
        T = 2.0
        cooling = 0.999
        min_T = 0.1
        stuck = 0
        
        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
                
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
                to_remove = np.where(conflicts_v4[v] & in_set)[0]
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors_v4[u]] -= 1
                in_set[v] = True
                current_size += delta
                count_conflicts[neighbors_v4[v]] += 1
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck = 0
                else:
                    stuck += 1
            else:
                stuck += 1
                
            if stuck > 1000:
                T = max(T, 2.0)
                stuck = 0
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size

    n_restarts = 0
    max_restarts = 20
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        in_set = greedy_independent_set(None)
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
            
        # Local search
        remaining = deadline - 10 - time.time()
        if remaining > 2:
            sa_result, sa_size = local_search(in_set, max_time=min(remaining, 8.0))
            if sa_size > best_size:
                best_size = sa_size
                best_in_set = sa_result.copy()
                
    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
