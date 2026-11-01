import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 112

    # 1. Generate candidates
    # Weight 4 vectors
    v4_candidates = []
    support_indices = []
    for positions in itertools.combinations(range(D), 4):
        sup_idx = len(support_indices)
        support_indices.append(positions)
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
    
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    N4 = len(vectors_v4)
    N_sup = len(support_indices)
    
    # Weight 1 vectors
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    # 2. Build Support Graph
    # Node i in support graph corresponds to support_indices[i]
    # Edge if intersection size == 3
    supp_graph = np.zeros((N_sup, N_sup), dtype=bool)
    for i in range(N_sup):
        set_i = set(support_indices[i])
        for j in range(i+1, N_sup):
            set_j = set(support_indices[j])
            if len(set_i.intersection(set_j)) == 3:
                supp_graph[i, j] = True
                supp_graph[j, i] = True
                
    # 3. Greedy Independent Set on Support Graph
    # This gives a set of supports that don't 3-intersect each other.
    # We can take all 16 vectors from these supports.
    
    def greedy_supp_init(seed=0):
        rng = np.random.default_rng(seed)
        in_set = np.zeros(N_sup, dtype=bool)
        alive = np.ones(N_sup, dtype=bool)
        degrees = supp_graph.sum(axis=1).astype(np.int32)
        
        while np.any(alive):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0: break
            degs = degrees[alive_indices]
            min_deg = np.min(degs)
            cands = alive_indices[degs == min_deg]
            chosen = cands[rng.integers(len(cands))]
            in_set[chosen] = True
            nbrs = np.where(supp_graph[chosen] & alive)[0]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= supp_graph[rem][:, to_remove].sum(axis=1)
                
        return in_set

    # 4. Build Full Conflict Graph
    # Conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbors
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    
    # 5. Initialize Full Set from Support Init
    def supp_to_full(supp_in_set):
        full_in_set = np.zeros(N4, dtype=bool)
        # Map support index to vector indices
        # Vectors are grouped by 16
        for s_idx in range(N_sup):
            if supp_in_set[s_idx]:
                start = s_idx * 16
                end = start + 16
                full_in_set[start:end] = True
        return full_in_set

    best_full_set = np.zeros(N4, dtype=bool)
    best_size = 0
    
    # Try a few support inits
    for seed in range(10):
        supp_set = greedy_supp_init(seed)
        full_set = supp_to_full(supp_set)
        size = int(full_set.sum())
        if size > best_size:
            best_size = size
            best_full_set = full_set.copy()
            
    # 6. SA Local Search on Full Graph
    rng = np.random.default_rng(12345)
    
    def sa_local_search(in_set_init, max_time=5.0, T_start=1.0):
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        T = T_start
        cooling = 0.999
        min_T = 0.01
        
        count_conflicts = np.zeros(N4, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors[i]] += 1
            
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        stuck = 0
        
        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0: break
            
            if rng.random() < 0.5:
                # Add
                counts = count_conflicts[not_in_set]
                min_c = np.min(counts)
                cands = not_in_set[counts == min_c]
                v = cands[rng.integers(len(cands))]
                k = int(min_c)
                delta = 1 - k
                if delta > 0 or rng.random() < np.exp(delta/T):
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    count_conflicts[neighbors[v]] += 1
            else:
                # Remove
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) == 0: break
                v = in_set_indices[rng.integers(len(in_set_indices))]
                delta = -1
                if rng.random() < np.exp(delta/T):
                    in_set[v] = False
                    current_size += delta
                    count_conflicts[neighbors[v]] -= 1
                    
            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()
                stuck = 0
            else:
                stuck += 1
            if stuck > 1000:
                T = max(T, 1.0)
                stuck = 0
            T = max(min_T, T * cooling)
            
        return local_best_in_set, local_best_size

    # Run SA from best supp init and some random inits
    remaining = deadline - 10 - time.time()
    if remaining > 0:
        # SA from best
        sa_res, sa_size = sa_local_search(best_full_set, max_time=min(remaining*0.5, 10.0))
        if sa_size > best_size:
            best_size = sa_size
            best_full_set = sa_res.copy()
            
        # SA from random
        for _ in range(3):
            if time.time() > deadline - 5: break
            rand_supp = np.random.default_rng().integers(0, 2, size=N_sup).astype(bool)
            # This is a bad init, but SA can fix it?
            # Better: random greedy on full graph
            in_set = np.zeros(N4, dtype=bool)
            alive = np.ones(N4, dtype=bool)
            degs = np.array([len(n) for n in neighbors], dtype=np.int32)
            while np.any(alive) and time.time() < deadline - 5:
                alive_idx = np.where(alive)[0]
                if len(alive_idx)==0: break
                d = degs[alive_idx]
                min_d = np.min(d)
                c = alive_idx[d==min_d]
                v = c[rng.integers(len(c))]
                in_set[v] = True
                nbrs = neighbors[v]
                to_rem = np.unique(np.concatenate(([v], nbrs)))
                alive[to_rem] = False
                rem = np.where(alive)[0]
                if len(rem)>0:
                    degs[rem] -= conflicts[rem][:, to_rem].sum(axis=1)
                    
            if time.time() > deadline - 5: break
            sa_res, sa_size = sa_local_search(in_set, max_time=min(remaining*0.2, 5.0))
            if sa_size > best_size:
                best_size = sa_size
                best_full_set = sa_res.copy()

    # Final result
    final_v4 = vectors_v4[best_full_set]
    result = np.vstack([final_v4, vectors_v1])
    return result
