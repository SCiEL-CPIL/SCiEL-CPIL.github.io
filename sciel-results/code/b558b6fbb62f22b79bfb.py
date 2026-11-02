import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # 1. Generate Candidates
    # W4: 2640 vectors
    v4_list = []
    for pos in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            if sum(signs) % 2 == 0:
                v = [0]*D
                for p, s in zip(pos, signs):
                    v[p] = s
                v4_list.append(v)
    v4 = np.array(v4_list, dtype=np.int64)
    N4 = len(v4)
    
    # W2: 22 vectors
    v2_list = []
    for pos in range(D):
        for s in [1, -1]:
            v = [0]*D
            v[pos] = 2*s
            v2_list.append(v)
    v2 = np.array(v2_list, dtype=np.int64)

    # 2. Build Conflict Graph for W4
    # Conflict if dot > 2
    dots = v4 @ v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    degrees = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)

    def greedy_min_degree():
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degs = degrees.copy()
        
        while np.any(alive):
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0: break
            min_d = np.min(degs[alive_idx])
            cands = alive_idx[degs[alive_idx] == min_d]
            v = cands[rng.integers(len(cands))]
            
            in_set[v] = True
            nbrs = neighbors[v]
            to_remove = np.unique(np.concatenate(([v], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degs[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def sa_local_search(init_set, max_time):
        in_set = init_set.copy()
        current_size = in_set.sum()
        
        # Precompute conflict counts for current set
        count_conf = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conf[neighbors[i]] += 1
            
        T = 3.0
        cooling = 0.9995
        min_T = 0.05
        stuck = 0
        local_best_size = current_size
        local_best_set = in_set.copy()
        
        t_start = time.time()
        while time.time() - t_start < max_time:
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0: break
            
            counts = count_conf[not_in_set]
            min_c = np.min(counts)
            cands = not_in_set[counts == min_c]
            v = cands[rng.integers(len(cands))]
            k = min_c
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
                        count_conf[neighbors[u]] -= 1
                        
                in_set[v] = True
                current_size += delta
                count_conf[neighbors[v]] += 1
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_set = in_set.copy()
                    stuck = 0
                else:
                    stuck += 1
            else:
                stuck += 1
                
            if stuck > 1500:
                T = max(T, 2.0)
                stuck = 0
            T = max(min_T, T * cooling)
            
        return local_best_set, local_best_size

    # 3. Multi-Start Search
    n_restarts = 0
    max_restarts = 100 # Will be limited by time
    
    while time.time() < deadline - 10:
        n_restarts += 1
        
        # Greedy Init
        init = greedy_min_degree()
        size_init = init.sum()
        
        if size_init > best_size:
            best_size = size_init
            best_in_set = init.copy()
            
        # SA Polish
        remaining = deadline - 10 - time.time()
        if remaining < 2: break
        
        sa_time = min(remaining * 0.7, 10.0) # Max 10s per SA
        if sa_time < 1.0: sa_time = 1.0
        
        sa_set, sa_size = sa_local_search(init, sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_set.copy()
            
        # Optional: Second SA pass if we improved significantly?
        # Inspiration 1 does a polish pass. Let's try if time permits.
        remaining = deadline - 5 - time.time()
        if remaining > 5 and sa_size > size_init:
             polish_time = min(remaining * 0.5, 5.0)
             if polish_time > 1.0:
                 pol_set, pol_size = sa_local_search(sa_set, polish_time)
                 if pol_size > best_size:
                     best_size = pol_size
                     best_in_set = pol_set.copy()

    # 4. Final Result
    # Combine best W4 set with ALL W2 vectors
    final_w4 = v4[best_in_set]
    result = np.vstack([final_w4, v2])
    
    return result
