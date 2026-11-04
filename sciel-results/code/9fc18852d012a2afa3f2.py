import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)
    
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    
    conflict = np.zeros((N_S, N_S), dtype=bool)
    adj = [set() for _ in range(N_S)]
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = support_masks[i] & support_masks[j]
            if inter.bit_count() >= 3:
                conflict[i, j] = True
                conflict[j, i] = True
                adj[i].add(j)
                adj[j].add(i)
    
    # Convert adjacency sets to sorted lists for faster iteration
    adj_lists = [sorted(adj[i]) for i in range(N_S)]
    
    rng = np.random.RandomState(42)
    
    def greedy_fill(x, blocked):
        changed = True
        while changed:
            changed = False
            for v in range(N_S):
                if not x[v] and not blocked[v]:
                    x[v] = True
                    for u in adj_lists[v]:
                        blocked[u] = True
                    changed = True
        return x, blocked

    def local_search(x, max_iter=10000, time_limit=None):
        current_x = x.copy()
        current_score = int(np.sum(current_x))
        
        blocked = np.zeros(N_S, dtype=bool)
        for v in np.where(current_x)[0]:
            for u in adj_lists[v]:
                blocked[u] = True
        
        current_x, blocked = greedy_fill(current_x, blocked)
        current_score = int(np.sum(current_x))
        
        best_x = current_x.copy()
        best_score = current_score
        
        T = 5.0
        cooling = 0.9995
        
        for it in range(max_iter):
            if time_limit is not None and time.time() > time_limit:
                break
            
            r = rng.random()
            
            if r < 0.3:
                n_remove = 1 + rng.randint(0, 2)
                in_set = np.where(current_x)[0]
                if len(in_set) >= n_remove:
                    to_remove = in_set[rng.choice(len(in_set), size=n_remove, replace=False)]
                    for v in to_remove:
                        current_x[v] = False
                    blocked = np.zeros(N_S, dtype=bool)
                    for u in np.where(current_x)[0]:
                        for w in adj_lists[u]:
                            blocked[w] = True
                    current_x, blocked = greedy_fill(current_x, blocked)
                    current_score = int(np.sum(current_x))
                    if current_score > best_score:
                        best_score = current_score
                        best_x = current_x.copy()
                        
            elif r < 0.6:
                in_set = np.where(current_x)[0]
                if len(in_set) == 0:
                    continue
                u = in_set[rng.randint(len(in_set))]
                temp_x = current_x.copy()
                temp_x[u] = False
                temp_blocked = np.zeros(N_S, dtype=bool)
                for w in np.where(temp_x)[0]:
                    for z in adj_lists[w]:
                        temp_blocked[z] = True
                candidates = [v for v in range(N_S) if not current_x[v] and not temp_blocked[v]]
                if candidates:
                    v = candidates[rng.randint(len(candidates))]
                    current_x[u] = False
                    current_x[v] = True
                    blocked = np.zeros(N_S, dtype=bool)
                    for w in np.where(current_x)[0]:
                        for z in adj_lists[w]:
                            blocked[z] = True
                    current_x, blocked = greedy_fill(current_x, blocked)
                    current_score = int(np.sum(current_x))
                    if current_score > best_score:
                        best_score = current_score
                        best_x = current_x.copy()
                        
            else:
                in_set = np.where(current_x)[0]
                if len(in_set) < 2:
                    continue
                u1_idx, u2_idx = rng.choice(len(in_set), size=2, replace=False)
                u1, u2 = in_set[u1_idx], in_set[u2_idx]
                temp_x = current_x.copy()
                temp_x[u1] = False
                temp_x[u2] = False
                temp_blocked = np.zeros(N_S, dtype=bool)
                for w in np.where(temp_x)[0]:
                    for z in adj_lists[w]:
                        temp_blocked[z] = True
                candidates = [v for v in range(N_S) if not current_x[v] and not temp_blocked[v]]
                if len(candidates) >= 2:
                    v1_idx, v2_idx = rng.choice(len(candidates), size=2, replace=False)
                    v1, v2 = candidates[v1_idx], candidates[v2_idx]
                    if not conflict[v1, v2]:
                        current_x[u1] = False
                        current_x[u2] = False
                        current_x[v1] = True
                        current_x[v2] = True
                        blocked = np.zeros(N_S, dtype=bool)
                        for w in np.where(current_x)[0]:
                            for z in adj_lists[w]:
                                blocked[z] = True
                        current_x, blocked = greedy_fill(current_x, blocked)
                        current_score = int(np.sum(current_x))
                        if current_score > best_score:
                            best_score = current_score
                            best_x = current_x.copy()
            
            T *= cooling
        
        return best_x

    best_support_set = set()
    best_size = 0
    
    num_restarts = 60
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        if r < 15:
            x0 = np.zeros(N_S, dtype=bool)
            blocked = np.zeros(N_S, dtype=bool)
            remaining = list(range(N_S))
            rng_local = np.random.RandomState(42 + r)
            while remaining:
                candidates = [v for v in remaining if not blocked[v]]
                if not candidates:
                    break
                v = candidates[rng_local.randint(len(candidates))]
                x0[v] = True
                for u in adj_lists[v]:
                    blocked[u] = True
                remaining = [v for v in remaining if not blocked[v] and not x0[v]]
        else:
            x0 = np.zeros(N_S, dtype=bool)
            order = rng.permutation(N_S)
            blocked = np.zeros(N_S, dtype=bool)
            for v in order:
                if not x0[v] and not blocked[v]:
                    x0[v] = True
                    for u in adj_lists[v]:
                        blocked[u] = True
        
        remaining_time = deadline - time.time()
        max_iter = min(10000, max(2000, int(remaining_time * 200)))
        
        x_best = local_search(x0, max_iter=max_iter, time_limit=deadline - 5.0)
        size = int(np.sum(x_best))
        
        if size > best_size:
            best_size = size
            best_support_set = set(np.where(x_best)[0].tolist())

    if time.time() < deadline - 3.0 and best_support_set:
        polish_x = set(best_support_set)
        changed = True
        while changed and time.time() < deadline - 3.0:
            changed = False
            for v in range(N_S):
                if v in polish_x:
                    continue
                compatible = all(not conflict[v, u] for u in polish_x)
                if compatible:
                    polish_x.add(v)
                    changed = True
        if len(polish_x) > best_size:
            best_size = len(polish_x)
            best_support_set = polish_x

    vectors = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors.append(v2)
        
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)
            
    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)
        
    return np.array(vectors, dtype=np.int8)
