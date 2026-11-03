"""Kissing number in dimension 11: structural reformulation as MIS on support graph + SA."""
import numpy as np
import itertools
import time
import random


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    D = 11

    # Generate all 4-subsets
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)  # 330
    
    # Build conflict graph: edge if |intersection| == 3
    adj = [set() for _ in range(n_sup)]
    for i in range(n_sup):
        s_i = set(supports[i])
        for j in range(i + 1, n_sup):
            s_j = set(supports[j])
            if len(s_i & s_j) == 3:
                adj[i].add(j)
                adj[j].add(i)
    
    # Precompute adjacency as lists for faster iteration
    adj_list = [list(s) for s in adj]
    
    # Greedy initial solution
    def greedy_is(seed=42):
        rng = random.Random(seed)
        remaining = set(range(n_sup))
        current_set = set()
        while remaining:
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0:
                        break
            current_set.add(best_v)
            to_remove = {best_v} | (adj[best_v] & remaining)
            remaining -= to_remove
        return current_set
    
    best_set = greedy_is(42)
    best_size = len(best_set)
    
    # Simulated annealing with multiple restarts
    np.random.seed(42)
    random.seed(42)
    
    S = best_set.copy()
    current_size = len(S)
    T = 2.0
    cooling = 0.9998
    
    step = 0
    max_steps = 300000
    while step < max_steps and time.time() < deadline - 30.0:
        step += 1
        v = random.randint(0, n_sup - 1)
        
        if v in S:
            # Try removing v
            if random.random() < 0.15:
                S.discard(v)
                current_size -= 1
            continue
        
        # v not in S
        N = adj[v] & S
        n_N = len(N)
        
        if n_N == 0:
            S.add(v)
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_set = S.copy()
        else:
            delta = 1 - n_N
            if delta >= 0:
                S -= N
                S.add(v)
                current_size += delta
                if current_size > best_size:
                    best_size = current_size
                    best_set = S.copy()
            else:
                # delta < 0
                if T > 0.01:
                    p = np.exp(delta / T)
                    if random.random() < p:
                        S -= N
                        S.add(v)
                        current_size += delta
                        if current_size > best_size:
                            best_size = current_size
                            best_set = S.copy()
        
        T = max(0.01, T * cooling)
    
    # Multiple restarts of greedy + SA
    for restart in range(30):
        if time.time() > deadline - 25.0:
            break
        np.random.seed(42 + restart)
        random.seed(42 + restart)
        
        # Randomized greedy
        remaining = set(range(n_sup))
        current_set = set()
        while remaining:
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0:
                        break
            current_set.add(best_v)
            to_remove = {best_v} | (adj[best_v] & remaining)
            remaining -= to_remove
        
        S = current_set.copy()
        current_size = len(S)
        T = 2.0
        cooling = 0.9998
        
        for step2 in range(80000):
            if time.time() > deadline - 25.0:
                break
            v = random.randint(0, n_sup - 1)
            
            if v in S:
                if random.random() < 0.15:
                    S.discard(v)
                    current_size -= 1
                continue
            
            N = adj[v] & S
            n_N = len(N)
            
            if n_N == 0:
                S.add(v)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_set = S.copy()
            else:
                delta = 1 - n_N
                if delta >= 0:
                    S -= N
                    S.add(v)
                    current_size += delta
                    if current_size > best_size:
                        best_size = current_size
                        best_set = S.copy()
                else:
                    if T > 0.01:
                        p = np.exp(delta / T)
                        if random.random() < p:
                            S -= N
                            S.add(v)
                            current_size += delta
                            if current_size > best_size:
                                best_size = current_size
                                best_set = S.copy()
            
            T = max(0.01, T * cooling)
    
    # Construct vectors from best_set: all 16 sign patterns for each chosen support
    vectors = []
    for idx in best_set:
        sup = supports[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in enumerate(sup):
                v[s] = signs[k]
            vectors.append(v)
    
    # Now try to add weight-3, weight-2, weight-1 vectors that are compatible with all
    # Generate weight-3 vectors
    weight3_candidates = []
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            weight3_candidates.append(v)
    
    # Generate weight-2 vectors
    weight2_candidates = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            weight2_candidates.append(v)
    
    # Generate weight-1 vectors
    weight1_candidates = []
    for pos in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[pos] = s
            weight1_candidates.append(v)
    
    # Check compatibility of each additional candidate with all existing vectors
    existing = np.array(vectors, dtype=np.int32)
    if len(existing) > 0:
        existing_norms_sq = np.sum(existing ** 2, axis=1)
        max_norm_sq = int(existing_norms_sq.max())
        
        # Try adding weight-3 vectors
        for w3 in weight3_candidates:
            if time.time() > deadline - 5.0:
                break
            w3_arr = np.array(w3, dtype=np.int32)
            w3_norm_sq = int(np.sum(w3_arr ** 2))
            # Check against all existing
            dots = existing @ w3_arr
            min_dist_sq = w3_norm_sq + existing_norms_sq - 2 * dots
            if np.all(min_dist_sq >= max_norm_sq):
                vectors.append(w3)
                existing = np.vstack([existing, w3_arr])
                existing_norms_sq = np.sum(existing ** 2, axis=1)
                max_norm_sq = int(existing_norms_sq.max())
        
        # Try adding weight-2 vectors
        for w2 in weight2_candidates:
            if time.time() > deadline - 3.0:
                break
            w2_arr = np.array(w2, dtype=np.int32)
            w2_norm_sq = int(np.sum(w2_arr ** 2))
            dots = existing @ w2_arr
            min_dist_sq = w2_norm_sq + existing_norms_sq - 2 * dots
            if np.all(min_dist_sq >= max_norm_sq):
                vectors.append(w2)
                existing = np.vstack([existing, w2_arr])
                existing_norms_sq = np.sum(existing ** 2, axis=1)
                max_norm_sq = int(existing_norms_sq.max())
        
        # Try adding weight-1 vectors
        for w1 in weight1_candidates:
            if time.time() > deadline - 2.0:
                break
            w1_arr = np.array(w1, dtype=np.int32)
            w1_norm_sq = int(np.sum(w1_arr ** 2))
            dots = existing @ w1_arr
            min_dist_sq = w1_norm_sq + existing_norms_sq - 2 * dots
            if np.all(min_dist_sq >= max_norm_sq):
                vectors.append(w1)
                existing = np.vstack([existing, w1_arr])
                existing_norms_sq = np.sum(existing ** 2, axis=1)
                max_norm_sq = int(existing_norms_sq.max())
    
    if not vectors:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)
    
    result = np.array(vectors, dtype=np.int64)
    return result
