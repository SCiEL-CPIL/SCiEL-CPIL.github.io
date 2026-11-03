import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_vectors = []
    best_score = 0
    
    def generate_vectors(w):
        vecs = []
        for idxs in itertools.combinations(range(D), w):
            for signs in itertools.product([1, -1], repeat=w):
                v = [0] * D
                for i, s in zip(idxs, signs):
                    v[i] = s
                vecs.append(v)
        return vecs

    def solve_greedy(candidates, w, shuffle_seed=None):
        if not candidates:
            return np.array([], dtype=np.int64)
        
        threshold = w // 2
        n = len(candidates)
        
        # Shuffle candidates
        indices = list(range(n))
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
            random.shuffle(indices)
        else:
            random.shuffle(indices)
            
        M = np.array([candidates[i] for i in indices], dtype=np.int8)
        
        # Compute Gram matrix
        # M is (N, D). G = M @ M.T is (N, N)
        # D=11. N up to 15000.
        # Memory check for w=5: 15000^2 * 4 bytes = 900MB.
        # Let's limit to w=4 for safety, or check memory.
        # For w=4, N=5280. 5280^2 * 4 = 111MB. Safe.
        
        G = M @ M.T
        
        in_set = np.zeros(n, dtype=bool)
        
        for i in range(n):
            # Check if i conflicts with any in_set
            # G[i, in_set] > threshold
            # If in_set is empty, G[i, in_set] is empty, any() is False.
            if not np.any(G[i, in_set] > threshold):
                in_set[i] = True
                
        return M[in_set]

    # Try w=3
    if time.time() < deadline - 20:
        cands_3 = generate_vectors(3)
        # Run a few times
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            res = solve_greedy(cands_3, 3, seed)
            score = len(res)
            if score > best_score:
                best_score = score
                best_vectors = res

    # Try w=4
    if time.time() < deadline - 30:
        cands_4 = generate_vectors(4)
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            res = solve_greedy(cands_4, 4, seed)
            score = len(res)
            if score > best_score:
                best_score = score
                best_vectors = res

    # Fallback
    if not best_vectors:
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        best_vectors = vectors

    return np.array(best_vectors, dtype=np.int64)
