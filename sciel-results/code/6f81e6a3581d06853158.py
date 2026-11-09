# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 110

    def elapsed_ok():
        return (time.time() - start_time) < DEADLINE

    D = 11
    candidates = []

    # Generate all integer vectors with norm squared 8
    # Type 1: two +/-2, rest 0
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                candidates.append(v)

    # Type 2: eight +/-1, rest 0
    for support in itertools.combinations(range(D), 8):
        for mask in range(256):
            v = [0] * D
            for idx, pos in enumerate(support):
                if mask & (1 << idx):
                    v[pos] = -1
                else:
                    v[pos] = 1
            candidates.append(v)

    # Type 3: one +/-2, four +/-1, rest 0
    for pos_2 in range(D):
        for pos_set_1 in itertools.combinations([p for p in range(D) if p != pos_2], 4):
            for s2 in (1, -1):
                for mask in range(16):
                    v = [0] * D
                    v[pos_2] = 2 * s2
                    for idx, pos in enumerate(pos_set_1):
                        if mask & (1 << idx):
                            v[pos] = -1
                        else:
                            v[pos] = 1
                    candidates.append(v)
                        
    best_count = 0
    best_arr = np.array([], dtype=np.int64).reshape(0, D)
    
    # Try multiple random shuffles to find a better packing
    for seed in range(10): # Try 10 different orders
        if not elapsed_ok():
            break
        random.seed(seed)
        shuffled = candidates.copy()
        random.shuffle(shuffled)
        
        vectors = []
        arr = np.array([], dtype=np.int64).reshape(0, D)
        
        for v in shuffled:
            if not elapsed_ok():
                break
            v_arr = np.array(v, dtype=np.int64)
            if len(arr) == 0:
                vectors.append(v)
                arr = v_arr.reshape(1, D)
            else:
                dots = arr @ v_arr
                if np.all(dots <= 4):
                    vectors.append(v)
                    arr = np.vstack([arr, v_arr])
                    
        if len(vectors) > best_count:
            best_count = len(vectors)
            best_arr = np.array(vectors, dtype=np.int64)
            
    return best_arr
# EVOLVE-BLOCK-END
