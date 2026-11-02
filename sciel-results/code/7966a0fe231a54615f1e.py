import time
import numpy as np
from itertools import combinations

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def generate_R4():
        vecs = []
        for comb in combinations(range(D), 4):
            for signs in product([1, -1], repeat=4):
                v = np.zeros(D, dtype=np.int64)
                for idx, s in zip(comb, signs):
                    v[idx] = s
                vecs.append(v)
        return np.array(vecs)

    def generate_R5():
        vecs = []
        # Type 1: 2 and 1
        for i in range(D):
            for j in range(D):
                if i == j: continue
                for si in [1, -1]:
                    for sj in [1, -1]:
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = 2 * si
                        v[j] = sj
                        vecs.append(v)
        # Type 2: five 1s
        for comb in combinations(range(D), 5):
            for signs in product([1, -1], repeat=5):
                v = np.zeros(D, dtype=np.int64)
                for idx, s in zip(comb, signs):
                    v[idx] = s
                vecs.append(v)
        return np.array(vecs)

    def greedy_select(vectors, R, num_restarts=5):
        limit = R // 2
        best_sel = None
        best_len = 0
        
        for r in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
                
            np.random.shuffle(vectors)
            selected = []
            sel_arr = np.array([]).reshape(0, D)
            
            for v in vectors:
                if len(sel_arr) == 0:
                    selected.append(v)
                    sel_arr = v.reshape(1, D)
                else:
                    dots = sel_arr @ v
                    if np.max(dots) <= limit:
                        selected.append(v)
                        sel_arr = np.vstack((sel_arr, v))
                        
            if len(selected) > best_len:
                best_len = len(selected)
                best_sel = np.array(selected)
                
        return best_sel

    # Generate candidates
    V4 = generate_R4()
    V5 = generate_R5()
    
    best_result = None
    max_len = 0
    
    # Try R=4
    res4 = greedy_select(V4, 4)
    if res4 is not None and len(res4) > max_len:
        max_len = len(res4)
        best_result = res4
        
    # Try R=5
    if time.time() < deadline - 10.0:
        res5 = greedy_select(V5, 5)
        if res5 is not None and len(res5) > max_len:
            max_len = len(res5)
            best_result = res5
            
    if best_result is None:
        # Fallback to R=2
        vecs = []
        for i in range(D):
            for j in range(i+1, D):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        vecs.append(v)
        best_result = np.array(vecs)
        
    return best_result

from itertools import product
