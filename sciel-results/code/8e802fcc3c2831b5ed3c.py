import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    max_time = 115  # Safety margin
    
    # Try n from 20 down to 10
    for n in range(20, 9, -1):
        if time.time() - start_time > max_time:
            break
            
        # Try to find a valid graph of size n
        for restart in range(3):
            if time.time() - start_time > max_time:
                break
                
            # Random initialization
            A = np.random.rand(n, n) < 0.5
            np.fill_diagonal(A, 0)
            A = (A | A.T).astype(bool)
            
            # Local search
            steps_since_violation = 0
            for step in range(20000):
                if time.time() - start_time > max_time:
                    break
                    
                violation = None
                viol_size = 0
                
                # 1. Random size 6
                for _ in range(20):
                    S = np.random.choice(n, 6, replace=False)
                    sub_A = A[S, S]
                    degs = sub_A.sum(axis=1)
                    if degs.min() == degs.max():
                        violation = S
                        viol_size = 6
                        break
                
                # 2. Exhaustive size 6
                if violation is None:
                    for S in itertools.combinations(range(n), 6):
                        S_arr = np.array(S)
                        sub_A = A[S_arr, S_arr]
                        degs = sub_A.sum(axis=1)
                        if degs.min() == degs.max():
                            violation = S_arr
                            viol_size = 6
                            break
                
                if violation is not None:
                    u_idx, v_idx = np.random.choice(len(violation), 2, replace=False)
                    u = violation[u_idx]
                    v = violation[v_idx]
                    A[u, v] = not A[u, v]
                    A[v, u] = not A[v, u]
                    steps_since_violation = 0
                else:
                    steps_since_violation += 1
                    
                    if steps_since_violation > 100:
                        found_large_violation = False
                        for k in range(7, n + 1):
                            for _ in range(50):
                                S = np.random.choice(n, k, replace=False)
                                sub_A = A[S, S]
                                degs = sub_A.sum(axis=1)
                                if degs.min() == degs.max():
                                    violation = S
                                    viol_size = k
                                    found_large_violation = True
                                    break
                            if found_large_violation:
                                break
                        
                        if found_large_violation:
                            u_idx, v_idx = np.random.choice(len(violation), 2, replace=False)
                            u = violation[u_idx]
                            v = violation[v_idx]
                            A[u, v] = not A[u, v]
                            A[v, u] = not A[v, u]
                            steps_since_violation = 0
                        else:
                            if steps_since_violation > 500:
                                is_valid = True
                                for k in range(7, n + 1):
                                    for S in itertools.combinations(range(n), k):
                                        S_arr = np.array(S)
                                        sub_A = A[S_arr, S_arr]
                                        degs = sub_A.sum(axis=1)
                                        if degs.min() == degs.max():
                                            is_valid = False
                                            violation = S_arr
                                            break
                                    if not is_valid:
                                        break
                                
                                if is_valid:
                                    edges = []
                                    for i in range(n):
                                        for j in range(i + 1, n):
                                            if A[i, j]:
                                                edges.append([i, j])
                                    return {'n': n, 'edges': edges}
                                else:
                                    u_idx, v_idx = np.random.choice(len(violation), 2, replace=False)
                                    u = violation[u_idx]
                                    v = violation[v_idx]
                                    A[u, v] = not A[u, v]
                                    A[v, u] = not A[v, u]
                                    steps_since_violation = 0

    return {'n': 5, 'edges': []}
