import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    def get_edges(A):
        n = A.shape[0]
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i,j] == 1:
                    edges.append([i, j])
        return edges

    def is_regular(A, combo):
        # combo is a tuple of indices
        sub_A = A[np.ix_(combo, combo)]
        degs = sub_A.sum(axis=1)
        return np.all(degs == degs[0])

    def count_violations(A, n):
        count = 0
        # k=6
        for combo in itertools.combinations(range(n), 6):
            if is_regular(A, combo):
                count += 1
        if count > 0:
            return count
        # k=7..n
        for k in range(7, n+1):
            for combo in itertools.combinations(range(n), k):
                if is_regular(A, combo):
                    count += 1
        return count

    def find_bad_subset(A, n):
        for k in range(6, n+1):
            for combo in itertools.combinations(range(n), k):
                if is_regular(A, combo):
                    return combo
        return None

    def local_search(n, max_iters=100):
        A = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i+1, n):
                if np.random.rand() < 0.5:
                    A[i,j] = 1
                    A[j,i] = 1
        
        v = count_violations(A, n)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                return None, v
            
            if v == 0:
                return A, 0
            
            bad_combo = find_bad_subset(A, n)
            if bad_combo is None:
                return A, 0
            
            # Try flipping edges in bad_combo
            improved = False
            # Shuffle indices for randomness
            idxs = list(range(len(bad_combo)))
            np.random.shuffle(idxs)
            
            for i in range(len(idxs)):
                u = bad_combo[idxs[i]]
                v_idx = bad_combo[idxs[(i+1)%len(idxs)]]
                
                # Flip
                A[u, v_idx] = 1 - A[u, v_idx]
                A[v_idx, u] = A[u, v_idx]
                
                v_new = count_violations(A, n)
                if v_new < v:
                    v = v_new
                    improved = True
                    break
                else:
                    # Revert
                    A[u, v_idx] = 1 - A[u, v_idx]
                    A[v_idx, u] = A[u, v_idx]
            
            if not improved:
                # Random restart or random flip
                if np.random.rand() < 0.1:
                    # Restart
                    A = np.zeros((n, n), dtype=int)
                    for i in range(n):
                        for j in range(i+1, n):
                            if np.random.rand() < 0.5:
                                A[i,j] = 1
                                A[j,i] = 1
                    v = count_violations(A, n)
                else:
                    # Random flip
                    u, v_idx = np.random.randint(0, n, 2)
                    if u != v_idx:
                        A[u, v_idx] = 1 - A[u, v_idx]
                        A[v_idx, u] = A[u, v_idx]
                        v_new = count_violations(A, n)
                        if v_new < v:
                            v = v_new
                        else:
                            A[u, v_idx] = 1 - A[u, v_idx]
                            A[v_idx, u] = A[u, v_idx]
        
        return A, v

    # Try n from 20 down to 5
    for n in range(20, 4, -1):
        if time.time() > deadline - 5:
            break
        
        # Try a few restarts for this n
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            A, v = local_search(n, max_iters=50)
            if v == 0:
                return {'n': n, 'edges': get_edges(A)}
        
    return {'n': 5, 'edges': []}
