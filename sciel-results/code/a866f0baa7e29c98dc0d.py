import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Helper to validate and return if valid
    def try_return(n, A):
        if time.time() > deadline - 5:
            return None
        
        # Full systematic check
        for k in range(6, n + 1):
            for S in combinations(range(n), k):
                if is_regular_induced(A, list(S)):
                    return None # Invalid
                if time.time() > deadline - 10:
                    return None # Timeout, assume invalid
        # Valid
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] == 1:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}

    # Try n=21
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        A = np.zeros((21, 21), dtype=int)
        for i in range(21):
            for j in range(i + 1, 21):
                if random.random() < 0.5:
                    A[i, j] = 1
                    A[j, i] = 1
        
        # Local Search
        for step in range(5000):
            if time.time() > deadline - 10:
                break
            
            # Find bad subset
            bad_subset = None
            
            # Random sampling
            for _ in range(200):
                if time.time() > deadline - 10:
                    break
                k = random.randint(6, 15)
                S = random.sample(range(21), k)
                if is_regular_induced(A, S):
                    bad_subset = S
                    break
            
            if bad_subset is None:
                # Try systematic check for small k
                for k in [6, 7, 8]:
                    for S in combinations(range(21), k):
                        if is_regular_induced(A, list(S)):
                            bad_subset = list(S)
                            break
                        if time.time() > deadline - 10:
                            break
                    if bad_subset:
                        break
                    if time.time() > deadline - 10:
                        break
                
            if bad_subset is None:
                # Likely valid, do full check
                res = try_return(21, A)
                if res:
                    return res
                # If full check failed or timed out, restart or break
                break
            
            # Flip edge
            u, v = random.sample(bad_subset, 2)
            A[u, v] = 1 - A[u, v]
            A[v, u] = A[u, v]

    # Try n=20
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        A = np.zeros((20, 20), dtype=int)
        for i in range(20):
            for j in range(i + 1, 20):
                if random.random() < 0.5:
                    A[i, j] = 1
                    A[j, i] = 1
        
        for step in range(5000):
            if time.time() > deadline - 10:
                break
            
            bad_subset = None
            for _ in range(200):
                if time.time() > deadline - 10:
                    break
                k = random.randint(6, 15)
                S = random.sample(range(20), k)
                if is_regular_induced(A, S):
                    bad_subset = S
                    break
            
            if bad_subset is None:
                for k in [6, 7, 8]:
                    for S in combinations(range(20), k):
                        if is_regular_induced(A, list(S)):
                            bad_subset = list(S)
                            break
                        if time.time() > deadline - 10:
                            break
                    if bad_subset:
                        break
                    if time.time() > deadline - 10:
                        break
                
            if bad_subset is None:
                res = try_return(20, A)
                if res:
                    return res
                break
            
            u, v = random.sample(bad_subset, 2)
            A[u, v] = 1 - A[u, v]
            A[v, u] = A[u, v]

    # Fallback n=10
    for attempt in range(100):
        if time.time() > deadline - 2:
            break
        A = np.zeros((10, 10), dtype=int)
        for i in range(10):
            for j in range(i + 1, 10):
                if random.random() < 0.5:
                    A[i, j] = 1
                    A[j, i] = 1
        
        res = try_return(10, A)
        if res:
            return res

    return {'n': 5, 'edges': []}

def is_regular_induced(A, S):
    if len(S) < 2:
        return True
    idx = np.array(S)
    sub_A = A[np.ix_(idx, idx)]
    degrees = sub_A.sum(axis=1)
    return np.all(degrees == degrees[0])
