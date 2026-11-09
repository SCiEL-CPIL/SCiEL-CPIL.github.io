import time
import random
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def to_result(n, adj):
        edges = [[int(u), int(v)] for u in range(n) for v in range(u + 1, n) if adj[u, v] == 1]
        return {'n': int(n), 'edges': edges}

    def get_combinations(n, k):
        return list(itertools.combinations(range(n), k))

    def find_bad_subset(adj, n, k, subsets):
        # subsets is a list of tuples of length k
        for S in subsets:
            sub = adj[np.ix_(S, S)]
            degs = sub.sum(axis=1)
            if np.all(degs == degs[0]):
                return S
        return None

    def is_valid(adj, n, subsets_by_k):
        for k in range(6, n + 1):
            if k not in subsets_by_k:
                continue
            if find_bad_subset(adj, n, k, subsets_by_k[k]) is not None:
                return False
        return True

    def repair(adj, n, subsets_by_k, max_iter=1000):
        for i in range(max_iter):
            if time.time() > deadline - 5:
                return False
            
            # Check k=6 first
            bad = find_bad_subset(adj, n, 6, subsets_by_k.get(6, []))
            if bad is not None:
                u, v = random.sample(list(bad), 2)
                adj[u, v] ^= 1
                adj[v, u] ^= 1
                continue
            
            # Check k=7
            bad = find_bad_subset(adj, n, 7, subsets_by_k.get(7, []))
            if bad is not None:
                u, v = random.sample(list(bad), 2)
                adj[u, v] ^= 1
                adj[v, u] ^= 1
                continue

            # Check larger k
            found = False
            for k in range(8, n + 1):
                if k not in subsets_by_k:
                    continue
                bad = find_bad_subset(adj, n, k, subsets_by_k[k])
                if bad is not None:
                    u, v = random.sample(list(bad), 2)
                    adj[u, v] ^= 1
                    adj[v, u] ^= 1
                    found = True
                    break
            if not found:
                # No bad subsets
                return True
        return False

    # Precompute subsets for n=20 and n=21? 
    # C(21, 10) = 352716. This is large.
    # C(20, 10) = 184756.
    # Storing these lists takes memory but is fast to iterate.
    
    # Try n=21
    if time.time() < deadline - 30:
        n = 21
        subsets_by_k = {}
        for k in range(6, n + 1):
            subsets_by_k[k] = get_combinations(n, k)
            
        adj = (np.random.rand(n, n) < 0.5).astype(np.uint8)
        np.fill_diagonal(adj, 0)
        adj = (adj | adj.T).astype(np.uint8)
        
        if repair(adj, n, subsets_by_k, max_iter=500):
            return to_result(n, adj)

    # Try n=20
    if time.time() < deadline - 10:
        n = 20
        subsets_by_k = {}
        for k in range(6, n + 1):
            subsets_by_k[k] = get_combinations(n, k)
            
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            adj = (np.random.rand(n, n) < 0.5).astype(np.uint8)
            np.fill_diagonal(adj, 0)
            adj = (adj | adj.T).astype(np.uint8)
            
            if repair(adj, n, subsets_by_k, max_iter=500):
                return to_result(n, adj)

    # Fallback
    n = 5
    adj = np.zeros((n, n), dtype=np.uint8)
    return to_result(n, adj)
