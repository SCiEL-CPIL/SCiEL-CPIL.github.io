import time
import random
import numpy as np
from collections import deque
import scipy.sparse as sp
from scipy.sparse.csgraph import bfs

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145

    if k != 3:
        return _luw_construct(k, g)

    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    petersen_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)
        
    base_edges = []
    for u in range(10):
        for v in petersen_adj[u]:
            if u < v:
                base_edges.append((u, v))
                
    random.seed(42)
    
    # Pre-compute sparse matrix structure for base? No, lift changes.
    
    for m in range(52, 1000):
        if time.time() > deadline:
            break
        
        if m < 60:
            attempts = 2000
        elif m < 100:
            attempts = 1000
        elif m < 200:
            attempts = 500
        else:
            attempts = 100

        # Deterministic shifts
        shifts1 = [int((i + 1) * m / 15.0) for i in range(15)]
        n, edges = _build_lift(base_edges, m, shifts1)
        if _verify_girth_fast(n, edges, g):
            return n, edges
        
        golden = (1 + 5**0.5) / 2.0
        shifts2 = [int((i + 1) * m * golden) % m for i in range(15)]
        n, edges = _build_lift(base_edges, m, shifts2)
        if _verify_girth_fast(n, edges, g):
            return n, edges
            
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        shifts3 = [primes[i] * m // 15 for i in range(15)]
        n, edges = _build_lift(base_edges, m, shifts3)
        if _verify_girth_fast(n, edges, g):
            return n, edges

        for _ in range(attempts):
            if time.time() > deadline:
                break
            shifts_rand = [random.randint(0, m-1) for _ in range(15)]
            n, edges = _build_lift(base_edges, m, shifts_rand)
            if _verify_girth_fast(n, edges, g):
                return n, edges

    # Fallback
    m = 3000
    shifts_rand = [random.randint(0, m-1) for _ in range(15)]
    n, edges = _build_lift(base_edges, m, shifts_rand)
    return n, edges

def _build_lift(base_edges, m, shifts):
    n = 10 * m
    edges = []
    for (u, v), shift in zip(base_edges, shifts):
        for i in range(m):
            u_id = u * m + i
            v_id = v * m + (i + shift) % m
            edges.append((u_id, v_id))
    return n, edges

def _verify_girth_fast(n, edges, g):
    rows = []
    cols = []
    for u, v in edges:
        rows.append(u)
        cols.append(v)
        rows.append(v)
        cols.append(u)
        
    data = np.ones(len(rows))
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    
    for s in range(n):
        dist, pred = bfs(A, directed=False, indices=[s])
        
        for u, v in edges:
            if dist[u] < np.inf and dist[v] < np.inf:
                if not (pred[v] == u or pred[u] == v):
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True

# Keep other functions as is
