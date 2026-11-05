"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
from math import comb

def _sample_check(n, adj, num_samples=50):
    """Quick sample check for regular 6-subsets."""
    for _ in range(num_samples):
        idx = random.sample(range(n), 6)
        sub_adj = adj[np.ix_(idx, idx)]
        degrees = sub_adj.sum(axis=1)
        if len(set(degrees)) == 1:
            return False
    return True

def _full_check(n, adj):
    """Full check for all subsets of size 6 and 7, sample 8+."""
    for combo in combinations(range(n), 6):
        idx = list(combo)
        sub_adj = adj[np.ix_(idx, idx)]
        degrees = sub_adj.sum(axis=1)
        if len(set(degrees)) == 1:
            return False
    
    for combo in combinations(range(n), 7):
        idx = list(combo)
        sub_adj = adj[np.ix_(idx, idx)]
        degrees = sub_adj.sum(axis=1)
        if len(set(degrees)) == 1:
            return False
    
    for sz in range(8, n + 1):
        if sz > n:
            break
        total = comb(n, sz)
        samples = min(total, 100)
        for _ in range(samples):
            idx = random.sample(range(n), sz)
            sub_adj = adj[np.ix_(idx, idx)]
            degrees = sub_adj.sum(axis=1)
            if len(set(degrees)) == 1:
                return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    best_n = 5
    best_edges = []
    
    def get_max_attempts(n):
        cost = comb(n, 6) + comb(n, 7)
        if cost < 1000:
            return 5000
        elif cost < 10000:
            return 1000
        elif cost < 50000:
            return 200
        else:
            return 50
    
    for n in range(6, 23):
        if time.time() > deadline - 20:
            break
        
        found = False
        max_attempts = get_max_attempts(n)
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            density = 0.2 + 0.6 * random.random()
            adj = np.zeros((n, n), dtype=np.uint8)
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < density:
                        adj[u, v] = 1
                        adj[v, u] = 1
            
            if not _sample_check(n, adj, 50):
                continue
            
            if _full_check(n, adj):
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u, v]:
                            edges.append([u, v])
                best_n = n
                best_edges = edges
                found = True
                break
        
        if found:
            continue
        else:
            if time.time() > deadline - 20:
                break
            
            for trial in range(5):
                if time.time() > deadline - 20:
                    break
                
                density = 0.2 + 0.6 * random.random()
                adj = np.zeros((n, n), dtype=np.uint8)
                for u in range(n):
                    for v in range(u + 1, n):
                        if random.random() < density:
                            adj[u, v] = 1
                            adj[v, u] = 1
                
                for step in range(200):
                    if time.time() > deadline - 20:
                        break
                    
                    if _sample_check(n, adj, 30) and _full_check(n, adj):
                        edges = []
                        for u in range(n):
                            for v in range(u + 1, n):
                                if adj[u, v]:
                                    edges.append([u, v])
                        best_n = n
                        best_edges = edges
                        found = True
                        break
                    
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u != v:
                        adj[u, v] = 1 - adj[u, v]
                        adj[v, u] = adj[u, v]
                
                if found:
                    break
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
