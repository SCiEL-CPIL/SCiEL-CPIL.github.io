"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_vertices = P['minimum_vertices']
    
    # Build adjacency matrix as numpy uint8
    adj = np.zeros((n, n), dtype=np.uint8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    
    # Precompute row sums for quick degree lookups in full graph
    full_degrees = adj.sum(axis=1)
    
    deadline = time.time() + 80  # 80s safety margin within 90s limit
    best_score = 0
    best_subset = list(range(min_vertices))
    
    # First, try exhaustive search for size 8
    # C(17,8) = 24310, very fast
    for subset in itertools.combinations(range(n), 8):
        if time.time() > deadline:
            break
        # Compute degrees in induced subgraph
        degs = []
        for v in subset:
            d = 0
            for u in subset:
                if u != v and adj[v, u]:
                    d += 1
            degs.append(d)
        distinct = len(set(degs))
        if distinct > best_score:
            best_score = distinct
            best_subset = list(subset)
    
    # If we already have a good score, try larger subsets
    # For size 9: C(17,9) = 24310
    if time.time() < deadline - 2:
        for subset in itertools.combinations(range(n), 9):
            if time.time() > deadline:
                break
            degs = []
            for v in subset:
                d = 0
                for u in subset:
                    if u != v and adj[v, u]:
                        d += 1
                degs.append(d)
            distinct = len(set(degs))
            if distinct > best_score:
                best_score = distinct
                best_subset = list(subset)
    
    # For size 10: C(17,10) = 19448
    if time.time() < deadline - 2:
        for subset in itertools.combinations(range(n), 10):
            if time.time() > deadline:
                break
            degs = []
            for v in subset:
                d = 0
                for u in subset:
                    if u != v and adj[v, u]:
                        d += 1
                degs.append(d)
            distinct = len(set(degs))
            if distinct > best_score:
                best_score = distinct
                best_subset = list(subset)
    
    # For size 11: C(17,11) = 12376
    if time.time() < deadline - 2:
        for subset in itertools.combinations(range(n), 11):
            if time.time() > deadline:
                break
            degs = []
            for v in subset:
                d = 0
                for u in subset:
                    if u != v and adj[v, u]:
                        d += 1
                degs.append(d)
            distinct = len(set(degs))
            if distinct > best_score:
                best_score = distinct
                best_subset = list(subset)
    
    # For size 12: C(17,12) = 6188
    if time.time() < deadline - 2:
        for subset in itertools.combinations(range(n), 12):
            if time.time() > deadline:
                break
            degs = []
            for v in subset:
                d = 0
                for u in subset:
                    if u != v and adj[v, u]:
                        d += 1
                degs.append(d)
            distinct = len(set(degs))
            if distinct > best_score:
                best_score = distinct
                best_subset = list(subset)
    
    # For size 13: C(17,13) = 2380
    if time.time() < deadline - 1:
        for subset in itertools.combinations(range(n), 13):
            if time.time() > deadline:
                break
            degs = []
            for v in subset:
                d = 0
                for u in subset:
                    if u != v and adj[v, u]:
                        d += 1
                degs.append(d)
            distinct = len(set(degs))
            if distinct > best_score:
                best_score = distinct
                best_subset = list(subset)
    
    # For size 14: C(17,14) = 680
    if time.time() < deadline - 1:
        for subset in itertools.combinations(range(n), 14):
            if time.time() > deadline:
                break
            degs = []
            for v in subset:
                d = 0
                for u in subset:
                    if u != v and adj[v, u]:
                        d += 1
                degs.append(d)
            distinct = len(set(degs))
            if distinct > best_score:
                best_score = distinct
                best_subset = list(subset)
    
    # For size 15: C(17,15) = 136
    if time.time() < deadline - 1:
        for subset in itertools.combinations(range(n), 15):
            if time.time() > deadline:
                break
            degs = []
            for v in subset:
                d = 0
                for u in subset:
                    if u != v and adj[v, u]:
                        d += 1
                degs.append(d)
            distinct = len(set(degs))
            if distinct > best_score:
                best_score = distinct
                best_subset = list(subset)
    
    # For size 16: C(17,16) = 17
    if time.time() < deadline - 1:
        for subset in itertools.combinations(range(n), 16):
            if time.time() > deadline:
                break
            degs = []
            for v in subset:
                d = 0
                for u in subset:
                    if u != v and adj[v, u]:
                        d += 1
                degs.append(d)
            distinct = len(set(degs))
            if distinct > best_score:
                best_score = distinct
                best_subset = list(subset)
    
    # For size 17: C(17,17) = 1
    if time.time() < deadline - 1:
        subset = tuple(range(n))
        degs = []
        for v in subset:
            d = 0
            for u in subset:
                if u != v and adj[v, u]:
                    d += 1
            degs.append(d)
        distinct = len(set(degs))
        if distinct > best_score:
            best_score = distinct
            best_subset = list(subset)
    
    # Ensure we return at least min_vertices
    if len(best_subset) < min_vertices:
        best_subset = list(range(min_vertices))
    
    return list(best_subset)
# EVOLVE-BLOCK-END
