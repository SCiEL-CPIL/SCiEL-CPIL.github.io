import time
import random
import numpy as np
from itertools import combinations

_START = time.time()
_DEADLINE = _START + 125.0  # safety margin before 150s kill

def _time_left():
    return _DEADLINE - time.time()

def _verify_fast(adj, n, max_check=400000):
    """Check if any subset of size >= 6 induces a regular subgraph.
    Returns (is_valid, violations).
    Uses vectorized degree computation via numpy."""
    if n < 6:
        return True, 0
    A = np.array(adj, dtype=np.int8)
    # Check size 6 subsets (most numerous, most likely to be regular)
    total = 0
    for combo in combinations(range(n), 6):
        total += 1
        if total > max_check:
            return False, -1  # too many to check quickly
        sub = A[list(combo), :][:, list(combo)]
        degs = sub.sum(axis=1)
        if len(set(degs.tolist())) == 1:
            return False, 1
    # If n > 6, also need to check larger subsets, but they're rarer
    # For n up to 22, size-6 check catches most violations
    # Check size 7 if n >= 7 (fewer subsets)
    if n >= 7:
        for combo in combinations(range(n), 7):
            sub = A[list(combo), :][:, list(combo)]
            degs = sub.sum(axis=1)
            if len(set(degs.tolist())) == 1:
                return False, 1
    if n >= 8:
        for combo in combinations(range(n), 8):
            sub = A[list(combo), :][:, list(combo)]
            degs = sub.sum(axis=1)
            if len(set(degs.tolist())) == 1:
                return False, 1
    if n >= 9:
        for combo in combinations(range(n), 9):
            sub = A[list(combo), :][:, list(combo)]
            degs = sub.sum(axis=1)
            if len(set(degs.tolist())) == 1:
                return False, 1
    if n >= 10:
        for combo in combinations(range(n), 10):
            sub = A[list(combo), :][:, list(combo)]
            degs = sub.sum(axis=1)
            if len(set(degs.tolist())) == 1:
                return False, 1
    if n >= 11:
        for combo in combinations(range(n), 11):
            sub = A[list(combo), :][:, list(combo)]
            degs = sub.sum(axis=1)
            if len(set(degs.tolist())) == 1:
                return False, 1
    if n >= 12:
        for combo in combinations(range(n), 12):
            sub = A[list(combo), :][:, list(combo)]
            degs = sub.sum(axis=1)
            if len(set(degs.tolist())) == 1:
                return False, 1
    if n >= 13:
        for combo in combinations(range(n), 13):
            sub = A[list(combo), :][:, list(combo)]
            degs = sub.sum(axis=1)
            if len(set(degs.tolist())) == 1:
                return False, 1
    if n >= 14:
        for combo in combinations(range(n), 14):
            sub = A[list(combo), :][:, list(combo)]
            degs = sub.sum(axis=1)
            if len(set(degs.tolist())) == 1:
                return False, 1
    if n >= 15:
        for combo in combinations(range(n), 15):
            sub = A[list(combo), :][:, list(combo)]
            degs = sub.sum(axis=1)
            if len(set(degs.tolist())) == 1:
                return False, 1
    if n >= 16:
        for combo in combinations(range(n), 16):
            sub = A[list(combo), :][:, list(combo)]
            degs = sub.sum(axis=1)
            if len(set(degs.tolist())) == 1:
                return False, 1
    if n >= 17:
        for combo in combinations(range(n), 17):
            sub = A[list(combo), :][:, list(combo)]
            degs = sub.sum(axis=1)
            if len(set(degs.tolist())) == 1:
                return False, 1
    if n >= 18:
        for combo in combinations(range(n), 18):
            sub = A[list(combo), :][:, list(combo)]
            degs = sub.sum(axis=1)
            if len(set(degs.tolist())) == 1:
                return False, 1
    if n >= 19:
        for combo in combinations(range(n), 19):
            sub = A[list(combo), :][:, list(combo)]
            degs = sub.sum(axis=1)
            if len(set(degs.tolist())) == 1:
                return False, 1
    if n >= 20:
        for combo in combinations(range(n), 20):
            sub = A[list(combo), :][:, list(combo)]
            degs = sub.sum(axis=1)
            if len(set(degs.tolist())) == 1:
                return False, 1
    if n >= 21:
        for combo in combinations(range(n), 21):
            sub = A[list(combo), :][:, list(combo)]
            degs = sub.sum(axis=1)
            if len(set(degs.tolist())) == 1:
                return False, 1
    if n >= 22:
        for combo in combinations(range(n), 22):
            sub = A[list(combo), :][:, list(combo)]
            degs = sub.sum(axis=1)
            if len(set(degs.tolist())) == 1:
                return False, 1
    return True, 0

def _find_violation_fast(adj, n):
    """Find one violating subset. Returns (mask, size) or None."""
    if n < 6:
        return None
    A = np.array(adj, dtype=np.int8)
    for size in range(6, min(n, 10) + 1):
        for combo in combinations(range(n), size):
            sub = A[list(combo), :][:, list(combo)]
            degs = sub.sum(axis=1)
            if len(set(degs.tolist())) == 1:
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                return (mask, size)
    # Check larger sizes
    for size in range(min(n, 11), n + 1):
        for combo in combinations(range(n), size):
            sub = A[list(combo), :][:, list(combo)]
            degs = sub.sum(axis=1)
            if len(set(degs.tolist())) == 1:
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                return (mask, size)
    return None

def _random_graph(n, seed=None):
    """Generate a random graph G(n, 0.5)."""
    if seed is not None:
        random.seed(seed)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _local_search(adj, n, max_steps=500):
    """Local search: flip edges to break regular subgraphs."""
    for step in range(max_steps):
        if _time_left() < 5:
            return False
        viol = _find_violation_fast(adj, n)
        if viol is None:
            return True
        mask, size = viol
        # Extract vertices
        verts = []
        m = mask
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            verts.append(idx)
            m ^= lb
        if len(verts) < 2:
            continue
        # Flip a random edge within the violating subset
        u_idx, v_idx = random.sample(range(len(verts)), 2)
        u, v = verts[u_idx], verts[v_idx]
        if (adj[u] >> v) & 1:
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    return False

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 125.0
    
    best_n = 5
    best_edges = []
    
    # Try n from 22 down to 6
    for target_n in range(22, 5, -1):
        if _time_left() < 10:
            break
        n = target_n
        
        # Number of random tries based on time budget
        # Larger n needs more tries but each verification is more expensive
        if n >= 21:
            max_tries = 50
        elif n >= 19:
            max_tries = 100
        else:
            max_tries = 200
        
        found = False
        for trial in range(max_tries):
            if _time_left() < 5:
                break
            
            adj = _random_graph(n)
            
            # Quick check
            is_valid, _ = _verify_fast(adj, n)
            if is_valid:
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[i] >> j) & 1:
                            edges.append([i, j])
                if n > best_n:
                    best_n = n
                    best_edges = edges
                found = True
                break
            
            # If not valid, try local search (only for smaller n where it's fast)
            if n <= 18 and _time_left() > 15:
                if _local_search(adj, n, max_steps=200):
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (adj[i] >> j) & 1:
                                edges.append([i, j])
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                    found = True
                    break
        
        if found:
            # Found valid graph for this n, stop (since we're going down)
            break
    
    # Fallback
    if best_n <= 5:
        # Return a small valid graph
        return {'n': 5, 'edges': [[0,1],[1,2],[2,3],[3,4]]}
    
    return {'n': best_n, 'edges': best_edges}
