"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_optimal(deadline)
    
    return _generic_greedy(q, n, radius, deadline)

def _solve_optimal(deadline):
    """Highly optimized solver for C_3(8,1)."""
    q, n, radius = 3, 8, 1
    total = q ** n  # 6561
    
    # Precompute all words as numpy array
    words = np.zeros((total, n), dtype=np.int8)
    for i in range(total):
        v = i
        for j in range(n - 1, -1, -1):
            words[i, j] = v % 3
            v //= 3
    
    # Precompute neighbor lists as integer arrays
    # Each word covers itself + 16 neighbors = 17 words
    neigh = np.zeros((total, 17), dtype=np.int32)
    for i in range(total):
        w = words[i]
        neigh[i, 0] = i
        count = 1
        for pos in range(n):
            orig = w[pos]
            for nv in range(3):
                if nv != orig:
                    nw = w.copy()
                    nw[pos] = nv
                    # Compute index
                    idx = 0
                    for j in range(n):
                        idx = idx * 3 + nw[j]
                    neigh[i, count] = idx
                    count += 1
    
    # Greedy set cover with numpy
    uncovered = np.ones(total, dtype=bool)
    code = []
    
    np.random.seed(123)
    
    # Precompute neighbor sets for fast coverage counting
    # Use a trick: for each candidate, count uncovered neighbors
    for iteration in range(200):
        if time.time() > deadline - 25:
            break
        if not uncovered.any():
            break
        
        # Sample candidates
        unc_idx = np.where(uncovered)[0]
        if len(unc_idx) > 400:
            candidates = np.random.choice(unc_idx, 400, replace=False)
        else:
            candidates = unc_idx
        
        best_i = -1
        best_cov = -1
        
        for c in candidates:
            # Count uncovered neighbors
            cov = np.sum(uncovered[neigh[c]])
            if cov > best_cov:
                best_cov = cov
                best_i = c
        
        if best_i < 0 or best_cov <= 0:
            break
        
        code.append(words[best_i].copy())
        uncovered[neigh[best_i]] = False
    
    # Local search: remove redundant codewords
    # Build a coverage count
    cov_count = np.zeros(total, dtype=np.int32)
    for cw in code:
        idx = 0
        for j in range(n):
            idx = idx * 3 + cw[j]
        cov_count[neigh[idx]] += 1
    
    for pass_n in range(100):
        if time.time() > deadline - 5:
            break
        removed = False
        for i in range(len(code)):
            if time.time() > deadline - 5:
                break
            
            # Check if code[i] is redundant
            # A codeword is redundant if all its neighbors have cov_count > 1
            cw = code[i]
            idx = 0
            for j in range(n):
                idx = idx * 3 + cw[j]
            
            # Check neighbors
            is_redundant = True
            for nb in neigh[idx]:
                if cov_count[nb] <= 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove it
                cov_count[neigh[idx]] -= 1
                code.pop(i)
                removed = True
                break  # Restart
        
        if not removed:
            break
    
    return [cw.tolist() for cw in code]

def _generic_greedy(q, n, radius, deadline):
    total = q ** n
    code = []
    uncovered = set(range(total))
    
    while uncovered and time.time() < deadline - 10:
        best = -1
        best_c = -1
        cands = list(uncovered)
        if len(cands) > 100:
            cands = random.sample(cands, 100)
        
        for c in cands:
            w = []
            v = c
            for j in range(n - 1, -1, -1):
                w.append(v % q)
                v //= q
            w = w[::-1]
            
            cnt = 0
            for u in uncovered:
                uw = []
                v = u
                for j in range(n - 1, -1, -1):
                    uw.append(v % q)
                    v //= q
                uw = uw[::-1]
                d = sum(1 for a, b in zip(w, uw) if a != b)
                if d <= radius:
                    cnt += 1
            if cnt > best_c:
                best_c = cnt
                best = c
        
        if best < 0:
            break
        
        w = []
        v = best
        for j in range(n - 1, -1, -1):
            w.append(v % q)
            v //= q
        w = w[::-1]
        code.append(w)
        
        new_unc = set()
        for u in uncovered:
            uw = []
            v = u
            for j in range(n - 1, -1, -1):
                uw.append(v % q)
                v //= q
            uw = uw[::-1]
            d = sum(1 for a, b in zip(w, uw) if a != b)
            if d > radius:
                new_unc.add(u)
        uncovered = new_unc
    
    return code
# EVOLVE-BLOCK-END
