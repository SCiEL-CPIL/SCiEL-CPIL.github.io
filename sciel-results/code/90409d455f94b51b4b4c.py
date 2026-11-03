"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import random
import itertools
from fractions import Fraction

def _build_target_tensor(n, m, p):
    """Build the target multiplication tensor as a dict of non-zero entries."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[(a, b, c)] = 1
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=0):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    T = _build_target_tensor(n, m, p)
    rank = len(U[0]) if U else 0
    for (a, b, c) in T:
        s = 0
        for r in range(rank):
            s += U[a][r] * V[b][r] * W[c][r]
        if abs(s - 1) > tol:
            return False
    nm, mp, pn = n * m, m * p, p * n
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in T:
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s) > tol:
                        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_binary_search(r, n, m, p, deadline=None):
    """Search for binary (0/1) decomposition using combinatorial patterns."""
    nm, mp, pn = n * m, m * p, p * n
    T = _build_target_tensor(n, m, p)
    
    # Try structured binary patterns based on index relationships
    for pattern_type in range(5):
        if deadline and time.time() > deadline:
            return None
        
        U = [[0] * r for _ in range(nm)]
        V = [[0] * r for _ in range(mp)]
        W = [[0] * r for _ in range(pn)]
        
        for idx in range(r):
            if deadline and time.time() > deadline:
                return None
            
            # Different pattern generators
            if pattern_type == 0:
                # Simple modular pattern
                i = idx % n
                j = (idx // n) % m
                k = (idx // (n * m)) % p
            elif pattern_type == 1:
                # Interleaved pattern
                i = (idx * 7) % n
                j = (idx * 11) % m
                k = (idx * 13) % p
            elif pattern_type == 2:
                # Diagonal pattern
                i = idx % n
                j = (idx // n) % m
                k = (idx + i) % p
            elif pattern_type == 3:
                # Cross pattern
                i = idx % n
                j = (idx % m)
                k = (idx // (n * m)) % p
            else:
                # Random with seed
                random.seed(idx * 12345 + 67890)
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
            
            a = i * m + j
            b = j * p + k
            c = k * n + i
            U[a][idx] = 1
            V[b][idx] = 1
            W[c][idx] = 1
        
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def _try_half_integer_search(r, n, m, p, deadline=None):
    """Search for half-integer decomposition using local refinement."""
    nm, mp, pn = n * m, m * p, p * n
    T = _build_target_tensor(n, m, p)
    
    for attempt in range(30):
        if deadline and time.time() > deadline:
            return None
        
        # Initialize with random half-integers
        U = [[random.choice([0, 0.5, 1, 1.5]) for _ in range(r)] for _ in range(nm)]
        V = [[random.choice([0, 0.5, 1, 1.5]) for _ in range(r)] for _ in range(mp)]
        W = [[random.choice([0, 0.5, 1, 1.5]) for _ in range(r)] for _ in range(pn)]
        
        # Local search to refine
        best_res = 999
        for it in range(100):
            if deadline and time.time() > deadline:
                return None
            
            # Compute residual
            max_res = 0
            for (a, b, c) in T:
                s = 0
                for rr in range(r):
                    s += U[a][rr] * V[b][rr] * W[c][rr]
                max_res = max(max_res, abs(s - 1))
            for _ in range(10):
                a = random.randint(0, nm - 1)
                b = random.randint(0, mp - 1)
                c = random.randint(0, pn - 1)
                if (a, b, c) not in T:
                    s = 0
                    for rr in range(r):
                        s += U[a][rr] * V[b][rr] * W[c][rr]
                    max_res = max(max_res, abs(s))
            
            if max_res < best_res:
                best_res = max_res
                if best_res == 0:
                    return {"U": U, "V": V, "W": W}
            
            # Perturb one entry
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = random.randint(0, nm - 1)
                arr = U
            elif mat == 'V':
                row = random.randint(0, mp - 1)
                arr = V
            else:
                row = random.randint(0, pn - 1)
                arr = W
            col = random.randint(0, r - 1)
            old_val = arr[row][col]
            new_val = old_val + random.choice([-0.5, 0.5, -1, 1])
            arr[row][col] = new_val
        
        if best_res == 0:
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try ranks from lower bound up
    # Theoretical lower bound considerations suggest trying from 18
    for r in range(18, best_rank + 1):
        if time.time() > deadline:
            break
        
        # First try binary patterns
        sol = _try_binary_search(r, n, m, p, deadline)
        if sol is not None:
            best = sol
            best_rank = r
            break
        
        # Then try half-integer search
        if time.time() < deadline - 10:
            sol = _try_half_integer_search(r, n, m, p, deadline)
            if sol is not None:
                best = sol
                best_rank = r
                break
    
    return best
# EVOLVE-BLOCK-END
