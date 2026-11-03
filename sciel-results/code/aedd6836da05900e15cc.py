# EVOLVE-BLOCK-START
import time
import random
import itertools
from fractions import Fraction
import numpy as np

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

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-12):
    """Verify using NumPy for speed."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    
    nm, mp, pn = n * m, m * p, p * n
    
    # Check all target entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(s - 1.0) > tol:
                    return False
    
    # Check non-target entries
    T_keys = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_keys.add((a, b, c))
    
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in T_keys:
                    s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                    if abs(s) > tol:
                        return False
    return True

def _verify_decomposition_exact(U, V, W, n, m, p):
    """Verify using exact Fraction arithmetic."""
    U_frac = [[Fraction(U[i][r]) for r in range(len(U[0]))] for i in range(len(U))]
    V_frac = [[Fraction(V[i][r]) for r in range(len(V[0]))] for i in range(len(V))]
    W_frac = [[Fraction(W[i][r]) for r in range(len(W[0]))] for i in range(len(W))]
    
    nm, mp, pn = n * m, m * p, p * n
    T_keys = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_keys.add((a, b, c))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = Fraction(0)
                for r in range(len(U[0])):
                    s += U_frac[a][r] * V_frac[b][r] * W_frac[c][r]
                if s != 1:
                    return False
    
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in T_keys:
                    s = Fraction(0)
                    for r in range(len(U[0])):
                        s += U_frac[a][r] * V_frac[b][r] * W_frac[c][r]
                    if s != 0:
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

def _try_structured_patterns(r, n, m, p, deadline=None):
    """Try various structured binary and half-integer patterns."""
    nm, mp, pn = n * m, m * p, p * n
    T = _build_target_tensor(n, m, p)
    
    patterns = []
    
    # Generate candidate (i,j,k) triples for each rank element
    for pattern_id in range(5): # Reduced from 20 to save time
        if deadline and time.time() > deadline:
            return None
        
        U = [[0] * r for _ in range(nm)]
        V = [[0] * r for _ in range(mp)]
        W = [[0] * r for _ in range(pn)]
        
        valid = True
        for idx in range(r):
            if deadline and time.time() > deadline:
                return None
            
            if pattern_id == 0:
                i = idx % n
                j = (idx // n) % m
                k = (idx // (n * m)) % p
            elif pattern_id == 1:
                k = idx % p
                j = (idx // p) % m
                i = (idx // (p * m)) % n
            elif pattern_id == 2:
                i = (idx * 3) % n
                j = (idx * 5) % m
                k = (idx * 7) % p
            elif pattern_id == 3:
                i = idx % n
                j = (idx // n) % m
                k = (idx // (n * m)) % p
            else:
                i = idx % n
                j = (idx // n) % m
                k = (idx // (n * m)) % p

            a = i * m + j
            b = j * p + k
            c = k * n + i
            U[a][idx] = 1
            V[b][idx] = 1
            W[c][idx] = 1
        
        if _verify_decomposition_fast(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 10 # Much tighter deadline for search, rely on schoolbook
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Theoretical lower bound: max(n*m, m*p, p*n) = 18
    # Try ranks from lower bound up to schoolbook rank - 1
    # We limit the search to be very fast.
    for r in range(18, best_rank):
        if time.time() > deadline:
            break
        
        # Only try structured patterns, which are very fast
        sol = _try_structured_patterns(r, n, m, p, deadline)
        if sol is not None:
            # Verify exactly before accepting
            if _verify_decomposition_exact(sol['U'], sol['V'], sol['W'], n, m, p):
                best = sol
                best_rank = r
                break
    
    return best
# EVOLVE-BLOCK-END
