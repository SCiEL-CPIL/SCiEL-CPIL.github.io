"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

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
    
    # Check non-target entries (sample for speed, but check all for correctness)
    # For small tensors, check all
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
    
    # Pattern 1: Direct index mapping
    # Pattern 2: Modular arithmetic with different moduli
    # Pattern 3: Based on Kronecker product structure
    # Pattern 4: Exploit that n=2 is small
    
    patterns = []
    
    # Generate candidate (i,j,k) triples for each rank element
    for pattern_id in range(20):
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
                # Cyclic through all (i,j,k)
                i = idx % n
                j = (idx // n) % m
                k = (idx // (n * m)) % p
            elif pattern_id == 1:
                # Different ordering
                k = idx % p
                j = (idx // p) % m
                i = (idx // (p * m)) % n
            elif pattern_id == 2:
                # Interleave
                i = (idx * 3) % n
                j = (idx * 5) % m
                k = (idx * 7) % p
            elif pattern_id == 3:
                # Pair i values
                i = idx % n
                j = (idx // n) % m
                k = (idx // (n * m)) % p
            elif pattern_id == 4:
                # Focus on first i=0
                i = 0
                j = idx % m
                k = (idx // m) % p
            elif pattern_id == 5:
                # Focus on second i=1
                i = 1
                j = idx % m
                k = (idx // m) % p
            elif pattern_id == 6:
                # Mixed
                i = idx % n
                j = (idx * 2) % m
                k = (idx * 3) % p
            elif pattern_id == 7:
                # Quadratic
                i = (idx * idx) % n
                j = (idx * idx * 2) % m
                k = (idx * idx * 3) % p
            elif pattern_id == 8:
                # Based on sum
                s = idx
                i = s % n
                s //= n
                j = s % m
                s //= m
                k = s % p
            elif pattern_id == 9:
                # Reverse
                i = (n - 1 - idx) % n
                j = (m - 1 - (idx // n)) % m
                k = (p - 1 - (idx // (n * m))) % p
            elif pattern_id == 10:
                # Prime-based
                i = (idx * 17) % n
                j = (idx * 19) % m
                k = (idx * 23) % p
            elif pattern_id == 11:
                # Fibonacci-like
                a, b = 0, 1
                for _ in range(idx):
                    a, b = b, (a + b) % (n * m * p)
                i = a % n
                j = (a // n) % m
                k = (a // (n * m)) % p
            elif pattern_id == 12:
                # Split into blocks
                block = idx // 6
                offset = idx % 6
                i = offset % n
                j = (offset // n) % m
                k = block % p
            elif pattern_id == 13:
                # Weighted
                i = (idx * 1) % n
                j = (idx * 2) % m
                k = (idx * 4) % p
            elif pattern_id == 14:
                # XOR-like
                i = (idx ^ (idx >> 2)) % n
                j = (idx ^ (idx >> 3)) % m
                k = (idx ^ (idx >> 4)) % p
            elif pattern_id == 15:
                # Sequential with wrap
                i = idx % n
                j = (idx + 1) % m
                k = (idx + 2) % p
            elif pattern_id == 16:
                # Block diagonal
                i = (idx // (m * p)) % n
                j = (idx // p) % m
                k = idx % p
            elif pattern_id == 17:
                # Random seeded
                random.seed(pattern_id * 1000 + idx)
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
            elif pattern_id == 18:
                # Concentrate on specific j
                j = idx % m
                i = (idx // m) % n
                k = (idx // (m * n)) % p
            else:
                # Concentrate on specific k
                k = idx % p
                i = (idx // p) % n
                j = (idx // (p * n)) % m
            
            a = i * m + j
            b = j * p + k
            c = k * n + i
            U[a][idx] = 1
            V[b][idx] = 1
            W[c][idx] = 1
        
        if _verify_decomposition_fast(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def _try_local_refinement(r, n, m, p, deadline=None):
    """Start from near-solution and refine using local search with exact arithmetic."""
    nm, mp, pn = n * m, m * p, p * n
    T = _build_target_tensor(n, m, p)
    
    for attempt in range(50):
        if deadline and time.time() > deadline:
            return None
        
        # Initialize with small random half-integers
        U = [[random.choice([0, 0.5, -0.5, 1, -1]) for _ in range(r)] for _ in range(nm)]
        V = [[random.choice([0, 0.5, -0.5, 1, -1]) for _ in range(r)] for _ in range(mp)]
        W = [[random.choice([0, 0.5, -0.5, 1, -1]) for _ in range(r)] for _ in range(pn)]
        
        best_res = float('inf')
        best_state = None
        
        for it in range(200):
            if deadline and time.time() > deadline:
                return None
            
            # Compute max residual
            max_res = 0
            for (a, b, c) in T:
                s = 0
                for rr in range(r):
                    s += U[a][rr] * V[b][rr] * W[c][rr]
                max_res = max(max_res, abs(s - 1))
            
            # Sample non-target entries
            for _ in range(20):
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
                best_state = ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
                if max_res < 1e-10:
                    # Verify exactly
                    if _verify_decomposition_exact(U, V, W, n, m, p):
                        return {"U": U, "V": V, "W": W}
            
            # Perturb one entry
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                row = random.randint(0, nm - 1)
                arr = U
            elif mat_idx == 1:
                row = random.randint(0, mp - 1)
                arr = V
            else:
                row = random.randint(0, pn - 1)
                arr = W
            col = random.randint(0, r - 1)
            old_val = arr[row][col]
            delta = random.choice([-0.5, 0.5, -1, 1, -1.5, 1.5])
            arr[row][col] = old_val + delta
        
        # Try to snap best state to half-integers and verify
        if best_state and best_res < 0.1:
            U_snap = [[round(v * 2) / 2 for v in row] for row in best_state[0]]
            V_snap = [[round(v * 2) / 2 for v in row] for row in best_state[1]]
            W_snap = [[round(v * 2) / 2 for v in row] for row in best_state[2]]
            if _verify_decomposition_exact(U_snap, V_snap, W_snap, n, m, p):
                return {"U": U_snap, "V": V_snap, "W": W_snap}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # Try ranks from lower bound up
    for r in range(18, best_rank + 1):
        if time.time() > deadline:
            break
        
        # First try structured patterns (fast)
        sol = _try_structured_patterns(r, n, m, p, deadline)
        if sol is not None:
            best = sol
            best_rank = r
            break
        
        # Then try local refinement (slower)
        if time.time() < deadline - 20:
            sol = _try_local_refinement(r, n, m, p, deadline)
            if sol is not None:
                best = sol
                best_rank = r
                break
    
    return best
# EVOLVE-BLOCK-END
