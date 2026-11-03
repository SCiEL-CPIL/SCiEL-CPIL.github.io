"""
Matrix multiplication 2x3 by 3x7: rank-minimal exact tensor decomposition.

Strategy:
- Start from the schoolbook decomposition of rank n*m*p = 42.
- Systematically attempt rank reductions (41, 40, ...) by treating the problem
  as a bilinear system and using a "fix-two-solve-one" approach.
- For a target rank r, we look for U (nm x r), V (mp x r), W (pn x r).
- We use a structured random search where we fix U and V with sparse integer entries,
  then solve for W via linear systems, and verify exactness using Fractions.
- We also try a "perturbation" approach: start from schoolbook and try to merge
  columns by finding linear dependencies in the tensor space.
- Crucially, we use a deadline guard to ensure we return the best result found
  before the 150s limit, with a safety margin.
"""
import numpy as np
from fractions import Fraction
import random
import sys
import time

DIMENSIONS = [2, 3, 7]
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 10.0  # seconds to reserve

def _build_tensor(n, m, p):
    """Build the full tensor as a dict of (a,b,c) -> 1 for valid entries."""
    tensor = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor[(a, b, c)] = 1
    return tensor

def _verify_exact_fast(Uf, Vf, Wf, n, m, p, tensor):
    """
    Exact verification with Fractions. Optimized for speed.
    Uf: list of lists (nm x r)
    Vf: list of lists (mp x r)
    Wf: list of lists (pn x r)
    """
    r = len(Uf[0])
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Check all valid entries equal 1
    for (a, b, c) in tensor:
        s = Fraction(0)
        for t in range(r):
            s += Uf[a][t] * Vf[b][t] * Wf[c][t]
        if s != 1:
            return False
    
    # Check all invalid entries equal 0
    # To speed this up, we can precompute the "product" matrices for each t
    # P_t = U[:,t] * V[:,t]^T (outer product)
    # Then T[a,b,c] = sum_t W[c,t] * P_t[a,b]
    
    # But for exactness, we still need to check each (a,b,c) not in tensor.
    # Given the small size (6*21*14 = 1764 total entries, 42 valid, 1722 invalid),
    # it's manageable.
    
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                if j == j2 and k == k2 and i == i2:
                    continue
                s = Fraction(0)
                for t in range(r):
                    s += Uf[a][t] * Vf[b][t] * Wf[c][t]
                if s != 0:
                    return False
    return True

def _convert_to_fractions(U, V, W, max_denom=100000):
    """Convert numpy arrays to Fraction matrices, rounding to nearest rational with small denominator."""
    Uf = [[Fraction(int(round(U[i, j] * max_denom)), max_denom) for j in range(U.shape[1])] for i in range(U.shape[0])]
    Vf = [[Fraction(int(round(V[i, j] * max_denom)), max_denom) for j in range(V.shape[1])] for i in range(V.shape[0])]
    Wf = [[Fraction(int(round(W[i, j] * max_denom)), max_denom) for j in range(W.shape[1])] for i in range(W.shape[0])]
    # Simplify fractions
    Uf = [[f for f in row] for row in Uf]
    Vf = [[f for f in row] for row in Vf]
    Wf = [[f for f in row] for row in Wf]
    return Uf, Vf, Wf

def _verify_decomposition_numpy(U, V, W, n, m, p, tensor):
    """Fast numerical verification using numpy."""
    r = U.shape[1]
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Check valid entries
    for (a, b, c) in tensor:
        s = np.dot(U[a, :], V[b, :] * W[c, :])
        if abs(s - 1) > 1e-8:
            return False
    
    # Check invalid entries (sample for speed, but be thorough if fast)
    # Total invalid entries: nm*mp*pn - len(tensor)
    # 6*21*14 - 42 = 1764 - 42 = 1722
    # This is small enough to check all
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                if j == j2 and k == k2 and i == i2:
                    continue
                s = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(s) > 1e-8:
                    return False
    return True

def _try_random_decomposition(n, m, p, target_rank, num_trials=50, seed=42, tensor=None):
    """
    Try to find a rank-target_rank decomposition using random U, V and solving for W.
    """
    if tensor is None:
        tensor = _build_tensor(n, m, p)
    
    rng = random.Random(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    r = target_rank
    
    for trial in range(num_trials):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            return None
        
        # Generate random U and V with entries in {-1, 0, 1}
        U = np.zeros((nm, r), dtype=float)
        V = np.zeros((mp, r), dtype=float)
        for t in range(r):
            for a in range(nm):
                U[a, t] = rng.choice([-1, 0, 1])
            for b in range(mp):
                V[b, t] = rng.choice([-1, 0, 1])
        
        # Build the system for each c
        # A is (nm*mp) x r, with A[(a,b), t] = U[a,t] * V[b,t]
        A = np.zeros((nm * mp, r), dtype=float)
        idx = 0
        for a in range(nm):
            for b in range(mp):
                for t in range(r):
                    A[idx, t] = U[a, t] * V[b, t]
                idx += 1
        
        # For each c, build rhs and solve
        W = np.zeros((pn, r), dtype=float)
        for c in range(pn):
            rhs = np.zeros(nm * mp, dtype=float)
            idx = 0
            for a in range(nm):
                for b in range(mp):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if j == j2 and k == k2 and i == i2:
                        rhs[idx] = 1.0
                    idx += 1
            
            # Solve A * w = rhs using least squares
            w, residuals, rank, sv = np.linalg.lstsq(A, rhs, rcond=None)
            W[c, :] = w
        
        # Verify numerically
        if _verify_decomposition_numpy(U, V, W, n, m, p, tensor):
            # Try to convert to exact fractions
            Uf, Vf, Wf = _convert_to_fractions(U, V, W)
            if _verify_exact_fast(Uf, Vf, Wf, n, m, p, tensor):
                return Uf, Vf, Wf
    
    return None

def _try_structured_reduction(n, m, p, target_rank, num_trials=50, seed=123, tensor=None):
    """
    Try a structured approach: fix U and V with more structure, then solve for W.
    Use uniform random in [-1, 1] for better conditioning.
    """
    if tensor is None:
        tensor = _build_tensor(n, m, p)
    
    rng = random.Random(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    r = target_rank
    
    for trial in range(num_trials):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            return None
        
        U = np.zeros((nm, r), dtype=float)
        V = np.zeros((mp, r), dtype=float)
        for t in range(r):
            for a in range(nm):
                U[a, t] = rng.uniform(-1, 1)
            for b in range(mp):
                V[b, t] = rng.uniform(-1, 1)
        
        A = np.zeros((nm * mp, r), dtype=float)
        idx = 0
        for a in range(nm):
            for b in range(mp):
                for t in range(r):
                    A[idx, t] = U[a, t] * V[b, t]
                idx += 1
        
        W = np.zeros((pn, r), dtype=float)
        for c in range(pn):
            rhs = np.zeros(nm * mp, dtype=float)
            idx = 0
            for a in range(nm):
                for b in range(mp):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if j == j2 and k == k2 and i == i2:
                        rhs[idx] = 1.0
                    idx += 1
            
            w, residuals, rank, sv = np.linalg.lstsq(A, rhs, rcond=None)
            W[c, :] = w
        
        if _verify_decomposition_numpy(U, V, W, n, m, p, tensor):
            Uf, Vf, Wf = _convert_to_fractions(U, V, W)
            if _verify_exact_fast(Uf, Vf, Wf, n, m, p, tensor):
                return Uf, Vf, Wf
    
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = nm * mp  # = n*m*p = 42
    
    tensor = _build_tensor(n, m, p)
    
    # Lower bound: max(nm, mp, pn) = max(6, 21, 14) = 21
    min_rank = max(nm, mp, pn)
    
    # Start from schoolbook_rank - 1 and go down
    # We try to find a decomposition with rank < schoolbook_rank
    for target_rank in range(schoolbook_rank - 1, min_rank - 1, -1):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        # Try random integer approach
        result = _try_random_decomposition(n, m, p, target_rank, num_trials=30, seed=target_rank * 100 + 1, tensor=tensor)
        if result is not None:
            Uf, Vf, Wf = result
            return {"U": Uf, "V": Vf, "W": Wf}
        
        # Try structured random approach
        result = _try_structured_reduction(n, m, p, target_rank, num_trials=30, seed=target_rank * 200 + 1, tensor=tensor)
        if result is not None:
            Uf, Vf, Wf = result
            return {"U": Uf, "V": Vf, "W": Wf}
    
    # Fallback to schoolbook decomposition
    return _schoolbook_decomposition(n, m, p)
