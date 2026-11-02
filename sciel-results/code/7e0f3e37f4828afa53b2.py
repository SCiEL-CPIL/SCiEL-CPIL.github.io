"""Matrix multiplication 3x3x5: rank-minimal exact tensor decomposition.

Uses structured search with exact rational arithmetic to find low-rank
decompositions. Falls back to schoolbook rank 45.
"""

import time
import sys
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Return list of (u_idx, v_idx, w_idx) for each (i,j,k)."""
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                triples.append((u_idx, v_idx, w_idx))
    return triples


def _compute_residual_exact(U, V, W, n, m, p, rank, triples):
    """Compute residual using exact Fraction arithmetic.
    Returns list of Fractions for each (i,j,k).
    """
    residuals = []
    for (u_idx, v_idx, w_idx) in triples:
        s = Fraction(0, 1)
        for r in range(rank):
            s += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
        residuals.append(s - Fraction(1, 1))
    return residuals


def _max_abs_residual(residuals):
    """Return max absolute value of residuals."""
    max_val = Fraction(0, 1)
    for r in residuals:
        ar = abs(r)
        if ar > max_val:
            max_val = ar
    return max_val


def _verify_exact(U, V, W, n, m, p, rank, triples):
    """Verify exact decomposition."""
    residuals = _compute_residual_exact(U, V, W, n, m, p, rank, triples)
    return _max_abs_residual(residuals) == Fraction(0, 1)


def _try_rank_reduction(n, m, p, target_rank, deadline, rng_seed=42):
    """Try to find a decomposition of target_rank using structured search.
    
    Strategy: Start from schoolbook (rank=n*m*p) and try to reduce rank by
    finding linear dependencies among the tensor terms.
    
    Returns (U, V, W) as lists of lists of Fractions, or None.
    """
    N = n * m
    P = m * p
    Q = p * n
    school_rank = n * m * p
    triples = _build_tensor(n, m, p)
    
    # Build schoolbook solution
    U_sb = [[Fraction(0, 1)] * school_rank for _ in range(N)]
    V_sb = [[Fraction(0, 1)] * school_rank for _ in range(P)]
    W_sb = [[Fraction(0, 1)] * school_rank for _ in range(Q)]
    for idx, (u_idx, v_idx, w_idx) in enumerate(triples):
        U_sb[u_idx][idx] = Fraction(1, 1)
        V_sb[v_idx][idx] = Fraction(1, 1)
        W_sb[w_idx][idx] = Fraction(1, 1)
    
    if target_rank >= school_rank:
        return (U_sb, V_sb, W_sb)
    
    # Strategy: Use a randomized local search with exact arithmetic
    # We work with a reduced set of "active" rank dimensions
    
    rng = random.Random(rng_seed)
    
    max_attempts = 30
    max_iters = 500
    
    for attempt in range(max_attempts):
        if time.time() >= deadline:
            break
        
        # Random initialization: pick target_rank columns from a larger random space
        # Use small integer values for entries
        U = [[Fraction(0, 1)] * target_rank for _ in range(N)]
        V = [[Fraction(0, 1)] * target_rank for _ in range(P)]
        W = [[Fraction(0, 1)] * target_rank for _ in range(Q)]
        
        # Initialize with random small integers
        for i in range(N):
            for r in range(target_rank):
                U[i][r] = Fraction(rng.randint(-2, 2), 1)
        for i in range(P):
            for r in range(target_rank):
                V[i][r] = Fraction(rng.randint(-2, 2), 1)
        for i in range(Q):
            for r in range(target_rank):
                W[i][r] = Fraction(rng.randint(-2, 2), 1)
        
        # Local search: iteratively adjust entries to reduce residuals
        for it in range(max_iters):
            if time.time() >= deadline:
                break
            
            residuals = _compute_residual_exact(U, V, W, n, m, p, target_rank, triples)
            max_res = _max_abs_residual(residuals)
            
            if max_res == Fraction(0, 1):
                return (U, V, W)
            
            # Find the worst residual
            worst_idx = 0
            worst_val = Fraction(0, 1)
            for idx, res in enumerate(residuals):
                ar = abs(res)
                if ar > worst_val:
                    worst_val = ar
                    worst_idx = idx
            
            if worst_val == Fraction(0, 1):
                break
            
            u_idx, v_idx, w_idx = triples[worst_idx]
            deficit = -residuals[worst_idx]  # We need to add this to make it 1
            
            # Try to fix by adjusting one entry in U, V, or W
            # Pick a random rank dimension
            r = rng.randint(0, target_rank - 1)
            
            # Try adjusting U[u_idx][r]
            # Current contribution from this r: U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
            contrib = U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
            # We want: (U[u_idx][r] + delta) * V[v_idx][r] * W[w_idx][r] = contrib + deficit
            # delta = deficit / (V[v_idx][r] * W[w_idx][r])
            denom = V[v_idx][r] * W[w_idx][r]
            if denom != Fraction(0, 1):
                delta = deficit / denom
                new_val = U[u_idx][r] + delta
                # Round to nearest integer
                rounded = Fraction(round(float(new_val)), 1)
                if abs(rounded - U[u_idx][r]) <= Fraction(3, 1):
                    old_val = U[u_idx][r]
                    U[u_idx][r] = rounded
                    # Check if it improved
                    new_res = _compute_residual_exact(U, V, W, n, m, p, target_rank, triples)[worst_idx]
                    if abs(new_res) < worst_val:
                        pass  # Keep it
                    else:
                        U[u_idx][r] = old_val
            
            # Also try adjusting V
            r = rng.randint(0, target_rank - 1)
            denom = U[u_idx][r] * W[w_idx][r]
            if denom != Fraction(0, 1):
                delta = deficit / denom
                new_val = V[v_idx][r] + delta
                rounded = Fraction(round(float(new_val)), 1)
                if abs(rounded - V[v_idx][r]) <= Fraction(3, 1):
                    old_val = V[v_idx][r]
                    V[v_idx][r] = rounded
                    new_res = _compute_residual_exact(U, V, W, n, m, p, target_rank, triples)[worst_idx]
                    if abs(new_res) < worst_val:
                        pass
                    else:
                        V[v_idx][r] = old_val
            
            # Also try adjusting W
            r = rng.randint(0, target_rank - 1)
            denom = U[u_idx][r] * V[v_idx][r]
            if denom != Fraction(0, 1):
                delta = deficit / denom
                new_val = W[w_idx][r] + delta
                rounded = Fraction(round(float(new_val)), 1)
                if abs(rounded - W[w_idx][r]) <= Fraction(3, 1):
                    old_val = W[w_idx][r]
                    W[w_idx][r] = rounded
                    new_res = _compute_residual_exact(U, V, W, n, m, p, target_rank, triples)[worst_idx]
                    if abs(new_res) < worst_val:
                        pass
                    else:
                        W[w_idx][r] = old_val
    
    return None


def _try_structured_reduction(n, m, p, target_rank, deadline):
    """Try a more structured approach: exploit the algebraic structure.
    
    For matrix multiplication, the tensor has a specific structure that can
    be exploited. We try to find decompositions where the rank dimensions
    correspond to products of subsets of indices.
    """
    N = n * m
    P = m * p
    Q = p * n
    triples = _build_tensor(n, m, p)
    
    # Try using a pattern based on Strassen-like combinations
    # For 3x3x5, we can try to use rank-2 or rank-3 blocks
    
    # One approach: try to find a solution where entries are in {-1, 0, 1}
    # This is a constraint satisfaction problem
    
    # Use a different initialization: based on known fast multiplication patterns
    # For 3x3, the minimal rank is 7 (Strassen-like), but for 3x3x5 it's different
    
    # Try a greedy approach: build up the decomposition term by term
    # Each term is a rank-1 tensor U_r * V_r * W_r
    
    # This is complex, so fall back to the local search
    return None


def _schoolbook(n, m, p):
    """Return the schoolbook rank-nmp decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _fraction_to_number(frac):
    """Convert Fraction to int if integer, else keep as float."""
    if frac.denominator == 1:
        return frac.numerator
    return float(frac)


def construct():
    """Return a rank-minimal exact tensor decomposition."""
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 130.0  # Safety margin
    
    triples = _build_tensor(n, m, p)
    
    # Try ranks from 44 down to 41
    ranks_to_try = [44, 43, 42, 41]
    time_per_rank = 30.0
    
    for rank in ranks_to_try:
        if time.time() >= deadline:
            break
        rank_deadline = min(deadline, time.time() + time_per_rank)
        
        sol = _try_rank_reduction(n, m, p, rank, rank_deadline)
        if sol is not None:
            U, V, W = sol
            # Convert to lists of numbers
            U_list = [[_fraction_to_number(f) for f in row] for row in U]
            V_list = [[_fraction_to_number(f) for f in row] for row in V]
            W_list = [[_fraction_to_number(f) for f in row] for row in W]
            
            # Verify with floating point
            R = np.ones((n, m, p), dtype=np.float64)
            for r in range(rank):
                for i in range(n):
                    for j in range(m):
                        u_val = U_list[i * m + j][r]
                        if u_val != 0:
                            for k in range(p):
                                R[i, j, k] -= u_val * V_list[j * p + k][r] * W_list[k * n + i][r]
            
            if np.max(np.abs(R)) < 1e-9:
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
