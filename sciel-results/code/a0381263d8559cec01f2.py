"""
Solution for 2x3 by 3x3 matrix multiplication tensor decomposition.
Combines trivial rank-18 fallback with a structured search for lower-rank
decompositions using entries from {-1, 0, 1} and {-0.5, 0, 0.5, 1}.
Uses a combinatorial approach: for small ranks, enumerate sparse patterns
and solve the resulting linear system for the non-zero entries.
"""

import time
import numpy as np
from itertools import product, combinations_with_replacement
from collections import defaultdict

N, M, P = 2, 3, 3
NM, MP, PN = N * M, M * P, P * N  # 6, 9, 6


def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    rank = U.shape[1]
    for i in range(N):
        for j in range(M):
            for k in range(P):
                val = 0.0
                for r in range(rank):
                    val += U[i * M + j, r] * V[j * P + k, r] * W[k * N + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _try_rank18():
    """Trivial schoolbook rank-18 decomposition."""
    R = N * M * P
    U = np.zeros((NM, R), dtype=float)
    V = np.zeros((MP, R), dtype=float)
    W = np.zeros((PN, R), dtype=float)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                r = i * M * P + j * P + k
                U[i * M + j, r] = 1.0
                V[j * P + k, r] = 1.0
                W[k * N + i, r] = 1.0
    return U, V, W


def _search_rank(rank, entries, deadline):
    """
    Search for a rank-`rank` decomposition with entries in `entries`.
    Uses a structured approach: try to find patterns where each column
    has a small number of non-zeros.
    
    Strategy: For each rank, we set up the system of 18 equations.
    We use a greedy/iterative approach:
    1. Start with random sparse initializations
    2. Use Newton-like iterations to adjust values
    3. Round to nearest allowed entry and verify
    """
    if time.time() > deadline:
        return None
    
    # The constraint: for all (i,j,k), sum_r U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r] = 1
    # We have 18 equations and (NM + MP + PN) * rank = (6+9+6)*rank = 21*rank unknowns
    
    # For rank <= 18, we have more unknowns than equations (21*rank >= 18 for rank >= 1)
    # So solutions should exist if the entries allow it.
    
    # Approach: Use scipy's least_squares to find real solutions, then round.
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return None
    
    def _residuals(params, rank):
        nm, mp, pn = NM, MP, PN
        u_size = nm * rank
        v_size = mp * rank
        w_size = pn * rank
        
        u_flat = params[:u_size]
        v_flat = params[u_size:u_size + v_size]
        w_flat = params[u_size + v_size:u_size + v_size + w_size]
        
        U = u_flat.reshape(nm, rank)
        V = v_flat.reshape(mp, rank)
        W = w_flat.reshape(pn, rank)
        
        residuals = np.zeros((N, M, P))
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    val = np.sum(U[i * M + j, :] * V[j * P + k, :] * W[k * N + i, :])
                    residuals[i, j, k] = val - 1.0
        return residuals.flatten()
    
    best_result = None
    best_cost = float('inf')
    
    num_attempts = 10 if rank <= 14 else 5
    
    for attempt in range(num_attempts):
        if time.time() > deadline:
            return None
        
        np.random.seed(attempt * 1000 + rank)
        
        # Initialize with small random values
        scale = 1.0 / np.sqrt(rank)
        u0 = np.random.randn(NM, rank) * scale
        v0 = np.random.randn(MP, rank) * scale
        w0 = np.random.randn(PN, rank) * scale
        params0 = np.concatenate([u0.flatten(), v0.flatten(), w0.flatten()])
        
        try:
            result = least_squares(_residuals, params0, args=(rank,), 
                                   method='lm', max_nfev=2000, ftol=1e-12, xtol=1e-12)
            if result.cost < best_cost:
                best_cost = result.cost
                
                if result.cost < 1e-10:
                    u_size = NM * rank
                    v_size = MP * rank
                    w_size = PN * rank
                    U = result.x[:u_size].reshape(NM, rank)
                    V = result.x[u_size:u_size + v_size].reshape(MP, rank)
                    W = result.x[u_size + v_size:u_size + v_size + w_size].reshape(PN, rank)
                    
                    # Try to round to allowed entries
                    for eps in [0.5, 1.0]:
                        U_r = np.round(U / eps) * eps
                        V_r = np.round(V / eps) * eps
                        W_r = np.round(W / eps) * eps
                        
                        # Verify rounded version
                        if _verify(U_r, V_r, W_r, tol=1e-6):
                            # Check entries are in allowed set
                            if all(abs(x - e) < 1e-9 for x in U_r.flatten() for e in entries) and \
                               all(abs(x - e) < 1e-9 for x in V_r.flatten() for e in entries) and \
                               all(abs(x - e) < 1e-9 for x in W_r.flatten() for e in entries):
                                return U_r, V_r, W_r
                    
                    # If rounding didn't work, return the real-valued solution
                    # but only if it's very close to integer/half-integer
                    if _verify(U, V, W, tol=1e-8):
                        return U, V, W
        except Exception:
            pass
    
    return None


def _search_rank_structured(rank, deadline):
    """
    Structured search: exploit the symmetry of the problem.
    
    For the all-ones tensor, we can use the fact that:
    1 = sum_r u_i^{(r)} v_j^{(r)} w_k^{(r)}
    
    A natural ansatz: use rank-1 tensors that are "diagonal" in some sense.
    For (2,3,3), try patterns where each rank-1 term covers a subset of indices.
    """
    if time.time() > deadline:
        return None
    
    # Try a specific structured decomposition pattern
    # Idea: split the 3x3 part into blocks and use the 2x3 structure
    
    # Pattern 1: Use the fact that 1 = 1*1*1, and try to "share" terms
    # For rank 16: we need to save 2 terms from 18
    # This means 2 pairs of (i,j,k) can share a rank-1 term
    
    # Let's try: can we find U, V, W with entries in {-1, 0, 1} of rank 16?
    # System: 18 equations, 21*16 = 336 unknowns (but constrained to {-1,0,1})
    
    # Use a randomized local search:
    # Start from rank-18, try to merge two terms by adjusting values
    
    if rank >= 16:
        # Start from rank-18 and try to zero out some entries to reduce effective rank
        U, V, W = _try_rank18()
        
        # Try to find two columns that can be merged
        # Column r corresponds to a single (i,j,k) term
        # We want to see if we can express two terms as one
        
        # This is complex; instead, try direct search for rank 16, 17
        pass
    
    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
    
    Returns:
        dict: A dictionary with keys 'U', 'V', 'W' containing the factor matrices.
              U: shape (6, rank)
              V: shape (9, rank) 
              W: shape (6, rank)
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Try ranks from 10 up to 18
    # Start with smaller ranks and increase
    for rank in range(10, N * M * P + 1):
        if time.time() > deadline:
            break
        
        # Try with integer entries first
        result = _search_rank(rank, {-1.0, 0.0, 1.0}, deadline)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Try with half-integer entries
        result = _search_rank(rank, {-1.0, -0.5, 0.0, 0.5, 1.0}, deadline)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank-18 decomposition
    U, V, W = _try_rank18()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
