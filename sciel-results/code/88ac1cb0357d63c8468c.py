"""
Rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.

Strategy:
- Use the schoolbook decomposition as a safe fallback (rank 42).
- Attempt to find a lower-rank decomposition via structured search.
- Key insight: We can exploit the structure by considering the problem as
  computing 7 independent 2x3 * 3x1 multiplications, but sharing structure.
- Use a combination of:
  1. Fast numerical verification with numpy
  2. Structured search trying to reduce rank from schoolbook
  3. Exact rational verification only for candidates that pass numerical check
"""

import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast numerical verification using numpy. Returns True if likely correct."""
    rank = U.shape[1]
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # For each (i, j, k), check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # Vectorized: build the full tensor and check
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(total - 1.0) > tol:
                    return False
    return True

def _verify_decomposition_exact(U, V, W, n, m, p):
    """Exact rational verification. Only call on candidates that pass fast check."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    u_val = Fraction(str(U[i*m+j][c]))
                    v_val = Fraction(str(V[j*p+k][c]))
                    w_val = Fraction(str(W[k*n+i][c]))
                    total += u_val * v_val * w_val
                if total != 1:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _try_constructed_decomposition(n, m, p, deadline):
    """
    Try a structured approach:
    For 2x3 * 3x7, we can think of this as:
    - 7 columns of B, each 3x1
    - For each column, compute A @ b (2x3 @ 3x1 -> 2x1)
    
    The 2x3 * 3x1 multiplication has a known minimal rank.
    Actually, let's try a different structured approach:
    Use the fact that we can share some computations across columns.
    
    One idea: partition the 7 columns into groups and use shared bilinear forms.
    """
    # For now, try a simple structured construction:
    # Use rank = 2*3 + 3*7 - 3 = 6 + 21 - 3 = 24? No, that's not quite right.
    # 
    # Actually, let's try: for each of the 3 intermediate indices j,
    # we have n*p = 14 products A[i,j]*B[j,k] for i in {0,1}, k in {0..6}.
    # Total schoolbook: 3 * 14 = 42.
    #
    # Can we do better? For fixed j, we're computing a 2x7 outer product
    # of vector A[:,j] (length 2) and vector B[j,:] (length 7).
    # The rank of this bilinear map is at most 2*7 = 14, but actually
    # for a 2x7 outer product, the minimal rank is... 
    # The tensor for 2x1 * 1x7 -> 2x7 has rank 14 (schoolbook) 
    # but can it be lower? For n=2, p=7, the rank of the 2x7 outer product
    # tensor is 14 (no known improvement for non-square).
    #
    # So per j, we need at least 14, total at least 42? That can't be right
    # because we can share across j.
    #
    # Let me try a different approach: just do a smarter random search
    # with proper initialization.
    pass

def _polish_decomposition(U, V, W, n, m, p, max_iters=200, tol=1e-12):
    """
    Gradient-based polish: minimize sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
    using simple gradient descent with adaptive step size.
    Returns (U, V, W, loss) after polishing.
    """
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # Precompute index mappings
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    lr = 0.01
    prev_loss = float('inf')
    
    for it in range(max_iters):
        # Compute loss and gradients
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for (i, j, k) in triples:
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            
            # current value
            val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
            err = val - 1.0
            loss += err * err
            
            # Gradients: d/dU[u_idx,c] = err * V[v_idx,c] * W[w_idx,c]
            gw = V[v_idx] * W[w_idx]
            grad_U[u_idx] += err * gw
            grad_V[v_idx] += err * U[u_idx] * W[w_idx]
            grad_W[w_idx] += err * U[u_idx] * V[v_idx]
        
        loss *= 0.5  # for the 0.5 * sum convention
        
        # Adaptive learning rate
        grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
        if grad_norm < 1e-15:
            break
        
        # Simple gradient step
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Adaptive lr
        if loss < prev_loss * 0.95:
            lr = min(lr * 1.1, 0.1)
        elif loss > prev_loss:
            lr *= 0.5
        prev_loss = loss
        
        if loss < tol:
            break
    
    return U, V, W, loss


def _try_polish_from_schoolbook(n, m, p, target_rank, deadline):
    """
    Initialize from a subset of schoolbook terms + random fill, then polish.
    """
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p
    
    U = np.zeros((nm, target_rank), dtype=np.float64)
    V = np.zeros((mp, target_rank), dtype=np.float64)
    W = np.zeros((pn, target_rank), dtype=np.float64)
    
    # Fill with small random values
    U += np.random.randn(nm, target_rank) * 0.5
    V += np.random.randn(mp, target_rank) * 0.5
    W += np.random.randn(pn, target_rank) * 0.5
    
    # Set up a few schoolbook-like anchors to help convergence
    num_anchors = min(target_rank, 10)
    for c in range(num_anchors):
        i = c % n
        j = (c // n) % m
        k = (c // (n * m)) % p
        U[i*m+j, c] = 1.0
        V[j*p+k, c] = 1.0
        W[k*n+i, c] = 1.0
    
    # Polish
    U, V, W, loss = _polish_decomposition(U, V, W, n, m, p, max_iters=300, tol=1e-10)
    
    if loss < 1e-8:
        # Snap to nearest half-integers and verify
        U_snapped = np.round(U * 2) / 2
        V_snapped = np.round(V * 2) / 2
        W_snapped = np.round(W * 2) / 2
        
        if _verify_decomposition_fast(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-9):
            U_list = U_snapped.tolist()
            V_list = V_snapped.tolist()
            W_list = W_snapped.tolist()
            if _verify_decomposition_exact(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
    
    return None


def _random_search_for_rank(n, m, p, target_rank, max_attempts, deadline):
    """
    Try to find a decomposition of given rank using random search + polish.
    """
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Try polish-based approach
        if attempt < max_attempts // 2:
            result = _try_polish_from_schoolbook(n, m, p, target_rank, deadline)
            if result is not None:
                return result
            continue
        
        # Pure random search with small integers
        U = np.random.randint(-3, 4, (nm, target_rank)).astype(np.float64)
        V = np.random.randint(-3, 4, (mp, target_rank)).astype(np.float64)
        W = np.random.randint(-3, 4, (pn, target_rank)).astype(np.float64)
        
        # Fast numerical check
        if _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-6):
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if _verify_decomposition_exact(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
        
        # Try polishing the random init
        U_p, V_p, W_p, loss = _polish_decomposition(U, V, W, n, m, p, max_iters=100, tol=1e-10)
        if loss < 1e-8:
            U_snapped = np.round(U_p * 2) / 2
            V_snapped = np.round(V_p * 2) / 2
            W_snapped = np.round(W_p * 2) / 2
            if _verify_decomposition_fast(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-9):
                U_list = U_snapped.tolist()
                V_list = V_snapped.tolist()
                W_list = W_snapped.tolist()
                if _verify_decomposition_exact(U_list, V_list, W_list, n, m, p):
                    return U_list, V_list, W_list
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p
    
    # Lower bound: max(nm, mp, pn) = max(6, 21, 14) = 21
    # Try ranks from 21 to 41, with more budget for lower ranks
    
    for target_rank in range(21, schoolbook_rank):
        if time.time() > deadline - 15:
            break
        
        # Allocate more attempts to lower ranks (more likely to be achievable)
        remaining_time = deadline - 15 - time.time()
        if remaining_time <= 0:
            break
        
        # Budget per rank: more for lower ranks
        rank_priority = 1.0 / (target_rank - 20)  # higher priority for lower rank
        base_attempts = 30
        attempts = int(base_attempts * rank_priority)
        attempts = max(10, min(attempts, 100))
        
        # Time budget per rank
        time_per_rank = min(remaining_time * 0.3, 20)
        rank_deadline = time.time() + time_per_rank
        
        result = _random_search_for_rank(n, m, p, target_rank, max_attempts=attempts, deadline=rank_deadline)
        if result is not None:
            U_list, V_list, W_list = result
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no lower rank found, return schoolbook
    return _schoolbook()
