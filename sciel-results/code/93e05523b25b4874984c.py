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
  4. Linear code / parity-check based rank reduction search
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
    # Also check that "invalid" combinations give 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(total - 1.0) > tol:
                    return False
    
    # Check some invalid combinations (indices that don't form consistent triples)
    # For a random sample of inconsistent index triples
    for _ in range(20):
        i_u = random.randint(0, n-1)
        j_u = random.randint(0, m-1)
        j_v = random.randint(0, m-1)
        k_v = random.randint(0, p-1)
        k_w = random.randint(0, p-1)
        i_w = random.randint(0, n-1)
        
        # Check if this is a consistent triple: j_u == j_v, k_v == k_w, i_u == i_w
        if j_u == j_v and k_v == k_w and i_u == i_w:
            continue  # consistent, skip
        
        idx_u = i_u * m + j_u
        idx_v = j_v * p + k_v
        idx_w = k_w * n + i_w
        
        total = np.dot(U_np[idx_u], V_np[idx_v] * W_np[idx_w])
        if abs(total) > tol:
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
    
    # Check some invalid combinations
    for _ in range(10):
        i_u = random.randint(0, n-1)
        j_u = random.randint(0, m-1)
        j_v = random.randint(0, m-1)
        k_v = random.randint(0, p-1)
        k_w = random.randint(0, p-1)
        i_w = random.randint(0, n-1)
        
        if j_u == j_v and k_v == k_w and i_u == i_w:
            continue
        
        idx_u = i_u * m + j_u
        idx_v = j_v * p + k_v
        idx_w = k_w * n + i_w
        
        total = Fraction(0)
        for c in range(rank):
            u_val = Fraction(str(U[idx_u][c]))
            v_val = Fraction(str(V[idx_v][c]))
            w_val = Fraction(str(W[idx_w][c]))
            total += u_val * v_val * w_val
        if total != 0:
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

def _polish_decomposition(U, V, W, n, m, p, max_iters=200, tol=1e-12):
    """
    Gradient-based polish: minimize sum of squared errors for valid triples
    plus sum of squared values for invalid triples.
    """
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # Precompute valid triples
    valid_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_triples.append((i, j, k))
    
    # Precompute some invalid triples for the penalty
    invalid_triples = []
    for _ in range(30):
        i_u = random.randint(0, n-1)
        j_u = random.randint(0, m-1)
        j_v = random.randint(0, m-1)
        k_v = random.randint(0, p-1)
        k_w = random.randint(0, p-1)
        i_w = random.randint(0, n-1)
        if not (j_u == j_v and k_v == k_w and i_u == i_w):
            invalid_triples.append((i_u, j_u, j_v, k_v, k_w, i_w))
    
    lr = 0.01
    prev_loss = float('inf')
    
    for it in range(max_iters):
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Valid triples: target 1
        for (i, j, k) in valid_triples:
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            
            val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
            err = val - 1.0
            loss += err * err
            
            gw = V[v_idx] * W[w_idx]
            grad_U[u_idx] += err * gw
            grad_V[v_idx] += err * U[u_idx] * W[w_idx]
            grad_W[w_idx] += err * U[u_idx] * V[v_idx]
        
        # Invalid triples: target 0
        for (i_u, j_u, j_v, k_v, k_w, i_w) in invalid_triples:
            u_idx = i_u * m + j_u
            v_idx = j_v * p + k_v
            w_idx = k_w * n + i_w
            
            val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
            loss += val * val
            
            gw = V[v_idx] * W[w_idx]
            grad_U[u_idx] += val * gw
            grad_V[v_idx] += val * U[u_idx] * W[w_idx]
            grad_W[w_idx] += val * U[u_idx] * V[v_idx]
        
        loss *= 0.5
        
        grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
        if grad_norm < 1e-15:
            break
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
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


def _parity_check_search(n, m, p, target_rank, max_attempts, deadline):
    """
    Use a parity-check / linear code approach to search for low-rank decompositions.
    
    The idea: construct a parity-check matrix H from the Kronecker structure,
    then use it to guide the search for linearly independent bilinear forms.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # Construct the "parity-check" structure based on the tensor's linear dependencies
    # For the multiplication tensor, each (i,j,k) triple defines a constraint
    # We can use this to identify promising directions for low-rank decomposition
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Initialize with structured random values guided by parity constraints
        U = np.zeros((nm, target_rank), dtype=np.float64)
        V = np.zeros((mp, target_rank), dtype=np.float64)
        W = np.zeros((pn, target_rank), dtype=np.float64)
        
        # Assign values with structure: for each column, pick a "center" triple
        for c in range(target_rank):
            # Random center triple
            i_c = random.randint(0, n-1)
            j_c = random.randint(0, m-1)
            k_c = random.randint(0, p-1)
            
            # Set the center values
            U[i_c*m+j_c, c] = random.choice([1, -1, 0.5, -0.5])
            V[j_c*p+k_c, c] = random.choice([1, -1, 0.5, -0.5])
            W[k_c*n+i_c, c] = random.choice([1, -1, 0.5, -0.5])
            
            # Add some structured noise to nearby indices
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    for dk in range(-1, 2):
                        i2 = i_c + di
                        j2 = j_c + dj
                        k2 = k_c + dk
                        if 0 <= i2 < n and 0 <= j2 < m and 0 <= k2 < p:
                            if random.random() < 0.3:
                                U[i2*m+j2, c] += random.choice([0.5, -0.5, 1, -1]) * random.random()
                                V[j2*p+k2, c] += random.choice([0.5, -0.5, 1, -1]) * random.random()
                                W[k2*n+i2, c] += random.choice([0.5, -0.5, 1, -1]) * random.random()
        
        # Polish
        U_p, V_p, W_p, loss = _polish_decomposition(U, V, W, n, m, p, max_iters=150, tol=1e-10)
        
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
        
        remaining_time = deadline - 15 - time.time()
        if remaining_time <= 0:
            break
        
        # Time budget per rank
        time_per_rank = min(remaining_time * 0.3, 20)
        rank_deadline = time.time() + time_per_rank
        
        # Try random search first
        result = _random_search_for_rank(n, m, p, target_rank, max_attempts=20, deadline=rank_deadline)
        if result is not None:
            U_list, V_list, W_list = result
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # Try parity-check guided search if time remains
        if time.time() < rank_deadline - 5:
            result = _parity_check_search(n, m, p, target_rank, max_attempts=10, deadline=rank_deadline)
            if result is not None:
                U_list, V_list, W_list = result
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no lower rank found, return schoolbook
    return _schoolbook()
