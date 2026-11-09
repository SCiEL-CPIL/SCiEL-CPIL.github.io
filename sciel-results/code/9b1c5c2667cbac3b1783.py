"""Combined approach: structural Strassen base + numerical refinement for lower rank."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # First, try to find a decomposition with rank < 28 using numerical methods
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    
    # Try ranks from 24 down to 20
    best_result = None
    
    for target_rank in [24, 25, 26, 27]:
        if time.time() > deadline - 30:
            break
        
        result = _try_find_decomposition(n, m, p, target_rank, deadline)
        if result is not None:
            U, V, W = result
            # Verify
            if _verify(U, V, W, n, m, p):
                best_result = {"U": U, "V": V, "W": W}
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback: rank 28 block-Strassen
    U, V, W = _strassen_28(n, m, p)
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Verify the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                if abs(s - 1.0) > 1e-6:
                    return False
    # Check integer/half-integer constraint
    for mat in [U, V, W]:
        for row in mat:
            for x in row:
                if abs(x * 2 - round(x * 2)) > 1e-6:
                    return False
    return True


def _try_find_decomposition(n, m, p, rank, deadline):
    """Try to find a rank-`rank` decomposition using structured initialization + local search."""
    np.random.seed(42)
    
    # Strategy: Use the known 28-rank solution as a starting point,
    # then try to merge/eliminate terms to reduce rank.
    # Or: use random initialization with multiple restarts.
    
    n_starts = 30
    iters_per_start = 5000
    
    for s in range(n_starts):
        if time.time() > deadline - 25:
            break
        
        # Initialize with random values
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # Local search with gradient-like updates
        best_err = float('inf')
        stuck_count = 0
        
        for it in range(iters_per_start):
            if time.time() > deadline - 20:
                break
            
            # Compute error efficiently
            err = _compute_error(U, V, W, n, m, p)
            
            if err < best_err:
                best_err = err
                stuck_count = 0
                if err < 1e-10:
                    # Found a solution, try to round
                    U_r, V_r, W_r = _round_to_halfint(U, V, W)
                    if U_r is not None:
                        if _verify(U_r, V_r, W_r, n, m, p):
                            return U_r, V_r, W_r
            else:
                stuck_count += 1
            
            if stuck_count > 500:
                break
            
            # Random coordinate update
            mat_choice = np.random.randint(3)
            if mat_choice == 0:
                idx = np.random.randint(n * m)
                t = np.random.randint(rank)
                old = U[idx, t]
                U[idx, t] += np.random.randn() * 0.1
                new_err = _compute_error(U, V, W, n, m, p)
                if new_err > err * 1.001 + 1e-15:
                    U[idx, t] = old
            elif mat_choice == 1:
                idx = np.random.randint(m * p)
                t = np.random.randint(rank)
                old = V[idx, t]
                V[idx, t] += np.random.randn() * 0.1
                new_err = _compute_error(U, V, W, n, m, p)
                if new_err > err * 1.001 + 1e-15:
                    V[idx, t] = old
            else:
                idx = np.random.randint(p * n)
                t = np.random.randint(rank)
                old = W[idx, t]
                W[idx, t] += np.random.randn() * 0.1
                new_err = _compute_error(U, V, W, n, m, p)
                if new_err > err * 1.001 + 1e-15:
                    W[idx, t] = old
    
    return None


def _compute_error(U, V, W, n, m, p):
    """Compute the squared error of the decomposition."""
    err = 0.0
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                err += (s - 1.0) ** 2
    return err


def _round_to_halfint(U, V, W):
    """Try to round to nearest half-integers and check if still valid."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U_r]
    V_list = [[float(x) for x in row] for row in V_r]
    W_list = [[float(x) for x in row] for row in W_r]
    
    # Check if rounding preserved the solution
    n = U_r.shape[0] // 4  # This won't work generically, let's pass dims
    # Actually, we need dims. Let's just return and let caller verify.
    return U_list, V_list, W_list


def _strassen_28(n, m, p):
    """Construct the rank-28 Strassen-based decomposition."""
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1), (3, -1)]},
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1), (3, 1)]},
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
    ]
    
    blocks = [
        {'j_offset': 0, 'k_offset': 0},
        {'j_offset': 2, 'k_offset': 0},
        {'j_offset': 0, 'k_offset': 2},
        {'j_offset': 2, 'k_offset': 2}
    ]
    
    col = 0
    for block in blocks:
        j_off = block['j_offset']
        k_off = block['k_offset']
        for term in strassen_terms:
            for (idx, coeff) in term['U']:
                i = idx // 2
                j = idx % 2
                global_i = i
                global_j = j + j_off
                U[global_i * m + global_j][col] += coeff
            for (idx, coeff) in term['V']:
                j = idx // 2
                k = idx % 2
                global_j = j + j_off
                global_k = k + k_off
                V[global_j * p + global_k][col] += coeff
            for (idx, coeff) in term['W']:
                k = idx // 2
                i = idx % 2
                global_k = k + k_off
                global_i = i
                W[global_k * n + global_i][col] += coeff
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    U = [[int(round(x)) for x in row] for row in U]
    V = [[int(round(x)) for x in row] for row in V]
    W = [[int(round(x)) for x in row] for row in W]
    
    return U, V, W
