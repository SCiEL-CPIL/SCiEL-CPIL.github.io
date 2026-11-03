"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_decomposition():
    """Build the rank-42 Strassen-based decomposition."""
    n, m, p = 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1},
    ]
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1},
    ]
    W_contrib = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1},
    ]
    
    col = 0
    for a in range(2):
        for b in range(3):
            for s in range(7):
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    n, m, p = 2, 4, 6
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the multiplication tensor from U, V, W
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where i indexes A (n*m), j indexes B (m*p), k indexes C (p*n)
    # C[k*n+i] = sum_j A[i*m+j] * B[j*p+k]
    # So T[i,j,k] should equal 1 if k is the output index for A row i and B col k, else 0
    
    # The tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # We need T[i,j,k] = delta_{i//m, j//p} * (j % p == k % n) ... 
    # Actually: C[i, k] = sum_j A[i,j] * B[j,k]
    # In flattened: C[k*n + i] = sum_j A[i*m + j] * B[j*p + k]
    # So the tensor T[i_flat, j_flat, k_flat] = 1 if i_flat = i*m+j, j_flat = j*p+k, k_flat = k*n+i for some valid i,j,k
    # More precisely: T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0
    
    # Let's compute the tensor from U,V,W
    # T = sum_r outer(U[:,r], V[:,r], W[:,r])
    # Use einsum: T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    T_computed = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Build the target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_target[a_idx, b_idx, c_idx] = 1.0
    
    diff = np.max(np.abs(T_computed - T_target))
    return diff < 1e-10, diff

def _try_rank_reduction(U_init, V_init, W_init, target_rank, max_time=60.0):
    """Try to reduce rank by zeroing out columns and verifying."""
    start_time = time.time()
    U_np = np.array(U_init, dtype=float).copy()
    V_np = np.array(V_init, dtype=float).copy()
    W_np = np.array(W_init, dtype=float).copy()
    n, m, p = 2, 4, 6
    
    # Target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    current_rank = U_np.shape[1]
    best_rank = current_rank
    best_result = {"U": U_init, "V": V_init, "W": W_init}
    
    # Try removing one column at a time
    # Use a while loop to allow restarting after successful removal
    removed_any = True
    while removed_any:
        removed_any = False
        for r_remove in range(current_rank):
            if time.time() - start_time > max_time * 0.8:
                if current_rank < best_rank:
                    return {
                        "U": U_np.tolist(),
                        "V": V_np.tolist(),
                        "W": W_np.tolist()
                    }
                else:
                    return None
            
            U_rem = np.delete(U_np, r_remove, axis=1)
            V_rem = np.delete(V_np, r_remove, axis=1)
            W_rem = np.delete(W_np, r_remove, axis=1)
            
            # Check if T_target is in the span of the remaining columns
            T_rem = np.einsum('ir,jr,kr->ijk', U_rem, V_rem, W_rem)
            residual = T_target - T_rem
            
            if np.abs(residual).max() < 1e-10:
                # Residual is zero, so we can remove this column!
                best_rank = min(best_rank, current_rank - 1)
                U_np = U_rem
                V_np = V_rem
                W_np = W_rem
                current_rank -= 1
                removed_any = True
                break
        
        if time.time() - start_time > max_time * 0.8:
            break
    
    if best_rank < 42:
        return {
            "U": U_np.tolist(),
            "V": V_np.tolist(),
            "W": W_np.tolist()
        }
    return None

def _standard_decomposition():
    """Build the standard rank-48 decomposition."""
    n, m, p = 2, 4, 6
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx][col] = 1
                V[b_idx][col] = 1
                W[c_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    
    # Start with the known rank-42 decomposition
    result_42 = _strassen_decomposition()
    
    # Verify the initial decomposition
    valid, diff = _verify_decomposition(result_42["U"], result_42["V"], result_42["W"])
    
    if not valid:
        # Fallback to standard rank-48
        result_std = _standard_decomposition()
        valid_std, _ = _verify_decomposition(result_std["U"], result_std["V"], result_std["W"])
        if valid_std:
            return result_std
        return result_42 # Should not happen, but return something
    
    best_result = result_42
    best_rank = 42
    
    # Try rank reduction
    if time.time() - start_time < deadline - 30.0:
        reduced = _try_rank_reduction(
            result_42["U"], result_42["V"], result_42["W"],
            target_rank=41,
            max_time=min(60.0, deadline - (time.time() - start_time) - 10.0)
        )
        if reduced is not None:
            # Verify the reduced decomposition
            valid_r, diff_r = _verify_decomposition(reduced["U"], reduced["V"], reduced["W"])
            if valid_r:
                new_rank = len(reduced["U"][0])
                if new_rank < best_rank:
                    best_result = reduced
                    best_rank = new_rank
    
    # Final check
    if time.time() < deadline - 5.0:
        valid_final, diff_final = _verify_decomposition(
            best_result["U"], best_result["V"], best_result["W"]
        )
        if not valid_final:
            # Fallback to standard rank-48
            result_std = _standard_decomposition()
            valid_std, _ = _verify_decomposition(result_std["U"], result_std["V"], result_std["W"])
            if valid_std:
                return result_std
            # If even standard fails (should not), return best effort
            return best_result
    
    return best_result

# EVOLVE-BLOCK-END
