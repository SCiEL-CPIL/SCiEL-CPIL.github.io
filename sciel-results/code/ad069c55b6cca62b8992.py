"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _get_rank17_solution():
    """Returns the known rank-17 solution using Strassen block."""
    n, m, p = 2, 3, 3
    n_a, n_b, n_c = 6, 9, 6
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    def set_term(r, u_dict, v_dict, w_dict):
        for a, val in u_dict.items():
            U[a, r] = val
        for b, val in v_dict.items():
            V[b, r] = val
        for c, val in w_dict.items():
            W[c, r] = val
            
    # Part 1: Strassen for A1(2x2) x B1_left(2x2) -> C[0:2, 0:2]
    set_term(0, {0:1, 4:1}, {0:1, 4:1}, {0:1, 3:1})
    set_term(1, {3:1, 4:1}, {0:1}, {1:1, 3:-1})
    set_term(2, {0:1}, {1:1, 4:-1}, {2:1, 3:1})
    set_term(3, {4:1}, {3:1, 0:-1}, {0:1, 1:1})
    set_term(4, {0:1, 1:1}, {4:1}, {0:-1, 2:1})
    set_term(5, {3:1, 0:-1}, {0:1, 1:1}, {3:1})
    set_term(6, {1:1, 4:-1}, {3:1, 4:1}, {0:1})
    
    # Part 2: Direct for A1(2x2) x B1_right(2x1) -> C[:, 2]
    set_term(7, {0:1}, {2:1}, {4:1})
    set_term(8, {1:1}, {5:1}, {4:1})
    set_term(9, {3:1}, {2:1}, {5:1})
    set_term(10, {4:1}, {5:1}, {5:1})
    
    # Part 3: Direct for A2(2x1) x B2(1x3) -> C[0:2, 0:3]
    set_term(11, {2:1}, {6:1}, {0:1})
    set_term(12, {2:1}, {7:1}, {2:1})
    set_term(13, {2:1}, {8:1}, {4:1})
    set_term(14, {5:1}, {6:1}, {1:1})
    set_term(15, {5:1}, {7:1}, {3:1})
    set_term(16, {5:1}, {8:1}, {5:1})
    
    return U, V, W

def _try_absorb_term(U17, V17, W17, T, remove_idx, deadline):
    """
    Try to absorb the contribution of term `remove_idx` into the other 16 terms.
    We fix the supports of the remaining 16 terms (same as in rank-17) and solve
    for scalar multipliers that make the sum equal T.
    
    The idea: sum_{r != remove_idx} alpha_r * (u_r * v_r * w_r) = T
    This is a linear system in the alpha_r's, but only if the 16 rank-1 tensors
    span T. In general they won't, so we also try small integer perturbations
    to the supports.
    """
    n_a, n_b, n_c = U17.shape[0], V17.shape[0], W17.shape[0]
    rank17 = U17.shape[1]
    
    # Build the 16 rank-1 tensors (excluding remove_idx)
    keep_indices = [r for r in range(rank17) if r != remove_idx]
    
    # Flatten T
    T_flat = T.flatten()
    num_entries = T_flat.size
    
    # Build matrix A where column r is the flattened rank-1 tensor u_r * v_r * w_r
    A = np.zeros((num_entries, len(keep_indices)))
    for j, r in enumerate(keep_indices):
        A[:, j] = np.einsum('a,b,c->abc', U17[:, r], V17[:, r], W17[:, r]).flatten()
    
    # Solve least squares
    try:
        alpha, residuals, rank_A, sv = np.linalg.lstsq(A, T_flat, rcond=None)
    except np.linalg.LinAlgError:
        return None
    
    # Check if the solution is exact (within tolerance)
    T_recon = A @ alpha
    if np.allclose(T_recon, T_flat, atol=1e-9):
        # Build the rank-16 solution
        U16 = np.zeros((n_a, 16))
        V16 = np.zeros((n_b, 16))
        W16 = np.zeros((n_c, 16))
        for j, r in enumerate(keep_indices):
            U16[:, j] = alpha[j] * U17[:, r]
            V16[:, j] = alpha[j] * V17[:, r]
            W16[:, j] = alpha[j] * W17[:, r]
        return U16, V16, W16
    
    # If direct absorption fails, try small integer perturbations to supports
    # For each of the 16 kept terms, try adding/removing one non-zero entry
    # This is a heuristic: we try to find a nearby rank-16 decomposition
    
    # Get the current supports
    supports_U = [set(np.nonzero(U17[:, r])[0]) for r in keep_indices]
    supports_V = [set(np.nonzero(V17[:, r])[0]) for r in keep_indices]
    supports_W = [set(np.nonzero(W17[:, r])[0]) for r in keep_indices]
    
    # Try perturbing one support at a time
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Pick a random term and a random perturbation
        j = np.random.randint(0, 16)
        r = keep_indices[j]
        
        # Try adding a random entry to one of the three vectors
        mat_choice = np.random.randint(0, 3)
        if mat_choice == 0:
            new_idx = np.random.randint(0, n_a)
            if new_idx not in supports_U[j]:
                supports_U[j].add(new_idx)
                # Rebuild and solve
                U_test = np.zeros((n_a, 16))
                V_test = np.zeros((n_b, 16))
                W_test = np.zeros((n_c, 16))
                for jj, rr in enumerate(keep_indices):
                    if jj == j:
                        U_test[new_idx, jj] = 1
                        for idx in supports_V[j]:
                            V_test[idx, jj] = 1
                        for idx in supports_W[j]:
                            W_test[idx, jj] = 1
                    else:
                        U_test[:, jj] = U17[:, rr]
                        V_test[:, jj] = V17[:, rr]
                        W_test[:, jj] = W17[:, rr]
                
                A_test = np.zeros((num_entries, 16))
                for jj in range(16):
                    A_test[:, jj] = np.einsum('a,b,c->abc', U_test[:, jj], V_test[:, jj], W_test[:, jj]).flatten()
                
                try:
                    alpha_test, _, _, _ = np.linalg.lstsq(A_test, T_flat, rcond=None)
                    T_recon_test = A_test @ alpha_test
                    if np.allclose(T_recon_test, T_flat, atol=1e-9):
                        U16 = np.zeros((n_a, 16))
                        V16 = np.zeros((n_b, 16))
                        W16 = np.zeros((n_c, 16))
                        for jj in range(16):
                            U16[:, jj] = alpha_test[jj] * U_test[:, jj]
                            V16[:, jj] = alpha_test[jj] * V_test[:, jj]
                            W16[:, jj] = alpha_test[jj] * W_test[:, jj]
                        return U16, V16, W16
                except:
                    pass
                # Undo
                supports_U[j].discard(new_idx)
        elif mat_choice == 1:
            new_idx = np.random.randint(0, n_b)
            if new_idx not in supports_V[j]:
                supports_V[j].add(new_idx)
                U_test = np.zeros((n_a, 16))
                V_test = np.zeros((n_b, 16))
                W_test = np.zeros((n_c, 16))
                for jj, rr in enumerate(keep_indices):
                    if jj == j:
                        for idx in supports_U[j]:
                            U_test[idx, jj] = 1
                        V_test[new_idx, jj] = 1
                        for idx in supports_W[j]:
                            W_test[idx, jj] = 1
                    else:
                        U_test[:, jj] = U17[:, rr]
                        V_test[:, jj] = V17[:, rr]
                        W_test[:, jj] = W17[:, rr]
                
                A_test = np.zeros((num_entries, 16))
                for jj in range(16):
                    A_test[:, jj] = np.einsum('a,b,c->abc', U_test[:, jj], V_test[:, jj], W_test[:, jj]).flatten()
                
                try:
                    alpha_test, _, _, _ = np.linalg.lstsq(A_test, T_flat, rcond=None)
                    T_recon_test = A_test @ alpha_test
                    if np.allclose(T_recon_test, T_flat, atol=1e-9):
                        U16 = np.zeros((n_a, 16))
                        V16 = np.zeros((n_b, 16))
                        W16 = np.zeros((n_c, 16))
                        for jj in range(16):
                            U16[:, jj] = alpha_test[jj] * U_test[:, jj]
                            V16[:, jj] = alpha_test[jj] * V_test[:, jj]
                            W16[:, jj] = alpha_test[jj] * W_test[:, jj]
                        return U16, V16, W16
                except:
                    pass
                supports_V[j].discard(new_idx)
        else:
            new_idx = np.random.randint(0, n_c)
            if new_idx not in supports_W[j]:
                supports_W[j].add(new_idx)
                U_test = np.zeros((n_a, 16))
                V_test = np.zeros((n_b, 16))
                W_test = np.zeros((n_c, 16))
                for jj, rr in enumerate(keep_indices):
                    if jj == j:
                        for idx in supports_U[j]:
                            U_test[idx, jj] = 1
                        for idx in supports_V[j]:
                            V_test[idx, jj] = 1
                        W_test[new_idx, jj] = 1
                    else:
                        U_test[:, jj] = U17[:, rr]
                        V_test[:, jj] = V17[:, rr]
                        W_test[:, jj] = W17[:, rr]
                
                A_test = np.zeros((num_entries, 16))
                for jj in range(16):
                    A_test[:, jj] = np.einsum('a,b,c->abc', U_test[:, jj], V_test[:, jj], W_test[:, jj]).flatten()
                
                try:
                    alpha_test, _, _, _ = np.linalg.lstsq(A_test, T_flat, rcond=None)
                    T_recon_test = A_test @ alpha_test
                    if np.allclose(T_recon_test, T_flat, atol=1e-9):
                        U16 = np.zeros((n_a, 16))
                        V16 = np.zeros((n_b, 16))
                        W16 = np.zeros((n_c, 16))
                        for jj in range(16):
                            U16[:, jj] = alpha_test[jj] * U_test[:, jj]
                            V16[:, jj] = alpha_test[jj] * V_test[:, jj]
                            W16[:, jj] = alpha_test[jj] * W_test[:, jj]
                        return U16, V16, W16
                except:
                    pass
                supports_W[j].discard(new_idx)
    
    return None

def _structured_rank16_search(start_time, deadline):
    """
    Structured search for rank-16: try absorbing each of the 17 terms into the other 16.
    """
    n, m, p = 2, 3, 3
    T = _build_tensor(n, m, p)
    U17, V17, W17 = _get_rank17_solution()
    
    search_deadline = deadline - 5
    
    for remove_idx in range(17):
        if time.time() > search_deadline:
            break
        
        result = _try_absorb_term(U17, V17, W17, T, remove_idx, search_deadline)
        if result is not None:
            U16, V16, W16 = result
            if _verify_decomposition(U16, V16, W16, T):
                return U16, V16, W16
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140 second deadline with safety margin
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    # 1. Attempt structured rank-16 search
    result_16 = _structured_rank16_search(start_time, deadline)
    if result_16 is not None:
        U, V, W = result_16
        T = _build_tensor(n, m, p)
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # 2. Fallback to verified Rank 17
    U, V, W = _get_rank17_solution()
    T = _build_tensor(n, m, p)
    
    if not _verify_decomposition(U, V, W, T):
        # Final fallback: Schoolbook (Rank 18)
        rank_fb = n * m * p
        U_fb = np.zeros((n_a, rank_fb))
        V_fb = np.zeros((n_b, rank_fb))
        W_fb = np.zeros((n_c, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
