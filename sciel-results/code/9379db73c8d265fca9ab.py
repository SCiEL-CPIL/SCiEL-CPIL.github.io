"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_tensor(n, m, p):
    """Build the tensor T where T[a,b,c]=1 iff indices are consistent."""
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

def _try_rank16_search(T, deadline):
    """
    Attempt to find a rank-16 decomposition by modifying the rank-17 solution.
    This is a heuristic search that tries to absorb one term into the others.
    """
    n_a, n_b, n_c = 6, 9, 6
    U17, V17, W17 = _get_rank17_solution()
    T_flat = T.flatten()
    num_entries = T_flat.size
    
    # Try removing each term and solving for coefficients
    for remove_idx in range(17):
        if time.time() > deadline:
            break
        
        keep_indices = [r for r in range(17) if r != remove_idx]
        
        # Build the linear system
        A = np.zeros((num_entries, 16))
        for j, r in enumerate(keep_indices):
            A[:, j] = np.einsum('a,b,c->abc', U17[:, r], V17[:, r], W17[:, r]).flatten()
        
        # Solve least squares
        try:
            alpha, residuals, rank, sv = np.linalg.lstsq(A, T_flat, rcond=None)
            if np.allclose(A @ alpha, T_flat, atol=1e-9):
                # Found a valid rank-16 solution
                U16 = np.zeros((n_a, 16))
                V16 = np.zeros((n_b, 16))
                W16 = np.zeros((n_c, 16))
                for j, r in enumerate(keep_indices):
                    U16[:, j] = alpha[j] * U17[:, r]
                    V16[:, j] = alpha[j] * V17[:, r]
                    W16[:, j] = alpha[j] * W17[:, r]
                return U16, V16, W16
        except:
            pass
    
    return None

def _try_rank15_search(T, deadline):
    """
    Attempt to find a rank-15 decomposition using a more aggressive search.
    This tries to modify multiple terms simultaneously.
    """
    n_a, n_b, n_c = 6, 9, 6
    U17, V17, W17 = _get_rank17_solution()
    T_flat = T.flatten()
    num_entries = T_flat.size
    
    # Try removing pairs of terms
    for i in range(17):
        if time.time() > deadline:
            return None
        for j in range(i+1, 17):
            keep_indices = [r for r in range(17) if r != i and r != j]
            
            A = np.zeros((num_entries, 15))
            for k, r in enumerate(keep_indices):
                A[:, k] = np.einsum('a,b,c->abc', U17[:, r], V17[:, r], W17[:, r]).flatten()
            
            try:
                alpha, residuals, rank, sv = np.linalg.lstsq(A, T_flat, rcond=None)
                if np.allclose(A @ alpha, T_flat, atol=1e-9):
                    U15 = np.zeros((n_a, 15))
                    V15 = np.zeros((n_b, 15))
                    W15 = np.zeros((n_c, 15))
                    for k, r in enumerate(keep_indices):
                        U15[:, k] = alpha[k] * U17[:, r]
                        V15[:, k] = alpha[k] * V17[:, r]
                        W15[:, k] = alpha[k] * W17[:, r]
                    return U15, V15, W15
            except:
                pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145 second deadline with safety margin
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    T = _build_tensor(n, m, p)
    
    # 1. Attempt rank-16 search
    result_16 = _try_rank16_search(T, deadline)
    if result_16 is not None:
        U, V, W = result_16
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # 2. Attempt rank-15 search
    if time.time() < deadline - 10:
        result_15 = _try_rank15_search(T, deadline)
        if result_15 is not None:
            U, V, W = result_15
            if _verify_decomposition(U, V, W, T):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # 3. Fallback to verified Rank 17
    U, V, W = _get_rank17_solution()
    
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
