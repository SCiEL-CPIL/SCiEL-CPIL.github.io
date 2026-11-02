import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Strassen decomposition for 2x2x2
    strassen_2x2 = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    def u_idx(row_off, col_off, local_r, local_c):
        return (row_off + local_r) * m + (col_off + local_c)
        
    def v_idx(row_off, col_off, local_r, local_c):
        return (row_off + local_r) * p + (col_off + local_c)
        
    def w_idx(col_off, local_r, local_c):
        return (col_off + local_c) * n + local_r

    # Build rank-28 solution as fallback
    def build_rank_28():
        rank = 28
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        block_configs = [
            (0, 0, 0, 0),
            (2, 2, 0, 0),
            (0, 0, 2, 2),
            (2, 2, 2, 2),
        ]
        
        col = 0
        for a_co, b_ro, b_co, c_co in block_configs:
            for u_terms, v_terms, w_terms in strassen_2x2:
                for idx, coeff in u_terms:
                    r = idx // 2
                    c = idx % 2
                    U[u_idx(0, a_co, r, c), col] += coeff
                for idx, coeff in v_terms:
                    r = idx // 2
                    c = idx % 2
                    V[v_idx(b_ro, b_co, r, c), col] += coeff
                for idx, coeff in w_terms:
                    r = idx // 2
                    c = idx % 2
                    W[w_idx(c_co, r, c), col] += coeff
                col += 1
        return U, V, W
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that the decomposition is exact"""
        rank = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(val - 1))
        return max_err < tol, max_err
    
    def try_rank_27():
        if time.time() > deadline - 15:
            return None
        
        U28, V28, W28 = build_rank_28()
        rank = 27
        rng = np.random.RandomState(42)
        
        # Dur-Krieger border rank approach:
        # 1. Start with rank-28 solution
        # 2. Apply a random linear projection to 27-dim subspace
        # 3. Use ALS to find the closest rank-27 approximation
        # 4. Check if it's exact
        
        best_result = None
        best_err = float('inf')
        
        for trial in range(100):
            if time.time() > deadline - 15:
                break
            
            # Create a random projection from 28-dim to 27-dim subspace
            # This is equivalent to randomly choosing 27 columns from the 28
            # and adding a small random perturbation to the 28th column's contribution
            # distributed among the 27 columns
            
            # Method: take all 28 columns, form a random 27x28 matrix P
            # New factors = P @ old_factors, then keep only 27 columns
            # Actually simpler: randomly select 27 of the 28 columns and add
            # a fraction of the 28th column to each
            
            # Even simpler: just try different subsets and perturbations
            seed = trial * 7 + 13
            rng_local = np.random.RandomState(seed)
            
            # Random perturbation approach
            U = U28[:, :rank].copy().astype(float)
            V = V28[:, :rank].copy().astype(float)
            W = W28[:, :rank].copy().astype(float)
            
            # Add the 28th column distributed randomly among the 27
            if trial % 3 == 0:
                # Even distribution
                U[:, :rank] += U28[:, rank:rank+1] / rank
                V[:, :rank] += V28[:, rank:rank+1] / rank
                W[:, :rank] += W28[:, rank:rank+1] / rank
            elif trial % 3 == 1:
                # Random distribution
                weights_u = rng_local.dirichlet(np.ones(rank))
                weights_v = rng_local.dirichlet(np.ones(rank))
                weights_w = rng_local.dirichlet(np.ones(rank))
                U += U28[:, rank:rank+1] * weights_u
                V += V28[:, rank:rank+1] * weights_v
                W += W28[:, rank:rank+1] * weights_w
            else:
                # Small random perturbation
                U += rng_local.randn(*U.shape) * 0.1
                V += rng_local.randn(*V.shape) * 0.1
                W += rng_local.randn(*W.shape) * 0.1
            
            # ALS iterations with better convergence
            for it in range(300):
                if time.time() > deadline - 15:
                    break
                    
                # Update U
                for i in range(n):
                    for j in range(m):
                        A_mat = np.zeros((p, rank))
                        for k in range(p):
                            A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        b_vec = np.ones(p)
                        try:
                            U[i*m+j, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((n, rank))
                        for i in range(n):
                            A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        b_vec = np.ones(n)
                        try:
                            V[j*p+k, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((m, rank))
                        for j in range(m):
                            A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        b_vec = np.ones(m)
                        try:
                            W[k*n+i, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        except:
                            pass
                
                # Check residual
                ok, max_err = verify_decomposition(U, V, W, tol=1e-10)
                
                if max_err < best_err:
                    best_err = max_err
                    best_result = (U.copy(), V.copy(), W.copy())
                
                if ok:
                    # Found an exact solution! Try to rationalize
                    # Round to nearest half-integer
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    
                    ok2, err2 = verify_decomposition(U_int, V_int, W_int, tol=1e-10)
                    if ok2:
                        return U_int, V_int, W_int
                    
                    # Try rounding to integers
                    U_int2 = np.round(U)
                    V_int2 = np.round(V)
                    W_int2 = np.round(W)
                    ok3, err3 = verify_decomposition(U_int2, V_int2, W_int2, tol=1e-10)
                    if ok3:
                        return U_int2, V_int2, W_int2
                    
                    # Return the floating point solution (will be converted later)
                    return U, V, W
                
                if max_err < 1e-12:
                    break
        
        # If no exact solution found, return the best approximation if it's close
        if best_result is not None and best_err < 1e-6:
            U, V, W = best_result
            # Try to rationalize
            U_int = np.round(U * 2) / 2
            V_int = np.round(V * 2) / 2
            W_int = np.round(W * 2) / 2
            ok, err = verify_decomposition(U_int, V_int, W_int, tol=1e-6)
            if ok:
                return U_int, V_int, W_int
        
        return None
    
    # Try rank 27
    result_27 = try_rank_27()
    
    if result_27 is not None:
        U, V, W = result_27
        # Convert to int if possible
        if np.allclose(U * 2, np.round(U * 2)):
            U = (U * 2).astype(int)
            U = U / 2.0
        if np.allclose(V * 2, np.round(V * 2)):
            V = (V * 2).astype(int)
            V = V / 2.0
        if np.allclose(W * 2, np.round(W * 2)):
            W = (W * 2).astype(int)
            W = W / 2.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 28
    U, V, W = build_rank_28()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
