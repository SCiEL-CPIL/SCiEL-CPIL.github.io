"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target matrix multiplication tensor
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook decomposition (rank = n*m*p = 60)
    rank_schoolbook = n * m * p
    U_school = np.zeros((nm, rank_schoolbook))
    V_school = np.zeros((mp, rank_schoolbook))
    W_school = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school

    # Try to find a lower rank decomposition using ALS (Alternating Least Squares)
    # We minimize ||T - sum_r U[:,r] * V[:,r] * W[:,r]||_F^2
    
    # The tensor can be reshaped: for each r, the contribution is U[:,r] (nm x 1) * V[:,r]^T (1 x mp) * W[:,r]^T (1 x pn)
    # Actually, the tensor decomposition is: T = sum_r u_r * v_r * w_r (outer product)
    # In matrix form, if we fix U and V, we can solve for W, etc.
    
    # Let's use a more efficient formulation.
    # T is a 3D array. We want T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    # This is equivalent to: for each r, we add U[:,r] * V[:,r]^T * W[:,r]^T to the tensor.
    # Wait, the dimensions are:
    # U is (nm, R), V is (mp, R), W is (pn, R)
    # Term r: U[:,r] (nm,1) * V[:,r]^T (1,mp) * W[:,r]^T (1,pn) -> (nm, mp, pn)
    
    # ALS Algorithm:
    # 1. Initialize U, V, W randomly.
    # 2. Repeat:
    #    a. Fix V, W. Solve for U.
    #       Minimize ||T - sum_r U[:,r] V[:,r]^T W[:,r]^T||_F^2
    #       This is a linear least squares problem for each column of U?
    #       Actually, we can vectorize.
    #       Let's reshape T to (nm, mp*pn).
    #       Let M_VW be a matrix of size (mp*pn, R) where M_VW[:, r] = vec(V[:,r] * W[:,r]^T)
    #       Then T_flat (nm, mp*pn) = U (nm, R) * M_VW^T (R, mp*pn)
    #       So U = argmin ||T_flat - U M_VW^T||_F^2
    #       Solution: U = T_flat @ M_VW @ pinv(M_VW^T @ M_VW)  (if M_VW has full column rank)
    #       Or simply solve the normal equations: (M_VW @ M_VW^T) U^T = M_VW @ T_flat^T
    #       Actually, U = T_flat @ M_VW @ inv(M_VW^T @ M_VW) is not quite right because U is on the left.
    #       Let's check: T_flat = U @ M_VW^T.
    #       Multiply by M_VW on right: T_flat @ M_VW = U @ M_VW^T @ M_VW.
    #       So U = (T_flat @ M_VW) @ inv(M_VW^T @ M_VW).
        
    #    b. Fix U, W. Solve for V similarly.
    #       Reshape T to (mp, nm*pn).
    #       Let M_UW be (nm*pn, R) where M_UW[:, r] = vec(U[:,r] * W[:,r]^T)
    #       Wait, the order of indices matters.
    #       T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    #       Fix U, W. For each j, T[:, j, :] is a matrix of size (nm, pn).
    #       T[:, j, :] = sum_r V[j, r] (U[:,r] * W[:,r]^T)
    #       Let M_UW be a matrix of size (nm*pn, R) where M_UW[:, r] = vec(U[:,r] * W[:,r]^T) (column-major or row-major?)
    #       Let's use row-major flatten: T_flat (mp, nm*pn) where T_flat[j, idx] = T[i, j, k] with idx = i*pn + k.
    #       Then T_flat = V @ M_UW^T.
    #       So V = (T_flat @ M_UW) @ inv(M_UW^T @ M_UW).
        
    #    c. Fix U, V. Solve for W similarly.
    #       Reshape T to (pn, nm*mp).
    #       W = (T_flat @ M_UV) @ inv(M_UV^T @ M_UV).

    def solve_als_rank(R, max_iters=50, tol=1e-10, seed=0):
        nonlocal best_rank, best_U, best_V, best_W
        
        np.random.seed(seed)
        U = np.random.randn(nm, R)
        V = np.random.randn(mp, R)
        W = np.random.randn(pn, R)
        
        # Precompute some matrices if possible? No, they depend on U, V, W.
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > start_time + DEADLINE - SAFETY_MARGIN:
                return False
            
            # Update U
            # M_VW: (mp*pn, R). M_VW[:, r] = vec(V[:,r] * W[:,r]^T)
            # V[:,r] is (mp, 1), W[:,r]^T is (1, pn). Outer product is (mp, pn).
            # Flatten to (mp*pn,).
            # We can form M_VW as: np.einsum('jr,kr->jkr', V, W).reshape(mp*pn, R)
            # Or more efficiently:
            M_VW = np.einsum('jr,kr->jkr', V, W).reshape(mp * pn, R)
            T_flat = T.reshape(nm, mp * pn)
            # Solve U @ M_VW^T = T_flat
            # U = T_flat @ M_VW @ inv(M_VW^T @ M_VW)
            try:
                A = M_VW.T @ M_VW
                b = T_flat @ M_VW
                U_new = np.linalg.solve(A, b.T).T
            except np.linalg.LinAlgError:
                U_new = np.linalg.lstsq(M_VW.T, T_flat.T, rcond=None)[0].T
            
            # Update V
            # M_UW: (nm*pn, R). M_UW[:, r] = vec(U[:,r] * W[:,r]^T)
            M_UW = np.einsum('ir,kr->ikr', U_new, W).reshape(nm * pn, R)
            # Reshape T to (mp, nm*pn). T_flat_v[j, idx] = T[i, j, k] with idx = i*pn + k.
            T_flat_v = np.einsum('ijk->jki', T).reshape(mp, nm * pn)
            try:
                A = M_UW.T @ M_UW
                b = T_flat_v @ M_UW
                V_new = np.linalg.solve(A, b.T).T
            except np.linalg.LinAlgError:
                V_new = np.linalg.lstsq(M_UW.T, T_flat_v.T, rcond=None)[0].T
            
            # Update W
            # M_UV: (nm*mp, R). M_UV[:, r] = vec(U[:,r] * V[:,r]^T)
            M_UV = np.einsum('ir,jr->ijr', U_new, V_new).reshape(nm * mp, R)
            # Reshape T to (pn, nm*mp). T_flat_w[k, idx] = T[i, j, k] with idx = i*mp + j.
            T_flat_w = np.einsum('ijk->kij', T).reshape(pn, nm * mp)
            try:
                A = M_UV.T @ M_UV
                b = T_flat_w @ M_UV
                W_new = np.linalg.solve(A, b.T).T
            except np.linalg.LinAlgError:
                W_new = np.linalg.lstsq(M_UV.T, T_flat_w.T, rcond=None)[0].T
            
            # Check convergence
            # Calculate error
            T_approx = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
            err = np.linalg.norm(T - T_approx)
            
            if abs(prev_err - err) < tol:
                break
            
            prev_err = err
            U, V, W = U_new, V_new, W_new
        
        # Check if the decomposition is exact (or very close)
        T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.linalg.norm(T - T_approx)
        
        if err < 1e-6:
            # Verify exactness with higher precision if possible, but for now 1e-6 is good
            # The problem requires "exact" decomposition with integer/half-integer entries?
            # "integer/half-integer entries that exactly decompose"
            # ALS produces floats. We need to check if they are close to integers or half-integers.
            # If not, we might need to round or use a different method.
            # However, the schoolbook solution uses integers.
            # If ALS finds a solution, it might be the same as schoolbook or a better one.
            # Let's check if the entries are close to half-integers.
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            T_approx_rounded = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
            err_rounded = np.linalg.norm(T - T_approx_rounded)
            
            if err_rounded < 1e-6:
                # We found a valid half-integer decomposition
                if R < best_rank:
                    best_rank = R
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    return True
            else:
                # The ALS solution is valid in float, but not half-integer.
                # The task requires integer/half-integer entries.
                # So this solution is invalid for the specific constraint.
                # Unless the evaluator is lenient? "exactly decompose" usually implies exact arithmetic.
                # Let's stick to the constraint.
                pass
        
        return False

    # Try ranks from schoolbook - 1 downwards
    # Start with a few random seeds for each rank
    for R in range(rank_schoolbook - 1, max(nm, mp, pn) - 1, -1):
        if time.time() > start_time + DEADLINE - SAFETY_MARGIN:
            break
        
        found = False
        for seed in range(5):
            if time.time() > start_time + DEADLINE - SAFETY_MARGIN:
                break
            if solve_als_rank(R, max_iters=20, seed=seed):
                found = True
                break
        
        if found:
            break

    # Return the best result found
    # Convert to lists as required by the interface
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
