import numpy as np
import scipy.optimize as opt
import time
from scipy.linalg import svd

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Build the target tensor T[a, b, c, d, k, i] = delta_b_c * delta_a_i * delta_d_k
    # Shape: (n*m, m*p, p*n) -> (8, 20, 10)
    # T[a*m+b, c*p+d, k*n+i] = 1 if b==c and a==i and d==k, else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(p):
            for b in range(m):
                a = i  # delta_a_i
                d = k  # delta_d_k
                c = b  # delta_b_c
                T[a*m+b, c*p+d, k*n+i] = 1.0
    
    def get_initial_solution_rank36():
        rank = 36
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        def add_strassen_222(i0, j0, k0):
            nonlocal col
            # M1
            U[i0*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M2
            U[(i0+1)*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + (i0+1), col] += 1; W[(k0+1)*n + (i0+1), col] -= 1
            col += 1
            # M3
            U[i0*m + j0, col] += 1
            V[j0*p + (k0+1), col] += 1; V[(j0+1)*p + (k0+1), col] -= 1
            W[(k0+1)*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M4
            U[(i0+1)*m + (j0+1), col] += 1
            V[(j0+1)*p + k0, col] += 1; V[j0*p + k0, col] -= 1
            W[k0*n + i0, col] += 1; W[k0*n + (i0+1), col] += 1
            col += 1
            # M5
            U[i0*m + j0, col] += 1; U[i0*m + (j0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] -= 1; W[(k0+1)*n + i0, col] += 1
            col += 1
            # M6
            U[(i0+1)*m + j0, col] += 1; U[i0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1; V[j0*p + (k0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M7
            U[i0*m + (j0+1), col] += 1; U[(i0+1)*m + (j0+1), col] -= 1
            V[(j0+1)*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            col += 1

        def add_schoolbook_2x4x1(i0, j0, k0):
            nonlocal col
            for di in range(2):
                for dj in range(4):
                    U[(i0 + di)*m + (j0 + dj), col] = 1
                    V[(j0 + dj)*p + k0, col] = 1
                    W[k0*n + (i0 + di), col] = 1
                    col += 1

        add_strassen_222(0, 0, 0)
        add_strassen_222(0, 2, 0)
        add_strassen_222(0, 0, 2)
        add_strassen_222(0, 2, 2)
        add_schoolbook_2x4x1(0, 0, 4)
        
        return U, V, W

    def find_linear_dependencies(U, V, W, tol=1e-10):
        """Find linear dependencies among rank-1 terms using tensor complement."""
        R = U.shape[1]
        # Flatten each rank-1 term as a vector in the tensor space
        # T = sum_r u_r ⊗ v_r ⊗ w_r
        # Check if any subset is linearly dependent
        # Build matrix where each column is the flattened rank-1 tensor
        terms = []
        for r in range(R):
            term = np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).flatten()
            terms.append(term)
        terms_mat = np.array(terms).T  # shape: (n*m*p*n, R)
        
        # SVD to find null space
        U_sv, S_vals, Vt = svd(terms_mat, full_matrices=False)
        
        # Find singular values below tolerance
        null_mask = S_vals < tol * max(S_vals[0], 1e-15)
        null_dim = np.sum(null_mask)
        
        if null_dim > 0:
            # Null space vectors give linear dependencies
            null_vecs = Vt[~null_mask:R, :].T if null_dim < R else np.zeros((R, 0))
            # Actually, null space is the right singular vectors corresponding to zero singular values
            null_space = Vt[-null_dim:, :].T if null_dim > 0 else np.zeros((R, 0))
        else:
            null_space = np.zeros((R, 0))
        
        return null_dim, null_space

    U36, V36, W36 = get_initial_solution_rank36()
    
    # Find linear dependencies in the rank-36 solution
    null_dim_36, null_space_36 = find_linear_dependencies(U36, V36, W36)
    
    def verify_solution_copy(U, V, W, tol=1e-9):
        T_copy = T.copy()
        R = U.shape[1]
        for mu in range(R):
            T_copy -= np.einsum('a,b,c->abc', U[:,mu], V[:,mu], W[:,mu])
        return np.max(np.abs(T_copy)) < tol

    def snap_to_half_integers(U, V, W):
        U_snapped = np.round(U * 2) / 2.0
        V_snapped = np.round(V * 2) / 2.0
        W_snapped = np.round(W * 2) / 2.0
        return U_snapped, V_snapped, W_snapped
    
    def try_reduce_rank_via_dependency(U, V, W, target_rank, tol=1e-8):
        """Try to reduce rank by exploiting linear dependencies."""
        R = U.shape[1]
        if R <= target_rank:
            return U, V, W, True
        
        # Find dependencies
        null_dim, null_space = find_linear_dependencies(U, V, W, tol=1e-10)
        
        if null_dim > 0:
            # We can eliminate null_dim columns
            new_rank = R - null_dim
            if new_rank >= target_rank:
                # Keep only the first target_rank columns after removing dependent ones
                # Identify which columns to remove based on null space
                # For simplicity, just truncate
                U_new = U[:, :target_rank]
                V_new = V[:, :target_rank]
                W_new = W[:, :target_rank]
                return U_new, V_new, W_new, True
        return U, V, W, False

    def residual_vector_vectorized(x, R):
        """Compute residual using vectorized tensor operations."""
        size_U = n * m * R
        size_V = m * p * R
        
        U = x[0:size_U].reshape(n*m, R)
        V = x[size_U:size_U+size_V].reshape(m*p, R)
        W = x[size_U+size_V:].reshape(p*n, R)
        
        T_rec = np.einsum('am,bm,cm->abc', U, V, W, optimize=True)
        return (T_rec - T).flatten()
    
    def gradient_residual_vectorized(x, R):
        """Compute gradient of 0.5*||residual||^2 using vectorized operations."""
        size_U = n * m * R
        size_V = m * p * R
        
        U = x[0:size_U].reshape(n*m, R)
        V = x[size_U:size_U+size_V].reshape(m*p, R)
        W = x[size_U+size_V:].reshape(p*n, R)
        
        T_rec = np.einsum('am,bm,cm->abc', U, V, W, optimize=True)
        residual = T_rec - T
        
        grad_U = np.einsum('abc,bm,cm->am', residual, V, W, optimize=True)
        grad_V = np.einsum('abc,am,cm->bm', residual, U, W, optimize=True)
        grad_W = np.einsum('abc,am,bm->cm', residual, U, V, optimize=True)
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    start_time = time.time()
    time_limit = 135
    
    best_rank = 36
    best_result = None
    
    # Pre-compute structured initializations using dependency analysis
    structured_inits = {}
    for R in range(35, 20, -1):
        # Use the rank-36 solution truncated, plus perturbations informed by null space
        U_init_struct = U36[:, :R].copy()
        V_init_struct = V36[:, :R].copy()
        W_init_struct = W36[:, :R].copy()
        structured_inits[R] = (U_init_struct, V_init_struct, W_init_struct)
    
    for R in range(35, 20, -1):
        if time.time() - start_time > time_limit:
            break
            
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        n_attempts = 20 if R >= 30 else 10
        
        for attempt in range(n_attempts):
            if time.time() - start_time > time_limit:
                break
            
            if attempt == 0:
                # Structured initialization from rank-36 solution
                U_init, V_init, W_init = structured_inits[R]
            elif attempt == 1:
                # Perturbed structured init
                U_init, V_init, W_init = structured_inits[R]
                U_init = U_init + np.random.normal(0, 0.1, (n*m, R))
                V_init = V_init + np.random.normal(0, 0.1, (m*p, R))
                W_init = W_init + np.random.normal(0, 0.1, (p*n, R))
            elif attempt == 2:
                U_init = np.random.normal(0, 1.0, (n*m, R))
                V_init = np.random.normal(0, 1.0, (m*p, R))
                W_init = np.random.normal(0, 1.0, (p*n, R))
            elif attempt == 3:
                U_init = np.random.normal(0, 0.5, (n*m, R))
                V_init = np.random.normal(0, 0.5, (m*p, R))
                W_init = np.random.normal(0, 0.5, (p*n, R))
            elif attempt == 4:
                U_init = np.random.choice([-1, 0, 0, 0, 1], (n*m, R))
                V_init = np.random.choice([-1, 0, 0, 0, 1], (m*p, R))
                W_init = np.random.choice([-1, 0, 0, 0, 1], (p*n, R))
            elif attempt == 5:
                U_init = np.random.normal(0, 2.0, (n*m, R))
                V_init = np.random.normal(0, 2.0, (m*p, R))
                W_init = np.random.normal(0, 2.0, (p*n, R))
            else:
                scale = 0.3 + 0.2 * (attempt % 5)
                U_init = np.random.normal(0, scale, (n*m, R))
                V_init = np.random.normal(0, scale, (m*p, R))
                W_init = np.random.normal(0, scale, (p*n, R))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = opt.minimize(
                    lambda x: 0.5 * np.sum(residual_vector_vectorized(x, R)**2),
                    x0,
                    method='L-BFGS-B',
                    jac=lambda x: gradient_residual_vectorized(x, R),
                    maxiter=3000,
                    maxfun=6000,
                    options={'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                x_opt = result.x
                U_opt = x_opt[0:size_U].reshape(n*m, R)
                V_opt = x_opt[size_U:size_U+size_V].reshape(m*p, R)
                W_opt = x_opt[size_U+size_V:].reshape(p*n, R)
                
                T_rec = np.einsum('am,bm,cm->abc', U_opt, V_opt, W_opt, optimize=True)
                max_res = np.max(np.abs(T_rec - T))
                
                if max_res < 1e-6:
                    U_snap, V_snap, W_snap = snap_to_half_integers(U_opt, V_opt, W_opt)
                    T_rec_snap = np.einsum('am,bm,cm->abc', U_snap, V_snap, W_snap, optimize=True)
                    if np.max(np.abs(T_rec_snap - T)) < 1e-9:
                        best_rank = R
                        best_result = (U_snap, V_snap, W_snap)
                        break
                
                elif max_res < 1e-3:
                    # High-precision polish pass
                    x_polish = x_opt.copy()
                    result2 = opt.minimize(
                        lambda x: 0.5 * np.sum(residual_vector_vectorized(x, R)**2),
                        x_polish,
                        method='L-BFGS-B',
                        jac=lambda x: gradient_residual_vectorized(x, R),
                        maxiter=8000,
                        maxfun=16000,
                        options={'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
                    )
                    
                    x_polished = result2.x
                    U_pol = x_polished[0:size_U].reshape(n*m, R)
                    V_pol = x_polished[size_U:size_U+size_V].reshape(m*p, R)
                    W_pol = x_polished[size_U+size_V:].reshape(p*n, R)
                    
                    T_rec_pol = np.einsum('am,bm,cm->abc', U_pol, V_pol, W_pol, optimize=True)
                    max_res_pol = np.max(np.abs(T_rec_pol - T))
                    
                    if max_res_pol < 1e-8:
                        U_snap, V_snap, W_snap = snap_to_half_integers(U_pol, V_pol, W_pol)
                        T_rec_snap = np.einsum('am,bm,cm->abc', U_snap, V_snap, W_snap, optimize=True)
                        if np.max(np.abs(T_rec_snap - T)) < 1e-9:
                            best_rank = R
                            best_result = (U_snap, V_snap, W_snap)
                            break
                    
            except Exception:
                pass
    
    if best_result is not None:
        U_snap, V_snap, W_snap = best_result
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    # Fallback: verify rank-36 solution
    U_snap, V_snap, W_snap = snap_to_half_integers(U36, V36, W36)
    T_rec = np.einsum('am,bm,cm->abc', U_snap, V_snap, W_snap, optimize=True)
    if np.max(np.abs(T_rec - T)) < 1e-9:
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}
