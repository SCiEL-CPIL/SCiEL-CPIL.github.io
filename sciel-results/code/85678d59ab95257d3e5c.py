import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    # Initialize best with schoolbook (rank 18)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1

    def compute_residuals(x, R):
        # x is flattened U, V, W
        U = x[:n*m*R].reshape(n*m, R)
        V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
        W = x[n*m*R + m*p*R :].reshape(p*n, R)
        
        # Residuals for each i, k, j, j'
        # M_ik[j, j'] = sum_r W[k*n+i, r] U[i*m+j, r] V[j'*p+k, r]
        # Target is delta_{j, j'}
        
        residuals = []
        for i in range(n):
            for k in range(p):
                # w_vec is W[k*n+i, :] shape (R,)
                w_vec = W[k * n + i, :]
                for j in range(m):
                    # u_vec is U[i*m+j, :] shape (R,)
                    u_vec = U[i * m + j, :]
                    for jp in range(m):
                        # v_vec is V[jp*p+k, :] shape (R,)
                        v_vec = V[jp * p + k, :]
                        # term = sum_r w_vec[r] * u_vec[r] * v_vec[r]
                        val = np.dot(w_vec, u_vec * v_vec)
                        target = 1.0 if j == jp else 0.0
                        residuals.append(val - target)
        return np.array(residuals)

    def check_solution(U, V, W, R, atol=1e-6):
        # Exact check with high precision
        for i in range(n):
            for k in range(p):
                w_vec = W[k * n + i, :]
                M = np.zeros((m, m))
                for j in range(m):
                    u_vec = U[i * m + j, :]
                    for jp in range(m):
                        v_vec = V[jp * p + k, :]
                        M[j, jp] = np.dot(w_vec, u_vec * v_vec)
                # Check if M is close to Identity
                if not np.allclose(M, np.eye(m), atol=atol):
                    return False
        return True

    def round_to_half_int(M):
        return np.round(M * 2) / 2

    def polish_solution(x0, R, max_iter=2000, tol=1e-10):
        """High-precision polish using least_squares with tight tolerance"""
        try:
            res = opt.least_squares(compute_residuals, x0, args=(R,), 
                                   method='lm', max_nfev=max_iter, 
                                   xtol=tol, ftol=tol, gtol=tol)
            return res.x
        except Exception:
            return x0

    def try_rank(R, max_attempts=30):
        nonlocal best_rank, best_U, best_V, best_W
        if R >= best_rank:
            return
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return
            
            # Random start with diverse strategies
            total_size = n*m*R + m*p*R + p*n*R
            
            if attempt % 5 == 0:
                # Small integers in {-1, 0, 1}
                x0 = np.random.randint(-1, 2, size=total_size).astype(float)
            elif attempt % 5 == 1:
                # Random floats in [-0.5, 0.5]
                x0 = np.random.uniform(-0.5, 0.5, size=total_size)
            elif attempt % 5 == 2:
                # Gaussian with moderate scale
                x0 = np.random.randn(total_size) * 0.3
            elif attempt % 5 == 3:
                # Sparse initialization - only a few non-zero entries
                x0 = np.zeros(total_size)
                num_nonzero = max(3, total_size // 4)
                indices = np.random.choice(total_size, num_nonzero, replace=False)
                x0[indices] = np.random.choice([-1, 0.5, 1, -0.5], size=num_nonzero)
            else:
                # Perturbed schoolbook-like structure
                x0 = np.random.randint(-1, 2, size=total_size).astype(float)
                x0 += np.random.randn(total_size) * 0.05
            
            # Optimize
            try:
                res = opt.least_squares(compute_residuals, x0, args=(R,), 
                                       method='lm', max_nfev=1500)
            except Exception:
                continue
            
            x_opt = res.x
            U = x_opt[:n*m*R].reshape(n*m, R)
            V = x_opt[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
            W = x_opt[n*m*R + m*p*R :].reshape(p*n, R)
            
            # Round to half-integers
            U_int = round_to_half_int(U)
            V_int = round_to_half_int(V)
            W_int = round_to_half_int(W)
            
            if check_solution(U_int, V_int, W_int, R, atol=1e-5):
                # Found a candidate - try to polish for exactness
                # Reconstruct flat vector from rounded solution
                x_rounded = np.concatenate([
                    U_int.flatten(), 
                    V_int.flatten(), 
                    W_int.flatten()
                ])
                
                # Polish with high precision
                x_polished = polish_solution(x_rounded, R, max_iter=3000, tol=1e-12)
                
                U_pol = x_polished[:n*m*R].reshape(n*m, R)
                V_pol = x_polished[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_pol = x_polished[n*m*R + m*p*R :].reshape(p*n, R)
                
                # Round again after polish
                U_pol_int = round_to_half_int(U_pol)
                V_pol_int = round_to_half_int(V_pol)
                W_pol_int = round_to_half_int(W_pol)
                
                if check_solution(U_pol_int, V_pol_int, W_pol_int, R, atol=1e-8):
                    best_rank = R
                    best_U = U_pol_int.tolist()
                    best_V = V_pol_int.tolist()
                    best_W = W_pol_int.tolist()
                    return
                elif check_solution(U_int, V_int, W_int, R, atol=1e-6):
                    # Accept the pre-polish solution if it passes
                    best_rank = R
                    best_U = U_int.tolist()
                    best_V = V_int.tolist()
                    best_W = W_int.tolist()
                    return

    # Try ranks from lower bounds upward
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(6, 9, 6) = 9
    # But for matrix multiplication, stronger lower bounds exist
    # Try from rank 10 upward
    
    for R in range(10, 18):
        if time.time() > deadline:
            break
        # Allocate more attempts to lower ranks (more promising)
        attempts_for_rank = 40 if R <= 12 else 25
        try_rank(R, max_attempts=attempts_for_rank)
        if best_rank <= R:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
