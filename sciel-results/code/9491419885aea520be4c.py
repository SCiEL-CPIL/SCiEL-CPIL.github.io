"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 by 4x6 matrix multiplication.
    
    The tensor T_{i,j,k} corresponds to the (i,j,k) entry of the multiplication tensor.
    We seek U (shape n*m x r), V (shape m*p x r), W (shape p*n x r) such that
    for all i, j, k: T_{i,j,k} = sum_{c=1}^{r} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    We use a randomized approach with integer/half-integer constraints by scaling by 2.
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 24
    pn = p * n  # 12
    
    # The multiplication tensor: T[i,j,k] = 1 for all valid i,j,k
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Strategy: Use randomized linear algebra with half-integer constraints.
    # Work with doubled values (integers) to enforce half-integer entries.
    
    def verify_decomposition(U, V, W, r):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(r))
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True
    
    def try_rank(r, num_trials=30, seed=42):
        """Try to find a decomposition of rank r using randomized least squares."""
        rng = np.random.RandomState(seed)
        
        for trial in range(num_trials):
            # Initialize with random values in a reasonable range
            # Use small values to encourage sparsity and integer solutions
            U_init = rng.randint(-1, 2, size=(nm, r)).astype(float)
            V_init = rng.randint(-1, 2, size=(mp, r)).astype(float)
            W_init = rng.randint(-1, 2, size=(pn, r)).astype(float)
            
            # Define the residual function for least_squares
            def residual(params):
                U_d = params[:nm*r].reshape(nm, r)
                V_d = params[nm*r:nm*r+mp*r].reshape(mp, r)
                W_d = params[nm*r+mp*r:].reshape(pn, r)
                
                # Compute T_approx[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                T_approx = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_approx[i,j,k] = np.sum(
                                U_d[i*m+j, :] * V_d[j*p+k, :] * W_d[k*n+i, :]
                            )
                
                # Target: all ones
                return (T_approx - 1.0).flatten()
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use least squares to solve
            try:
                result = least_squares(residual, x0, method='lm', max_nfev=2000, ftol=1e-12, xtol=1e-12)
                
                # Check if we got a good solution
                if np.linalg.norm(result.fun) < 1e-6:
                    U_sol = result.x[:nm*r].reshape(nm, r)
                    V_sol = result.x[nm*r:nm*r+mp*r].reshape(mp, r)
                    W_sol = result.x[nm*r+mp*r:].reshape(pn, r)
                    
                    # Check if solution is close to half-integers (2*value is integer)
                    U_int = np.round(2 * U_sol)
                    V_int = np.round(2 * V_sol)
                    W_int = np.round(2 * W_sol)
                    
                    if (np.allclose(U_sol, U_int/2, atol=1e-4) and
                        np.allclose(V_sol, V_int/2, atol=1e-4) and
                        np.allclose(W_sol, W_int/2, atol=1e-4)):
                        
                        # Verify exactness with the rounded values
                        U_val = (U_int / 2.0).tolist()
                        V_val = (V_int / 2.0).tolist()
                        W_val = (W_int / 2.0).tolist()
                        
                        if verify_decomposition(U_val, V_val, W_val, r):
                            # Clean up values
                            def clean_val(v):
                                if abs(v - round(v)) < 1e-9:
                                    return int(round(v))
                                else:
                                    return v
                            
                            U_list = [[clean_val(x) for x in row] for row in U_val]
                            V_list = [[clean_val(x) for x in row] for row in V_val]
                            W_list = [[clean_val(x) for x in row] for row in W_val]
                            
                            return {"U": U_list, "V": V_list, "W": W_list}
            except Exception:
                continue
        
        return None
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: at least max(n*m, m*p, p*n) = 24 for this case
    # Actually, the rank must be at least the dimension of the output space in some sense
    # For matrix multiplication, a known lower bound is n*m*p / max(n,m,p) but that's not tight
    # Let's try from 24 up to 47
    
    for r in range(24, 48):
        result = try_rank(r, num_trials=25, seed=100 + r)
        if result is not None:
            return result
    
    # If we couldn't find a better decomposition, return schoolbook
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
# EVOLVE-BLOCK-END
