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
    
    # The tensor entries: T[i,j,k] = 1 if i,j,k are valid (they always are for identity-like structure)
    # Actually for matrix multiplication C = A*B where A is n x m and B is m x p,
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 (it's the identity in the middle dimension sense)
    # More precisely, the multiplication tensor has entries T_{i,j,k} = delta_{j,j} which is always 1
    # for valid indices. Wait, let me reconsider.
    
    # The standard matrix multiplication tensor: 
    # For multiplying (n x m) by (m x p) to get (n x p):
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1 for all valid i,j,k
    # because the "contraction" is over j, and the tensor itself is just the identity 
    # in the sense that each term A[i,j]*B[j,k] contributes with coefficient 1.
    
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # We want to find rank r decomposition:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Strategy: Try ranks from lower bound upward, using random initialization and
    # solving the resulting system. Since we need integer/half-integer entries, we'll
    # work with doubled values (integers) and check if they're even or odd appropriately.
    
    # Lower bound: The border rank of matrix multiplication is at least max(n*m, m*p, p*n)
    # Actually, a known lower bound is n*m*p / max(n,m,p) or similar. Let's just try to find
    # a good decomposition.
    
    # For 2x4x6, schoolbook is 48. Let's try to find something better.
    # 
    # Approach: Use the fact that we can sometimes share computations.
    # For example, if we have repeated patterns in U, V, W.
    
    # Let me try a structured approach:
    # Consider that n=2, m=4, p=6.
    # 
    # One idea: Use a decomposition based on the structure of the matrices.
    # 
    # Let's try a randomized search with small ranks first.
    
    def try_rank(r, num_trials=50, seed=42):
        """Try to find a decomposition of rank r."""
        rng = np.random.RandomState(seed)
        
        for trial in range(num_trials):
            # Initialize with random half-integers (represented as integers/2)
            # Work with doubled values to keep integers
            U_doubled = rng.randint(-2, 3, size=(nm, r))
            V_doubled = rng.randint(-2, 3, size=(mp, r))
            W_doubled = rng.randint(-2, 3, size=(pn, r))
            
            # Define the residual function
            def residual(params):
                U_d = params[:nm*r].reshape(nm, r)
                V_d = params[nm*r:nm*r+mp*r].reshape(mp, r)
                W_d = params[nm*r+mp*r:].reshape(pn, r)
                
                # Compute T_approx[i,j,k] = sum_c U_d[i*m+j, c] * V_d[j*p+k, c] * W_d[k*n+i, c]
                # This is a tensor contraction
                # Reshape to 3D tensor
                # T_approx has shape (n, m, p)
                T_approx = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_approx[i,j,k] = np.sum(
                                U_d[i*m+j, :] * V_d[j*p+k, :] * W_d[k*n+i, :]
                            )
                
                # Target: all ones
                return (T_approx - 1.0).flatten()
            
            x0 = np.concatenate([U_doubled.flatten(), V_doubled.flatten(), W_doubled.flatten()])
            
            # Use least squares to solve
            try:
                result = least_squares(residual, x0, method='lm', max_nfev=1000)
                if result.success or np.linalg.norm(result.fun) < 1e-6:
                    # Check if solution is close to half-integers
                    U_sol = result.x[:nm*r].reshape(nm, r)
                    V_sol = result.x[nm*r:nm*r+mp*r].reshape(mp, r)
                    W_sol = result.x[nm*r+mp*r:].reshape(pn, r)
                    
                    # Check if 2*values are close to integers
                    U_int = np.round(2 * U_sol)
                    V_int = np.round(2 * V_sol)
                    W_int = np.round(2 * W_sol)
                    
                    if (np.allclose(U_sol, U_int/2, atol=1e-4) and
                        np.allclose(V_sol, V_int/2, atol=1e-4) and
                        np.allclose(W_sol, W_int/2, atol=1e-4)):
                        
                        # Verify exactness
                        U_val = U_int / 2.0
                        V_val = V_int / 2.0
                        W_val = W_int / 2.0
                        
                        T_check = np.zeros((n, m, p))
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    T_check[i,j,k] = np.sum(
                                        U_val[i*m+j, :] * V_val[j*p+k, :] * W_val[k*n+i, :]
                                    )
                        
                        if np.allclose(T_check, 1.0, atol=1e-8):
                            # Convert to lists of fractions
                            U_list = [[U_int[i,c] / 2 for c in range(r)] for i in range(nm)]
                            V_list = [[V_int[i,c] / 2 for c in range(r)] for i in range(mp)]
                            W_list = [[W_int[i,c] / 2 for c in range(r)] for i in range(pn)]
                            
                            # Clean up: use integers where possible
                            def clean_val(v):
                                if abs(v - round(v)) < 1e-9:
                                    return int(round(v))
                                else:
                                    return v
                            
                            U_list = [[clean_val(x) for x in row] for row in U_list]
                            V_list = [[clean_val(x) for x in row] for row in V_list]
                            W_list = [[clean_val(x) for x in row] for row in W_list]
                            
                            return {"U": U_list, "V": V_list, "W": W_list}
            except Exception:
                continue
        
        return None
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound for matrix multiplication tensor rank is at least max(n*m, m*p, p*n) / something
    # Actually, a simple lower bound: the rank is at least the dimension of the space spanned
    # For our purposes, let's try from 24 (half of schoolbook) up to 47
    
    for r in range(24, 48):
        result = try_rank(r, num_trials=20, seed=42 + r)
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
