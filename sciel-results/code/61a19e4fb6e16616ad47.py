import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # The tensor T for matrix multiplication C = A * B
    # T[i, j, k] = 1 for all i, j, k.
    # We want U (n*m x R), V (m*p x R), W (p*n x R)
    # such that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Strategy: Try to find a decomposition with rank R = 3*p + 1 = 46.
    # If that fails, try R = 47, ..., up to 52 (current best is 53).
    
    # Pre-compute indices for the tensor
    # We will flatten the tensor T into a vector of length n*m*p
    # The equation is linear in W given U, V.
    # Actually, it's a 3-way tensor.
    # We can fix U and V and solve for W? No, it's non-linear in all.
    # But we can use alternating minimization or just full least squares.
    
    # Let's define the function to compute the error given U, V, W
    def compute_error(params):
        R = p * 3 + 1 # We will adjust R in the loop, but for now assume fixed
        # Unpack params
        # U: n*m x R, V: m*p x R, W: p*n x R
        # Total params: (n*m + m*p + p*n) * R
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U : size_U + size_V]
        W_flat = params[size_U + size_V : size_U + size_V + size_W]
        
        U = U_flat.reshape((n * m, R))
        V = V_flat.reshape((m * p, R))
        W = W_flat.reshape((p * n, R))
        
        # Compute the tensor product sum
        # T_computed[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We can do this efficiently using einsum
        # T_computed = np.einsum('ic,jkc,kic->ijk', U, V, W) 
        # Wait, indices:
        # U: i*m+j -> let's call index a = i*m+j
        # V: j*p+k -> index b = j*p+k
        # W: k*n+i -> index c = k*n+i
        # We want T[i,j,k] = sum_r U[a,r] V[b,r] W[c,r]
        
        # Reshape U, V, W to 3D for easier einsum?
        # U: (n, m, R)
        # V: (m, p, R)
        # W: (p, n, R)
        
        U_3d = U.reshape((n, m, R))
        V_3d = V.reshape((m, p, R))
        W_3d = W.reshape((p, n, R))
        
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        T_comp = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        # Target T is all ones
        T_target = np.ones((n, m, p))
        
        # Return flattened error
        return (T_comp - T_target).flatten()
    
    # We will try ranks from 3*p+1 (46) up to 52
    best_result = None
    best_rank = 53 # Fallback to known good
    
    # Generate initial guess
    # Use Strassen for first 2 k's, zeros for rest?
    # Or random?
    # Random is better for escaping local minima.
    
    np.random.seed(42)
    
    for R in range(3 * p + 1, 53): # 46 to 52
        if time.time() > deadline:
            break
            
        # Initial guess
        # Let's try a mix of structured and random
        U_init = np.zeros((n * m, R))
        V_init = np.zeros((m * p, R))
        W_init = np.zeros((p * n, R))
        
        # Fill with small random noise
        U_init += np.random.normal(0, 0.1, U_init.shape)
        V_init += np.random.normal(0, 0.1, V_init.shape)
        W_init += np.random.normal(0, 0.1, W_init.shape)
        
        # Add the Strassen structure for the first 2 columns to help convergence?
        # Strassen for k=0,1
        # This might bias the solution. Let's try pure random first for R=46.
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Use least_squares
        # method='lm' (Levenberg-Marquardt) is good for small problems, but R=46 -> 3*30*46 ~ 4140 params.
        # 'trf' (Trust Region Reflective) is more robust.
        
        try:
            res = least_squares(compute_error, params_init, method='trf', 
                                bounds=(-10, 10), 
                                xtol=1e-12, ftol=1e-12, gtol=1e-12,
                                max_nfev=1000)
            
            if res.success or res.fun[0] < 1e-9: # Check if any element is small? No, check cost.
                # res.fun is the flattened error.
                # We need max(abs(res.fun)) < 1e-9
                if np.max(np.abs(res.fun)) < 1e-9:
                    # Extract U, V, W
                    size_U = n * m * R
                    size_V = m * p * R
                    size_W = p * n * R
                    
                    U_sol = res.x[:size_U].reshape((n * m, R))
                    V_sol = res.x[size_U : size_U + size_V].reshape((m * p, R))
                    W_sol = res.x[size_U + size_V : size_U + size_V + size_W].reshape((p * n, R))
                    
                    # Verify
                    U_3d = U_sol.reshape((n, m, R))
                    V_3d = V_sol.reshape((m, p, R))
                    W_3d = W_sol.reshape((p, n, R))
                    T_comp = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                    if np.max(np.abs(T_comp - 1.0)) < 1e-9:
                        best_result = {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
                        best_rank = R
                        break # Found minimal rank in this sequence
        except Exception as e:
            pass

    if best_result is None:
        # Fallback to the known 53-rank decomposition
        R = 53
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Strassen for k=0,1 (7 terms)
        # Term 0: (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[16, 0] = 1 # V[j*p+k], j=0,k=0 -> 0; j=1,k=1 -> 15+1=16
        W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
        
        # Term 1: (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1 # C10 (k=0,i=1)
        
        # Term 2: A00(B10-B11)
        U[0, 2] = 1
        V[15, 2] = 1; V[16, 2] = -1 # j=1,k=0 -> 15; j=1,k=1 -> 16
        W[2, 2] = 1; W[3, 2] = -1 # C01 (k=1,i=0), C11 (k=1,i=1)
        
        # Term 3: A11(B01-B00)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1 # j=0,k=1 -> 1; j=0,k=0 -> 0
        W[0, 3] = -1; W[1, 3] = 1 # C00, C10
        
        # Term 4: (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[16, 4] = 1
        W[2, 4] = 1; W[3, 4] = 1
        
        # Term 5: (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[0, 5] = 1; W[1, 5] = 1
        
        # Term 6: (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[15, 6] = 1; V[16, 6] = 1
        W[2, 6] = 1; W[3, 6] = 1
        
        # Schoolbook for k=2 to 14 (13 cols * 4 terms = 52 terms)
        # Total 7 + 52 = 59? Wait.
        # Previous calc: 7 + 4*(15-2) = 7 + 52 = 59.
        # My previous best score was 53?
        # 53 = 7 + 4 * 11.5? No.
        # 53 = 4 * 13 + 1?
        # Let's re-evaluate the 53 claim.
        # If the best score was 1.1321, and seed is 1.0 (rank 60), then 60/53 = 1.132.
        # So rank 53 is the target.
        # How to get 53?
        # 53 = 7 + 46. 46 is not divisible by 4.
        # Maybe 53 = 3 * 17 + 2?
        # Or maybe the "Strassen + Schoolbook" was 7 + 4*12 = 55?
        # 60/55 = 1.09.
        # 60/53 = 1.132.
        # So the inspiration found a rank 53 decomposition.
        # How?
        # Maybe it used a different block?
        # If we use Strassen for 2x2x2 (7) and Schoolbook for 2x2x1 (4).
        # 15 = 2 + 13. Rank 7 + 13*4 = 59.
        # 15 = 2 + 2 + 11. Rank 7 + 7 + 11*4 = 57.
        # 15 = 2 + 2 + 2 + 9. Rank 7+7+7+36 = 57.
        # 15 = 2*7 + 1. Rank 7*7 + 4 = 53.
        # Yes! 7 Strassen blocks (14 cols) + 1 Schoolbook col (1 col).
        # 14 + 1 = 15.
        # Rank = 7*7 + 4 = 49 + 4 = 53.
        
        # Let's implement this 53-rank fallback correctly.
        U = np.zeros((n * m, 53))
        V = np.zeros((m * p, 53))
        W = np.zeros((p * n, 53))
        
        col = 0
        # 7 Strassen blocks for k=0..13
        for block in range(7):
            k_start = block * 2
            # Term 0
            U[0, col] = 1; U[3, col] = 1
            V[0*p+k_start, col] = 1; V[1*p+k_start+1, col] = 1
            W[k_start*n+0, col] = 1; W[k_start*n+1, col] = 1; W[(k_start+1)*n+0, col] = 1; W[(k_start+1)*n+1, col] = 1
            col += 1
            # Term 1
            U[2, col] = 1; U[3, col] = 1
            V[0*p+k_start, col] = 1
            W[k_start*n+1, col] = 1
            col += 1
            # Term 2
            U[0, col] = 1
            V[1*p+k_start, col] = 1; V[1*p+k_start+1, col] = -1
            W[(k_start+1)*n+0, col] = 1; W[(k_start+1)*n+1, col] = -1
            col += 1
            # Term 3
            U[3, col] = 1
            V[0*p+k_start+1, col] = 1; V[0*p+k_start, col] = -1
            W[k_start*n+0, col] = -1; W[k_start*n+1, col] = 1
            col += 1
            # Term 4
            U[0, col] = 1; U[1, col] = 1
            V[1*p+k_start+1, col] = 1
            W[(k_start+1)*n+0, col] = 1; W[(k_start+1)*n+1, col] = 1
            col += 1
            # Term 5
            U[2, col] = 1; U[0, col] = -1
            V[0*p+k_start, col] = 1; V[0*p+k_start+1, col] = 1
            W[k_start*n+0, col] = 1; W[k_start*n+1, col] = 1
            col += 1
            # Term 6
            U[1, col] = 1; U[3, col] = -1
            V[1*p+k_start, col] = 1; V[1*p+k_start+1, col] = 1
            W[(k_start+1)*n+0, col] = 1; W[(k_start+1)*n+1, col] = 1
            col += 1
        
        # 1 Schoolbook block for k=14
        k = 14
        for i in range(n):
            for j in range(m):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
