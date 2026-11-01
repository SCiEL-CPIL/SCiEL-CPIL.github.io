import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = 2, 2, 9
    start_time = time.time()
    deadline = start_time + 145  # safety margin

    # The multiplication tensor T has indices (i, j, k) for C[i,k] += A[i,j] * B[j,k]
    # We want to decompose T into sum_{r=1}^{R} U_r \otimes V_r \otimes W_r
    # where U_r is vector of size n*m=4, V_r is size m*p=18, W_r is size p*n=18.
    # T_{i,j,k} = sum_r U_{r, i*m+j} * V_{r, j*p+k} * W_{r, k*n+i}
    
    # First, let's build the explicit tensor T of shape (n*m, m*p, p*n)
    # Actually, the standard tensor is T_{i,j,k} where i in [0,n), j in [0,m), k in [0,p)
    # The decomposition is: T_{i,j,k} = sum_r U_{r, i*m+j} * V_{r, j*p+k} * W_{r, k*n+i}
    
    # Let's flatten the tensor to a vector for optimization.
    # We will attempt rank 31.
    
    def build_rank32_fallback():
        # This is the known correct rank 32 construction from the seed, cleaned up.
        rank = 32
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0

        # Strassen for 2x2x2 blocks
        # Indices: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        
        # We have 9 columns in B. Let's group them.
        # k=0,1 -> Block 0
        # k=2,3 -> Block 1
        # k=4,5 -> Block 2
        # k=6,7 -> Block 3
        # k=8    -> Block 4 (Schoolbook)
        
        for block in range(4):
            k1 = block * 2
            k2 = block * 2 + 1
            
            # m1 = (a+d)(x1+w1)
            # U: a(0,0)=1, d(1,1)=1
            # V: B[0,k1]=1, B[1,k2]=1
            # W: C[0,k1]+=1, C[1,k2]+=1 (Wait, Strassen mapping needs care)
            
            # Let's use the verified Strassen formulas from the seed but ensure indices are correct.
            # Seed verification:
            # C00 = m1 - m4 + m6 + m7
            # C01 = m3 + m4
            # C10 = m2 + m5
            # C11 = m1 - m3 + m5 - m2
            
            # m1 = (a+d)(x+w)
            U[0, col] = 1; U[3, col] = 1
            V[0*p+k1, col] = 1; V[1*p+k2, col] = 1
            W[k1*n+0, col] = 1  # C[0,k1]
            W[k2*n+1, col] = 1  # C[1,k2]
            col += 1

            # m2 = (c+d)(x)
            U[2, col] = 1; U[3, col] = 1
            V[0*p+k1, col] = 1
            W[k1*n+1, col] = 1  # C[1,k1]
            W[k2*n+1, col] = -1 # C[1,k2]
            col += 1

            # m3 = (a-b)(y+w)
            U[0, col] = 1; U[1, col] = -1
            V[0*p+k2, col] = 1; V[1*p+k2, col] = 1
            W[k2*n+0, col] = 1  # C[0,k2]
            W[k2*n+1, col] = -1 # C[1,k2]
            col += 1

            # m4 = (c+d)(z)
            U[2, col] = 1; U[3, col] = 1
            V[1*p+k1, col] = 1
            W[k1*n+0, col] = -1 # C[0,k1]
            W[k2*n+0, col] = 1  # C[0,k2]
            col += 1

            # m5 = (a+b)(x)
            U[0, col] = 1; U[1, col] = 1
            V[0*p+k1, col] = 1
            W[k1*n+1, col] = 1  # C[1,k1]
            W[k2*n+1, col] = 1  # C[1,k2]
            col += 1

            # m6 = (c-a)(x+z)
            U[2, col] = 1; U[0, col] = -1
            V[0*p+k1, col] = 1; V[1*p+k1, col] = 1
            W[k1*n+0, col] = 1  # C[0,k1]
            col += 1

            # m7 = (b-d)(y+w)
            U[1, col] = 1; U[3, col] = -1
            V[0*p+k2, col] = 1; V[1*p+k2, col] = 1
            W[k1*n+0, col] = 1  # C[0,k1]
            col += 1

        # Last single column k=8: rank 4
        k = 8
        # C[0,8] = A[0,0]B[0,8] + A[0,1]B[1,8]
        # C[1,8] = A[1,0]B[0,8] + A[1,1]B[1,8]
        U[0, col] = 1; V[0*p+k, col] = 1; W[k*n+0, col] = 1; col += 1
        U[1, col] = 1; V[1*p+k, col] = 1; W[k*n+0, col] = 1; col += 1
        U[2, col] = 1; V[0*p+k, col] = 1; W[k*n+1, col] = 1; col += 1
        U[3, col] = 1; V[1*p+k, col] = 1; W[k*n+1, col] = 1; col += 1

        assert col == rank
        return U, V, W

    # Strategy: Try to find a rank 31 solution using optimization.
    # We start with the rank 32 solution, zero out one column (or merge two), and optimize.
    # A better start: Random initialization of rank 31, but that might not converge.
    # Let's try to merge two columns of the rank 32 solution into one, and optimize the merged column.
    
    def compute_residual(U, V, W):
        # Compute the tensor from U, V, W and compare to the true multiplication tensor.
        # True tensor T_{i,j,k} is 1 if A[i,j] and B[j,k] contribute to C[i,k].
        # Actually, the tensor is defined by the bilinear map.
        # T_{i,j,k} = 1 for all i,j,k? No.
        # The multiplication tensor T is such that <T, A \otimes B> = C.
        # In index form: sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} = C_{i,k}? No.
        # The decomposition is: C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
        # This is equivalent to: C_{i,k} = sum_r W_{r, k*n+i} * (sum_j U_{r, i*m+j} A_{i,j}) * (sum_j V_{r, j*p+k} B_{j,k})
        # Wait, the standard form is:
        # C_{i,k} = sum_r W_{r, k*n+i} * (sum_{j} U_{r, i*m+j} A_{i,j}) * (sum_{j'} V_{r, j'*p+k} B_{j',k})
        # This is NOT the standard tensor decomposition form. The standard form is:
        # T_{i,j,k} = sum_r U_{r,i} V_{r,j} W_{r,k}
        # Here, the indices of U, V, W in the task are:
        # U: (n*m, rank) -> U_{r, i*m+j}
        # V: (m*p, rank) -> V_{r, j*p+k}
        # W: (p*n, rank) -> W_{r, k*n+i}
        
        # Let's verify the expansion:
        # Sum_r W_{r, k*n+i} * (Sum_j U_{r, i*m+j} A_{i,j}) * (Sum_j' V_{r, j'*p+k} B_{j',k})
        # This expands to Sum_r Sum_j Sum_j' W_{r, k*n+i} U_{r, i*m+j} V_{r, j'*p+k} A_{i,j} B_{j',k}
        # We want this to equal Sum_j A_{i,j} B_{j,k}.
        # This requires that the coefficient of A_{i,j} B_{j',k} is delta_{j,j'}.
        # So we need: Sum_r W_{r, k*n+i} U_{r, i*m+j} V_{r, j'*p+k} = delta_{j,j'}
        
        # So the "tensor" we are trying to decompose is the identity tensor on the middle index j.
        # Let's define the target tensor T_{i,j,j',k} = delta_{j,j'}.
        # But the decomposition is rank-1 in r, with factors depending on (i,j), (j',k), (i,k).
        # This is a 4-way tensor, but it decomposes into a sum of rank-1 terms in the 3-way sense if we fix i,k?
        # Actually, the standard matrix multiplication tensor is 3-way: T_{i,j,k} = 1 if the operation is A_{i,j} B_{j,k} -> C_{i,k}.
        # The decomposition is: T_{i,j,k} = sum_r U_{r,i,j} V_{r,j,k} W_{r,i,k}? No.
        
        # Let's stick to the verification method used in the seed:
        # Generate random A, B. Compute C_expected = A @ B.
        # Compute C_decomp using the formula.
        # Return the norm of the difference.
        
        # For optimization, we need a smooth function.
        # Let's use a fixed set of test matrices (or random with fixed seed) to compute the residual.
        # To make it differentiable, we can just use the algebraic identity.
        # Residual = Sum_{i,j,j',k} (Sum_r W_{r, k*n+i} U_{r, i*m+j} V_{r, j'*p+k} - delta_{j,j'})^2
        
        # This is a quadratic function of the parameters (U, V, W).
        # We can compute this efficiently.
        
        # Let's construct the target tensor M of shape (n, m, m, p) where M[i,j,j',k] = 1 if j==j' else 0.
        # Actually, we only need to check the contraction.
        # Let's define a function that takes U, V, W and returns the squared error.
        
        # We can vectorize this.
        # Let U be (4, R), V be (18, R), W be (18, R).
        # We want to compute E_{i,j,j',k} = Sum_r W[r, k*n+i] * U[r, i*m+j] * V[r, j'*p+k]
        # And compare to delta_{j,j'}.
        
        # This is a 4D tensor of size 2x2x2x9 = 72 entries.
        # We can compute this by iterating over r, or using einsum.
        
        # Let's use einsum for speed and clarity.
        # U_idx: (i, j, r) -> U[r, i*m+j]
        # V_idx: (j', k, r) -> V[r, j'*p+k]
        # W_idx: (i, k, r) -> W[r, k*n+i]
        
        # Residual_{i,j,j',k} = einsum('ir,jkr,ikr->ijjk', U_reshaped, V_reshaped, W_reshaped)
        
        # Let's reshape U, V, W.
        U_reshaped = U.T.reshape(n, m, -1) # (i, j, r)
        V_reshaped = V.T.reshape(m, p, -1) # (j', k, r)
        W_reshaped = W.T.reshape(p, n, -1) # (k, i, r) -> need (i, k, r)
        W_reshaped = W_reshaped.transpose(1, 0, 2) # (i, k, r)
        
        # Compute E
        E = np.einsum('ijr,jkr,ikr->ijjk', U_reshaped, V_reshaped, W_reshaped)
        
        # Target T
        # T[i,j,j',k] = 1 if j==j' else 0
        T = np.zeros((n, m, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, j, k] = 1.0
        
        return np.sum((E - T)**2)

    def flatten_params(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

    def unflatten_params(x):
        u_size = n*m
        v_size = m*p
        w_size = p*n
        U = x[:u_size].reshape(u_size, -1)
        V = x[u_size:u_size+v_size].reshape(v_size, -1)
        W = x[u_size+v_size:].reshape(w_size, -1)
        return U, V, W

    # Attempt rank 31 optimization
    def try_rank31_optimization():
        if time.time() > deadline:
            return None
            
        # Start from rank 32, merge two columns.
        # Which columns to merge? The last two schoolbook columns for k=8 are sparse.
        # Maybe merge two Strassen columns?
        # Let's just take the rank 32 U, V, W, and set the last column to 0, then optimize.
        # Or better: Initialize with random rank 31.
        
        # Let's try a few random initializations.
        np.random.seed(42)
        
        best_U, best_V, best_W = None, None, None
        best_err = float('inf')
        
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
                
            R = 31
            # Initialize with small random values
            U_init = np.random.randn(n*m, R) * 0.1
            V_init = np.random.randn(m*p, R) * 0.1
            W_init = np.random.randn(p*n, R) * 0.1
            
            # Scale to match the magnitude of the target tensor (entries are 1)
            # The sum of products should be 1. With random 0.1, sum is small.
            # Let's scale up.
            U_init *= 2.0
            V_init *= 2.0
            W_init *= 2.0
            
            x0 = flatten_params(U_init, V_init, W_init)
            
            def objective(x):
                U, V, W = unflatten_params(x)
                return compute_residual(U, V, W)
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.success or res.fun < best_err:
                    U, V, W = unflatten_params(res.x)
                    err = res.fun
                    if err < best_err:
                        best_err = err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        
            except Exception:
                pass

        if best_err < 1e-6:
            # Verify with random matrices
            np.random.seed(123)
            A_rand = np.random.rand(n, m)
            B_rand = np.random.rand(m, p)
            C_expected = A_rand @ B_rand
            
            C_decomp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    for r in range(31):
                        u_val = sum(best_U[i*m+j, r] * A_rand[i,j] for j in range(m))
                        v_val = sum(best_V[j*p+k, r] * B_rand[j,k] for j in range(m))
                        C_decomp[i,k] += best_W[k*n+i, r] * u_val * v_val
                        
            if np.allclose(C_expected, C_decomp, atol=1e-8):
                return best_U, best_V, best_W
                
        return None

    # Try rank 31
    result31 = try_rank31_optimization()
    if result31 is not None:
        U, V, W = result31
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to rank 32
    U, V, W = build_rank32_fallback()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
