"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 16, 24, 24
    
    # Build the target tensor T[i,j,k] = 1 if it's a valid multiplication entry
    # T is a 3-tensor: T[i,j,k] = 1 for all i,j,k (it's the structure tensor)
    # Actually the matmul tensor is: for each (i,j,k), the coefficient is 1
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # and = 0 for all "invalid" combinations... wait, no.
    # The tensor T[i,j,k] = 1 for all valid (i,j,k) in the matmul.
    # Actually the matmul tensor has T[i,j,k] = 1 for ALL i in [0,n), j in [0,m), k in [0,p)
    # because (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor is simply all-ones of shape (n, m, p).
    
    # We need rank-r decomposition: U (nm x r), V (mp x r), W (pn x r)
    # such that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    
    # Strategy: Start with a random rank-r decomposition and optimize
    # Use the fact that we can set up an optimization problem
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        r = U.shape[1]
        # Compute the reconstructed tensor
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # U is (nm, r), V is (mp, r), W is (pn, r)
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # Use einsum
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return T
    
    def is_valid(U, V, W, tol=1e-9):
        T = verify(U, V, W)
        return np.allclose(T, 1.0, atol=tol, rtol=0)
    
    def rank_of(U, V, W):
        return U.shape[1]
    
    # Try various ranks, starting low
    # Lower bound: the tensor is all-ones (n,m,p) = (4,4,6)
    # The rank of the all-ones tensor... it's at least max(n,m,p) = 6
    # But we need it to factor through the specific index structure
    
    best_rank = n * m * p  # schoolbook
    best_decomp = None
    
    # First, get the schoolbook decomposition as fallback
    U_sb = np.zeros((nm, best_rank))
    V_sb = np.zeros((mp, best_rank))
    W_sb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_decomp = (U_sb, V_sb, W_sb)
    best_rank = best_rank
    
    # Now try to find lower rank decompositions
    # Use a random initialization + optimization approach
    
    np.random.seed(42)
    
    # Try ranks from a reasonable lower bound up
    # The all-ones tensor of shape (4,4,6) has rank... 
    # Actually, we can think of it as: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    # but with the constraint that u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] sums correctly
    
    # Alternative approach: use the structure more directly
    # For each r, define vectors a_r (length n*m), b_r (length m*p), c_r (length p*n)
    # We need: sum_r a_r[i*m+j] * b_r[j*p+k] * c_r[k*n+i] = 1
    
    # Let's try a parametrization where we fix some structure
    # Key insight: if we can write 1 = sum_r f_r(i,j) * g_r(j,k) * h_r(k,i)
    # where f_r, g_r, h_r are separable functions
    
    # Try rank r from 24 upward (arbitrary lower bound based on dimensions)
    for target_rank in range(24, min(97, 100)):
        if time.time() > deadline - 5:
            break
        
        found = False
        n_attempts = 50 if target_rank < 60 else 20
        
        for attempt in range(n_attempts):
            if time.time() > deadline - 5:
                break
            
            r = target_rank
            # Random initialization
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            # Optimize using a simple gradient-free or iterative approach
            # Use the residual: R[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1
            # We want to minimize ||R||_F^2
            
            # Use a simple iterative update
            lr = 0.01
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                U3 = U.reshape(n, m, r)
                V3 = V.reshape(m, p, r)
                W3 = W.reshape(p, n, r)
                
                # Compute residual
                T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                R = T - 1.0
                
                # Gradient with respect to U:
                # dR[i,j,k]/dU[i*m+j,r] = V3[j,k,r] * W3[k,i,r]
                # dL/dU[i*m+j,r] = sum_k R[i,j,k] * V3[j,k,r] * W3[k,i,r]
                grad_U = np.einsum('ijk,jkr,kir->ijr', R, V3, W3)
                grad_U = grad_U.reshape(nm, r)
                
                # Gradient with respect to V:
                # dR[i,j,k]/dV[j*p+k,r] = U3[i,j,r] * W3[k,i,r]
                grad_V = np.einsum('ijk,ijr,kir->jkr', R, U3, W3)
                grad_V = grad_V.reshape(mp, r)
                
                # Gradient with respect to W:
                # dR[i,j,k]/dW[k*n+i,r] = U3[i,j,r] * V3[j,k,r]
                grad_W = np.einsum('ijk,ijr,jkr->kir', R, U3, V3)
                grad_W = grad_W.reshape(pn, r)
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if np.max(np.abs(R)) < 1e-10:
                    break
            
            # Check if we found a valid decomposition
            T_final = verify(U, V, W)
            if np.allclose(T_final, 1.0, atol=1e-9, rtol=0):
                # Verify more carefully
                if np.max(np.abs(T_final - 1.0)) < 1e-8:
                    found = True
                    best_rank = r
                    best_decomp = (U, V, W)
                    break
        
        if found:
            break
    
    # Convert to lists of lists with exact fractions if possible
    U, V, W = best_decomp
    
    # Try to clean up near-zero and near-integer values
    def clean_matrix(M):
        M = M.copy()
        # Round values that are very close to integers or half-integers
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M[i, j]
                if abs(val) < 1e-10:
                    M[i, j] = 0.0
                else:
                    # Try to snap to nearest half-integer
                    snapped = round(val * 2) / 2
                    if abs(val - snapped) < 1e-8:
                        M[i, j] = snapped
        return M
    
    U_clean = clean_matrix(U)
    V_clean = clean_matrix(V)
    W_clean = clean_matrix(W)
    
    # Verify the cleaned version
    T_check = verify(U_clean, V_clean, W_clean)
    if not np.allclose(T_check, 1.0, atol=1e-8, rtol=0):
        # Use the uncleaned version
        U_clean, V_clean, W_clean = U, V, W
    
    # Convert to list format
    U_list = [[U_clean[i, j] for j in range(U_clean.shape[1])] for i in range(U_clean.shape[0])]
    V_list = [[V_clean[i, j] for j in range(V_clean.shape[1])] for i in range(V_clean.shape[0])]
    W_list = [[W_clean[i, j] for j in range(W_clean.shape[1])] for i in range(W_clean.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

