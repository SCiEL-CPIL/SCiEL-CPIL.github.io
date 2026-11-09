"""
Strategy: Numerical Rank Search with Rounding
1. Assume a rank r decomposition exists.
2. Generate random U (n*m, r) and V (m*p, r) matrices.
3. Compute the tensor T (flattened) and the Khatri-Rao product of U and V.
4. Solve for W via least squares: T = (U ⊙ V) @ W.
5. Check if the reconstruction error is small.
6. If error is small, round U, V, W to nearest integers/half-integers.
7. Verify if the rounded matrices exactly reconstruct the tensor.
8. Return the smallest rank that works.
"""
import numpy as np
import time
import random

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # T indices: (i, j, k) for A[i,j] * B[j,k] -> C[i,k]
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank) -- Note: Task says W[k*n+i, r] which is (k, i) indexing?
    # "W[k*n+i,r]" implies index is k*n + i. So W is (p*n, rank).
    # The tensor T has entries T[i, j, k] = 1 if we are looking at the coefficient of A[i,j]B[j,k] in C[i,k].
    # Actually, the decomposition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # Wait, the matrix multiplication tensor is defined such that:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The rank decomposition of the multiplication tensor T (where T[i,j,k] = 1) is:
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[i,k,r]
    # Note the order of W indices. The prompt says W[k*n+i, r]. So W is indexed by (k, i).
    # So the term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    
    # Let's flatten the tensor T into a vector of length n*m*p.
    # Index mapping for T: (i, j, k) -> i*m*p + j*p + k
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # We want to find U, V, W of rank r such that:
    # For all i, j, k: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    
    # This is equivalent to:
    # Let T be a matrix of size (n*m*p, 1) filled with 1s? No.
    # The tensor T is the "multiplication tensor".
    # T_{i,j,k} = 1.
    # We are looking for a rank-r decomposition of this specific tensor.
    
    # Let's construct the "Khatri-Rao" matrix M of size (n*m*p, r) for a given U, V.
    # M[row, r] = U[i*m+j, r] * V[j*p+k, r] where row corresponds to (i,j,k).
    # Then we need M @ W = 1 (vector of ones).
    # Wait, W has size (p*n, r). The index in W is k*n+i.
    # So for a fixed (i,j,k), the contribution is sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    # This is not a simple linear system in W if we treat all (i,j,k) as one big vector,
    # because W[k*n+i, r] is shared across different j values for the same (k,i).
    
    # Let's restructure.
    # For each r, we have vectors u_r (size n*m), v_r (size m*p), w_r (size p*n).
    # The product tensor is sum_r u_r v_r^T w_r^T (with appropriate reshaping).
    # Specifically, T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    
    # We can fix U and V, and solve for W.
    # The equation is linear in W.
    # For each r, the term is U[:, r] (vector of n*m) * V[:, r] (vector of m*p) * W[:, r] (vector of p*n).
    # This is a trilinear form.
    
    # Let's try a different approach: 
    # We know the rank is at most 32 (schoolbook).
    # We suspect it might be lower (e.g., 28).
    # Let's try to find a rank-28 decomposition numerically.
    
    # Algorithm:
    # 1. Initialize U, V randomly.
    # 2. Solve for W using least squares? 
    #    The system is: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    #    This is a linear system in the entries of W.
    #    Let's vectorize W into a vector w of length p*n*r.
    #    The coefficient for W[k*n+i, r] in the equation for (i,j,k) is U[i*m+j, r] * V[j*p+k, r].
    #    So we have a matrix A of size (n*m*p, p*n*r) where A[(i,j,k), (k,i,r)] = U[i*m+j, r] * V[j*p+k, r].
    #    And we solve A @ w = 1.
    #    
    #    This is a huge matrix (32 x 28*8 = 32 x 224) for rank 28. Manageable.
    #    But we also need to optimize U and V.
    #    
    #    Alternative: Alternating Least Squares (ALS).
    #    - Fix V, W. Solve for U.
    #    - Fix U, W. Solve for V.
    #    - Fix U, V. Solve for W.
    #    
    #    However, the "solve for U" step is not straightforward because the equations are coupled.
    #    
    #    Let's stick to the "Fix U, V, solve for W" approach and then check error.
    #    If error is low, we have a candidate.
    #    Then we round U, V, W.
    #    Then we verify.
    
    # To improve the chance of finding a low-rank solution, we can use random restarts.
    # We will try ranks from 24 up to 32.
    # For each rank, we do a few random restarts.
    
    def verify(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Check if sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k
        # Use einsum for efficiency
        # U indices: (im, r), V: (jp, r), W: (kn, r)
        # We want to compute T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        
        # Reshape U to (n, m, r)
        U_reshaped = U.reshape(n, m, -1)
        # Reshape V to (m, p, r)
        V_reshaped = V.reshape(m, p, -1)
        # Reshape W to (p, n, r) -- Note: W is indexed by k*n+i, so k is first, i is second.
        W_reshaped = W.reshape(p, n, -1)
        
        # Compute product
        # Term r: U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Sum over r.
        
        # Let's do it in steps to avoid huge intermediate tensors if possible, but 2x4x4 is small.
        # Intermediate 1: X[i, j, k, r] = U[i, j, r] * V[j, k, r]
        # This is size 2*4*4*28 = 896. Small.
        
        # X = np.einsum('ijr,jkr->ijk r', U_reshaped, V_reshaped) -- syntax error, need space or comma
        # X = np.einsum('ijr,jkr->ijk r', U_reshaped, V_reshaped)
        
        # Actually, let's just loop over r to be safe and simple, since r is small.
        T_reconstructed = np.zeros((n, m, p))
        for r in range(U.shape[1]):
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            
            # Add u_r[i,j] * v_r[j,k] * w_r[k,i]
            # Product of u_r and v_r is (n, p) matrix M_r[i,k] = sum_j u_r[i,j] v_r[j,k]
            M_r = u_r @ v_r
            # Then multiply by w_r[k,i]
            # T_reconstructed[i,j,k] += u_r[i,j] * v_r[j,k] * w_r[k,i]
            # This is not just M_r * w_r because w_r depends on k and i, but the sum is over j.
            # Wait, the decomposition is:
            # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # It is NOT a matrix multiplication of the summed products.
            # It is a tensor product sum.
            
            # So for each r, we form a tensor T_r[i,j,k] = U[i,j,r] V[j,k,r] W[k,i,r]
            # And sum them up.
            
            # T_r[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r]
            # We can compute this as:
            # Temp[i,j,k] = U[i,j,r] * V[j,k,r]
            # Then T_r[i,j,k] = Temp[i,j,k] * W[k,i,r]
            
            # U[i,j,r] is u_r[i,j]
            # V[j,k,r] is v_r[j,k]
            # W[k,i,r] is w_r[k,i]
            
            # Let's compute Temp = u_r (n,m) and v_r (m,p).
            # We need to broadcast u_r[i,j] * v_r[j,k] to (n, m, p).
            # u_r[:, :, np.newaxis] * v_r[np.newaxis, :, :] -> (n, m, p)
            temp = u_r[:, :, np.newaxis] * v_r[np.newaxis, :, :]
            # Now multiply by w_r[k,i]
            # w_r is (p, n). We need it to be (1, 1, p, n) to broadcast with (n, m, p)?
            # No, temp is (n, m, p). w_r is (p, n).
            # We want to multiply temp[i,j,k] by w_r[k,i].
            # So we need w_r reshaped to (1, 1, p, n) and then permuted?
            # w_r[k,i] -> axis 2 is k, axis 3 is i.
            # temp has axes i, j, k.
            # So we need w_r to have axes (i, j, k) effectively.
            # w_r[k, i] -> we need to align k with axis 2 of temp, and i with axis 0 of temp.
            # So w_r_reshaped = w_r.T (n, p) -> (n, 1, p)? No.
            # w_r is (p, n). w_r.T is (n, p).
            # We want w_r[k, i] to be at position [i, ?, k].
            # So we need an array W_b of shape (n, 1, p) such that W_b[i, 0, k] = w_r[k, i].
            # W_b = w_r.T[:, np.newaxis, :] -> shape (n, 1, p).
            # W_b[i, 0, k] = w_r.T[i, k] = w_r[k, i]. Correct.
            
            w_b = w_r.T[:, np.newaxis, :]
            T_r = temp * w_b
            T_reconstructed += T_r
            
        # Check if T_reconstructed is all 1s
        return np.allclose(T_reconstructed, 1.0, atol=1e-9)

    def round_to_half_int(x):
        # Round to nearest 0.5
        return np.round(x * 2) / 2

    best_rank = n * m * p  # Fallback to schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from 24 to 32
    for r in range(24, 33):
        found = False
        # Try a few random restarts
        for restart in range(5):
            # Initialize U, V randomly
            # Use small integers to help rounding?
            U = np.random.randint(-1, 2, size=(n*m, r)).astype(float)
            V = np.random.randint(-1, 2, size=(m*p, r)).astype(float)
            
            # Solve for W
            # We need to solve A @ w = 1
            # A is (n*m*p, p*n*r)
            # A[(i,j,k), (k,i,r)] = U[i*m+j, r] * V[j*p+k, r]
            
            # Let's build A efficiently.
            # U: (n*m, r)
            # V: (m*p, r)
            
            # We can iterate over i, j, k to build rows of A?
            # n*m*p = 32. p*n*r = 8*r.
            # 32 x (8r) matrix.
            
            A = np.zeros((n*m*p, p*n*r))
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row_idx = i*m*p + j*p + k
                        u_val = U[i*m+j, :] # vector of size r
                        v_val = V[j*p+k, :] # vector of size r
                        
                        # For each r, coeff is u_val[r] * v_val[r]
                        # This coeff goes into column (k*n+i)*r + r_idx?
                        # W index is k*n+i.
                        # So W[k*n+i, r] is at flat index (k*n+i)*r + r.
                        
                        coeffs = u_val * v_val # size r
                        # Place coeffs into A[row_idx, (k*n+i)*r : (k*n+i)*r + r]
                        col_start = (k*n+i)*r
                        A[row_idx, col_start:col_start+r] = coeffs
            
            # Solve A @ w = 1
            # Use least squares
            try:
                w, residuals, rank, sv = np.linalg.lstsq(A, np.ones(n*m*p), rcond=None)
            except:
                continue
                
            # Reshape w to W (p*n, r)
            W = w.reshape(p*n, r)
            
            # Check reconstruction error
            # T_reconstructed = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # We can check error quickly.
            T_rec = np.zeros((n, m, p))
            for rr in range(r):
                u_r = U[:, rr].reshape(n, m)
                v_r = V[:, rr].reshape(m, p)
                w_r = W[:, rr].reshape(p, n)
                temp = u_r[:, :, np.newaxis] * v_r[np.newaxis, :, :]
                w_b = w_r.T[:, np.newaxis, :]
                T_rec += temp * w_b
                
            err = np.max(np.abs(T_rec - 1.0))
            
            if err < 1e-6:
                # Found a valid floating point decomposition
                # Now try to round
                U_rounded = round_to_half_int(U)
                V_rounded = round_to_half_int(V)
                W_rounded = round_to_half_int(W)
                
                if verify(U_rounded, V_rounded, W_rounded):
                    # Success!
                    best_rank = r
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    found = True
                    break
        if found:
            break
            
    # If no better rank found, use schoolbook
    if best_U is None:
        best_rank = n * m * p
        U = np.zeros((n*m, best_rank))
        V = np.zeros((m*p, best_rank))
        W = np.zeros((p*n, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_U = U
        best_V = V
        best_W = W

    # Convert to lists of lists for output
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
