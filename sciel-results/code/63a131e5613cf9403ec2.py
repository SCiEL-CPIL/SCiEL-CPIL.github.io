import numpy as np
import time
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import sys

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        c = np.convolve(h, h)
        return abs(2.0 * N * np.max(c) / (s * s))

    best_val = np.inf
    best_h = np.ones(N)

    # Update with seed value
    val_seed = compute_value(best_h)
    if val_seed < best_val:
        best_val = val_seed
        best_h = best_h.copy()

    # We will iterate over all possible lags k where the maximum of the autoconvolution
    # might occur. For each k, we find the h that minimizes the quadratic form
    # Q_k(h) = sum h[i] h[i+k] subject to sum(h) = 1.
    # This is solved via the KKT conditions.
    
    # Pre-allocate for speed? Not really needed for N=400.
    
    for k in range(N):
        if time.time() > deadline - 5.0:
            break
            
        # Construct the symmetric matrix A_k such that h^T A_k h = sum_{i} h[i] h[i+k]
        # Note: h^T A h = sum_{i,j} h[i] A[i,j] h[j]
        # We want sum_{i} h[i] h[i+k].
        # This is sum_{i=0}^{N-1-k} h[i] h[i+k].
        # In matrix form, A[i, i+k] = 0.5 and A[i+k, i] = 0.5 for i=0..N-1-k.
        # If k=0, A[i,i] = 1.
        
        if k == 0:
            # A is identity
            # Minimize h^T I h subject to 1^T h = 1
            # Solution is h = 1/N * ones
            h_candidate = np.ones(N) / N
        else:
            # Construct sparse A_k
            # Diagonals: k and -k
            # Length of off-diagonals is N-k
            main_diag = np.zeros(N)
            off_diag = np.ones(N - k)
            
            # For k > 0, the diagonals are at offset k and -k
            A_k = diags([off_diag, off_diag], [k, -k], shape=(N, N), format='csr')
            
            # We need to solve:
            # 2 * A_k * h + lambda * 1 = 0
            # 1^T * h = 1
            #
            # This is a linear system:
            # [ 2*A_k   1 ] [ h ]   [ 0 ]
            # [ 1^T     0 ] [ lam ] = [ 1 ]
            
            # Build the sparse KKT matrix
            # Block (0,0): 2 * A_k
            # Block (0,1): 1 (column vector of ones)
            # Block (1,0): 1^T (row vector of ones)
            # Block (1,1): 0
            
            # Note: A_k is symmetric.
            
            # We can solve this using scipy.sparse.linalg
            # However, A_k might be singular.
            # The KKT matrix is indefinite, but typically non-singular if A_k is non-singular on the subspace orthogonal to 1?
            # Actually, if A_k is singular, the system might still have a unique solution due to the constraint.
            # Let's try to solve it. If it fails, we skip this k.
            
            # Construct the full sparse matrix
            # Size (N+1) x (N+1)
            
            # It is often faster to just solve A_k h = -lambda/2 * 1
            # But lambda is unknown.
            
            # Alternative: 
            # Minimize h^T A h subject to c^T h = 1.
            # Solution h = A^{-1} c / (c^T A^{-1} c) IF A is invertible.
            # If A is singular, we use the pseudo-inverse or solve the KKT system.
            
            # Let's try solving A_k x = 1.
            try:
                x = spsolve(A_k, np.ones(N))
                denom = np.dot(np.ones(N), x)
                if abs(denom) < 1e-12:
                    # A_k is singular or denom is zero. 
                    # Fallback: Use a small ridge regularization or skip.
                    # Adding epsilon to diagonal makes it positive definite? 
                    # No, A_k has negative eigenvalues potentially.
                    # Let's try KKT directly.
                    raise np.linalg.LinAlgError("Singular or near-singular")
                else:
                    h_candidate = x / denom
            except Exception:
                # Solve KKT system directly
                try:
                    # Build KKT matrix
                    # 2*A_k
                    A_k_2 = 2.0 * A_k
                    # Vector of ones
                    vec_1 = np.ones(N)
                    
                    # Sparse construction
                    # Row 0..N-1, Col 0..N-1: A_k_2
                    # Row 0..N-1, Col N: 1
                    # Row N, Col 0..N-1: 1
                    # Row N, Col N: 0
                    
                    # We can use scipy.sparse.block or manual construction.
                    # Manual is safer for small N.
                    # Actually, N=400, dense is fine? 401x401 dense is ~1.6MB. 
                    # 400 iterations * 1.6MB * ops. 
                    # Dense solve is very fast.
                    
                    A_k_dense = A_k.toarray()
                    KKT = np.zeros((N+1, N+1))
                    KKT[:N, :N] = 2.0 * A_k_dense
                    KKT[:N, N] = 1.0
                    KKT[N, :N] = 1.0
                    KKT[N, N] = 0.0
                    
                    b = np.zeros(N+1)
                    b[N] = 1.0
                    
                    sol = np.linalg.solve(KKT, b)
                    h_candidate = sol[:N]
                except Exception:
                    continue
        
        # Check validity and compute value
        s = np.sum(h_candidate)
        if abs(s) < 1e-12:
            continue
        
        # Normalize to ensure sum is 1? The KKT solution already satisfies sum=1.
        # But due to numerical errors, let's renormalize.
        h_candidate = h_candidate / s
        
        val_candidate = compute_value(h_candidate)
        
        if val_candidate < best_val:
            best_val = val_candidate
            best_h = h_candidate.copy()

    # Optional: Local refinement using L-BFGS-B on the best candidate found
    # This can help if the "active lag" assumption was slightly off or if there's a better local minimum nearby.
    if time.time() < deadline - 10.0:
        # Determine the active lag of the best candidate
        c_best = np.convolve(best_h, best_h)
        k_active = np.argmax(c_best)
        
        # Define objective for this specific lag
        def obj_grad(h):
            s = np.sum(h)
            if s < 1e-12:
                return np.inf, np.zeros(N)
            q = np.dot(h[:N-k_active], h[k_active:])
            grad_q = np.zeros(N)
            grad_q[:N-k_active] += h[k_active:]
            grad_q[k_active:] += h[:N-k_active]
            val = 2.0 * N * q / (s * s)
            grad_f = 2.0 * N * (grad_q * s - 2.0 * q) / (s ** 3)
            return val, grad_f

        from scipy.optimize import minimize
        try:
            res = minimize(
                obj_grad, 
                best_h, 
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
            )
            if res.fun < best_val:
                h_refined = res.x
                s_ref = np.sum(h_refined)
                if abs(s_ref) > 1e-6:
                    val_ref = compute_value(h_refined)
                    if val_ref < best_val:
                        best_val = val_ref
                        best_h = h_refined
        except Exception:
            pass

    return best_h
