import numpy as np
import time
import random

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try to find a lower rank decomposition
    # The tensor T has T[i,j,k] = 1 for all i,j,k
    # We seek U (n*m x R), V (m*p x R), W (p*n x R)
    # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # We fix U and V randomly and solve for W
    # The system is linear in W
    # A * vec(W) = 1
    # A has shape (n*m*p, p*n*R)
    
    def try_rank(R, max_iters=100):
        # Build the structure of A
        # Rows indexed by (i, j, k)
        # Cols indexed by (k, i, r) -> index = (k*n + i)*R + r
        # Coeff A[i,j,k, k,i,r] = U[i*m+j, r] * V[j*p+k, r]
        
        # Precompute indices
        rows = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    rows.append((i, j, k))
        
        num_eq = len(rows)
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Random sparse U and V
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            
            # Initialize with small integers
            for r in range(R):
                # Pick a few non-zeros for U and V
                # This structure often leads to simpler solutions
                idx_u = random.sample(range(n * m), k=min(3, n * m))
                idx_v = random.sample(range(m * p), k=min(3, m * p))
                for idx in idx_u:
                    U[idx, r] = random.choice([-1, 1])
                for idx in idx_v:
                    V[idx, r] = random.choice([-1, 1])
            
            # Build A matrix
            # A is sparse
            # We can build it as a dense matrix for small R
            A = np.zeros((num_eq, p * n * R))
            
            for eq_idx, (i, j, k) in enumerate(rows):
                # The equation is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # W is flattened: W_flat[(k*n+i)*R + r]
                for r in range(R):
                    coeff = U[i * m + j, r] * V[j * p + k, r]
                    if coeff != 0:
                        w_idx = (k * n + i) * R + r
                        A[eq_idx, w_idx] = coeff
            
            # Solve A * W_flat = 1
            try:
                W_flat, residuals, rank, s = np.linalg.lstsq(A, np.ones(num_eq), rcond=None)
            except Exception:
                continue
            
            # Check residual
            if residuals and residuals[0] > 1e-9:
                continue
            
            # Check if solution is valid
            # Verify
            W = W_flat.reshape((p * n, R))
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Snap to half-integers if close
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify snapped version
                valid_snap = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_snap[i * m + j, :] * V_snap[j * p + k, :] * W_snap[k * n + i, :])
                            if abs(val - 1.0) > 1e-9:
                                valid_snap = False
                                break
                        if not valid_snap:
                            break
                    if not valid_snap:
                        break
                
                if valid_snap:
                    return U_snap, V_snap, W_snap
                else:
                    # Return floating point if exact, but problem asks for int/half-int
                    # If not close to half-int, skip
                    pass
        
        return None

    # Search for lower ranks
    for R in range(43, 30, -1):
        res = try_rank(R, max_iters=50)
        if res is not None:
            U, V, W = res
            # Convert to lists of lists for output
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
        
        if time.time() > deadline:
            break

    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
