import numpy as np
import time
from fractions import Fraction
from itertools import product
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]
START_TIME = time.time()
TIME_LIMIT = 120.0

def construct():
    n, m, p = DIMENSIONS
    
    # Build the target tensor T
    # T is a 3D array of shape (n*m, m*p, p*n)
    # But the decomposition is U (n*m, r) x V (m*p, r) x W (p*n, r)
    # The tensor equation is sum_r U[:, r] V[:, r]^T W[:, r]^T? No.
    # The tensor T has indices (a, b, c) where a is in U, b in V, c in W.
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) ... wait.
    # The multiplication is C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor T is defined such that T_{(i,j), (j,k), (k,i)} is not quite right.
    # The standard tensor for matrix multiplication M_{n,m,p} has:
    # T_{i, j, k} is not the right indices.
    # The legs are A, B, C.
    # A index: (i, j). B index: (j, k). C index: (k, i) ?
    # The contraction is over j.
    # So T_{(i,j), (j,k), (k,i)} = delta_{j, j'} delta_{k, k'} delta_{i, i'}?
    # No, T is the "multiplication tensor".
    # T_{a, b, c} = 1 if a=(i,j), b=(j,k), c=(k,i) is NOT correct.
    # The tensor is such that <T, U \otimes V \otimes W> = sum U_{ij} V_{jk} W_{ki} ?
    # No, we want sum U_{ij,r} V_{jk,r} W_{ki,r} to equal the product?
    # The identity is:
    # sum_j A_{ij} B_{jk} = C_{ik}.
    # This can be written as:
    # sum_{r} U_{ij,r} V_{jk,r} W_{ki,r} = delta_{j, \text{something}}?
    # Actually, the tensor T is defined by:
    # T_{(i,j), (j,k), (k,i)} = 1 is WRONG.
    # The tensor T has non-zero entries at:
    # T_{(i,j), (j,k), (k,i)} is not a single entry.
    # The tensor T is the sum of basis tensors:
    # T = sum_{i,j,k} E_{ij} \otimes E_{jk} \otimes E_{ki}? No.
    # T = sum_{i,j,k} (E_{ij} \otimes E_{jk} \otimes E_{ki}) is not right.
    # The correct tensor is:
    # T_{a, b, c} = 1 if a=(i,j), b=(j,k), c=(k,i) is not possible because c depends on k and i.
    # The indices are:
    # Leg 1: (i, j)
    # Leg 2: (j, k)
    # Leg 3: (k, i)
    # For a fixed i, j, k, the term A_{ij} B_{jk} contributes to C_{ik}.
    # So the tensor entry T_{(i,j), (j,k), (k,i)} should be 1?
    # But (k,i) is the index for C.
    # So T_{(i,j), (j,k), (k,i)} = 1.
    # Let's verify dimensions.
    # Leg 1 size: n*m = 6.
    # Leg 2 size: m*p = 12.
    # Leg 3 size: p*n = 8.
    # Total entries: 6 * 12 * 8 = 576.
    # Non-zero entries: for each i, j, k, there is one entry.
    # i in 0..1, j in 0..2, k in 0..3.
    # Total non-zeros: 2 * 3 * 4 = 24.
    # So T is sparse.
    
    def build_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    T[idx_u, idx_v, idx_w] = 1.0
        return T

    T = build_tensor()
    
    def build_rank_22():
        # ... existing code ...
        pass

    # Try to find rank 21
    best_rank = 22
    best_U, best_V, best_W = None, None, None
    
    # Fallback
    U22, V22, W22, r22 = build_rank_22()
    best_U, best_V, best_W = U22, V22, W22

    # Numerical search for rank 21
    # Use time budget
    time_left = TIME_LIMIT - (time.time() - START_TIME)
    if time_left > 10:
        r = 21
        # Flatten U, V, W
        # U: 6 x 21, V: 12 x 21, W: 8 x 21
        # Total vars: 6*21 + 12*21 + 8*21 = 21 * 26 = 546
        # Objective: || T - sum_r U_r V_r^T W_r^T ||_F
        # Note: The tensor product is U[:, r] V[:, r]^T W[:, r]^T?
        # No, the tensor sum is sum_r np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        def residuals(params):
            U_flat, V_flat, W_flat = params[:6*r], params[6*r:6*r+12*r], params[6*r+12*r:]
            U = U_flat.reshape(6, r)
            V = V_flat.reshape(12, r)
            W = W_flat.reshape(8, r)
            
            # Compute sum of outer products
            # This is expensive. 
            # T_computed = np.zeros((6, 12, 8))
            # for k in range(r):
            #     T_computed += np.einsum('i,j,k->ijk', U[:, k], V[:, k], W[:, k])
            
            # Vectorized:
            # T_computed = np.einsum('ik,jk,lk->ijl', U, V, W)
            T_comp = np.einsum('ik,jk,lk->ijl', U, V, W)
            return (T_comp - T).ravel()

        # Random init
        np.random.seed(42)
        for attempt in range(5):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            U_init = np.random.randn(6, r)
            V_init = np.random.randn(12, r)
            W_init = np.random.randn(8, r)
            params_init = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])
            
            try:
                res = least_squares(residuals, params_init, method='lm', max_nfev=1000)
                if res.cost < 1e-10:
                    # Found a solution?
                    # Check if it can be rounded
                    U_sol = res.x[:6*r].reshape(6, r)
                    V_sol = res.x[6*r:6*r+12*r].reshape(12, r)
                    W_sol = res.x[6*r+12*r:].reshape(8, r)
                    
                    # Round to nearest 0.5
                    U_round = np.round(U_sol * 2) / 2
                    V_round = np.round(V_sol * 2) / 2
                    W_round = np.round(W_sol * 2) / 2
                    
                    # Verify
                    T_check = np.einsum('ik,jk,lk->ijl', U_round, V_round, W_round)
                    if np.allclose(T_check, T, atol=1e-9):
                        # Convert to list of lists for output
                        # Ensure integers or half-integers
                        U_out = [[float(x) for x in row] for row in U_round]
                        V_out = [[float(x) for x in row] for row in V_round]
                        W_out = [[float(x) for x in row] for row in W_round]
                        best_U, best_V, best_W = U_out, V_out, W_out
                        best_rank = 21
                        break
            except Exception:
                pass

    return {"U": best_U, "V": best_V, "W": best_W}
