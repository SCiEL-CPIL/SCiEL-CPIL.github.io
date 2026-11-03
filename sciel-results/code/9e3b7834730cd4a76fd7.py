"""Matrix multiplication tensor decomposition for 2x3x8 using Kronecker product structure."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 8]

def _build_target_tensor(n, m, p):
    """Build the target matmul tensor T[i,j,k]=1 for all i,j,k."""
    T = np.ones((n, m, p))
    return T

def _validate_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _find_rank5_2x3x2():
    """Find a rank-5 decomposition of the 2x3x2 matmul tensor."""
    n, m, p = 2, 3, 2
    T = np.ones((n, m, p))
    
    # We need 5 terms: (u_r, v_r, w_r) with u_r in R^2, v_r in R^3, w_r in R^2
    # such that sum_r u_r[i]*v_r[j]*w_r[k] = 1 for all i,j,k
    
    # Parameterize: flatten all variables
    # Total variables: 5*(2+3+2) = 25
    # Constraints: 12 equations (but they're polynomial)
    
    # Use alternating minimization:
    # Fix V and W, solve for U (linear)
    # Fix U and W, solve for V (linear)
    # Fix U and V, solve for W (linear)
    
    rng = np.random.default_rng(42)
    
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    
    for trial in range(50):
        # Random initialization
        U = rng.standard_normal((n, 5))
        V = rng.standard_normal((m, 5))
        W = rng.standard_normal((p, 5))
        
        # Alternating minimization
        for it in range(200):
            # Solve for U: for each i, sum_r U[i,r]*V[j,r]*W[k,r] = 1 for all j,k
            # This is: for each i, the vector [U[i,0],...,U[i,4]] must satisfy
            # sum_r U[i,r] * (V[j,r]*W[k,r]) = 1 for all j,k
            # Let A[i] be a 5x5 matrix where A[i][r,s] = sum over (j,k) of V[j,r]*W[k,r]*V[j,s]*W[k,s]
            # Actually, it's a least squares problem:
            # For fixed i, we need: B_i @ U[i,:] = ones(m*p)
            # where B_i[j*p+k, r] = V[j,r]*W[k,r]
            
            for i in range(n):
                B = np.zeros((m*p, 5))
                for j in range(m):
                    for k in range(p):
                        for r in range(5):
                            B[j*p+k, r] = V[j, r] * W[k, r]
                # Solve B @ U[i,:] = ones(m*p)
                sol, residuals, rank_B, sv = np.linalg.lstsq(B, np.ones(m*p), rcond=None)
                U[i, :] = sol
            
            # Solve for V: for each j, sum_r V[j,r]*U[i,r]*W[k,r] = 1 for all i,k
            for j in range(m):
                B = np.zeros((n*p, 5))
                for i in range(n):
                    for k in range(p):
                        for r in range(5):
                            B[i*p+k, r] = U[i, r] * W[k, r]
                sol, residuals, rank_B, sv = np.linalg.lstsq(B, np.ones(n*p), rcond=None)
                V[j, :] = sol
            
            # Solve for W: for each k, sum_r W[k,r]*U[i,r]*V[j,r] = 1 for all i,j
            for k in range(p):
                B = np.zeros((n*m, 5))
                for i in range(n):
                    for j in range(m):
                        for r in range(5):
                            B[i*m+j, r] = U[i, r] * V[j, r]
                sol, residuals, rank_B, sv = np.linalg.lstsq(B, np.ones(n*m), rcond=None)
                W[k, :] = sol
        
        # Check error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i, r] * V[j, r] * W[k, r] for r in range(5))
                    err += (val - 1.0)**2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if err < 1e-10:
            break
    
    if best_err < 1e-9:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Try rank-20 approach: Kronecker product of rank-5 (2x3x2) and rank-4 (1x1x4)
    # The 1x1x4 tensor is just 4 independent scalar multiplications, rank 4
    
    result_20 = None
    
    # Step 1: Find rank-5 decomposition of 2x3x2
    U5, V5, W5 = _find_rank5_2x3x2()
    
    if U5 is not None:
        # Kronecker product with 1x1x4 identity
        # U for 2x3x8: shape (2*3, 20) = (6, 20)
        # V for 2x3x8: shape (3*8, 20) = (24, 20)  
        # W for 2x3x8: shape (8*2, 20) = (16, 20)
        
        # The Kronecker structure:
        # For 2x3x8 = (2x3x2) ⊗ (1x1x4):
        # U[i*m+j, r*4+s] = U5[i, r] * I1[s] where I1 is 1x1 identity (just 1)
        # Wait, the 1x1x4 tensor: A is 1x1, B is 1x4, C is 4x1
        # C[k,0] = A[0,0]*B[0,k] for k=0..3
        # This is just 4 independent products, rank 4
        # U1[0, s] = 1, V1[0, s] = 1, W1[s, 0] = 1 for s=0..3
        
        # So the Kronecker product:
        # U_full[i*3+j, r*4+s] = U5[i, r] * 1 = U5[i, r]
        # V_full[j*8+k, r*4+s] = V5[j, r] * 1 = V5[j, r]  (but B is 1x4, so B[0,k]=B[k])
        # Wait, I need to be more careful.
        
        # 2x3x2 tensor: A is 2x3, B is 3x2, C is 2x2
        # 1x1x4 tensor: A' is 1x1, B' is 1x4, C' is 4x1
        # Kronecker: A⊗A' is 2x3, B⊗B' is 3x8, C⊗C' is 8x2
        
        # Hmm, that doesn't quite work. Let me reconsider.
        
        # Actually, the decomposition is:
        # 2x3x8 matmul: C[i,k] = sum_j A[i,j]*B[j,k]
        # We can group k into pairs: k = 2*s + t where s=0..3, t=0..1
        # Then C[i, 2s+t] = sum_j A[i,j]*B[j, 2s+t]
        # This is like doing 4 independent 2x3x2 multiplications
        # Each uses the same A but different B columns
        
        # So the rank is 4 * rank(2x3x2) = 4*5 = 20
        
        # For the Kronecker structure:
        # U_full[i*3+j, r*4+s] = U5[i, r] * 1  (the 1x1 A' contributes 1)
        # V_full[j*8+(2s+t), r*4+s] = V5[j, r] * 1  (B'[0, 2s+t] = 1 for the s-th block)
        # Wait, B' is 1x4, so B'[0, s] for s=0..3. In the Kronecker, B⊗B' has B[j, 2s+t] = B[j,t]*B'[0,s]
        # But we want B to be arbitrary, so B'[0,s] should be 1 for all s.
        
        # Actually, the 1x1x4 tensor: A' is 1x1 with A'[0,0]=1, B' is 1x4, C' is 4x1
        # C'[s, 0] = A'[0,0]*B'[0, s] = B'[0, s]
        # The tensor T'[a, b, c] where a=0, b=0..3, c=0: T'[0, b, 0] = 1
        # Rank-4 decomposition: for each s, u'=[1], v'=[0,...,1,...,0] (1 at position s), w'=[1]
        
        # So the Kronecker product decomposition:
        # For each (r, s) where r=0..4, s=0..3:
        # u[i*3+j, r*4+s] = U5[i, r] * 1
        # v[j*8+(2s+t), r*4+s] = V5[j, r] * (1 if t==0 else 0)  -- wait, this isn't right
        
        # Let me think again. The tensor for 2x3x8 is:
        # T[i, j, k] = 1 for all i,j,k (in the flattened sense)
        # Where i∈{0,1}, j∈{0,1,2}, k∈{0,...,7}
        
        # We can write k = 2*s + t where s∈{0,1,2,3}, t∈{0,1}
        # T[i, j, 2s+t] = 1
        
        # The 2x3x2 tensor has T5[i, j, t] = 1 for i∈{0,1}, j∈{0,1,2}, t∈{0,1}
        # The 1x1x4 tensor has T4[s, b, c] where it's just identity-like
        
        # Actually, T[i,j,2s+t] = T5[i,j,t] * T4[s]
        # where T4[s] = 1 for all s
        
        # So the Kronecker product decomposition:
        # For r=0..4, s=0..3:
        # u[i*3+j, r*4+s] = U5[i, r]
        # v[j*8+(2s+t), r*4+s] = V5[j, r]  (for the t component)
        # w[(2s+t)*2+i, r*4+s] = W5[t, r]
        
        # Wait, I need to be careful with indices.
        # In the 2x3x8 problem:
        # U has shape (n*m, rank) = (6, 20), indexed by i*m+j = i*3+j
        # V has shape (m*p, rank) = (24, 20), indexed by j*p+k = j*8+k
        # W has shape (p*n, rank) = (16, 20), indexed by k*n+i = k*2+i
        
        # For the Kronecker product with 1x1x4:
        # The "4" comes from splitting k into s (which block of 2) and t (position within block)
        # k = 2*s + t
        
        # U5[i, r] for the 2x3x2 part (i∈{0,1}, r∈{0..4})
        # V5[j, r] for the 2x3x2 part (j∈{0,1,2}, r∈{0..4})
        # W5[t, r] for the 2x3x2 part (t∈{0,1}, r∈{0..4})
        
        # The 1x1x4 part:
        # U4[0, s] = 1 (A' is 1x1)
        # V4[0, s] = 1 (B' is 1x4, but we need V4[b, s] where b indexes B')
        # W4[s, 0] = 1 (C' is 4x1)
        
        # Kronecker product:
        # U_full[i*3+j, r*4+s] = U5[i, r] * U4[0, s] = U5[i, r]
        # V_full[j*8+(2s+t), r*4+s] = V5[j, r] * V4[t, s]  -- but V4 is 1x4, so V4[0, s]=1
        #   Actually, B is 3x8, B[j, k] where k=2s+t. In the Kronecker, B[j, 2s+t] = B5[j, t] * B4[0, s]
        #   So V_full[j*8+(2s+t), r*4+s] = V5[j, r] * 1 = V5[j, r]
        # W_full[(2s+t)*2+i, r*4+s] = W5[t, r] * W4[s, 0] = W5[t, r] * 1 = W5[t, r]
        
        # Let me construct this:
        U20 = np.zeros((6, 20))
        V20 = np.zeros((24, 20))
        W20 = np.zeros((16, 20))
        
        for r in range(5):
            for s in range(4):
                col = r * 4 + s
                for i in range(2):
                    for j in range(3):
                        U20[i*3+j, col] = U5[i, r]
                for j in range(3):
                    for t in range(2):
                        k = 2*s + t
                        V20[j*8+k, col] = V5[j, r]
                for t in range(2):
                    for i in range(2):
                        k = 2*s + t
                        W20[k*2+i, col] = W5[t, r]
        
        # Validate
        if _validate_decomposition(U20, V20, W20, 2, 3, 8, tol=1e-9):
            result_20 = {"U": U20.tolist(), "V": V20.tolist(), "W": W20.tolist()}
    
    # Fallback: schoolbook rank 48
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    if result_20 is not None:
        return result_20
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

