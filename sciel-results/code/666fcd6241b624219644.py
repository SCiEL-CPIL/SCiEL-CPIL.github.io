import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 4, 6]

def _build_strassen_2x2_2x2():
    """
    Build the rank-7 Strassen decomposition for 2x2 @ 2x2.
    Returns U, V, W each of shape (4, 7) with the trace convention:
    U indexed by (i*2+j), V by (j*2+k), W by (k*2+i).
    Strassen:
      M1 = (A00+A11)(B00+B11)
      M2 = (A10+A11)B00
      M3 = A00(B10-B11)
      M4 = A11(B01-B10)
      M5 = (A00+A01)B11
      M6 = (A10-A00)(B00+B01)
      M7 = (A01-A11)(B10+B11)
      C00 = M1+M4-M5+M7
      C01 = M3+M5
      C10 = M2+M4
      C11 = M1-M2+M3+M6
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=2, C11=3 (W indexed by k*2+i: C00=0, C01=1, C10=2, C11=3)
    # Wait, W index is k*n+i. For 2x2x2, n=2, so W[k*2+i].
    # C[i,k] where i is row, k is col. W index = k*2+i.
    # C00: i=0,k=0 -> W[0]
    # C01: i=0,k=1 -> W[2]
    # C10: i=1,k=0 -> W[1]
    # C11: i=1,k=1 -> W[3]
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1  # A00+A11
    V[0,0]=1; V[3,0]=1  # B00+B11
    W[0,0]=1; W[3,0]=-1; W[1,0]=1; W[2,0]=1  # C00+=M1, C11-=M1, C10+=M1, C01+=M1? 
    # Let me redo. C00 = M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    # W[0]=C00 coeff, W[2]=C01 coeff, W[1]=C10 coeff, W[3]=C11 coeff
    
    # M1: C00+=1, C11+=1
    W[0,0]=1; W[3,0]=1
    # M2: C10+=1, C11-=1
    W[1,1]=1; W[3,1]=-1
    # M3: C01+=1, C11+=1
    W[2,2]=1; W[3,2]=1
    # M4: C00+=1, C10+=1
    W[0,3]=1; W[1,3]=1
    # M5: C00-=1, C01+=1
    W[0,4]=-1; W[2,4]=1
    # M6: C11+=1
    W[3,5]=1
    # M7: C00+=1
    W[0,6]=1
    
    # Now U and V for each M:
    # M1: (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1
    # M2: (A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1
    # M3: A00(B10-B11)
    U[0,2]=1; V[2,2]=1; V[3,2]=-1
    # M4: A11(B01-B10)
    U[3,3]=1; V[1,3]=1; V[2,3]=-1
    # M5: (A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1
    # M6: (A10-A00)(B00+B01)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1
    # M7: (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify the decomposition."""
    Ur = U.reshape(n, m, -1)
    Vr = V.reshape(m, p, -1)
    Wr = W.reshape(p, n, -1)
    rank = Ur.shape[2]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Ur[i,j,:], Vr[j,k,:]*Wr[k,i,:])
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    # Build schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Strategy: Use Strassen on 2x2 sub-blocks
    # Split m=4 into two 2s: m1={0,1}, m2={2,3}
    # For each (i,k) pair, the 2x2x2 Strassen gives 7 terms
    # But we can share across i and k using linearity
    # 
    # Key idea: The 3x4 @ 4x6 can be seen as:
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    # = sum_{j=0}^{1} A[i,j] B[j,k] + sum_{j=2}^{3} A[i,j] B[j,k]
    # 
    # For the first term (j=0,1), this is a 3x2 @ 2x6 product
    # For the second term (j=2,3), this is a 3x2 @ 2x6 product
    #
    # A 3x2 @ 2x6 product: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is a sum of 2 outer products for each (i,k), total 3*6*2 = 36 terms
    # But we can use Strassen-like: treat as 3 independent 1x2 @ 2x6
    # 1x2 @ 2x6: C[k] = a0*B[0,k] + a1*B[1,k] for k=0..5
    # This has rank 2 (just 2 multiplications per output, but outputs are linearly related)
    # Actually rank of 1x2@2x6 tensor is 2 (it's just 2 independent linear forms)
    # So 3 rows * 2 = 6 for first block, 3*2 = 6 for second block, total 12? No.
    #
    # Let me think differently. The tensor for 3x2@2x6:
    # T[i,j,k] where i in {0,1,2}, j in {0,1}, k in {0,1,2,3,4,5}
    # T[i,j,k] = 1 for all valid indices
    # Rank of this tensor: it's the sum over j of (e_i)(e_j)(e_k) = 
    # For each j, it's a rank-1 tensor in the (i,k) space tensored with e_j
    # So rank = 2 * (rank of 3x6 all-ones tensor) = 2 * 1 = 2? No.
    # The 3x6 all-ones matrix has rank 1, so the tensor has rank 2.
    # So 3x2@2x6 has rank 2? That seems too low.
    # Actually T[i,j,k] = 1 = sum_j delta_j * 1 * 1 where the "1" in i-space is the all-ones vector
    # and "1" in k-space is the all-ones vector.
    # So T = sum_{j=0}^{1} (u_j)(v_j)(w_j) where u_j is all-ones in R^3, v_j = e_j in R^2, w_j is all-ones in R^6
    # That's rank 2. But that can't be right for matrix multiplication.
    #
    # Wait, I'm confusing the tensor. The multiplication tensor for A(3x2) @ B(2x6) is:
    # M[i,j,k] = 1 for all i,j,k (it's the structural tensor, not the data)
    # The decomposition is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = M[i,j,k] = 1
    # This is the "multiplication tensor" which has rank equal to the number of scalar multiplications.
    #
    # For 3x2 @ 2x6, the schoolbook rank is 3*2*6 = 36.
    # Can we do better? The tensor is M[i,j,k] = 1 for i in {0,1,2}, j in {0,1}, k in {0..5}.
    # This is a rank-1 tensor in the combined (i,j,k) space? No, it's the all-ones tensor.
    # The all-ones tensor of size 3x2x6 has rank 1! Because it's u v^T w^T where u=ones(3), v=ones(2), w=ones(6).
    # But that's not how matrix multiplication works. The multiplication tensor is NOT the all-ones tensor.
    #
    # Let me re-read the problem. The tensor T[i,j,k] = 1 for all i,j,k means:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    # This is exactly the condition for the decomposition to represent matrix multiplication.
    #
    # So the tensor we're decomposing is the all-ones tensor of size (n*m) x (m*p) x (p*n) with the specific index mapping.
    # The rank of this tensor is what we're minimizing.
    #
    # For the schoolbook, rank = n*m*p = 72.
    #
    # Now, can we use Strassen? Strassen works by finding a rank-7 decomposition of the 2x2x2 multiplication tensor.
    # The 2x2x2 multiplication tensor has 8 entries (all 1s in the structural sense) and rank 7.
    #
    # For 3x4 @ 4x6, we can pad to 4x4 @ 4x8 or use block structure.
    # 
    # Approach: Split m=4 into 2+2. Then C = A[:,0:2]@B[0:2,:] + A[:,2:4]@B[2:4,:]
    # Each term is 3x2 @ 2x6.
    # 
    # For 3x2 @ 2x6: split the 3 rows into 1+2, and 6 cols into 2+2+2.
    # A_top (1x2) @ B_j (2x2) for j=0,1,2: each is 1x2@2x2
    # A_bot (2x2) @ B_j (2x2) for j=0,1,2: each is 2x2@2x2 (Strassen, rank 7)
    #
    # 1x2 @ 2x2: C[k] = a0*B[0,k] + a1*B[1,k] for k=0,1
    # Tensor: T[i,j,k] where i=0, j in {0,1}, k in {0,1}. T[0,j,k]=1.
    # Rank: this is a 1x2x2 all-ones tensor. It can be written as:
    # sum over (j,k) pairs: 4 terms. But can we do better?
    # T[0,0,0]=1, T[0,0,1]=1, T[0,1,0]=1, T[0,1,1]=1
    # = u v w^T where u=[1], v=ones(2), w=ones(2)? That gives T[0,j,k]=1 for all j,k. Yes! Rank 1.
    # Wait, but the index mapping: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # For 1x2@2x2: n=1, m=2, p=2. U has 2 rows, V has 4 rows, W has 2 rows.
    # U[0*2+j, r] = U[j, r] for j=0,1
    # V[j*2+k, r] for j=0,1; k=0,1
    # W[k*1+i, r] = W[k, r] for k=0,1 (i=0)
    # Condition: sum_r U[j,r] * V[j*2+k,r] * W[k,r] = 1 for all j,k
    # This is: for each (j,k), sum_r U[j,r] V[j*2+k,r] W[k,r] = 1
    # Let a_r = U[0,r], b_r = U[1,r], c_{j,k,r} = V[j*2+k,r], d_r = W[0,r], e_r = W[1,r]
    # sum_r a_r c_{0,0,r} d_r = 1
    # sum_r a_r c_{0,1,r} e_r = 1
    # sum_r b_r c_{1,0,r} d_r = 1
    # sum_r b_r c_{1,1,r} e_r = 1
    # With rank 1: a c_{0,0} d = 1, a c_{0,1} e = 1, b c_{1,0} d = 1, b c_{1,1} e = 1
    # Set a=b=d=e=1, then c_{0,0}=1, c_{0,1}=1, c_{1,0}=1, c_{1,1}=1. Works! Rank 1.
    #
    # So 1x2@2x2 has rank 1? That seems surprising but the math checks out.
    # Actually this makes sense: a 1x2 vector times a 2x2 matrix is just 2 linear combinations,
    # and the "multiplication" can be done with 1 scalar multiplication if we allow the
    # coefficients to be arbitrary (which they are in the tensor decomposition).
    #
    # Hmm, but that would mean the rank of 3x2@2x6 is very low. Let me reconsider.
    #
    # Actually, I think I'm overcomplicating. Let me just try the numerical optimization
    # with a good initial guess based on Strassen blocks.
    
    size_u = n * m  # 12
    size_v = m * p  # 24
    size_w = p * n  # 18
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Try ranks from low to high
    for r in [64, 60, 56, 52, 48, 44, 40, 36, 32, 28, 24]:
        if time.time() > deadline:
            break
        
        for trial in range(2):
            if time.time() > deadline:
                break
                
            np.random.seed(42 * r + trial)
            U_init = np.random.randn(size_u, r) * 0.5
            V_init = np.random.randn(size_v, r) * 0.5
            W_init = np.random.randn(size_w, r) * 0.5
            
            params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def residuals(params, r=r):
                U = params[0:size_u*r].reshape(size_u, r)
                V = params[size_u*r:size_u*r + size_v*r].reshape(size_v, r)
                W = params[size_u*r + size_v*r:].reshape(size_w, r)
                
                D = np.zeros((n, m, p))
                for c in range(r):
                    Uc = U[:, c].reshape(n, m)
                    Vc = V[:, c].reshape(m, p)
                    Wc = W[:, c].reshape(p, n)
                    D += Uc[:, :, None] * Vc[None, :, :] * Wc.T[:, None, :]
                
                return (D - 1.0).flatten()
            
            try:
                result = least_squares(residuals, params0, method='lm', max_nfev=2000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                
                if result.cost < 1e-8:
                    U_sol = result.x[0:size_u*r].reshape(size_u, r)
                    V_sol = result.x[size_u*r:size_u*r + size_v*r].reshape(size_v, r)
                    W_sol = result.x[size_u*r + size_v*r:].reshape(size_w, r)
                    
                    # Snap to nearest half-integer
                    U_snap = np.round(U_sol * 2) / 2
                    V_snap = np.round(V_sol * 2) / 2
                    W_snap = np.round(W_sol * 2) / 2
                    
                    if _verify(U_snap, V_snap, W_snap, n, m, p):
                        best_result = (U_snap, V_snap, W_snap)
                        best_rank = r
                        break
                    
            except Exception:
                pass
        
        if best_result is not None:
            break
    
    if best_result is not None:
        U_out, V_out, W_out = best_result
        return {
            "U": U_out.tolist(),
            "V": V_out.tolist(),
            "W": W_out.tolist()
        }
    
    # Fallback: return schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
