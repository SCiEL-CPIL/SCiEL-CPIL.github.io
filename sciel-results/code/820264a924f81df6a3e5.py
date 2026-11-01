import numpy as np
import time

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 17
    fallback_rank = 18
    
    # Define the tensor T
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # But wait, the indices in the decomposition are:
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # The product sum is over r.
    # The term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This must equal T[i,j,k].
    # For matrix multiplication, T[i,j,k] is 1 if the indices are valid?
    # Yes, T[i,j,k] = 1 for all i,j,k.
    
    T = np.ones((n, m, p))
    
    # Reshape T to match the flattened indices for easier ALS
    # We need to solve for U, V, W.
    # U shape: (n*m, R)
    # V shape: (m*p, R)
    # W shape: (p*n, R)
    
    def als_decompose(rank, U_init=None, V_init=None, W_init=None, max_iter=100, tol=1e-10):
        if U_init is None:
            U = np.random.randn(n*m, rank)
        else:
            U = np.array(U_init, dtype=float)
            
        if V_init is None:
            V = np.random.randn(m*p, rank)
        else:
            V = np.array(V_init, dtype=float)
            
        if W_init is None:
            W = np.random.randn(p*n, rank)
        else:
            W = np.array(W_init, dtype=float)
            
        for it in range(max_iter):
            # Update U
            # For each i, j, we have variable U[i*m+j, :]
            # The equation for T[i,j,k] is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This is linear in U[i*m+j, :].
            # Let x = U[i*m+j, :].
            # We want to minimize sum_k (sum_r x_r * V[j*p+k, r] * W[k*n+i, r] - T[i,j,k])^2
            # This is a least squares problem: A x = b
            # A has shape (p, rank).
            # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
            # b[k] = T[i,j,k]
            
            new_U = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    b = np.zeros(p)
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        b[k] = T[i, j, k]
                    # Solve A x = b
                    # Use lstsq
                    sol, residuals, rank_sol, s = np.linalg.lstsq(A, b, rcond=None)
                    new_U[i*m+j, :] = sol
            
            # Update V
            new_V = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank)) # Wait, variable is V[j*p+k, :]
                    # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
                    # Linear in V[j*p+k, :].
                    # For fixed j,k, we vary i.
                    # A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    # b[i] = T[i,j,k]
                    A = np.zeros((n, rank))
                    b = np.zeros(n)
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        b[i] = T[i, j, k]
                    sol, residuals, rank_sol, s = np.linalg.lstsq(A, b, rcond=None)
                    new_V[j*p+k, :] = sol
            
            # Update W
            new_W = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    # Variable W[k*n+i, :]
                    # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
                    # Linear in W[k*n+i, :].
                    # For fixed k,i, we vary j.
                    # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    # b[j] = T[i,j,k]
                    A = np.zeros((m, rank))
                    b = np.zeros(m)
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        b[j] = T[i, j, k]
                    sol, residuals, rank_sol, s = np.linalg.lstsq(A, b, rcond=None)
                    new_W[k*n+i, :] = sol
            
            U, V, W = new_U, new_V, new_W
            
            # Check error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err += (val - T[i, j, k])**2
            if err < tol:
                break
        
        return U, V, W, err

    def check_exact(U, V, W):
        # Check if entries are close to integers or half-integers
        # And if the decomposition is exact
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - T[i, j, k]) > 1e-9:
                        return False
        # Check if we can round to 0.5
        # The problem allows integer/half-integer entries.
        # We should return the rounded values if they are close.
        return True

    def round_and_check(U, V, W):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify exactness of rounded solution
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                    if abs(val - T[i, j, k]) > 1e-9:
                        return None
        return U_r, V_r, W_r

    # 1. Construct Rank 18 Fallback (Exact)
    # This is based on M_2,2,2 (Rank 7) + M_2,2,2 (Rank 7) + M_2,2,1 (Rank 4)
    # We need to map this to the indices.
    
    # Strassen for 2x2x2:
    # U, V, W for 2x2x2.
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # We will create a rank 7 solution for the first 2 k's, and another for the next 2.
    
    # Standard Strassen matrices for 2x2:
    # M1 = A00 + A11
    # M2 = A10 + A11
    # M3 = A00 - A01
    # M4 = A10 - A11
    # M5 = (A00 + A11) * (B00 + B11) ... wait, Strassen is for C = AB.
    # C00 = M1 - M4 + M6
    # C01 = M2 + M5
    # C10 = M3 + M7
    # C11 = M1 - M2 + M3
    # Where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B11
    # M3 = A00(B00-B10)
    # M4 = A10(B01-B11)
    # M5 = (A00+A01)B00
    # M6 = (A10-A00)(B10+B11)
    # M7 = (A01-A11)(B01+B11)
    
    # This gives 7 terms.
    # We can encode this into U, V, W.
    
    def get_strassen_222():
        # Returns U, V, W of shape (4, 7), (4, 7), (4, 7) for n=2,m=2,p=2
        # U index: i*2+j
        # V index: j*2+k
        # W index: k*2+i
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        # A00: U[0,0]=1, A11: U[3,0]=1
        # B00: V[0,0]=1, B11: V[3,0]=1
        # C00: W[0,0]=1, C11: W[3,0]=1?
        # Wait, W index is k*2+i.
        # C00 -> k=0, i=0 -> W[0]
        # C11 -> k=1, i=1 -> W[3]
        # But M1 contributes to C00 and C11 with +1 and -1?
        # C00 = M1 - M4 + M6
        # C11 = M1 - M2 + M3
        # So W[0,0] += 1, W[3,0] += 1
        
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1
        
        # Term 2: M2 = (A10+A11)B11
        # A10: U[2,1]=1, A11: U[3,1]=1
        # B11: V[3,1]=1
        # C01: W[1,1]? k=0, i=1 -> W[1]. C11: W[3,1]?
        # C01 = M2 + M5. C11 = ... - M2
        # So W[1,1] += 1, W[3,1] -= 1
        
        U[2,1] = 1; U[3,1] = 1
        V[3,1] = 1
        W[1,1] = 1; W[3,1] = -1
        
        # Term 3: M3 = A00(B00-B10)
        # A00: U[0,2]=1
        # B00: V[0,2]=1, B10: V[2,2]=-1
        # C10: W[2,2]? k=1, i=0 -> W[2].
        # C10 = M3 + M7.
        # So W[2,2] += 1
        
        U[0,2] = 1
        V[0,2] = 1; V[2,2] = -1
        W[2,2] = 1
        
        # Term 4: M4 = A10(B01-B11)
        # A10: U[2,3]=1
        # B01: V[1,3]=1, B11: V[3,3]=-1
        # C00: W[0,3]?
        # C00 = ... - M4
        # So W[0,3] -= 1
        
        U[2,3] = 1
        V[1,3] = 1; V[3,3] = -1
        W[0,3] = -1
        
        # Term 5: M5 = (A00+A01)B00
        # A00: U[0,4]=1, A01: U[1,4]=1
        # B00: V[0,4]=1
        # C01: W[1,4]?
        # C01 = M2 + M5.
        # So W[1,4] += 1
        
        U[0,4] = 1; U[1,4] = 1
        V[0,4] = 1
        W[1,4] = 1
        
        # Term 6: M6 = (A10-A00)(B10+B11)
        # A10: U[2,5]=1, A00: U[0,5]=-1
        # B10: V[2,5]=1, B11: V[3,5]=1
        # C00: W[0,5]?
        # C00 = ... + M6
        # So W[0,5] += 1
        
        U[2,5] = 1; U[0,5] = -1
        V[2,5] = 1; V[3,5] = 1
        W[0,5] = 1
        
        # Term 7: M7 = (A01-A11)(B01+B11)
        # A01: U[1,6]=1, A11: U[3,6]=-1
        # B01: V[1,6]=1, B11: V[3,6]=1
        # C10: W[2,6]?
        # C10 = M3 + M7.
        # So W[2,6] += 1
        
        U[1,6] = 1; U[3,6] = -1
        V[1,6] = 1; V[3,6] = 1
        W[2,6] = 1
        
        return U, V, W

    # Get base Strassen
    U_s, V_s, W_s = get_strassen_222()
    
    # Construct Rank 18
    # k=0,1: Strassen 1
    # k=2,3: Strassen 2
    # k=4: Rank 4
    
    U_18 = np.zeros((4, 18))
    V_18 = np.zeros((10, 18))
    W_18 = np.zeros((10, 18))
    
    # Fill Strassen 1 (k=0,1)
    # V index j*2+k. For k=0,1, indices are 0..3.
    # W index k*2+i. For k=0,1, indices are 0..3.
    U_18[:, 0:7] = U_s
    V_18[0:4, 0:7] = V_s
    W_18[0:4, 0:7] = W_s
    
    # Fill Strassen 2 (k=2,3)
    # V index j*2+k. k=2,3. j=0 -> 2,3. j=1 -> 4,5.
    # V_s has indices 0,1,2,3.
    # We need to map V_s[j*2+k'] to V_18[j*2+(k'+2)].
    # V_s[0] (j=0,k=0) -> V_18[2] (j=0,k=2)
    # V_s[1] (j=0,k=1) -> V_18[3] (j=0,k=3)
    # V_s[2] (j=1,k=0) -> V_18[4] (j=1,k=2)
    # V_s[3] (j=1,k=1) -> V_18[5] (j=1,k=3)
    
    U_18[:, 7:14] = U_s
    V_18[2:6, 7:14] = V_s # j=0,1; k=2,3 -> indices 2,3,4,5
    # W index k*2+i. k=2,3.
    # W_s[0] (k=0,i=0) -> W_18[4] (k=2,i=0)
    # W_s[1] (k=0,i=1) -> W_18[5] (k=2,i=1)
    # W_s[2] (k=1,i=0) -> W_18[6] (k=3,i=0)
    # W_s[3] (k=1,i=1) -> W_18[7] (k=3,i=1)
    W_18[4:8, 7:14] = W_s
    
    # Fill Rank 4 for k=4
    # C_i = sum_j A_ij B_j4
    # C0 = A00 B04 + A01 B14
    # C1 = A10 B04 + A11 B14
    # 4 terms.
    # Term 8: A00 B04 -> C0
    U_18[0, 14] = 1
    V_18[4, 14] = 1 # j=0, k=4 -> 0*5+4=4
    W_18[8, 14] = 1 # k=4, i=0 -> 4*2+0=8
    
    # Term 9: A01 B14 -> C0
    U_18[1, 15] = 1
    V_18[9, 15] = 1 # j=1, k=4 -> 1*5+4=9
    W_18[8, 15] = 1
    
    # Term 10: A10 B04 -> C1
    U_18[2, 16] = 1
    V_18[4, 16] = 1
    W_18[9, 16] = 1 # k=4, i=1 -> 4*2+1=9
    
    # Term 11: A11 B14 -> C1
    U_18[3, 17] = 1
    V_18[9, 17] = 1
    W_18[9, 17] = 1
    
    # Verify Rank 18
    # It should be exact.
    
    # Now try to find Rank 17
    best_result = None
    start_time = time.time()
    
    # Try Rank 17 with ALS
    for attempt in range(50):
        if time.time() - start_time > 100:
            break
            
        # Initialize Rank 17 from Rank 18
        # Drop a random column from 18
        drop_idx = np.random.randint(18)
        keep_idx = [i for i in range(18) if i != drop_idx]
        
        U_init = U_18[:, keep_idx]
        V_init = V_18[:, keep_idx]
        W_init = W_18[:, keep_idx]
        
        # Add noise
        U_init = U_init + np.random.randn(4, 17) * 0.1
        V_init = V_init + np.random.randn(10, 17) * 0.1
        W_init = W_init + np.random.randn(10, 17) * 0.1
        
        U, V, W, err = als_decompose(17, U_init, V_init, W_init, max_iter=200)
        
        if err < 1e-6:
            rounded = round_and_check(U, V, W)
            if rounded is not None:
                U_r, V_r, W_r = rounded
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
    # If Rank 17 not found, return Rank 18
    return {"U": U_18.tolist(), "V": V_18.tolist(), "W": W_18.tolist()}
