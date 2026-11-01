"""Matrix multiplication 2,2,8 - Strassen-based rank 28 decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 8]

def _verify_strassen_2x2():
    """Verify the 7-term Strassen decomposition for 2x2 matrices."""
    # Standard Strassen with matrices indexed [row, col]
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+b)(e+f)
    # M2 = (a-b)(e+f)
    # M3 = (c+d)g
    # M4 = (c+d)h
    # M5 = a(g+h)
    # M6 = c(e-g)
    # M7 = (d-b)(f+h)
    # Result: [[M1+M4+M5-M7, M2+M4], [M3+M5, M1-M2+M3+M6]]
    
    # Build U, V, W for the 2x2x2 tensor with convention:
    # U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    # Here n=m=p=2, so U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    # For term r, U[:, r] is a 4-vector (rows of A), V[:, r] is a 4-vector (cols of B), W[:, r] is a 4-vector (output)
    # The tensor identity: sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Let's define each term as (u_vec, v_vec, w_vec) where u_vec[i*2+j] = coefficient of A[i,j]
    # v_vec[j*2+k] = coefficient of B[j,k], w_vec[k*2+i] = coefficient of C[i,k]
    
    # M1 = (a+b)(e+f): u = [1,1,0,0], v = [1,1,1,1] (for e,f,g,h: B[0,0],B[0,1],B[1,0],B[1,1])
    #   w: contributes to C[0,0] and C[0,1]
    #   C[0,0] gets M1, C[0,1] gets M2+M4... 
    #   Actually let me just use the formula directly.
    
    # C[0,0] = M1 + M4 + M5 - M7
    # C[0,1] = M2 + M4
    # C[1,0] = M3 + M5
    # C[1,1] = M1 - M2 + M3 + M6
    
    # So w vectors:
    # M1: w[0*2+0]=1 (C[0,0]), w[1*2+1]=1 (C[1,1]) -> [1,0,0,1] in [k*2+i] order: k=0,i=0 -> idx 0; k=1,i=1 -> idx 3
    #   w order is [k*n+i] = [k*2+i]: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    #   C[0,0] is at index k*2+i where i=0,k=0 -> 0
    #   C[0,1] is at index k*2+i where i=0,k=1 -> 2
    #   C[1,0] is at index k*2+i where i=1,k=0 -> 1
    #   C[1,1] is at index k*2+i where i=1,k=1 -> 3
    
    # M1: C[0,0]+=1, C[1,1]+=1 -> w = [1, 0, 0, 1]
    # M2: C[0,1]+=1, C[1,1]+=-1 -> w = [0, 0, 1, -1]
    # M3: C[1,0]+=1, C[1,1]+=1 -> w = [0, 1, 0, 1]
    # M4: C[0,0]+=1, C[0,1]+=1 -> w = [1, 0, 1, 0]
    # M5: C[0,0]+=1, C[1,0]+=1 -> w = [1, 1, 0, 0]
    # M6: C[1,1]+=1 -> w = [0, 0, 0, 1]
    # M7: C[0,0]+=-1 -> w = [-1, 0, 0, 0]
    
    U2 = np.array([
        [1, 1, 0, 0, 0, 0, 0],  # a
        [1, 0, 0, 0, 0, 0, 0],  # b  -- wait, M2=(a-b)(e+f)
        [0, 0, 1, 1, 0, 0, 0],  # c
        [0, 0, 0, 0, 0, 0, 0],  # d -- need to check
    ])
    
    # Let me redo this carefully.
    # M1 = (a+b)(e+f): u=[1,1,0,0], v=[1,1,1,1]
    # M2 = (a-b)(e+f): u=[1,-1,0,0], v=[1,1,1,1]
    # M3 = (c+d)g:      u=[0,0,1,1], v=[0,0,1,0]
    # M4 = (c+d)h:      u=[0,0,1,1], v=[0,0,0,1]
    # M5 = a(g+h):      u=[1,0,0,0], v=[0,0,1,1]
    # M6 = c(e-g):      u=[0,0,1,0], v=[1,0,-1,0]
    # M7 = (d-b)(f+h):  u=[0,-1,0,1], v=[0,1,0,1]
    
    U2 = np.array([
        [1, 1, 0, 0, 1, 0, 0],   # a: M1,M2,M5
        [1,-1, 0, 0, 0, 0,-1],   # b: M1,M2,M7
        [0, 0, 1, 1, 0, 1, 0],   # c: M3,M4,M6
        [0, 0, 1, 1, 0, 0, 1],   # d: M3,M4,M7
    ])
    
    V2 = np.array([
        [1, 1, 0, 0, 0, 1, 0],   # e: M1,M2,M6
        [1, 1, 0, 0, 0, 0, 1],   # f: M1,M2,M7
        [1, 1, 1, 0, 1,-1, 0],   # g: M1,M2,M3,M5,M6
        [1, 1, 0, 1, 1, 0, 1],   # h: M1,M2,M4,M5,M7
    ])
    
    W2 = np.array([
        [ 1, 0, 0, 1, 1, 0,-1],  # C[0,0]: M1,M4,M5,M7
        [ 0, 0, 1, 0, 1, 0, 0],  # C[1,0]: M3,M5
        [ 0, 1, 0, 1, 0, 0, 0],  # C[0,1]: M2,M4
        [ 1,-1, 1, 0, 0, 1, 0],  # C[1,1]: M1,M2,M3,M6
    ])
    
    # Verify: for random A, B, check that sum_r U2[:,r] outer V2[:,r] outer W2[:,r] reproduces A@B
    # The tensor T[i,j,k] = sum_j A[i,j] B[j,k]
    # Decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r where (u_r ⊗ v_r ⊗ w_r)[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i]
    # Wait, the convention is U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # So the product for term r at position (i,j,k) is U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # And C[i,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This should equal sum_j A[i,j] * B[j,k]
    
    np.random.seed(42)
    for _ in range(100):
        A = np.random.randn(2, 2)
        B = np.random.randn(2, 2)
        C = A @ B
        # Compute via decomposition
        C_decomp = np.zeros((2, 2))
        for r in range(7):
            for i in range(2):
                for k in range(2):
                    val = 0.0
                    for j in range(2):
                        val += U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r]
                    C_decomp[i, k] = val
        if not np.allclose(C, C_decomp, atol=1e-10):
            return False, U2, V2, W2
    return True, U2, V2, W2

def construct():
    start_time = time.time()
    deadline = start_time + 120  # seconds
    
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # Try Strassen-based rank 28
    valid, U2, V2, W2 = _verify_strassen_2x2()
    
    if valid:
        # Expand to 2x2x8: B is 2x8, viewed as 4 blocks of 2x2
        # B[j, k] where j in {0,1}, k in {0,...,7}
        # Block b (0..3) covers columns 2*b .. 2*b+1
        # For each block b, we use the same U2, V2, W2 but with V and W shifted
        # V[j*2 + k] for the 2x2 case becomes V[j*2 + (2*b + k_local)] for the full case
        # W[k*2 + i] becomes W[(2*b + k_local)*2 + i] = W[4*b + 2*k_local + i]
        
        rank = 7 * 4  # 28
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for b in range(4):
            for r in range(7):
                col = b * 7 + r
                # U: same for all blocks (A doesn't change)
                U[:, col] = U2[:, r]
                # V: shift column index by 2*b
                for j in range(2):
                    for k_local in range(2):
                        V[j * 2 + (2 * b + k_local), col] = V2[j * 2 + k_local, r]
                # W: shift column index by 2*b, so k = 2*b + k_local
                for k_local in range(2):
                    for i in range(2):
                        W[(2 * b + k_local) * 2 + i, col] = W2[k_local * 2 + i, r]
        
        # Verify the full decomposition
        np.random.seed(123)
        ok = True
        for _ in range(50):
            A = np.random.randn(2, 2)
            B = np.random.randn(2, 8)
            C = A @ B
            C_decomp = np.zeros((2, 8))
            for r in range(28):
                for i in range(2):
                    for k in range(8):
                        val = 0.0
                        for j in range(2):
                            val += U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
                        C_decomp[i, k] = val
            if not np.allclose(C, C_decomp, atol=1e-10):
                ok = False
                break
        
        if ok:
            # Convert to lists of integers/half-integers
            U_list = [[int(x) if x == int(x) else x for x in row] for row in U.tolist()]
            V_list = [[int(x) if x == int(x) else x for x in row] for row in V.tolist()]
            W_list = [[int(x) if x == int(x) else x for x in row] for row in W.tolist()]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    if time.time() > deadline - 5:
        raise TimeoutError("Deadline exceeded")
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
