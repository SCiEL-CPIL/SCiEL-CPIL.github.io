"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
_start_time = time.time()
TIME_LIMIT = 120.0

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    # Factor 1: Strassen 2x2x2 rank 7
    # Standard Strassen: 7 bilinear forms
    # U (8x7) for 2x2: row = i*2+j, col = r
    # Strassen u vectors: 
    #   M1=(a11+b22)(c11+c22): u=[1,0,0,1] wait let me use standard form
    # Standard Strassen for 2x2:
    #   m1=(a11+a22)(c11+c22)
    #   m2=(a21+a22)(c11)
    #   m3=(a11)(c21+c22)
    #   m4=(a22)(c12-c22)
    #   m5=(a11+a12)(c22)
    #   m6=(a21-a11)(c11+c12)
    #   m7=(a12-a22)(c21+c22)
    # U rows (i*2+j): 
    #   i=0,j=0 (a11): m1,m3,m5,m6,m7 -> [1,0,1,0,1,1,1]
    #   i=0,j=1 (a12): m5,m7 -> [0,0,0,0,1,0,1]
    #   i=1,j=0 (a21): m2,m6 -> [0,1,0,0,0,1,0]
    #   i=1,j=1 (a22): m1,m2,m4,m5,m7 -> [1,1,0,1,1,0,1]
    U_s = [
        [1, 0, 1, 0, 1, 1, 1],  # a11
        [0, 0, 0, 0, 1, 0, 1],  # a12
        [0, 1, 0, 0, 0, 1, 0],  # a21
        [1, 1, 0, 1, 1, 0, 1],  # a22
    ]
    # V rows (j*2+k):
    #   j=0,k=0 (c11): m1,m2,m6 -> [1,1,0,0,0,1,0]
    #   j=0,k=1 (c12): m4,m6,m7 -> [0,0,0,1,0,1,1]
    #   j=1,k=0 (c21): m3,m5,m7 -> [0,0,1,0,1,0,1]
    #   j=1,k=1 (c22): m1,m3,m4,m5 -> [1,0,1,1,1,0,0]
    V_s = [
        [1, 1, 0, 0, 0, 1, 0],  # c11
        [0, 0, 0, 1, 0, 1, 1],  # c12
        [0, 0, 1, 0, 1, 0, 1],  # c21
        [1, 0, 1, 1, 1, 0, 0],  # c22
    ]
    # W rows (k*2+i):
    #   k=0,i=0 (b11): m1-m2+m3+m7 -> [1,-1,1,0,0,0,1]
    #   k=0,i=1 (b12): m4+m5 -> [0,0,0,1,1,0,0]
    #   k=1,i=0 (b21): m2+m6 -> [0,1,0,0,0,1,0]
    #   k=1,i=1 (b22): m1-m5+m7 -> [1,0,0,0,-1,0,1]
    W_s = [
        [1, -1, 1, 0, 0, 0, 1],  # b11 (k=0,i=0)
        [0, 0, 0, 1, 1, 0, 0],  # b12 (k=0,i=1)
        [0, 1, 0, 0, 0, 1, 0],  # b21 (k=1,i=0)
        [1, 0, 0, 0, -1, 0, 1],  # b22 (k=1,i=1)
    ]
    
    # Verify Strassen: C[i,k] = sum_r U[i,j] * V[j,k] * W[k,i] for all i,j,k
    # Actually C[i,k] = sum_{r,j} U[i,j]*V[j,k]*W[r,k*n+i]... 
    # The tensor is: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # For 2x2x2: T[i,j,k] should be 1 if this is the (i,j,k) entry of the multiplication tensor
    # T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor)
    # C[i,k] = sum_j A[i,j]*B[j,k] = sum_j sum_r U[i,j]*V[j,k]*W[k*n+i,r] * ... 
    # Actually T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # For multiplication: T[i,j,k] should equal delta (it's 1 for each (i,j,k) pair)
    # Let's verify: sum_r U[i,j,r]*V[j,k,r]*W[k*n+i,r] should = 1 for all i,j,k
    
    # Let me verify numerically
    def verify_strassen(U_s, V_s, W_s):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for r in range(7):
                        s += U_s[i*2+j][r] * V_s[j*2+k][r] * W_s[k*2+i][r]
                    if s != 1:
                        return False, (i,j,k,s)
        return True, None
    
    ok, info = verify_strassen(U_s, V_s, W_s)
    if not ok:
        # Fallback to schoolbook
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
    
    # Now construct 2x4x6 using tensor product with 1x2x3 (rank 6)
    # The 1x2x3 part: multiply a 1x2 (i2) by 2x3 (j2,k2)
    # This is just: C[i2,k2] = sum_{j2} A[i2,j2]*B[j2,k2]
    # Rank 6 schoolbook for 1x2x3
    # But we can use a simpler decomposition since i2 is just 1 dimension
    # T2[i2, j2, k2] = 1 for all (since it's the multiplication tensor)
    # We need: sum_r2 U2[i2,j2,r2]*V2[j2,k2,r2]*W2[k2*1+i2,r2] = 1
    # Simple rank-6: for each (j2,k2), U2[i2,j2]=1, V2[j2,k2]=1, W2[k2,0]=1
    # That gives 2*3=6 terms
    
    # Actually, since i2 has size 1, we can be smarter.
    # T2[i2,j2,k2] = 1 for all i2 in {0}, j2 in {0,1}, k2 in {0,1,2}
    # Rank 6: r2 = j2*3+k2, U2[0,j2,r2]=1 if j2 matches, V2[j2,k2,r2]=1 if both match, W2[k2,0,r2]=1 if k2 matches
    # This is schoolbook rank 6.
    
    # Tensor product: rank = 7*6 = 42
    # But we can do better. Since i1 size is 2 and i2 size is 1, total i size is 2.
    # j1 size 2, j2 size 2, total j size 4.
    # k1 size 2, k2 size 3, total k size 6.
    # 
    # T[i1,i2, j1,j2, k1,k2] = T_strassen[i1,j1,k1] * T_1x2x3[i2,j2,k2]
    # = 1 * 1 = 1 for all valid indices
    # 
    # Decomposition: rank = 7*6 = 42
    # U[i1*1+i2, j1*2+j2, r1*6+r2] = U_s[i1*2+j1, r1] * U2[i2*2+j2, r2]
    # V[j1*2+j2, k1*3+k2, r1*6+r2] = V_s[j1*2+k1, r1] * V2[j2*3+k2, r2]  
    # W[k1*3+k2, i1*1+i2, r1*6+r2] = W_s[k1*2+i1, r1] * W2[k2*1+i2, r2]
    
    # For U2: 1x2, row = i2*2+j2, so i2=0, j2=0 or 1
    # U2[0*2+j2, r2] = U2[j2, r2]
    # V2: 2x3, row = j2*3+k2
    # W2: 3x1, row = k2*1+i2 = k2
    
    # Schoolbook for 1x2x3: rank 6
    # r2 = j2*3+k2 (6 values)
    # U2[j2, r2] = 1 if j2 matches the j2 in r2
    # V2[j2*3+k2, r2] = 1 if (j2,k2) matches
    # W2[k2, r2] = 1 if k2 matches
    
    # Actually, let me just build it directly.
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    for r1 in range(7):
        for r2 in range(6):
            r = r1 * 6 + r2
            j2 = r2 // 3
            k2 = r2 % 3
            for i1 in range(2):
                for j1 in range(2):
                    # i = i1 (since i2=0, n=2, i = i1*1+0 = i1)
                    # j = j1*2+j2
                    i = i1
                    j = j1 * 2 + j2
                    u_val = U_s[i * 2 + j]  # Wait, U_s is indexed by (i1, j1) as i1*2+j1
                    # U_s[i1*2+j1, r1]
                    u_val = U_s[i1 * 2 + j1][r1]
                    U[i * m + j][r] = u_val
            
            for j1 in range(2):
                for k1 in range(2):
                    # j = j1*2+j2, k = k1*3+k2
                    j = j1 * 2 + j2
                    k = k1 * 3 + k2
                    v_val = V_s[j1 * 2 + k1][r1]
                    V[j * p + k][r] = v_val
            
            for k1 in range(2):
                for i1 in range(2):
                    # k = k1*3+k2, i = i1
                    k = k1 * 3 + k2
                    i = i1
                    w_val = W_s[k1 * 2 + i1][r1]
                    W[k * n + i][r] = w_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
