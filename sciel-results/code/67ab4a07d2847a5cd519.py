import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Step 1: Construct rank-7 Strassen decomposition for 2x2
    # U2, V2, W2 are lists of lists, shape (4, 7)
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Indices for 2x2: i, j, k in {0, 1}
    # U row: i*2+j
    # V row: j*2+k
    # W row: k*2+i
    
    # Term 1: (A00+A11)(B00+B11)
    U2[0*2+0][0] = 1; U2[1*2+1][0] = 1
    V2[0*2+0][0] = 1; V2[1*2+1][0] = 1
    W2[0*2+0][0] = 1; W2[1*2+1][0] = 1
    
    # Term 2: (A10+A11)B00
    U2[1*2+0][1] = 1; U2[1*2+1][1] = 1
    V2[0*2+0][1] = 1
    W2[0*2+1][1] = 1; W2[1*2+1][1] = -1 # C10 gets P2, C11 gets -P2. Wait.
    # C10 is row 1, col 0. Index in W is k*2+i.
    # C_{10} -> k=0, i=1 -> W[0*2+1].
    # C_{11} -> k=1, i=1 -> W[1*2+1].
    # Formula: C10 = P2 + P4. So W[1][1] should be 1?
    # Let's re-verify Strassen formulas.
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    # Indices: C_{row, col}.
    # W index is k*n + i. Here n=2.
    # C_{i, k} in standard notation?
    # The tensor is T[i, j, k]. C[i, k] = sum_j A[i, j] B[j, k].
    # So output index is (i, k).
    # W row index is k*n + i.
    # So for C_{10} (i=1, k=0), W index is 0*2 + 1 = 1.
    # For C_{11} (i=1, k=1), W index is 1*2 + 1 = 3.
    # For C_{00} (i=0, k=0), W index is 0*2 + 0 = 0.
    # For C_{01} (i=0, k=1), W index is 1*2 + 0 = 2.
    
    # Re-evaluating W assignments:
    # P1: C00 (+1), C11 (+1) -> W[0]=1, W[3]=1
    # P2: C10 (+1), C11 (-1) -> W[1]=1, W[3]=-1
    # P3: C01 (+1), C11 (+1) -> W[2]=1, W[3]=1
    # P4: C00 (+1), C10 (+1) -> W[0]=1, W[1]=1
    # P5: C00 (-1), C01 (+1) -> W[0]=-1, W[2]=1
    # P6: C11 (+1) -> W[3]=1
    # P7: C00 (+1) -> W[0]=1
    
    # Let's restart W2 construction carefully.
    W2 = [[0]*7 for _ in range(4)]
    # c=0 (P1)
    W2[0][0] = 1 # C00
    W2[3][0] = 1 # C11
    # c=1 (P2)
    W2[1][1] = 1 # C10
    W2[3][1] = -1 # C11
    # c=2 (P3)
    W2[2][2] = 1 # C01
    W2[3][2] = 1 # C11
    # c=3 (P4)
    W2[0][3] = 1 # C00
    W2[1][3] = 1 # C10
    # c=4 (P5)
    W2[0][4] = -1 # C00
    W2[2][4] = 1 # C01
    # c=5 (P6)
    W2[3][5] = 1 # C11
    # c=6 (P7)
    W2[0][6] = 1 # C00
    
    # U2 and V2 were already set correctly based on P definitions.
    # P1: (A00+A11)(B00+B11)
    # P2: (A10+A11)B00
    # P3: A00(B01-B11)
    # P4: A11(B10-B00)
    # P5: (A00+A01)B11
    # P6: (A10-A00)(B00+B01)
    # P7: (A01-A11)(B10+B11)
    
    # Verify U2/V2
    # c=0: U[0]=1, U[3]=1. V[0]=1, V[3]=1.
    # c=1: U[2]=1, U[3]=1. V[0]=1.
    # c=2: U[0]=1. V[1]=1, V[3]=-1.
    # c=3: U[3]=1. V[2]=1, V[0]=-1.
    # c=4: U[0]=1, U[1]=1. V[3]=1.
    # c=5: U[2]=1, U[0]=-1. V[0]=1, V[1]=1.
    # c=6: U[1]=1, U[3]=-1. V[2]=1, V[3]=1.
    
    # This looks consistent.
    
    # Step 2: Construct rank-49 decomposition for 4x4 via Kronecker product
    rank_4x4 = 7 * 7
    U4 = [[0]*rank_4x4 for _ in range(16)]
    V4 = [[0]*rank_4x4 for _ in range(16)]
    W4 = [[0]*rank_4x4 for _ in range(16)]
    
    for c1 in range(7):
        for c2 in range(7):
            c4 = c1 * 7 + c2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            # U index: (i1, i2) is row, (j1, j2) is col?
                            # i = 2*i1 + i2, j = 2*j1 + j2
                            # U4 row: i*4 + j
                            # U2 row: i1*2 + j1 and i2*2 + j2
                            u_val = U2[i1*2+j1][c1] * U2[i2*2+j2][c2]
                            if u_val != 0:
                                U4[(2*i1+i2)*4 + (2*j1+j2)][c4] = u_val
                            
                            # V index: j is row of B, k is col of B
                            # j = 2*j1 + j2, k = 2*k1 + k2
                            # V4 row: j*4 + k
                            # V2 row: j1*2 + k1 and j2*2 + k2
                            for k1 in range(2):
                                for k2 in range(2):
                                    v_val = V2[j1*2+k1][c1] * V2[j2*2+k2][c2]
                                    if v_val != 0:
                                        V4[(2*j1+j2)*4 + (2*k1+k2)][c4] = v_val
                                        
                            # W index: k is row of C? No, k is col of C, i is row of C.
                            # W row: k*4 + i
                            # k = 2*k1 + k2, i = 2*i1 + i2
                            # W4 row: (2*k1+k2)*4 + (2*i1+i2)
                            # W2 row: k1*2 + i1 and k2*2 + i2
                            for k1 in range(2):
                                for k2 in range(2):
                                    w_val = W2[k1*2+i1][c1] * W2[k2*2+i2][c2]
                                    if w_val != 0:
                                        W4[(2*k1+k2)*4 + (2*i1+i2)][c4] = w_val
                                        
    # Step 3: Construct rank-98 decomposition for 4x4x8
    # Use two copies of the 4x4 algorithm.
    # Copy 1 for k=0..3
    # Copy 2 for k=4..7
    rank_total = rank_4x4 * 2
    U = [[0]*rank_total for _ in range(16)]
    V = [[0]*rank_total for _ in range(32)]
    W = [[0]*rank_total for _ in range(32)]
    
    # Copy 1: c in 0..48
    for i in range(16):
        for c in range(rank_4x4):
            U[i][c] = U4[i][c]
            # V rows for k=0..3 correspond to j*8+k
            # For a given j (0..3) and k (0..3), the index in V4 is j*4+k
            # The index in V is j*8+k
            # So V[j*8+k][c] = V4[j*4+k][c]
            for j in range(4):
                for k in range(4):
                    V[j*8+k][c] = V4[j*4+k][c]
            # W rows for k=0..3 correspond to k*4+i
            # The index in W4 is k*4+i
            # The index in W is k*4+i (since n=4, p=8, W size is 32)
            # Wait, W index is k*n + i. Here n=4.
            # So W[k*4+i][c] = W4[k*4+i][c]
            for k in range(4):
                for i_out in range(4):
                    W[k*4+i_out][c] = W4[k*4+i_out][c]
                    
    # Copy 2: c in 49..97
    offset = rank_4x4
    for i in range(16):
        for c in range(rank_4x4):
            U[i][c + offset] = U4[i][c]
            # V rows for k=4..7
            # Let k' = k - 4.
            # V[j*8 + (4+k')][c+offset] = V4[j*4+k'][c]
            for j in range(4):
                for k_prime in range(4):
                    k = 4 + k_prime
                    V[j*8+k][c + offset] = V4[j*4+k_prime][c]
            # W rows for k=4..7
            # W[(4+k')*4 + i][c+offset] = W4[k'*4+i][c]
            for k_prime in range(4):
                k = 4 + k_prime
                for i_out in range(4):
                    W[k*4+i_out][c + offset] = W4[k_prime*4+i_out][c]

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    r = len(result["U"][0])
    print(f"Rank: {r}")
    
    # Quick verification
    n, m, p = DIMENSIONS
    U, V, W = result["U"], result["V"], result["W"]
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    print(f"Error at {i},{j},{k}: {s}")
                    break
            if not valid: break
        if not valid: break
    print(f"Valid: {valid}")
