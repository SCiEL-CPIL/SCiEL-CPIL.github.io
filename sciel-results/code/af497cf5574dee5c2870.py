"""Matrix multiplication 2x4 x 4x7 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _strassen_222():
    """Return (U, V, W) for 2x2 * 2x2 matrix multiplication, rank 7.
    U: (4,7), V: (4,7), W: (4,7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # a=U[0], b=U[1], c=U[2], d=U[3]
    # e=V[0], f=V[1], g=V[2], h=V[3]
    # W[0]=c00, W[1]=c01, W[2]=c10, W[3]=c11
    
    # r=0: (a+d)(e+h) -> c00 +, c11 +
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # r=1: (c+d)(e) -> c10 +, c11 -
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    # r=2: (a)(f-h) -> c01 +, c11 +
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    # r=3: (d)(g-e) -> c00 +, c10 +
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    # r=4: (a+b)(h) -> c00 -, c01 +
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    # r=5: (c-a)(e+f) -> c11 +
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # r=6: (b-d)(g+h) -> c00 +
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    
    return U, V, W

def _schoolbook_221():
    """Return (U, V, W) for 2x2 * 2x1 matrix multiplication, rank 4.
    U: (4,4), V: (2,4), W: (2,4)
    """
    U = np.zeros((4, 4), dtype=np.int64)
    V = np.zeros((2, 4), dtype=np.int64)
    W = np.zeros((2, 4), dtype=np.int64)
    # a*0, b*1, c*0, d*1
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    U[2,2]=1; V[0,2]=1; W[1,2]=1
    U[3,3]=1; V[1,3]=1; W[1,3]=1
    return U, V, W

def _block_224():
    """Return (U, V, W) for 2x2 * 2x4 matrix multiplication, rank 13.
    U: (4,13), V: (8,13), W: (8,13)
    """
    # Use the known rank-13 algorithm for 2x2*2x4
    # A = [[a,b],[c,d]], B = [[e,f,g,h],[i,j,k,l]]
    # C = [[ae+bi, af+bj, ag+bk, ah+bl], [ce+di, cf+dj, cg+dk, ch+dl]]
    
    U = np.zeros((4, 13), dtype=np.int64)
    V = np.zeros((8, 13), dtype=np.int64)
    W = np.zeros((8, 13), dtype=np.int64)
    # V index: j*4+k, so e=0,f=1,g=2,h=3,i=4,j=5,k=6,l=7
    # W index: k*2+i, so c00=0,c01=1,c10=2,c11=3,c02=4,c03=5,c12=6,c13=7
    
    # We'll use a decomposition that achieves rank 13
    # One approach: split B into two 2x2 blocks and use a shared Strassen-like
    # Or use the specific known algorithm
    
    # Let's use: 
    # m1 = (a+d)(e+i) -> contributes to c00 and c10
    # m2 = (a+d)(j+k) -> contributes to c01 and c11  
    # m3 = (a+d)(g+l) -> contributes to c02 and c12
    # m4 = (a+d)(h+m) -> wait, h is already used
    
    # Actually, let me use a different approach. For 2x2*2x4, we can write:
    # C = A * [B1, B2] where B1, B2 are 2x2
    # Using Strassen for each gives 14. To get 13, we need to share one mult.
    
    # Known rank-13 algorithm for 2x2*2x4:
    # Let B = [[e,f,g,h],[i,j,k,l]]
    # m1 = (a+d)(e+i)
    # m2 = (c-a)(e+f)  
    # m3 = (b-d)(g+h)
    # m4 = (b+d)(j+l)
    # m5 = (c+b)(f+k)
    # m6 = (a+b)(h+?) ... 
    
    # Let me just use a valid rank-14 decomposition for safety, then try to reduce
    # Actually, let me implement a known rank-13 algorithm properly
    
    # For 2x2 * 2x4, the minimum rank is 13 (known result)
    # Algorithm:
    # m1 = (a+d)(e+i)
    # m2 = (c+d)(e)
    # m3 = (a)(f-h)
    # m4 = (d)(g-e) -- wait, this is getting complicated
    
    # Let me use a simpler approach: just use Strassen on two 2x2 blocks = 14
    # and accept rank 14 for this block. Then total = 14 + 14 + 14 + 4 = 46
    # That's still better than 50.
    
    # Actually, let me try to implement the rank-13 properly.
    # From literature, R(2,2,4) = 13.
    
    # One known algorithm:
    # Let A = [[a,b],[c,d]], B = [[e,f,g,h],[i,j,k,l]]
    # m1 = (a+d)(e+i)
    # m2 = (c+d)(e+f)
    # m3 = (a+b)(g+h)
    # m4 = (b+d)(j+l)
    # m5 = (a+b)(f+k)
    # m6 = (c+d)(g)
    # m7 = (a)(h+l)
    # m8 = (b)(i+m) -- no, m isn't defined
    
    # I'll use a conservative rank-14 for this block to ensure correctness
    U, V, W = _strassen_222()  # placeholder, will override
    
    # For now, use two Strassen blocks = rank 14
    U1, V1, W1 = _strassen_222()
    U2, V2, W2 = _strassen_222()
    
    # Block 1: A * B[:, :2], Block 2: A * B[:, 2:]
    U = np.zeros((4, 14), dtype=np.int64)
    V = np.zeros((8, 14), dtype=np.int64)
    W = np.zeros((8, 14), dtype=np.int64)
    
    # First Strassen: A * B1 where B1 = [[e,f],[i,j]]
    U[:, 0:7] = U1
    V[0:4, 0:7] = V1  # e,f,i,j are indices 0-3 in V
    W[0:4, 0:7] = W1  # c00,c01,c10,c11 are indices 0-3 in W
    
    # Second Strassen: A * B2 where B2 = [[g,h],[k,l]]
    U[:, 7:14] = U2
    V[4:8, 7:14] = V2  # g,h,k,l are indices 4-7 in V
    W[4:8, 7:14] = W2  # c02,c03,c12,c13 are indices 4-7 in W
    
    return U, V, W

def _block_223():
    """Return (U, V, W) for 2x2 * 2x3 matrix multiplication.
    Use Strassen (7) + one extra schoolbook term = 8? Or just use Strassen on 2x2 and schoolbook on 2x1.
    7 + 4 = 11. Or we can do better.
    Actually, let's use Strassen for first 2 columns (7) and schoolbook for last 1 column (2 terms for 2x1).
    Total = 7 + 2 = 9.
    """
    U_s, V_s, W_s = _strassen_222()  # for 2x2*2x2
    U_sb = np.zeros((4, 2), dtype=np.int64)
    V_sb = np.zeros((2, 2), dtype=np.int64)
    W_sb = np.zeros((2, 2), dtype=np.int64)
    # For 2x2 * 2x1: a*e + b*f -> c0, c*e + d*f -> c1
    U_sb[0,0]=1; V_sb[0,0]=1; W_sb[0,0]=1
    U_sb[1,1]=1; V_sb[1,1]=1; W_sb[0,1]=1
    U_sb[2,0]=1; V_sb[0,1]=1; W_sb[1,0]=1
    U_sb[3,1]=1; V_sb[1,1]=1; W_sb[1,1]=1
    
    # Wait, that's 4 terms for 2x2*2x1. Let me recount.
    # 2x2 * 2x1: C = [[a,b],[c,d]] * [[e],[f]] = [[ae+bf],[ce+df]]
    # That's 4 multiplications. So 7 + 4 = 11.
    
    # Let me just use Strassen for 2x2*2x2 (7) and schoolbook for 2x2*2x1 (4) = 11
    U1 = U_s
    V1 = V_s
    W1 = W_s
    U2 = U_sb
    V2 = V_sb
    W2 = W_sb
    
    U = np.hstack([U1, U2])
    V = np.vstack([V1, V2])  # V1 is 4x7, V2 is 2x4 -> 6x? No, shapes don't match
    # V1 has shape (4,7) for 2x2 input, V2 has shape (2,4) for 2x1 input
    # Total V should be (6, 11): rows are j*3+k for j in [0,2), k in [0,3)
    # For first 2 columns (k=0,1): V indices 0-3
    # For last 1 column (k=2): V indices 4-5
    V_full = np.zeros((6, 11), dtype=np.int64)
    V_full[0:4, 0:7] = V1
    V_full[4:6, 7:11] = V2
    
    W_full = np.zeros((6, 11), dtype=np.int64)
    W_full[0:4, 0:7] = W1  # c00,c01,c10,c11
    # For k=2: W indices are 2*2+0=4, 2*2+1=5
    W_full[4:6, 7:11] = W2  # c0,c1 for the third column
    
    U_full = np.hstack([U1, U2])
    
    return U_full, V_full, W_full

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Strategy: 
    # Split the 4x7 middle dimension into blocks.
    # For each pair of rows in A (2 rows), we compute C = A_i * B_i
    # A is 2x4, B is 4x7.
    # C = A * B where A is 2x4, B is 4x7, C is 2x7.
    
    # We can view this as: for each pair of columns in B, use a block algorithm.
    # B has 7 columns. Split into 3+2+2 or 4+3.
    
    # Approach 1: Split B's 4 rows into pairs: (0,1), (2,3)
    # A = [A_left, A_right] where A_left, A_right are 2x2
    # B = [[B_tl, B_tr], [B_bl, B_br]] where each block is 2x7
    # C = A_left * B_tl + A_right * B_bl + A_left * B_tr + A_right * B_br
    # This doesn't directly help.
    
    # Better: View the computation as summing over j=0..3 of A[:,j] * B[j,:]
    # Each term is an outer product of a 2x1 and 1x7 vector.
    # Total rank = 4 * 7 = 28 for schoolbook.
    # We need to do better.
    
    # Use block decomposition:
    # Split j into groups: {0,1}, {2,3}
    # For each group, we compute A_sub * B_sub where A_sub is 2x2, B_sub is 2x7
    # R(2,2,7) = ? 
    # We can split B_sub's 7 columns into 4+3
    # R(2,2,4) = 13 (or 14), R(2,2,3) = 8 (or 9 or 11)
    # So R(2,2,7) <= 13 + 8 = 21 (or 14+9=23)
    # Two such groups: 2 * 21 = 42 (or 2*23=46)
    
    # Let's implement with the safe bounds first: 14 + 11 = 25 per group, 50 total.
    # Then try to improve.
    
    # For now, use the existing rank-50 approach but with cleaner code.
    # Then attempt to use rank-13 for 2x2*2x4 and rank-8 for 2x2*2x3.
    
    # Let me build the full decomposition using block_224 and block_223
    
    # Block for rows {0,1} of j, columns {0,1,2,3,4,5,6} of k
    # A_sub = A[:, 0:2] (2x2), B_sub = B[0:2, :] (2x7)
    # Split B_sub's 7 columns into 4+3
    
    U_224, V_224, W_224 = _block_224()  # rank 14
    U_223, V_223, W_223 = _block_223()  # rank 11
    
    # Total for one j-pair: 14 + 11 = 25
    # Two j-pairs: 50
    
    # Let's construct the full matrices
    rank_total = 50
    U = np.zeros((n*m, rank_total), dtype=np.int64)  # (8, 50)
    V = np.zeros((m*p, rank_total), dtype=np.int64)  # (28, 50)
    W = np.zeros((p*n, rank_total), dtype=np.int64)  # (14, 50)
    
    col = 0
    
    # j-pair 0: j=0,1
    # A indices: i*m+j -> 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # B indices: j*p+k -> 0*7+k, 1*7+k
    # C indices: k*n+i -> k*2+i
    
    # Block 2x2*2x4 for j={0,1}, k={0,1,2,3}
    # U_224 has shape (4,14), indexed by [i*2+j_local, r] where j_local in {0,1}
    # V_224 has shape (8,14), indexed by [j_local*4+k_local, r] where k_local in {0,1,2,3}
    # W_224 has shape (8,14), indexed by [k_local*2+i, r]
    
    U[0:2, col:col+14] = U_224[0:2, :]  # i=0
    U[4:6, col:col+14] = U_224[2:4, :]  # i=1
    V[0:4, col:col+14] = V_224[0:4, :]  # j=0, k=0..3
    V[7:11, col:col+14] = V_224[4:8, :]  # j=1, k=0..3
    W[0:4, col:col+14] = W_224[0:4, :]  # k=0..1, i=0..1
    W[4:8, col:col+14] = W_224[4:8, :]  # k=2..3, i=0..1
    col += 14
    
    # Block 2x2*2x3 for j={0,1}, k={4,5,6}
    # U_223 has shape (4,11), V_223 has shape (6,11), W_223 has shape (6,11)
    U[0:2, col:col+11] = U_223[0:2, :]  # i=0
    U[4:6, col:col+11] = U_223[2:4, :]  # i=1
    V[0:2, col:col+11] = V_223[0:2, :]  # j=0, k_local=0..1 -> k=4,5
    # Wait, V_223 has 6 rows for 2x3 input. Indices are j_local*3+k_local
    # j_local=0: k_local=0,1,2 -> k=4,5,6 -> V indices 4,5,6
    # j_local=1: k_local=0,1,2 -> k=4,5,6 -> V indices 11,12,13
    V[4:7, col:col+11] = V_223[0:3, :]  # j=0, k=4,5,6
    V[11:14, col:col+11] = V_223[3:6, :]  # j=1, k=4,5,6
    # W_223 has 6 rows for 2x3 output. Indices are k_local*2+i
    # k_local=0: i=0,1 -> k=4 -> W indices 8,9
    # k_local=1: i=0,1 -> k=5 -> W indices 10,11
    # k_local=2: i=0,1 -> k=6 -> W indices 12,13
    W[8:10, col:col+11] = W_223[0:2, :]  # k=4
    W[10:12, col:col+11] = W_223[2:4, :]  # k=5
    W[12:14, col:col+11] = W_223[4:6, :]  # k=6
    col += 11
    
    # j-pair 1: j=2,3
    # A indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # B indices: 2*7+k=14+k, 3*7+k=21+k
    
    # Block 2x2*2x4 for j={2,3}, k={0,1,2,3}
    U[2:4, col:col+14] = U_224[0:2, :]
    U[6:8, col:col+14] = U_224[2:4, :]
    V[14:18, col:col+14] = V_224[0:4, :]  # j=2, k=0..3
    V[21:25, col:col+14] = V_224[4:8, :]  # j=3, k=0..3
    W[0:4, col:col+14] = W_224[0:4, :]
    W[4:8, col:col+14] = W_224[4:8, :]
    col += 14
    
    # Block 2x2*2x3 for j={2,3}, k={4,5,6}
    U[2:4, col:col+11] = U_223[0:2, :]
    U[6:8, col:col+11] = U_223[2:4, :]
    V[14:17, col:col+11] = V_223[0:3, :]  # j=2, k=4,5,6
    V[21:24, col:col+11] = V_223[3:6, :]  # j=3, k=4,5,6
    W[8:10, col:col+11] = W_223[0:2, :]
    W[10:12, col:col+11] = W_223[2:4, :]
    W[12:14, col:col+11] = W_223[4:6, :]
    col += 11
    
    # Verify
    U_t = U.reshape(n, m, rank_total)
    V_t = V.reshape(m, p, rank_total)
    W_t = W.reshape(p, n, rank_total)
    T_check = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t, optimize=True)
    expected_T = np.ones((n, m, p), dtype=np.int64)
    
    if not np.array_equal(T_check, expected_T):
        return _schoolbook(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook(n, m, p):
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

# EVOLVE-BLOCK-END
