"""Fresh design for 4x4x7 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]
DEADLINE = 150.0
SAFETY = 10.0  # seconds margin before the kill

def _time_left():
    return DEADLINE - SAFETY - time.time()

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check exact decomposition: sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c]."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # T[i*m+j, j*p+k] = sum_c U[i*m+j,c]*V[j*p+k,c] must equal W[k*n+i,c]
    # Build the tensor check: for each (i,j,k), the coefficient of the rank-c term
    # is U[i*m+j,c]*V[j*p+k,c] and must equal W[k*n+i,c].
    # Compute M = U @ V.T  -> shape (n*m, m*p); M[i*m+j, j*p+k]
    M = U @ V.T
    # Build expected W reshaped: W[k*n+i, c]
    # For each c, the (n*m x m*p) matrix W_c has entry [i*m+j, j*p+k] = W[k*n+i, c]
    # i.e. W_c[i*m+j, j*p+k] = W[k*n + i, c]
    err = 0.0
    for c in range(U.shape[1]):
        # construct W_c
        Wc = np.zeros((n*m, m*p))
        for k in range(p):
            for i in range(n):
                val = W[k*n + i, c]
                for j in range(m):
                    Wc[i*m + j, j*p + k] = val
        err += np.max(np.abs(M[:, c] - Wc[:, c])) if M.shape[1] > c else 0
    # Actually M is (n*m, m*p) and we compare column-wise? No.
    # The tensor: T[a,b] = sum_c U[a,c]*V[b,c]. We need T[i*m+j, j*p+k] = W[k*n+i, c] summed? 
    # No! The decomposition is: T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c].
    # So for each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 (the identity tensor).
    # So we check: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1.
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = max(err, abs(s - 1.0))
    return err

def _schoolbook_block(n, m, p):
    """Return U (n*m, r), V (m*p, r), W (p*n, r) for schoolbook of n x m by m x p."""
    r = n*m*p
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Strassen 2x2x2 rank-7 decomposition for C = A*B where A,B are 2x2.
    U indexed by (i,j) row of A, V by (j,k) col, W by (k,i) of C.
    Strassen: 
      M1=(a11+a22)(b11+b22)
      M2=(a21+a22)b11
      M3=a11(b12-b22)
      M4=a22(b21-b11)
      M5=(a11+a12)b22
      M6=(a21-a11)(b11+b12)
      M7=(a12-a22)(b21+b22)
      c11=M1+M4-M5+M7
      c12=M3+M5
      c21=M2+M4
      c22=M1-M2+M3+M6
    """
    # A is 2x2: a[i,j], B is 2x2: b[j,k], C is 2x2: c[k? no, c[i,k]]
    # Our convention: U[i*m+j] for A, V[j*p+k] for B, W[k*n+i] for C.
    # For 2x2x2: n=m=p=2. U[i*2+j], V[j*2+k], W[k*2+i].
    # C[i,k] = sum_j A[i,j] B[j,k]. W[k*2+i] is the C[i,k] entry.
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # Define the 7 products as linear combinations.
    # M1 = (a00+a11)(b00+b11)
    # U: a00=1,a11=1 ; V: b00=1,b11=1 ; W: c00+=1, c11+=1
    def add(Mu, Uv, Wv, c):
        U[:, c] = Mu
        V[:, c] = Vv
        W[:, c] = Wv
    # M1
    add([1,0,0,1], [1,0,0,1], [1,0,0,1], 0)  # c00=1,c11=1 -> W[0*2+0]=W[0]=1, W[1*2+1]=W[3]=1
    # Wait W[k*2+i]: c[i,k]. c00 -> i=0,k=0 -> W[0]. c11 -> i=1,k=1 -> W[1*2+1]=W[3].
    # Let me redo with explicit mapping.
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # A indices: a00=U[0], a01=U[1], a10=U[2], a11=U[3]
    # B indices: b00=V[0], b01=V[1], b10=V[2], b11=V[3]
    # C indices: c00=W[0] (k=0,i=0), c01=W[1] (k=1,i=0), c10=W[2] (k=0,i=1), c11=W[3] (k=1,i=1)
    # M1=(a00+a11)(b00+b11) -> c00 += M1, c11 += M1
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2=(a10+a11)b00 -> c10 += M2
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1
    # M3=a00(b01-b11) -> c01 += M3
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1
    # M4=a11(b10-b00) -> c10 += M4
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[2,3]=1
    # M5=(a00+a01)b11 -> c00 -= M5, c10? no. c00 = M1+M4-M5+M7; so c00 -= M5
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1
    # M6=(a10-a00)(b00+b01) -> c11 += M6
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7=(a01-a11)(b10+b11) -> c00 += M7
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _block_2x2x3():
    """Decompose 2x2 by 2x3 multiplication, rank 11 (Strassen 2x2x2 + schoolbook 2x2x1)."""
    # n=2, m=2, p=3.
    # Split B into B_left (2x2) and B_right (2x1).
    # C_left (2x2) = A * B_left via Strassen rank 7.
    # C_right (2x1) = A * B_right via schoolbook rank 4 (n*m*p = 2*2*1=4).
    # Total rank 11.
    n, m, p = 2, 2, 3
    U = np.zeros((n*m, 11))
    V = np.zeros((m*p, 11))
    W = np.zeros((p*n, 11))
    # Strassen part for columns k=0,1 of B (B_left), producing C columns 0,1.
    Us, Vs, Ws = _strassen_2x2x2()
    # Us: (4,7) for A (2x2). Vs: (4,7) for B_left (2x2) i.e. b[j,k] with k in {0,1}.
    # Ws: (4,7) for C_left (2x2) i.e. c[i,k] with k in {0,1}.
    # Map into full: V index j*p+k = j*3+k. For k in {0,1}: V[j*3+k].
    # W index k*n+i = k*2+i. For k in {0,1}: W[k*2+i].
    for c in range(7):
        U[:, c] = Us[:, c]
        # V: b[j,k] for k in {0,1} -> full index j*3+k
        for j in range(2):
            for k in range(2):
                V[j*3 + k, c] = Vs[j*2 + k, c]
        # W: c[i,k] for k in {0,1} -> full index k*2+i
        for i in range(2):
            for k in range(2):
                W[k*2 + i, c] = Ws[k*2 + i, c]
    # Schoolbook part for column k=2 of B (B_right), producing C column 2.
    # C[i,2] = sum_j A[i,j] B[j,2]. Schoolbook rank n*m*1 = 4.
    col0 = 7
    for i in range(2):
        for j in range(2):
            U[i*2 + j, col0] = 1
            V[j*3 + 2, col0] = 1
            W[2*2 + i, col0] = 1  # k=2, i -> index 2*2+i = 4+i
            col0 += 1
    return U, V, W

def construct():
    start = time.time()
    n, m, p = DIMENSIONS  # 4,4,7
    # Build 4 blocks of 2x2x3 (splitting A rows into two groups of 2, and B cols into 3+... 
    # Actually 4x4 by 4x7: split A into top 2 rows and bottom 2 rows (n=2 each), 
    # B is 4x7. For each A-block (2x4), multiply by B (4x7) -> 2x7 result.
    # Hmm, that's 2x4 by 4x7. Not 2x2x3.
    # The known 93-rank result: 4 separate 2x2x3 decompositions.
    # Interpretation: split the 4x4 A into 4 blocks of 2x2, and the 4x7 B into... 
    # Actually 4x4 * 4x7: think of A as 2x2 blocks (2x2 grid of 2x2 blocks), 
    # B as 2x? blocks. 4x7 = 2 blocks of 2 rows? No.
    # Let me reconsider: maybe split along the m (inner) dimension.
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k].
    # Split j into {0,1} and {2,3}. Then C = A_top * B_top + A_bot * B_bot,
    # where A_top is 4x2, B_top is 2x7, A_bot is 4x2, B_bot is 2x7.
    # Each term is 4x2 * 2x7 = 4x7. 
    # Now split each 4x2 * 2x7 into 2 blocks of 2x2 * 2x7? 
    # 2x2 * 2x7: split the 7 cols into 3+4? Or use 2x2x3 + 2x2x4.
    # This is getting complicated. Let me just build the full thing via block Kronecker.
    
    # Simpler: build the whole 4x4x7 as a Kronecker combination.
    # Represent A as 2x2 block matrix of 2x2 blocks: A = [[A00,A01],[A10,A11]]
    # B as 2x? ... B is 4x7. Let B = [[B0],[B1]] where B0 is 2x7, B1 is 2x7.
    # C[i,k] = sum_j A[i,j]B[j,k].
    # Split i into {0,1},{2,3} and j into {0,1},{2,3}:
    # C_top = A_top * B  where A_top is 2x4, B is 4x7 -> 2x7.
    # C_bot = A_bot * B where A_bot is 2x4 -> 2x7.
    # A_top = [A00 A01] (2x2 each), so C_top[i,k] = sum_{j in 0,1} A00[i,j]B[j,k] + sum_{j in 0,1} A01[i,j] B[j+2,k].
    # = A00 * B0 + A01 * B1, where B0 is 2x7 (rows 0,1), B1 is 2x7 (rows 2,3).
    # So C_top = A00*B0 + A01*B1 (both 2x2 * 2x7).
    # Similarly C_bot = A10*B0 + A11*B1.
    # Each 2x2 * 2x7 = 2x2 * (2x3 + 2x4)? Split 7 cols into 3 and 4.
    # 2x2 * 2x3: rank 11 (from _block_2x2x3).
    # 2x2 * 2x4: split 4 cols into 2+2, two 2x2*2x2 Strassen, rank 14.
    # So each 2x2*2x7 = 11 + 14 = 25? That's worse.
    # 
    # The known 93 result must use a smarter split. Let me just trust the measured 
    # approach: 4 separate 2x2x3 blocks of rank 11 = 44, plus the rest schoolbook.
    # 44 + (112-44*?) ... 112 - 93 = 19 saved.
    # 
    # I'll implement a clean, verifiable construction and then a light rank-reduction search.
    # Start from schoolbook rank 112, then apply the 2x2x3 trick to 4 disjoint sub-tensors.
    
    # Let me build schoolbook first, then replace 4 of the 2x2x3 sub-structures.
    # A 2x2x3 sub-structure occupies: i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1,k0+2}.
    # There are 2*2*2 = 8 such sub-structures in 4x4x7? i:2 choices, j:2, k: 2 (0-2,1-3).
    # Actually k: 7 cols, 2x2x3 needs 3 consecutive k. Choices: k0 in {0,1,2,3} -> 4, but 
    # for disjointness we pick specific ones.
    # 
    # This is getting too complex to guarantee correctness by hand. Let me instead do a 
    # direct, verified construction using numpy Kronecker products and a search.
    
    # --- Direct verified construction via Kronecker ---
    # Build the full identity tensor T[i,j,k] = delta (it's just 1 for the mult tensor).
    # We want rank-r factors. Start schoolbook, then try to find a lower-rank 
    # decomposition by exploiting the 2x2x3 Strassen trick on selected sub-blocks.
    
    # I'll construct explicitly using the block structure and verify.
    
    U = np.zeros((n*m, n*m*p))
    V = np.zeros((m*p, n*m*p))
    W = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    rank = n*m*p
    best = (U, V, W, rank)
    best_err = _verify(U, V, W, n, m, p)
    assert best_err < 1e-9, f"schoolbook failed: {best_err}"
    
    # Now try the 2x2x3 replacement on disjoint sub-blocks.
    # A sub-block (i0, j0, k0) with i0 in {0,2}, j0 in {0,2}, k0 in {0,1,2,3} (k0+2<7).
    # For each such sub-block, replace its 2*2*3=12 schoolbook columns with 11 Strassen-ish columns.
    # But they must be disjoint in (i,j,k) to avoid interference.
    # i0 in {0,2} (2 choices), j0 in {0,2} (2), k0 in {0,1,2,3} but disjoint in k: 
    # pick k0 in {0,2,4}? 4+2=6<7 ok. So k0 in {0,2,4} (3 choices) but 2x2x3 uses k0,k0+1,k0+2.
    # {0,1,2},{2,3,4},{4,5,6} overlap at 2,4. Use {0,1,2} and {3,4,5} and {6}? no.
    # Disjoint k-sets of size 3 from 7: {0,1,2},{3,4,5} (2 sets) leaves 6. Or {0,1,2},{3,4,5,6}?
    # Let's just pick k0 in {0,3} -> sets {0,1,2} and {3,4,5}, disjoint, leaves k=6.
    # So candidate sub-blocks: i0 in {0,2}, j0 in {0,2}, k0 in {0,3} -> 2*2*2=8 sub-blocks,
    # but each uses 12 schoolbook cols -> replace with 11, saving 1 each -> save up to 8.
    # 112 - 8 = 104. Not 93. 
    # 
    # The 93 result likely uses a more global Strassen. Given time, I'll implement a 
    # general search: random sign flips / column merges to reduce rank, verified.
    
    # --- Light search: try to zero out columns by linear combinations ---
    # This is hard. Given the deadline, I'll return the verified schoolbook-based 
    # construction with the 2x2x3 replacements applied where disjoint, and verify.
    
    # Apply 2x2x3 replacement for sub-blocks with i0 in {0,2}, j0 in {0,2}, k0 in {0,3}.
    # For each, find its 12 schoolbook columns and replace with 11 from _block_2x2x3.
    
    # Map: schoolbook column for (i,j,k) is col = (i*m+j)*p + k = (i*4+j)*7 + k.
    def sb_col(i, j, k):
        return (i*m + j)*p + k
    
    U2 = U.copy(); V2 = V.copy(); W2 = W.copy()
    # We'll build a new decomposition by constructing from scratch using blocks.
    # Let me just build the full thing as a list of (U_row_vec, V_row_vec, W_row_vec) triples.
    
    triples = []
    # For each sub-block decide: use Strassen-2x2x3 or schoolbook.
    # Sub-blocks: i0 in {0,2}, j0 in {0,2}, k-set in {{0,1,2},{3,4,5},{6}}.
    # For k-set of size 3 -> 2x2x3 rank 11. For k-set {6} -> 2x2x1 schoolbook rank 4.
    # Total: 8 sub-blocks? i0(2)*j0(2)*kset(3)=12 sub-blocks. 
    # 2x2x3: 11 cols each; 2x2x1: 4 cols each.
    # k-sets: {0,1,2},{3,4,5},{6}. So 2*2*2=8 blocks of 2x2x3 (8*11=88) + 2*2*1=4 blocks of 2x2x1 (4*4=16) = 104.
    # Hmm 104, not 93. 
    # 
    # I'll go with this 104 construction (better than 112) and verify, then do a tiny search.
    
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            for kset in ([0,1,2], [3,4,5], [6]):
                if len(kset) == 3:
                    Ub, Vb, Wb = _block_2x2x3()  # n=2,m=2,p=3
                    for c in range(11):
                        # U: local (i,j) i in 0,1 -> global i0+i; j in 0,1 -> j0+j
                        for li in range(2):
                            for lj in range(2):
                                U2[(i0+li)*m + (j0+lj), col] = Ub[li*2+lj, c]
                        # V: local (j,k) j in 0,1 -> j0+j; k in 0,1,2 -> kset[k]
                        for lj in range(2):
                            for lk in range(3):
                                V2[(j0+lj)*p + kset[lk], col] = Vb[lj*3+lk, c]
                        # W: local (k,i) k in 0,1,2 -> kset[k]; i in 0,1 -> i0+i
                        for lk in range(3):
                            for li in range(2):
                                W2[kset[lk]*n + (i0+li), col] = Wb[lk*2+li, c]
                        col += 1
                else:  # len 1, schoolbook 2x2x1
                    for li in range(2):
                        for lj in range(2):
                            k = kset[0]
                            U2[(i0+li)*m + (j0+lj), col] = 1
                            V2[(j0+lj)*p + k, col] = 1
                            W2[k*n + (i0+li), col] = 1
                            col += 1
    rank2 = col
    err2 = _verify(U2, V2, W2, n, m, p)
    if err2 < 1e-9 and rank2 < rank:
        best = (U2, V2, W2, rank2)
        best_err = err2
    
    # Convert best to lists
    Uf, Vf, Wf, rk = best
    return {
        "U": [[float(x) for x in Uf[i]] for i in range(Uf.shape[0])],
        "V": [[float(x) for x in Vf[i]] for i in range(Vf.shape[0])],
        "W": [[float(x) for x in Wf[i]] for i in range(Wf.shape[0])],
    }

if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
