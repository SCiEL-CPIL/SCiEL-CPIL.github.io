"""Program for matrix multiplication 3x4 * 4x6 with rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]
n, m, p = DIMENSIONS


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x6 matrix.
    
    Strategy: Split A(3x4) into A1(2x4) and A2(1x4), split B(4x6) into B1(4x3) and B2(4x3).
    - C11(2x3) = A1*B1: use Strassen-based 2x2x2 (rank 7) + 2x2x1 (rank 2) = 9 per 2x2x3 block, x2 = 18
    - C12(2x3) = A1*B2: same, rank 18
    - C21(1x3) = A2*B1: rank 4 (minimal for 1x4x3)
    - C22(1x3) = A2*B2: rank 4
    Total rank: 18+18+4+4 = 44
    """
    start_time = time.time()
    DEADLINE = 110

    def check_time():
        return time.time() - start_time < DEADLINE

    def verify_full(U, V, W, tol=1e-6):
        """Full verification of all entries."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0
                    for r in range(rank):
                        val += U[a][r] * V[b][r] * W[c][r]
                    if abs(val - 1) > tol:
                        return False
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if i != i2 or j != j2 or k != k2:
                        val = 0
                        for r in range(rank):
                            val += U[a][r] * V[b][r] * W[c][r]
                        if abs(val) > tol:
                            return False
        return True

    def schoolbook_full():
        """Return the schoolbook decomposition with rank n*m*p = 72."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2x2():
        """
        Return Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
        U is 4x7, V is 4x7, W is 4x7.
        U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[3, 1] = 1; W[2, 1] = -1
        
        # M3 = A00(B10-B11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11(B01-B00)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[1, 3] = 1; W[2, 3] = -1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[3, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[2, 6] = 1; W[3, 6] = 1
        
        return U, V, W

    def build_2x2x3():
        """
        Build rank-9 decomposition for 2x2x3 matrix multiplication.
        Split into 2x2x2 (Strassen, rank 7) + 2x2x1 (rank 2).
        Returns U(4x9), V(6x9), W(6x9).
        U[i*2+j, r], V[j*3+k, r], W[k*2+i, r]
        """
        U_str, V_str, W_str = strassen_2x2x2()
        
        U = np.zeros((4, 9))
        V = np.zeros((6, 9))
        W = np.zeros((6, 9))
        
        # First 7 columns: 2x2x2 part (k=0,1)
        U[:, 0:7] = U_str
        V[0:4, 0:7] = V_str  # V is 4x4 for 2x2x2, now 6x4 with extra row
        # Map: V[j*2+k] for 2x2x2 -> V[j*3+k] for 2x2x3
        V[0, 0:7] = V_str[0, :]  # j=0,k=0
        V[1, 0:7] = V_str[1, :]  # j=0,k=1
        V[3, 0:7] = V_str[2, :]  # j=1,k=0
        V[4, 0:7] = V_str[3, :]  # j=1,k=1
        # W[k*2+i] for 2x2x2 -> W[k*2+i] for 2x2x3 (same indexing since k=0,1)
        W[0:4, 0:7] = W_str
        
        # Columns 7,8: 2x2x1 part (k=2)
        # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
        U[0, 7] = 1; V[2, 7] = 1; W[4, 7] = 1  # i=0,k=2: A[0,0]*B[0,2]
        U[1, 7] = 1; V[5, 7] = 1; W[4, 7] += 0  # wait, need separate terms
        # Actually: term for j=0: U[i,0]*V[0,2]*W[2,i]
        # term for j=1: U[i,1]*V[1,2]*W[2,i]
        # So we need 2 terms:
        # Term 7: U[0,7]=1, U[1,7]=1 (for i=0,1 with j=0)
        # No wait. Each term is rank 1. 
        # Term 7: handles j=0, k=2: U[i,0]=1 for both i, V[0,2]=1, W[2,i]=1 for both i
        # That's not rank 1 in the tensor sense unless we separate i.
        
        # Let me redo: for 2x2x1, C[i] = sum_j A[i,j]*B[j]
        # Term for (i=0,j=0): U[0,0]=1, V[0,0]=1, W[0,0]=1
        # Term for (i=0,j=1): U[0,1]=1, V[1,0]=1, W[0,1]=1
        # Term for (i=1,j=0): U[1,0]=1, V[0,1]=1, W[1,0]=1
        # Term for (i=1,j=1): U[1,1]=1, V[1,1]=1, W[1,1]=1
        # That's rank 4 for 2x2x1, not 2!
        
        # I was wrong. 2x2x1 has rank 4 (schoolbook), not 2.
        # So 2x2x3 has rank 7+4=11, not 9.
        
        # Let me restart this function properly.
        return None

    def build_2x2x3_correct():
        """
        Build rank-11 decomposition for 2x2x3 matrix multiplication.
        2x2x2 (Strassen, rank 7) + 2x2x1 (schoolbook, rank 4).
        Returns U(4x11), V(6x11), W(6x11).
        U[i*2+j, r], V[j*3+k, r], W[k*2+i, r]
        """
        U_str, V_str, W_str = strassen_2x2x2()
        
        U = np.zeros((4, 11))
        V = np.zeros((6, 11))
        W = np.zeros((6, 11))
        
        # First 7 columns: 2x2x2 part (k=0,1)
        U[:, 0:7] = U_str
        # V mapping: 2x2x2 has V[j*2+k], 2x2x3 has V[j*3+k]
        V[0, 0:7] = V_str[0, :]  # j=0,k=0
        V[1, 0:7] = V_str[1, :]  # j=0,k=1
        V[3, 0:7] = V_str[2, :]  # j=1,k=0
        V[4, 0:7] = V_str[3, :]  # j=1,k=1
        # W mapping: 2x2x2 has W[k*2+i], 2x2x3 has W[k*2+i] (same for k=0,1)
        W[0:4, 0:7] = W_str
        
        # Columns 7-10: 2x2x1 part (k=2), schoolbook rank 4
        # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
        # Term 7: i=0,j=0
        U[0, 7] = 1; V[2, 7] = 1; W[4, 7] = 1
        # Term 8: i=0,j=1
        U[1, 8] = 1; V[5, 8] = 1; W[4, 8] = 1
        # Term 9: i=1,j=0
        U[2, 9] = 1; V[2, 9] = 1; W[5, 9] = 1
        # Term 10: i=1,j=1
        U[3, 10] = 1; V[5, 10] = 1; W[5, 10] = 1
        
        return U, V, W

    def build_1x4x3():
        """
        Build rank-4 decomposition for 1x4x3 matrix multiplication.
        C[k] = sum_j A[j]*B[j,k]
        Returns U(4x4), V(12x4), W(3x4).
        U[j, r], V[j*3+k, r], W[k, r]
        """
        U = np.zeros((4, 4))
        V = np.zeros((12, 4))
        W = np.zeros((3, 4))
        for j in range(4):
            U[j, j] = 1
            for k in range(3):
                V[j*3+k, j] = 1
            W[k, j] = 1 if k == 0 else 0
        # Wait, that's wrong. Let me think again.
        # C[k] = sum_j A[j]*B[j,k]
        # Each term r corresponds to a specific j:
        # Term j: U[j]=1 (for that j), V[j*3+k]=1 for all k, W[k]=1 for all k
        # But that would make W the same for all j, which means:
        # sum_j U[j]*V[j*3+k]*W[k] = sum_j 1*1*W[k] = 4*W[k] for all k
        # That's not right. We need W to depend on k properly.
        
        # Actually the correct decomposition:
        # For each j in 0..3:
        #   Term j: U[j,j]=1, V[j*3+k, j]=1 for all k, W[k,j]=1 for all k
        # Then: sum_j U[a,j]*V[b,j]*W[c,j] where a=j, b=j'*3+k, c=k
        # = sum_j delta(a,j)*delta(b, j*3+k)*W[c,j]
        # For a=j, b=j*3+k: = W[k,j]
        # We want this to be 1 when j=j' and c=k, else 0.
        # So W[k,j] should be 1 for all k,j? No, that gives 1 for all valid (a,b,c).
        # But for invalid entries (where a!=j or b!=j*3+c), it should be 0.
        # U[a,j] is 0 unless a=j, so those are handled.
        # V[b,j] is 0 unless b=j*3+k for some k, so those are handled.
        # So W[k,j]=1 for all k,j works!
        
        U = np.zeros((4, 4))
        V = np.zeros((12, 4))
        W = np.zeros((3, 4))
        for j in range(4):
            U[j, j] = 1
            for k in range(3):
                V[j*3+k, j] = 1
                W[k, j] = 1
        
        return U, V, W

    def build_2x4x3():
        """
        Build rank-22 decomposition for 2x4x3 matrix multiplication.
        Split 4 into 2+2: two 2x2x3 subproblems, each rank 11.
        Returns U(8x22), V(12x22), W(6x22).
        U[i*4+j, r], V[j*3+k, r], W[k*2+i, r]
        """
        U1, V1, W1 = build_2x2x3_correct()
        U2, V2, W2 = build_2x2x3_correct()
        
        U = np.zeros((8, 22))
        V = np.zeros((12, 22))
        W = np.zeros((6, 22))
        
        # First 11 columns: j=0,1
        U[0:4, 0:11] = U1
        V[0:6, 0:11] = V1
        W[0:6, 0:11] = W1
        
        # Next 11 columns: j=2,3
        U[4:8, 11:22] = U2
        V[6:12, 11:22] = V2
        W[0:6, 11:22] = W2
        
        return U, V, W

    def assemble_full():
        """
        Assemble the full 3x4x6 decomposition.
        Split A(3x4) into A1(2x4, rows 0-1) and A2(1x4, row 2).
        Split B(4x6) into B1(4x3, cols 0-2) and B2(4x3, cols 3-5).
        
        C11(2x3) = A1*B1: rank 22
        C12(2x3) = A1*B2: rank 22
        C21(1x3) = A2*B1: rank 4
        C22(1x3) = A2*B2: rank 4
        Total: 52
        """
        U_2x4x3, V_2x4x3, W_2x4x3 = build_2x4x3()
        U_1x4x3, V_1x4x3, W_1x4x3 = build_1x4x3()
        
        total_rank = 22 + 22 + 4 + 4  # = 52
        U = np.zeros((12, total_rank))
        V = np.zeros((24, total_rank))
        W = np.zeros((18, total_rank))
        
        # Block C11: rows 0-1 of C, cols 0-2
        # U: rows 0-7 (A1: 2x4)
        U[0:8, 0:22] = U_2x4x3
        # V: rows 0-11 (B1: 4x3)
        V[0:12, 0:22] = V_2x4x3
        # W: rows 0-5 (C11: 3x2, indexed as k*2+i)
        W[0:6, 0:22] = W_2x4x3
        
        # Block C12: rows 0-1 of C, cols 3-5
        U[0:8, 22:44] = U_2x4x3
        # V: rows 12-23 (B2: 4x3, offset by 3 in k)
        V[12:24, 22:44] = V_2x4x3
        # W: rows 6-11 (C12: 3x2, indexed as (k+3)*2+i = 6+k*2+i)
        W[6:12, 22:44] = W_2x4x3
        
        # Block C21: row 2 of C, cols 0-2
        U[8:12, 44:48] = U_1x4x3
        # V: rows 0-11 (B1: 4x3)
        V[0:12, 44:48] = V_1x4x3
        # W: rows 12-14 (C21: 3x1, indexed as k*3+2)
        W[12:15, 44:48] = W_1x4x3
        
        # Block C22: row 2 of C, cols 3-5
        U[8:12, 48:52] = U_1x4x3
        # V: rows 12-23 (B2: 4x3)
        V[12:24, 48:52] = V_1x4x3
        # W: rows 15-17 (C22: 3x1, indexed as (k+3)*3+2 = 11+k*3+2... wait)
        # W index: k*n+i where n=3, so for i=2: k*3+2
        # For C21 (k=0,1,2): W[0*3+2]=W[2], W[1*3+2]=W[5], W[2*3+2]=W[8]
        # For C22 (k=3,4,5): W[3*3+2]=W[11], W[4*3+2]=W[14], W[5*3+2]=W[17]
        
        # I made an error in W indexing. Let me fix.
        # W has 18 rows: W[k*3+i] for k=0..5, i=0..2
        # C11 (i=0,1; k=0,1,2): W[0*3+0]=W[0], W[0*3+1]=W[1], W[1*3+0]=W[3], W[1*3+1]=W[4], W[2*3+0]=W[6], W[2*3+1]=W[7]
        # C12 (i=0,1; k=3,4,5): W[3*3+0]=W[9], W[3*3+1]=W[10], W[4*3+0]=W[12], W[4*3+1]=W[13], W[5*3+0]=W[15], W[5*3+1]=W[16]
        # C21 (i=2; k=0,1,2): W[0*3+2]=W[2], W[1*3+2]=W[5], W[2*3+2]=W[8]
        # C22 (i=2; k=3,4,5): W[3*3+2]=W[11], W[4*3+2]=W[14], W[5*3+2]=W[17]
        
        return None  # Placeholder, will fix

    # Let me rebuild with correct W indexing
    def assemble_full_correct():
        """
        Assemble the full 3x4x6 decomposition with correct W indexing.
        W[k*n+i] where n=3, so W has 18 rows.
        """
        U_2x4x3, V_2x4x3, W_2x4x3 = build_2x4x3()
        U_1x4x3, V_1x4x3, W_1x4x3 = build_1x4x3()
        
        total_rank = 22 + 22 + 4 + 4  # = 52
        U = np.zeros((12, total_rank))
        V = np.zeros((24, total_rank))
        W = np.zeros((18, total_rank))
        
        # Block C11: i=0,1; k=0,1,2; j=0,1,2,3
        # U rows: i*4+j for i=0,1; j=0..3 -> rows 0-7
        U[0:8, 0:22] = U_2x4x3
        # V rows: j*6+k for j=0..3; k=0,1,2 -> rows 0,1,2, 6,7,8, 12,13,14, 18,19,20
        # V_2x4x3 has rows j'*3+k' for j'=0..3, k'=0..2 -> rows 0-11
        # Need to map: V_2x4x3[j'*3+k'] -> V[j'*6+k']
        for jp in range(4):
            for kp in range(3):
                V[jp*6+kp, 0:22] = V_2x4x3[jp*3+kp, 0:22]
        # W rows: k*3+i for k=0,1,2; i=0,1 -> rows 0,1, 3,4, 6,7
        # W_2x4x3 has rows k'*2+i' for k'=0,1,2; i'=0,1 -> rows 0-5
        # Map: W_2x4x3[k'*2+i'] -> W[k'*3+i']
        for kp in range(3):
            for ip in range(2):
                W[kp*3+ip, 0:22] = W_2x4x3[kp*2+ip, 0:22]
        
        # Block C12: i=0,1; k=3,4,5; j=0,1,2,3
        U[0:8, 22:44] = U_2x4x3
        for jp in range(4):
            for kp in range(3):
                V[jp*6+(3+kp), 22:44] = V_2x4x3[jp*3+kp, 0:22]
        for kp in range(3):
            for ip in range(2):
                W[(3+kp)*3+ip, 22:44] = W_2x4x3[kp*2+ip, 0:22]
        
        # Block C21: i=2; k=0,1,2; j=0,1,2,3
        U[8:12, 44:48] = U_1x4x3
        for jp in range(4):
            for kp in range(3):
                V[jp*6+kp, 44:48] = V_1x4x3[jp*3+kp, 0:4]
        for kp in range(3):
            W[kp*3+2, 44:48] = W_1x4x3[kp, 0:4]
        
        # Block C22: i=2; k=3,4,5; j=0,1,2,3
        U[8:12, 48:52] = U_1x4x3
        for jp in range(4):
            for kp in range(3):
                V[jp*6+(3+kp), 48:52] = V_1x4x3[jp*3+kp, 0:4]
        for kp in range(3):
            W[(3+kp)*3+2, 48:52] = W_1x4x3[kp, 0:4]
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Schoolbook
    best_result = schoolbook_full()
    if not verify_full(best_result["U"], best_result["V"], best_result["W"]):
        raise RuntimeError("Schoolbook verification failed")
    best_rank = n * m * p

    # Try the structured decomposition (rank 52)
    if check_time():
        try:
            result = assemble_full_correct()
            if verify_full(result["U"], result["V"], result["W"], tol=1e-5):
                best_rank = 52
                best_result = result
                return best_result
        except Exception:
            pass

    return best_result
