"""Matrix multiplication 3,5,6 rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _elapsed():
    return time.time() - _START

def _remaining(deadline):
    return deadline - time.time()

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W form an exact decomposition of the matrix multiplication tensor."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 if this corresponds to (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # Actually the standard convention: T is the bilinear map (A,B) -> AB
    # T[i,j,k] where i is row of A, j is col of A / row of B, k is col of B
    # Output (AB)[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor has indices (i, j, k) and T[i,j,k] = delta (it's 1 for each (i,j,k) triple)
    # The decomposition: (AB)[i,k] = sum_r (sum_j U[i,j,r]*A[i,j]) * (sum_j V[j,k,r]*B[j,k])
    # Wait, let me reconsider the convention.
    # 
    # Standard: C = A*B, C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: C[i,k] = sum_{i',j',k'} T[i',j',k'] A[i',j'] B[j',k']
    # So T[i',j',k'] = 1 if i'=i, k'=k, and the sum over j' gives the right answer.
    # Actually T is a 3-tensor with T[a,b,c] where a indexes A, b indexes B... 
    # 
    # Let me use the given convention: W[k*n+i, r]
    # U has shape (n*m, rank): indexed by (i,j) -> i*m+j
    # V has shape (m*p, rank): indexed by (j,k) -> j*p+k  
    # W has shape (p*n, rank): indexed by (k,i) -> k*n+i
    #
    # The decomposition: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # Wait no, that's not right either. Let me think more carefully.
    #
    # For matrix multiplication C = AB:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Tensor decomposition with rank r:
    # C[i,k] = sum_{r=1}^{R} (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k])
    #
    # For this to equal sum_j A[i,j] B[j,k] for ALL A, B:
    # We need: sum_r U[i,j,r] * V[j',k,r] = delta_{j,j'} for each (i,k)
    # i.e., for each (i,k): sum_r U[i,j,r] V[j',k,r] = delta_{j,j'}
    #
    # This is equivalent to: the sum over r of outer products U[:,j,r] * V[:,k,r]^T = I_m
    # for each (i,k) pair. But that's not quite right because U depends on i and V depends on k.
    #
    # More precisely, the condition is:
    # For all i, k: sum_r sum_{j,j'} U[i*m+j, r] * V[j'*p+k, r] * A[i,j] * B[j',k] = sum_j A[i,j] B[j,k]
    # This holds iff: for all i, k, j, j': sum_r U[i*m+j, r] * V[j'*p+k, r] = delta_{j,j'}
    #
    # So the verification is: for each (i,k) pair, the matrix M_{j,j'} = sum_r U[i*m+j, r] * V[j'*p+k, r]
    # must equal the identity matrix I_m.
    
    R = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(p):
            # M[j, j'] = sum_r U[i*m+j, r] * V[j'*p+k, r]
            M = np.zeros((m, m))
            for j in range(m):
                for jp in range(m):
                    M[j, jp] = np.dot(U[i*m+j, :], V[jp*p+k, :])
            if np.max(np.abs(M - np.eye(m))) > tol:
                return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-10):
    """Faster verification using matrix operations."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    
    # U reshaped: (n, m, R), V reshaped: (m, p, R)
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    
    # For each (i,k): sum_r U[i,:,r] * V[:,k,r]^T should be I_m
    # This is: sum_r U[i,:,r] outer V[:,k,r] = I_m
    # In matrix form: U[i] @ V[:,k].T ... wait
    # U[i] is (m, R), V[:,k] is (m, R)
    # sum_r U[i,j,r] * V[j',k,r] = (U[i] @ V[:,k].T)[j, j']
    
    for i in range(n):
        for k in range(p):
            M = U3[i] @ V3[:, k].T  # (m, R) @ (R, m) = (m, m)
            if np.max(np.abs(M - np.eye(m))) > tol:
                return False
    return True

def _schoolbook():
    n, m, p = 3, 5, 6
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _strassen_2x2x2():
    """Return the Strassen decomposition for 2x2x2 matrix multiplication (rank 7)."""
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    # Strassen's 7 products for 2x2 matrices:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    # U for A00: row 0 (i=0,j=0), for A11: row 3 (i=1,j=1)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1  # B00 (j=0,k=0), B11 (j=1,k=1)
    # W: C00 (k=0,i=0) gets +1, C11 (k=1,i=1) gets +1
    W[0, 0] = 1; W[3, 0] = 1  # k*n+i: (0,0)->0, (1,1)->3
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1  # A10 (i=1,j=0), A11 (i=1,j=1)
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10 (k=0,i=1) -> 0*2+1=1... wait
    # W index: k*n+i where n=2. C10 means i=1, k=0: k*n+i = 0*2+1 = 1
    W[1, 1] = 1
    
    # M3: A00(B10-B11)
    U[0, 2] = 1  # A00
    V[2, 2] = 1; V[3, 2] = -1  # B10 (j=1,k=0), B11 (j=1,k=1)
    # C01 (i=0,k=1): k*n+i = 1*2+0 = 2
    W[2, 2] = 1
    
    # M4: A11(B01-B00)
    U[3, 3] = 1  # A11
    V[1, 3] = 1; V[0, 3] = -1  # B01 (j=0,k=1), B00 (j=0,k=0)
    # C00 (i=0,k=0): 0*2+0=0, C11 (i=1,k=1): 1*2+1=3
    W[0, 3] = 1; W[3, 3] = 1
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1  # A00 (i=0,j=0), A01 (i=0,j=1)
    V[3, 4] = 1  # B11 (j=1,k=1)
    # C00 (i=0,k=0): 0, C01 (i=0,k=1): 2
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1  # A10 (i=1,j=0), A00 (i=0,j=0)
    V[0, 5] = 1; V[2, 5] = 1  # B00 (j=0,k=0), B10 (j=1,k=0)
    # C11 (i=1,k=1): 3
    W[3, 5] = 1
    
    # M7: (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1  # A01 (i=0,j=1), A11 (i=1,j=1)
    V[1, 6] = 1; V[3, 6] = 1  # B01 (j=0,k=1), B11 (j=1,k=1)
    # C00 (i=0,k=0): 0
    W[0, 6] = 1
    
    return U, V, W

def _block_strassen(n, m, p):
    """
    Use block decomposition with Strassen on 2x2 sub-blocks where possible.
    For 3x5*5x6, we can partition as:
    A (3x5) = [A1 (2x5); A2 (1x5)]
    B (5x6) = [B1 (5x4); B2 (5x2)]
    
    Or better: use 2x2 blocks where dimensions allow.
    
    Actually, let's think about this differently.
    n=3, m=5, p=6.
    
    One approach: partition A into rows [2,1] and B into cols [4,2]
    C = A*B = [A1; A2] * B = [A1*B; A2*B]
    This doesn't directly help with Strassen.
    
    Better: partition into 2x2 blocks where possible.
    A is 3x5, B is 5x6, C is 3x6.
    
    Let's try: A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x3, A21 is 1x2, A22 is 1x3
    B = [B11 B12; B21 B22] where B11 is 2x2, B12 is 2x4, B21 is 3x2, B22 is 3x4
    
    C11 (2x2) = A11*B11 + A12*B21
    C12 (2x4) = A11*B12 + A12*B22
    C21 (1x2) = A21*B11 + A22*B21
    C22 (1x4) = A21*B12 + A22*B22
    
    This is getting complicated. Let me try a simpler block approach.
    
    Alternative: Use the fact that we can decompose the multiplication by grouping.
    
    Actually, let me try a direct approach: use Strassen-like decomposition on the full problem.
    The key insight from the knowledge: "Block Strassen decomposition reduces rank from 90 to ~84"
    
    For 3x5x6, one natural decomposition:
    Split the middle dimension m=5 into 2+3.
    A (3x5) = [A_left (3x2), A_right (3x3)]
    B (5x6) = [B_top (2x6); B_bottom (3x6)]
    
    C = A_left * B_top + A_right * B_bottom
    
    A_left * B_top: 3x2 * 2x6 -> can use schoolbook rank 3*2*6=36
    A_right * B_bottom: 3x3 * 3x6 -> can use schoolbook rank 3*3*6=54
    Total: 90 (same as schoolbook, no gain)
    
    Or split differently. Let me try splitting n=3 into 2+1:
    A (3x5) = [A1 (2x5); A2 (1x5)]
    B (5x6) = B
    C1 = A1 * B: 2x5 * 5x6 -> schoolbook rank 60
    C2 = A2 * B: 1x5 * 5x6 -> schoolbook rank 30
    Total: 90
    
    Hmm. The Strassen gain comes from using rank-7 instead of rank-8 for 2x2x2.
    
    Let me think about when we can embed 2x2x2 Strassen blocks.
    
    For 3x5x6, consider partitioning:
    - n: 2+1
    - m: 2+3  
    - p: 2+4
    
    A = [A11 A12; A21 A22] with A11(2x2), A12(2x3), A21(1x2), A22(1x3)
    B = [B11 B12; B21 B22] with B11(2x2), B12(2x4), B21(3x2), B22(3x4)
    
    C11 (2x2) = A11*B11 + A12*B21
    C12 (2x4) = A11*B12 + A12*B22
    C21 (1x2) = A21*B11 + A22*B21
    C22 (1x4) = A21*B12 + A22*B22
    
    For C11 (2x2 result): A11*B11 is 2x2*2x2 (use Strassen, rank 7)
                         A12*B21 is 2x3*3x2 (schoolbook rank 12)
    Total for C11: 7 + 12 = 19
    
    For C12 (2x4): A11*B12 is 2x2*2x4 (schoolbook rank 16)
                   A12*B22 is 2x3*3x4 (schoolbook rank 24)
    Total for C12: 16 + 24 = 40
    
    For C21 (1x2): A21*B11 is 1x2*2x2 (schoolbook rank 4)
                   A22*B21 is 1x3*3x2 (schoolbook rank 6)
    Total for C21: 4 + 6 = 10
    
    For C22 (1x4): A21*B12 is 1x2*2x4 (schoolbook rank 8)
                   A22*B22 is 1x3*3x4 (schoolbook rank 12)
    Total for C22: 8 + 12 = 20
    
    Grand total: 19 + 40 + 10 + 20 = 89
    
    That's only 1 better than 90. Let me try different partitions.
    
    Try n: 2+1, m: 3+2, p: 4+2
    A = [A11 A12; A21 A22] with A11(2x3), A12(2x2), A21(1x3), A22(1x2)
    B = [B11 B12; B21 B22] with B11(3x4), B12(3x2), B21(2x4), B22(2x2)
    
    C11 (2x4) = A11*B11 + A12*B21: 2x3*3x4 (rank 24) + 2x2*2x4 (rank 16) = 40
    C12 (2x2) = A11*B12 + A12*B22: 2x3*3x2 (rank 12) + 2x2*2x2 (Strassen rank 7) = 19
    C21 (1x4) = A21*B11 + A22*B21: 1x3*3x4 (rank 12) + 1x2*2x4 (rank 8) = 20
    C22 (1x2) = A21*B12 + A22*B22: 1x3*3x2 (rank 6) + 1x2*2x2 (rank 4) = 10
    
    Total: 40 + 19 + 20 + 10 = 89
    
    Same. Let me try to get more 2x2x2 blocks.
    
    Try n: 2+1, m: 2+3, p: 2+4 (same as first, got 89)
    
    What if I use Strassen more aggressively? The issue is that most sub-blocks aren't square.
    
    Let me try a different strategy: directly construct a lower-rank decomposition using
    algebraic identities.
    
    Actually, let me just implement the block approach that gives 89 and verify it works.
    """
    n, m, p = 3, 5, 6
    
    # Partition: n=2+1, m=2+3, p=2+4
    n1, n2 = 2, 1
    m1, m2 = 2, 3
    p1, p2 = 2, 4
    
    # A = [A11 A12; A21 A22]
    # A11: (n1, m1) = (2,2), A12: (n1, m2) = (2,3)
    # A21: (n2, m1) = (1,2), A22: (n2, m2) = (1,3)
    #
    # B = [B11 B12; B21 B22]
    # B11: (m1, p1) = (2,2), B12: (m1, p2) = (2,4)
    # B21: (m2, p1) = (3,2), B22: (m2, p2) = (3,4)
    #
    # C = [C11 C12; C21 C22]
    # C11: (n1, p1) = (2,2) = A11*B11 + A12*B21
    # C12: (n1, p2) = (2,4) = A11*B12 + A12*B22
    # C21: (n2, p1) = (1,2) = A21*B11 + A22*B21
    # C22: (n2, p2) = (1,4) = A21*B12 + A22*B22
    
    # We'll build U, V, W incrementally
    # Each term contributes to U, V, W
    
    terms = []
    
    def add_term(n_rows, m_cols, p_cols, U_sub, V_sub, W_sub, 
                 n_offset, m_offset, p_offset):
        """Add a sub-problem to the global decomposition."""
        R = U_sub.shape[1]
        for r in range(R):
            u_vals = U_sub[:, r]
            v_vals = V_sub[:, r]
            w_vals = W_sub[:, r]
            
            u_row = np.zeros(n * m)
            v_row = np.zeros(m * p)
            w_row = np.zeros(p * n)
            
            for i in range(n_rows):
                for j in range(m_cols):
                    if u_vals[i * m_cols + j] != 0:
                        u_row[(n_offset + i) * m + (m_offset + j)] = u_vals[i * m_cols + j]
            
            for j in range(m_cols):
                for k in range(p_cols):
                    if v_vals[j * p_cols + k] != 0:
                        v_row[(m_offset + j) * p + (p_offset + k)] = v_vals[j * p_cols + k]
            
            for k in range(p_cols):
                for i in range(n_rows):
                    if w_vals[k * n_rows + i] != 0:
                        w_row[(p_offset + k) * n + (n_offset + i)] = w_vals[k * n_rows + i]
            
            terms.append((u_row, v_row, w_row))
    
    # C11 = A11*B11 + A12*B21
    # A11*B11: 2x2 * 2x2 -> Strassen (rank 7)
    U_s, V_s, W_s = _strassen_2x2x2()
    add_term(2, 2, 2, U_s, V_s, W_s, n_offset=0, m_offset=0, p_offset=0)
    
    # A12*B21: 2x3 * 3x2 -> schoolbook (rank 12)
    R = 2 * 3 * 2
    U_sub = np.zeros((2*3, R)); V_sub = np.zeros((3*2, R)); W_sub = np.zeros((2*2, R))
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U_sub[i*3+j, col] = 1
                V_sub[j*2+k, col] = 1
                W_sub[k*2+i, col] = 1
                col += 1
    add_term(2, 3, 2, U_sub, V_sub, W_sub, n_offset=0, m_offset=2, p_offset=0)
    
    # C12 = A11*B12 + A12*B22
    # A11*B12: 2x2 * 2x4 -> schoolbook (rank 16)
    R = 2 * 2 * 4
    U_sub = np.zeros((2*2, R)); V_sub = np.zeros((2*4, R)); W_sub = np.zeros((4*2, R))
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(4):
                U_sub[i*2+j, col] = 1
                V_sub[j*4+k, col] = 1
                W_sub[k*2+i, col] = 1
                col += 1
    add_term(2, 2, 4, U_sub, V_sub, W_sub, n_offset=0, m_offset=0, p_offset=2)
    
    # A12*B22: 2x3 * 3x4 -> schoolbook (rank 24)
    R = 2 * 3 * 4
    U_sub = np.zeros((2*3, R)); V_sub = np.zeros((3*4, R)); W_sub = np.zeros((4*2, R))
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(4):
                U_sub[i*3+j, col] = 1
                V_sub[j*4+k, col] = 1
                W_sub[k*2+i, col] = 1
                col += 1
    add_term(2, 3, 4, U_sub, V_sub, W_sub, n_offset=0, m_offset=2, p_offset=2)
    
    # C21 = A21*B11 + A22*B21
    # A21*B11: 1x2 * 2x2 -> schoolbook (rank 4)
    R = 1 * 2 * 2
    U_sub = np.zeros((1*2, R)); V_sub = np.zeros((2*2, R)); W_sub = np.zeros((2*1, R))
    col = 0
    for i in range(1):
        for j in range(2):
            for k in range(2):
                U_sub[i*2+j, col] = 1
                V_sub[j*2+k, col] = 1
                W_sub[k*1+i, col] = 1
                col += 1
    add_term(1, 2, 2, U_sub, V_sub, W_sub, n_offset=2, m_offset=0, p_offset=0)
    
    # A22*B21: 1x3 * 3x2 -> schoolbook (rank 6)
    R = 1 * 3 * 2
    U_sub = np.zeros((1*3, R)); V_sub = np.zeros((3*2, R)); W_sub = np.zeros((2*1, R))
    col = 0
    for i in range(1):
        for j in range(3):
            for k in range(2):
                U_sub[i*3+j, col] = 1
                V_sub[j*2+k, col] = 1
                W_sub[k*1+i, col] = 1
                col += 1
    add_term(1, 3, 2, U_sub, V_sub, W_sub, n_offset=2, m_offset=2, p_offset=0)
    
    # C22 = A21*B12 + A22*B22
    # A21*B12: 1x2 * 2x4 -> schoolbook (rank 8)
    R = 1 * 2 * 4
    U_sub = np.zeros((1*2, R)); V_sub = np.zeros((2*4, R)); W_sub = np.zeros((4*1, R))
    col = 0
    for i in range(1):
        for j in range(2):
            for k in range(4):
                U_sub[i*2+j, col] = 1
                V_sub[j*4+k, col] = 1
                W_sub[k*1+i, col] = 1
                col += 1
    add_term(1, 2, 4, U_sub, V_sub, W_sub, n_offset=2, m_offset=0, p_offset=2)
    
    # A22*B22: 1x3 * 3x4 -> schoolbook (rank 12)
    R = 1 * 3 * 4
    U_sub = np.zeros((1*3, R)); V_sub = np.zeros((3*4, R)); W_sub = np.zeros((4*1, R))
    col = 0
    for i in range(1):
        for j in range(3):
            for k in range(4):
                U_sub[i*3+j, col] = 1
                V_sub[j*4+k, col] = 1
                W_sub[k*1+i, col] = 1
                col += 1
    add_term(1, 3, 4, U_sub, V_sub, W_sub, n_offset=2, m_offset=2, p_offset=2)
    
    # Assemble
    total_rank = sum(len(t) for t in terms)
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    for u_row, v_row, w_row in terms:
        U[:, col] = u_row
        V[:, col] = v_row
        W[:, col] = w_row
        col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _snap_to_half_ints(mat, tol=1e-9):
    """Snap values to nearest half-integer if close enough."""
    mat = np.array(mat, dtype=float)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            # Check if close to a half-integer
            nearest = round(v * 2) / 2
            if abs(v - nearest) < tol:
                mat[i, j] = nearest
    return mat

def _to_list(x):
    if isinstance(x, dict):
        return {k: _to_list(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_list(v) for v in x]
    if isinstance(x, (int, float)):
        if isinstance(x, float) and x == int(x):
            return int(x)
        return x
    return x

def _random_search(n, m, p, target_rank, max_iter=200, seed=42):
    """Try to find a rank-target_rank decomposition using random initialization and ALS-like steps."""
    np.random.seed(seed)
    
    best_result = None
    best_err = float('inf')
    
    for attempt in range(5):  # 5 random restarts
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        # Normalize to avoid overflow/underflow
        for r in range(target_rank):
            norm = np.linalg.norm(U[:, r])
            if norm > 1e-10:
                U[:, r] /= norm
                V[:, r] *= norm
            norm = np.linalg.norm(V[:, r])
            if norm > 1e-10:
                V[:, r] /= norm
                W[:, r] *= norm
        
        # ALS-like optimization
        for it in range(max_iter):
            # Update U
            for i in range(n):
                for j in range(m):
                    # We need sum_r U[i*m+j, r] * V[j'*p+k, r] = delta_{j,j'} for each (i,k)
                    # This is complex. Let me use a different approach.
                    pass
            
            # Alternative: use gradient-based approach
            # The constraint is: for each (i,k): U3[i] @ V3[:,k].T = I_m
            # Let's minimize ||U3[i] @ V3[:,k].T - I_m||_F^2 over all (i,k)
            
            U3 = U.reshape(n, m, target_rank)
            V3 = V.reshape(m, p, target_rank)
            W3 = W.reshape(p, n, target_rank)
            
            # Gradient of loss w.r.t. U
            loss = 0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            
            for i in range(n):
                for k in range(p):
                    M = U3[i] @ V3[:, k].T - np.eye(m)
                    loss += np.sum(M * M)
                    # d/dU3[i] = M @ V3[:, k]
                    grad_U[i*m:(i+1)*m, :] += M @ V3[:, k]
                    # d/dV3[:,k] = M.T @ U3[i]
                    grad_V[:, k*p:(k+1)*p, :] += M.T @ U3[i]
            
            # Simple gradient descent with adaptive step
            lr = 0.01
            U -= lr * grad_U
            V -= lr * grad_V
            
            # Check convergence
            if loss < 1e-12:
                break
        
        # Verify
        U3 = U.reshape(n, m, target_rank)
        V3 = V.reshape(m, p, target_rank)
        max_err = 0
        for i in range(n):
            for k in range(p):
                M = U3[i] @ V3[:, k].T
                max_err = max(max_err, np.max(np.abs(M - np.eye(m))))
        
        if max_err < best_err:
            best_err = max_err
            best_result = (U.copy(), V.copy(), W.copy())
        
        if max_err < 1e-10:
            break
    
    if best_result is not None and best_err < 1e-10:
        U, V, W = best_result
        # Snap to half-integers
        U = _snap_to_half_ints(U)
        V = _snap_to_half_ints(V)
        W = _snap_to_half_ints(W)
        
        # Re-verify after snapping
        U3 = U.reshape(n, m, target_rank)
        V3 = V.reshape(m, p, target_rank)
        max_err = 0
        for i in range(n):
            for k in range(p):
                M = U3[i] @ V3[:, k].T
                max_err = max(max_err, np.max(np.abs(M - np.eye(m))))
        
        if max_err < 1e-10:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    global _START
    _START = time.time()
    deadline = _START + 140  # 10s safety margin
    
    n, m, p = 3, 5, 6
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try block Strassen
    if time.time() < deadline - 20:
        try:
            block_result = _block_strassen()
            if _verify(block_result["U"], block_result["V"], block_result["W"], n, m, p):
                best = block_result
                best_rank = len(block_result["U"][0])
        except Exception:
            pass
    
    # Try random search for lower ranks
    for target_rank in range(best_rank - 1, max(29, best_rank - 15), -1):
        if time.time() > deadline - 30:
            break
        try:
            result = _random_search(n, m, p, target_rank, max_iter=100, 
                                     seed=random.randint(0, 10000))
            if result is not None:
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = target_rank
                    break
        except Exception:
            pass
    
    return _to_list(best)

_START = time.time()

# EVOLVE-BLOCK-END
