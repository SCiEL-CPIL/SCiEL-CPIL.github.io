"""Tensor decomposition for 3x3 * 3x5 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, c] = 1
                V[j*p + k, c] = 1
                W[k*n + i, c] = 1
                c += 1
    return U, V, W

def _strassen_222():
    # Strassen for 2x2 * 2x2, rank 7
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U: (i,j) -> 2*2=4 rows
    # V: (j,k) -> 2*2=4 rows
    # W: (k,i) -> 2*2=4 rows (k*n+i, n=2)
    # Strassen formulas:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A11-A01)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A00+A11)*(B00+B11) -> C00, C11
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    W[0, 0] = 1  # C00 (k=0,i=0)
    W[3, 0] = 1  # C11 (k=1,i=1)
    
    # M2: (A10+A11)*B00 -> C10
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10 (k=1,i=0)
    
    # M3: A00*(B01-B11) -> C01, C11
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1 # B11
    W[1, 2] = 1  # C01 (k=0,i=1)
    W[3, 2] = 1  # C11 (k=1,i=1)
    
    # M4: A11*(B10-B00) -> C00, C10
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1 # B00
    W[0, 3] = 1  # C00 (k=0,i=0)
    W[2, 3] = 1  # C10 (k=1,i=0)
    
    # M5: (A00+A10)*B11 -> C00, C01
    U[0, 4] = 1  # A00
    U[2, 4] = 1  # A10
    V[3, 4] = 1  # B11
    W[0, 4] = -1 # C00 (k=0,i=0)
    W[1, 4] = 1  # C01 (k=0,i=1)
    
    # M6: (A11-A01)*(B00+B01) -> C11
    U[3, 5] = 1  # A11
    U[1, 5] = -1 # A01
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    W[3, 5] = 1  # C11 (k=1,i=1)
    
    # M7: (A01-A11)*(B10+B11) -> C00
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    W[0, 6] = 1  # C00 (k=0,i=0)
    
    return U, V, W

def _construct_43():
    """
    Block decomposition:
    A is 3x3, B is 3x5, C is 3x5.
    
    Split A into blocks:
    A = [[A00, A01],
         [A10, A11]]
    where A00 is 2x2, A01 is 2x1, A10 is 1x2, A11 is 1x1.
    
    Split B into blocks:
    B = [[B00],
         [B10],
         [B20]]
    where B00 is 2x5 (rows 0,1), B10 is 1x5 (row 2).
    
    Wait, the split for B should align with the column split of A.
    A columns: 0,1 (J1), 2 (J2).
    B rows: 0,1 (J1), 2 (J2).
    So B = [[B0], [B1]] where B0 is 2x5, B1 is 1x5.
    
    C = A*B = [[A00, A01], [A10, A11]] * [[B0], [B1]]
    C = [[A00*B0 + A01*B1],
         [A10*B0 + A11*B1]]
    
    C0 = A00*B0 + A01*B1  (2x5)
    C1 = A10*B0 + A11*B1  (1x5)
    
    Decomposition of C0:
    A00 is 2x2, B0 is 2x5.
    We can split B0 into B0a (2x2) and B0b (2x3).
    A00*B0 = A00*B0a + A00*B0b.
    A00*B0a: 2x2 * 2x2 -> use Strassen (rank 7).
    A00*B0b: 2x2 * 2x3 -> use rank 11 (Strassen for 2x2x2 on first 2 cols? No, 2x2x3).
    
    Wait, R(2,2,3) = 11.
    Let's check:
    C = A(2x2) * B(2x3).
    We can view B as [B1(2x2), B2(2x1)].
    C = A*B1 + A*B2.
    A*B1: rank 7 (Strassen).
    A*B2: 2x2 * 2x1. Rank 4.
    Total 11.
    So A00*B0a (2x2x2) rank 7.
    A00*B0b (2x2x3) rank 11.
    A01*B1: 2x1 * 1x5. Rank 10.
    Total for C0: 7 + 11 + 10 = 28.
    
    Decomposition of C1:
    A10 is 1x2, B0 is 2x5. A10*B0: rank 10.
    A11 is 1x1, B1 is 1x5. A11*B1: rank 5.
    Total for C1: 10 + 5 = 15.
    
    Total rank: 28 + 15 = 43.
    """
    n, m, p = 3, 3, 5
    rank = 43
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # 1. A00*B0a: 2x2 * 2x2 (A rows 0,1; cols 0,1; B rows 0,1; cols 0,1)
    U_sa, V_sa, W_sa = _strassen_222()
    # Map to global indices
    # A: i in {0,1}, j in {0,1} -> i*m+j
    # B: j in {0,1}, k in {0,1} -> j*p+k
    # C: k in {0,1}, i in {0,1} -> k*n+i
    for r in range(4):
        for c in range(7):
            # U index: r (0..3) corresponds to (i,j)
            # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            i = r // 2
            j = r % 2
            U[i*m + j, col + c] += U_sa[r, c]
            
            # V index: r (0..3) corresponds to (j,k)
            # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            j = r // 2
            k = r % 2
            V[j*p + k, col + c] += V_sa[r, c]
            
            # W index: r (0..3) corresponds to (k,i)
            # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            k = r // 2
            i = r % 2
            W[k*n + i, col + c] += W_sa[r, c]
    col += 7
    
    # 2. A00*B0b: 2x2 * 2x3 (A rows 0,1; cols 0,1; B rows 0,1; cols 2,3,4)
    # Use decomposition of R(2,2,3) = 11
    # Split B0b into B0b1 (2x2, cols 2,3) and B0b2 (2x1, col 4)
    # A00*B0b1: Strassen (rank 7)
    # A00*B0b2: Schoolbook 2x2x1 (rank 4)
    
    # 2a. A00*B0b1 (cols 2,3)
    U_sa, V_sa, W_sa = _strassen_222()
    for r in range(4):
        for c in range(7):
            i = r // 2
            j = r % 2
            U[i*m + j, col + c] += U_sa[r, c]
            
            j = r // 2
            k = r % 2
            V[j*p + (2 + k), col + c] += V_sa[r, c] # B cols 2,3
            
            k = r // 2
            i = r % 2
            W[k*n + i, col + c] += W_sa[r, c]
    col += 7
    
    # 2b. A00*B0b2 (col 4)
    # Schoolbook 2x2x1
    for i in range(2):
        for j in range(2):
            k = 0
            U[i*m + j, col] = 1
            V[j*p + 4, col] = 1
            W[k*n + i, col] = 1
            col += 1
    col += 0 # already incremented inside loop? No, col is shared.
    # Wait, col increment:
    # for i,j: 4 times.
    # Let's redo col management carefully.
    
    # Actually, let's just build the matrices directly with a helper.
    
def construct():
    n, m, p = 3, 3, 5
    rank = 43
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        U_sa, V_sa, W_sa = _strassen_222()
        for r in range(4):
            for c in range(7):
                i = r // 2 + i_off
                j = r % 2 + j_off
                U[i*m + j, col + c] += U_sa[r, c]
                
                j_v = r // 2 + j_off
                k_v = r % 2 + k_off
                V[j_v*p + k_v, col + c] += V_sa[r, c]
                
                k_w = r // 2 + k_off
                i_w = r % 2 + i_off
                W[k_w*n + i_w, col + c] += W_sa[r, c]
        col += 7
        
    def add_schoolbook(i_off, j_off, k_off, ni, nj, nk):
        nonlocal col
        for ii in range(ni):
            for jj in range(nj):
                for kk in range(nk):
                    i = i_off + ii
                    j = j_off + jj
                    k = k_off + kk
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1

    # 1. A00*B0a (i:0-1, j:0-1, k:0-1)
    add_strassen(0, 0, 0)
    
    # 2a. A00*B0b1 (i:0-1, j:0-1, k:2-3)
    add_strassen(0, 0, 2)
    
    # 2b. A00*B0b2 (i:0-1, j:0-1, k:4)
    add_schoolbook(0, 0, 4, 2, 2, 1)
    
    # 3. A01*B1 (i:0-1, j:2, k:0-4)
    # A01 is 2x1 (cols 2 of A, rows 0-1). B1 is 1x5 (row 2 of B).
    # Wait, A01 is A[0:2, 2:3]. B1 is B[2:3, :].
    # This is (2x1) * (1x5).
    # Rank 10.
    add_schoolbook(0, 2, 2, 2, 1, 5)
    
    # 4. A10*B0 (i:2, j:0-1, k:0-4)
    # A10 is A[2:3, 0:2]. B0 is B[0:2, :].
    # (1x2) * (2x5). Rank 10.
    add_schoolbook(2, 0, 0, 1, 2, 5)
    
    # 5. A11*B1 (i:2, j:2, k:0-4)
    # A11 is A[2:3, 2:3]. B1 is B[2:3, :].
    # (1x1) * (1x5). Rank 5.
    add_schoolbook(2, 2, 2, 1, 1, 5)
    
    # Total rank used: 7 + 7 + 4 + 10 + 10 + 5 = 43.
    # col should be 43.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
