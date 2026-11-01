"""Matrix multiplication 2x2x15: Strassen-based rank reduction."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

def _build_strassen_2x2x5():
    """Build a rank-14 decomposition for 2x2 @ 2x5 using two Strassen + two schoolbook columns."""
    n, m, p = 2, 2, 5
    # We'll use 5 "slots" for the 5 columns of B.
    # For columns 0,1: Strassen (rank 7)
    # For columns 2,3: Strassen (rank 7)
    # For column 4: schoolbook (rank 4) but we only need 2x2x1 = 4 terms, but since n*m=4, that's 4.
    # Total: 7 + 7 + 4 = 18. That's worse than schoolbook 20? No, 18 < 20.
    # Wait, can we do better?
    # Actually, for 2x2x2, Strassen is rank 7 vs schoolbook 8.
    # For 2x2x5: if we treat it as 2x2x2 + 2x2x2 + 2x2x1, that's 7+7+4 = 18.
    # Schoolbook is 2*2*5 = 20. So 18 is better.
    
    # But wait - the task says "rank-minimal". Is 18 minimal for 2x2x5?
    # The lower bound for 2x2xp is not well known. Let's just use 18.
    
    # Actually, let me reconsider. We need the full 2x2x15 decomposition.
    # Let's use Strassen for each 2x2x2 block of B.
    # 15 columns = 7 full 2-column blocks (14 cols) + 1 single column.
    # 7 * 7 + 4 = 49 + 4 = 53.
    # Schoolbook is 60. So 53 is better.
    
    # Let's build it directly for 2x2x15.
    return None  # We'll build directly in construct

def _strassen_2x2x2_base():
    """Return the 7 Strassen terms as (U_row, V_row, W_row) indices for a 2x2x2 block.
    
    For multiplying A[2,2] @ B[2,2] -> C[2,2]:
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    Strassen:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    U indices: row = i*m+j = i*2+j, so A00=0, A01=1, A10=2, A11=3
    V indices: row = j*p+k, for 2x2x2: B00=0, B01=1, B10=2, B11=3
    W indices: row = k*n+i, for 2x2x2: C00=0, C10=1, C01=2, C11=3
    Wait: W[k*n+i] where k is col of C (0,1) and i is row of C (0,1).
    C[0,0]: k=0,i=0 -> W[0]
    C[0,1]: k=1,i=0 -> W[1]
    C[1,0]: k=0,i=1 -> W[2]
    C[1,1]: k=1,i=1 -> W[3]
    
    Let me re-verify: output is C[i,k] where i in [0,n), k in [0,p).
    W[k*n+i]: for n=2, k=0: W[0]=C[0,0], W[1]=C[1,0]; k=1: W[2]=C[0,1], W[3]=C[1,1].
    So W[0]=C[0,0], W[1]=C[1,0], W[2]=C[0,1], W[3]=C[1,1].
    
    Now the Strassen formulas in terms of C indices:
    C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M6
    
    W[0] = C[0,0] = M1 + M4 - M5 + M7
    W[1] = C[1,0] = M2 + M4
    W[2] = C[0,1] = M3 + M5
    W[3] = C[1,1] = M1 - M2 + M3 + M6
    
    M1 = (A00+A11)(B00+B11): U has +1 at row 0, +1 at row 3; V has +1 at row 0, +1 at row 3
    M2 = (A10+A11)B00: U has +1 at row 2, +1 at row 3; V has +1 at row 0
    M3 = A00(B10-B11): U has +1 at row 0; V has +1 at row 2, -1 at row 3
    M4 = A11(B01-B00): U has +1 at row 3; V has +1 at row 1, -1 at row 0
    M5 = (A00+A01)B11: U has +1 at row 0, +1 at row 1; V has +1 at row 3
    M6 = (A10-A00)(B00+B10): U has +1 at row 2, -1 at row 0; V has +1 at row 0, +1 at row 2
    M7 = (A01-A11)(B01+B11): U has +1 at row 1, -1 at row 3; V has +1 at row 1, +1 at row 3
    """
    # Each entry is a list of (row, coeff) for that matrix
    U_terms = [
        [(0, 1), (3, 1)],       # M1
        [(2, 1), (3, 1)],       # M2
        [(0, 1)],               # M3
        [(3, 1)],               # M4
        [(0, 1), (1, 1)],       # M5
        [(2, 1), (0, -1)],      # M6
        [(1, 1), (3, -1)],      # M7
    ]
    V_terms = [
        [(0, 1), (3, 1)],       # M1
        [(0, 1)],               # M2
        [(2, 1), (3, -1)],      # M3
        [(1, 1), (0, -1)],      # M4
        [(3, 1)],               # M5
        [(0, 1), (2, 1)],       # M6
        [(1, 1), (3, 1)],       # M7
    ]
    W_terms = [
        [(0, 1), (1, 1), (3, -1), (2, 1)],   # M1: W[0]+W[1]+W[3]-W[2] wait...
        # Let me redo:
        # W[0] = C[0,0] = M1 + M4 - M5 + M7
        # W[1] = C[1,0] = M2 + M4
        # W[2] = C[0,1] = M3 + M5
        # W[3] = C[1,1] = M1 - M2 + M3 + M6
        # So for each M_s, which W rows does it appear in, and with what coefficient?
        # M1: W[0]+1, W[3]+1
        # M2: W[1]+1, W[3]-1
        # M3: W[2]+1, W[3]+1
        # M4: W[0]+1, W[1]+1
        # M5: W[0]-1, W[2]+1
        # M6: W[3]+1
        # M7: W[0]+1
    ]
    W_terms = [
        [(0, 1), (3, 1)],       # M1
        [(1, 1), (3, -1)],      # M2
        [(2, 1), (3, 1)],       # M3
        [(0, 1), (1, 1)],       # M4
        [(0, -1), (2, 1)],      # M5
        [(3, 1)],               # M6
        [(0, 1)],               # M7
    ]
    return U_terms, V_terms, W_terms

def construct():
    """Construct rank-53 decomposition using 7 Strassen blocks (14 cols) + 1 schoolbook col."""
    n, m, p = 2, 2, 15
    # Strassen uses 7 terms per 2-column block
    # 7 blocks * 7 + 4 (schoolbook for last col) = 53
    rank = 53
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    U_terms, V_terms, W_terms = _strassen_2x2x2_base()
    
    col = 0
    for block in range(7):
        # This block covers B columns 2*block and 2*block+1
        for s in range(7):
            idx = col + s
            # U: A rows are always 0-3 (since n*m=4)
            for (row, coeff) in U_terms[s]:
                U[row, idx] = coeff
            # V: B rows are j*p+k where j=0,1 and k=2*block, 2*block+1
            for (row, coeff) in V_terms[s]:
                # row is 0-3, mapping: 0->B00, 1->B01, 2->B10, 3->B11
                # In our indexing, B[j,k] is at V[j*p+k]
                j = row // 2
                k_local = row % 2
                k = 2 * block + k_local
                V[j * p + k, idx] = coeff
            # W: C rows are k*n+i where k is output col, i is output row
            for (row, coeff) in W_terms[s]:
                # row is 0-3, mapping: 0->C[0,0], 1->C[1,0], 2->C[0,1], 3->C[1,1]
                # In our indexing, C[i,k] is at W[k*n+i]
                k = row % 2  # Wait, let me recheck
                # W[k*n+i]: row index = k*n+i
                # row=0: k=0,i=0 -> C[0,0]
                # row=1: k=0,i=1 -> C[1,0]
                # row=2: k=1,i=0 -> C[0,1]
                # row=3: k=1,i=1 -> C[1,1]
                k = row // 2
                i = row % 2
                W[k * n + i, idx] = coeff
        col += 7
    
    # Last column (column 14 of B): schoolbook, 4 terms
    for s in range(4):
        idx = col + s
        i = s // 2
        j = s % 2
        U[i * m + j, idx] = 1.0
        V[j * p + 14, idx] = 1.0
        W[14 * n + i, idx] = 1.0
    col += 4
    
    # Convert to lists of lists for the required format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
