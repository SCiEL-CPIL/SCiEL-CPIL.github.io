"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150 - 10  # 140s deadline, 10s safety margin

    n, m, p = DIMENSIONS  # n=2, m=2, p=9

    # Block decomposition:
    # Split the p=9 dimension into 4 blocks of 2 and 1 block of 1.
    # For each 2x2x2 block, use Strassen (rank 7).
    # For the 2x2x1 block, use schoolbook (rank 4).
    # Total rank = 4*7 + 4 = 32.
    #
    # The trace convention is W[k*n+i, r], where k is the column index of B (0..p-1)
    # and i is the row index of C (0..n-1).
    #
    # U has shape (n*m, rank) = (4, rank)
    # V has shape (m*p, rank) = (18, rank)
    # W has shape (p*n, rank) = (18, rank)

    def _strassen_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (U_entries, V_entries, W_entries) where each entry is a list of (index, value).
        U index: i*m + j (i=row of A, j=col of A)
        V index: j*p + k (j=row of B, k=col of B)
        W index: k*n + i (k=col of B, i=row of C)
        """
        # Strassen formulas:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = (A11 - A22)(B11 + B12)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        #
        # C11 = M1 - M4 + M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6

        terms = []

        # M1 = (A00 + A11)(B00 + B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=+1, C11=+1
        terms.append({
            'U': [(0, 1), (3, 1)],  # A[0,0], A[1,1]
            'V': [(0, 1), (3, 1)],  # B[0,0], B[1,1]
            'W': [(0, 1), (5, 1)],  # W[k*n+i]: C[0,0]=k0,i0; C[1,1]=k1,i1 -> 1*2+1=3? Wait.
        })
        # Let me re-derive the W indices carefully.
        # W index = k*n + i where k is column of B, i is row of C.
        # C[0,0]: i=0, k=0 -> 0*2+0 = 0
        # C[0,1]: i=0, k=1 -> 1*2+0 = 2
        # C[1,0]: i=1, k=0 -> 0*2+1 = 1
        # C[1,1]: i=1, k=1 -> 1*2+1 = 3

        # Redo:
        # M1: contributes +1 to C[0,0] and +1 to C[1,1]
        # W indices: C[0,0]=0, C[1,1]=3
        terms[0] = {
            'U': [(0, 1), (3, 1)],  # A[0,0], A[1,1]
            'V': [(0, 1), (3, 1)],  # B[0,0], B[1,1]
            'W': [(0, 1), (3, 1)],  # C[0,0], C[1,1]
        }

        # M2 = (A10 + A11)B00
        # U: A[1,0]=1, A[1,1]=1
        # V: B[0,0]=1
        # W: C[1,0]=+1
        terms.append({
            'U': [(2, 1), (3, 1)],
            'V': [(0, 1)],
            'W': [(1, 1)],
        })

        # M3 = A00(B10 - B11)
        # U: A[0,0]=1
        # V: B[1,0]=1, B[1,1]=-1
        # W: C[0,1]=+1, C[1,1]=+1
        terms.append({
            'U': [(0, 1)],
            'V': [(2, 1), (3, -1)],
            'W': [(2, 1), (3, 1)],
        })

        # M4 = (A00 - A11)(B00 + B01)
        # U: A[0,0]=1, A[1,1]=-1
        # V: B[0,0]=1, B[0,1]=1
        # W: C[0,0]=-1, C[1,0]=-1
        terms.append({
            'U': [(0, 1), (3, -1)],
            'V': [(0, 1), (1, 1)],
            'W': [(0, -1), (1, -1)],
        })

        # M5 = (A00 + A01)B11
        # U: A[0,0]=1, A[0,1]=1
        # V: B[1,1]=1
        # W: C[0,0]=+1, C[0,1]=+1
        terms.append({
            'U': [(0, 1), (1, 1)],
            'V': [(3, 1)],
            'W': [(0, 1), (2, 1)],
        })

        # M6 = (A10 - A00)(B00 + B01)
        # U: A[1,0]=1, A[0,0]=-1
        # V: B[0,0]=1, B[0,1]=1
        # W: C[1,1]=+1
        terms.append({
            'U': [(2, 1), (0, -1)],
            'V': [(0, 1), (1, 1)],
            'W': [(3, 1)],
        })

        # M7 = (A01 - A11)(B10 + B11)
        # U: A[0,1]=1, A[1,1]=-1
        # V: B[1,0]=1, B[1,1]=1
        # W: C[0,0]=+1
        terms.append({
            'U': [(1, 1), (3, -1)],
            'V': [(2, 1), (3, 1)],
            'W': [(0, 1)],
        })

        return terms

    def _schoolbook_2x2x1_terms():
        """
        Return the 4 schoolbook terms for 2x2x1 matrix multiplication.
        C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        W index for k=0: C[i,0] = 0*n + i = i
        """
        terms = []
        # T1: A[0,0]*B[0,0] -> C[0,0]
        terms.append({
            'U': [(0, 1)],
            'V': [(0, 1)],
            'W': [(0, 1)],
        })
        # T2: A[0,1]*B[1,0] -> C[0,0]
        terms.append({
            'U': [(1, 1)],
            'V': [(1, 1)],
            'W': [(0, 1)],
        })
        # T3: A[1,0]*B[0,0] -> C[1,0]
        terms.append({
            'U': [(2, 1)],
            'V': [(0, 1)],
            'W': [(1, 1)],
        })
        # T4: A[1,1]*B[1,0] -> C[1,0]
        terms.append({
            'U': [(3, 1)],
            'V': [(1, 1)],
            'W': [(1, 1)],
        })
        return terms

    # Build the rank-32 decomposition
    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # 4 x 32
    V = [[0] * rank for _ in range(m * p)]  # 18 x 32
    W = [[0] * rank for _ in range(p * n)]  # 18 x 32

    col = 0
    strassen = _strassen_terms()

    # 4 blocks of 2x2x2 (k = 0..7 in steps of 2)
    for block in range(4):
        k_start = block * 2
        for term in strassen:
            for (idx, val) in term['U']:
                U[idx][col] += val
            for (idx, val) in term['V']:
                V[k_start + idx][col] += val
            for (idx, val) in term['W']:
                W[k_start + idx][col] += val
            col += 1

    # 1 block of 2x2x1 (k = 8)
    k_start = 8
    schoolbook = _schoolbook_2x2x1_terms()
    for term in schoolbook:
        for (idx, val) in term['U']:
            U[idx][col] += val
        for (idx, val) in term['V']:
            V[k_start + idx][col] += val
        for (idx, val) in term['W']:
            W[k_start + idx][col] += val
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
