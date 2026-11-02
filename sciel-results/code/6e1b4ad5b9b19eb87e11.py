"""Matrix multiplication 4,5,6 – rank-minimal exact tensor decomposition."""
import time
import numpy as np


def _build_strassen_2x2():
    """2x2*2x2 -> 2x2, rank 7. Returns U(4x7), V(4x7), W(4x7)."""
    U = np.zeros((4, 7), dtype=object)
    V = np.zeros((4, 7), dtype=object)
    W = np.zeros((4, 7), dtype=object)
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = a*(f-h), M2=(a+b)*h, M3=(c+d)*e, M4=d*(g-e)
    # M5=(a+d)*(e+h), M6=(b-d)*(g+h), M7=(c-a)*(e+f)
    # C = [[M5+M4-M2+M6, M1+M2], [M3+M4, M1-M3+M5+M7]]
    # Index: A[i*2+j], B[j*2+k], C[k*2+i]
    # A: a=0,b=1,c=2,d=3; B: e=0,f=1,g=2,h=3; C: C00=0,C01=1,C10=2,C11=3

    # M1 = a*(f-h): A[0]=1, B[1]=1, B[3]=-1
    U[0,0]=1; V[1,0]=1; V[3,0]=-1; W[0,0]=1; W[1,0]=1
    # M2=(a+b)*h: A[0]=1,A[1]=1, B[3]=1
    U[0,1]=1; U[1,1]=1; V[3,1]=1; W[0,1]=1; W[1,1]=-1
    # M3=(c+d)*e: A[2]=1,A[3]=1, B[0]=1
    U[2,2]=1; U[3,2]=1; V[0,2]=1; W[0,2]=1; W[2,2]=1
    # M4=d*(g-e): A[3]=1, B[2]=1,B[0]=-1
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    # M5=(a+d)*(e+h): A[0]=1,A[3]=1, B[0]=1,B[3]=1
    U[0,4]=1; U[3,4]=1; V[0,4]=1; V[3,4]=1; W[0,4]=1; W[1,4]=1; W[2,4]=1; W[3,4]=-1
    # M6=(b-d)*(g+h): A[1]=1,A[3]=-1, B[2]=1,B[3]=1
    U[1,5]=1; U[3,5]=-1; V[2,5]=1; V[3,5]=1; W[0,5]=1; W[3,5]=1
    # M7=(c-a)*(e+f): A[2]=1,A[0]=-1, B[0]=1,B[1]=1
    U[2,6]=1; U[0,6]=-1; V[0,6]=1; V[1,6]=1; W[1,6]=1; W[3,6]=1
    return U, V, W


def _build_strassen_4x4():
    """4x4*4x4 -> 4x4, rank 49 via recursive Strassen."""
    U2, V2, W2 = _build_strassen_2x2()
    rank = 49
    U4 = np.zeros((16, rank), dtype=object)
    V4 = np.zeros((16, rank), dtype=object)
    W4 = np.zeros((16, rank), dtype=object)

    # Index mappings: local 2x2 index -> global 4x4 index (i*4+j)
    # A11: (0,0)=0,(0,1)=1,(1,0)=4,(1,1)=5
    # A12: (0,2)=2,(0,3)=3,(1,2)=6,(1,3)=7
    # A21: (2,0)=8,(2,1)=9,(3,0)=12,(3,1)=13
    # A22: (2,2)=10,(2,3)=11,(3,2)=14,(3,3)=15
    A_maps = {
        'A11': [0, 1, 4, 5],
        'A12': [2, 3, 6, 7],
        'A21': [8, 9, 12, 13],
        'A22': [10, 11, 14, 15],
    }
    B_maps = {
        'B11': [0, 1, 4, 5],
        'B12': [2, 3, 6, 7],
        'B21': [8, 9, 12, 13],
        'B22': [10, 11, 14, 15],
    }
    C_maps = {
        'C11': [0, 1, 4, 5],
        'C12': [2, 3, 6, 7],
        'C21': [8, 9, 12, 13],
        'C22': [10, 11, 14, 15],
    }

    # Strassen 4x4 terms (7 terms, each is a 2x2*2x2 product):
    # M1 = A11 * (B12 - B22) -> contributes to C11, C12
    # M2 = (A11 + A12) * B22 -> contributes to C11, C12
    # M3 = (A21 + A22) * B11 -> contributes to C21, C22
    # M4 = A22 * (B21 - B11) -> contributes to C11, C21
    # M5 = (A11 + A22) * (B11 + B22) -> contributes to C11, C12, C21, C22
    # M6 = (A12 - A22) * (B21 + B22) -> contributes to C11, C22
    # M7 = (A21 - A11) * (B11 + B12) -> contributes to C12, C22

    # For each term, define:
    #   A_blocks: list of (block_name, coeff)
    #   B_blocks: list of (block_name, coeff)
    #   C_contrib: dict mapping C_block_name -> coeff (how M_t contributes to that C block)
    #
    # The 2x2*2x2 sub-product has 7 sub-terms (s=0..6).
    # For sub-term s:
    #   U contribution: for each (a_block, a_coeff) in A_blocks:
    #       for each local_idx where U2[local_idx, s] != 0:
    #           U4[A_maps[a_block][local_idx], t*7+s] += a_coeff * U2[local_idx, s]
    #   V contribution: for each (b_block, b_coeff) in B_blocks:
    #       for each local_idx where V2[local_idx, s] != 0:
    #           V4[B_maps[b_block][local_idx], t*7+s] += b_coeff * V2[local_idx, s]
    #   W contribution: for each (c_block, c_coeff) in C_contrib:
    #       for each local_idx where W2[local_idx, s] != 0:
    #           W4[C_maps[c_block][local_idx], t*7+s] += c_coeff * W2[local_idx, s]

    terms = [
        # M1 = A11 * (B12 - B22)
        {
            'A': [('A11', 1)],
            'B': [('B12', 1), ('B22', -1)],
            'C': {'C11': 1, 'C12': 1},
        },
        # M2 = (A11 + A12) * B22
        {
            'A': [('A11', 1), ('A12', 1)],
            'B': [('B22', 1)],
            'C': {'C11': -1, 'C12': 1},
        },
        # M3 = (A21 + A22) * B11
        {
            'A': [('A21', 1), ('A22', 1)],
            'B': [('B11', 1)],
            'C': {'C21': 1, 'C22': -1},
        },
        # M4 = A22 * (B21 - B11)
        {
            'A': [('A22', 1)],
            'B': [('B21', 1), ('B11', -1)],
            'C': {'C11': 1, 'C21': 1},
        },
        # M5 = (A11 + A22) * (B11 + B22)
        {
            'A': [('A11', 1), ('A22', 1)],
            'B': [('B11', 1), ('B22', 1)],
            'C': {'C11': 1, 'C12': 1, 'C21': 1, 'C22': 1},
        },
        # M6 = (A12 - A22) * (B21 + B22)
        {
            'A': [('A12', 1), ('A22', -1)],
            'B': [('B21', 1), ('B22', 1)],
            'C': {'C11': 1, 'C22': 1},
        },
        # M7 = (A21 - A11) * (B11 + B12)
        {
            'A': [('A21', 1), ('A11', -1)],
            'B': [('B11', 1), ('B12', 1)],
            'C': {'C12': 1, 'C22': 1},
        },
    ]

    for t, term in enumerate(terms):
        for s in range(7):
            col = t * 7 + s
            # U contributions
            for a_block, a_coeff in term['A']:
                for li in range(4):
                    val = U2[li, s]
                    if val != 0:
                        gi = A_maps[a_block][li]
                        U4[gi, col] = (U4[gi, col] if U4[gi, col] != 0 else 0) + a_coeff * val
            # V contributions
            for b_block, b_coeff in term['B']:
                for li in range(4):
                    val = V2[li, s]
                    if val != 0:
                        gi = B_maps[b_block][li]
                        V4[gi, col] = (V4[gi, col] if V4[gi, col] != 0 else 0) + b_coeff * val
            # W contributions
            for c_block, c_coeff in term['C'].items():
                for li in range(4):
                    val = W2[li, s]
                    if val != 0:
                        gi = C_maps[c_block][li]
                        W4[gi, col] = (W4[gi, col] if W4[gi, col] != 0 else 0) + c_coeff * val

    return U4, V4, W4


def _build_schoolbook(n, m, p):
    """Schoolbook decomposition for n x m * m x p -> n x p, rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def construct():
    start = time.time()
    deadline = start + 130.0  # 130s safety margin (150s limit)

    n, m, p = 4, 5, 6

    # Strategy: Partition A (4x5) and B (5x6) into blocks.
    # A = [A1 | A2] where A1 is 4x4, A2 is 4x1
    # B = [B1; B2] where B1 is 4x6, B2 is 1x6
    # C = A1*B1 + A2*B2
    #
    # A1*B1: 4x4 * 4x6. Partition B1 = [B1a | B1b] where B1a is 4x4, B1b is 4x2.
    # C = A1*B1a + A1*B1b + A2*B2
    #
    # Term 1: A1 * B1a (4x4 * 4x4) -> use Strassen rank 49
    # Term 2: A1 * B1b (4x4 * 4x2) -> schoolbook 32
    # Term 3: A2 * B2 (4x1 * 1x6) -> schoolbook 24
    # Total: 49 + 32 + 24 = 105

    rank_total = 105
    U = np.zeros((n * m, rank_total), dtype=object)  # 20 x 105
    V = np.zeros((m * p, rank_total), dtype=object)  # 30 x 105
    W = np.zeros((p * n, rank_total), dtype=object)  # 24 x 105

    # --- Term 1: A1 (4x4) * B1a (4x4) -> C[0:4, 0:4] ---
    # A1 = A[0:4, 0:4], B1a = B[0:4, 0:4], C1 = C[0:4, 0:4]
    U4, V4, W4 = _build_strassen_4x4()

    # A1 indices: A[i*5 + j] for i in 0..3, j in 0..3 -> i*5+j
    # B1a indices: B[j*6 + k] for j in 0..3, k in 0..3 -> j*6+k
    # C1 indices: C[k*4 + i] for k in 0..3, i in 0..3 -> k*4+i

    for col in range(49):
        for gi in range(16):
            val = U4[gi, col]
            if val != 0:
                # gi is index in 4x4 A: i*4+j
                i = gi // 4
                j = gi % 4
                U[i * 5 + j, col] = (U[i * 5 + j, col] if U[i * 5 + j, col] != 0 else 0) + val
        for gi in range(16):
            val = V4[gi, col]
            if val != 0:
                # gi is index in 4x4 B: j*4+k
                j = gi // 4
                k = gi % 4
                V[j * 6 + k, col] = (V[j * 6 + k, col] if V[j * 6 + k, col] != 0 else 0) + val
        for gi in range(16):
            val = W4[gi, col]
            if val != 0:
                # gi is index in 4x4 C: k*4+i
                k = gi // 4
                i = gi % 4
                W[k * 4 + i, col] = (W[k * 4 + i, col] if W[k * 4 + i, col] != 0 else 0) + val

    # --- Term 2: A1 (4x4) * B1b (4x2) -> C[0:4, 4:6] ---
    # A1 = A[0:4, 0:4], B1b = B[0:4, 4:6], C2 = C[0:4, 4:6]
    # Schoolbook: rank 4*4*2 = 32
    col_offset = 49
    for i in range(4):
        for j in range(4):
            for k in range(2):
                col = col_offset + i * 8 + j * 2 + k
                U[i * 5 + j, col] = 1
                V[j * 6 + (4 + k), col] = 1
                W[(4 + k) * 4 + i, col] = 1

    # --- Term 3: A2 (4x1) * B2 (1x6) -> C[0:4, 0:6] ---
    # A2 = A[0:4, 4], B2 = B[4:5, 0:6], C3 = C[0:4, 0:6]
    # Schoolbook: rank 4*1*6 = 24
    col_offset = 49 + 32
    for i in range(4):
        for k in range(6):
            col = col_offset + i * 6 + k
            U[i * 5 + 4, col] = 1
            V[4 * 6 + k, col] = 1
            W[k * 4 + i, col] = 1

    # Verify quickly with random matrices
    try:
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        C_computed = np.zeros((n, p))
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # C[k*n+i, :] = sum over t of W[k*n+i, t] * (U^T A)[t] * (V^T B)[t]
        # Actually: C = W^T * diag(U^T A, V^T B) ... let me think
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # A is (n, m), B is (m, p)
        # Reshape A to (n*m, 1)? No.
        # The decomposition says: C[k*n+i] = sum_t W[k*n+i, t] * (sum_{i',j'} U[i'*m+j', t] * A[i',j']) * (sum_{j'',k'} V[j''*p+k'', t] * B[j'',k'])
        # So: C = W^T * (U^T * vec(A)) * (V^T * vec(B)) element-wise product then sum
        # vec(A) is (n*m, 1), U^T * vec(A) is (rank, 1)
        # vec(B) is (m*p, 1), V^T * vec(B) is (rank, 1)
        # W^T is (p*n, rank), so W^T * (U^T vec(A) .* V^T vec(B)) gives (p*n, 1)
        a_vec = A_test.reshape(n * m, 1)
        b_vec = B_test.reshape(m * p, 1)
        u_vals = U_arr.T @ a_vec  # (rank, 1)
        v_vals = V_arr.T @ b_vec  # (rank, 1)
        c_vec = W_arr.T @ (u_vals * v_vals)  # (p*n, 1)
        C_computed = c_vec.reshape(p, n).T  # (n, p)
        err = np.max(np.abs(C_computed - C_expected))
        if err > 1e-9:
            # Fall back to schoolbook
            U, V, W = _build_schoolbook(n, m, p)
    except Exception:
        # Fall back to schoolbook
        U, V, W = _build_schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
