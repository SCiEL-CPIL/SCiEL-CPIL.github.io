"""Matrix multiplication 3,5,6 — block-decomposed exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Split dimensions into blocks: n=3 -> [2,1], m=5 -> [2,2,1], p=6 -> [2,2,2]
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 2]
    assert sum(n_parts) == n
    assert sum(m_parts) == m
    assert sum(p_parts) == p

    # Precompute row/col indices for each part
    n_offsets = np.cumsum([0] + n_parts)
    m_offsets = np.cumsum([0] + m_parts)
    p_offsets = np.cumsum([0] + p_parts)

    U_rows = n * m
    V_rows = m * p
    W_rows = p * n

    # Build the 2x2x2 Strassen decomposition (rank 7)
    # Strassen: C = (A1+B2)(B1+B3) - (A1-B3)(B1-B3) - (A2+B3)B2 + (A1-B1)(B2+B3)
    # Let me use the standard 7-multiplication Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    # For a 2x2x2 block, U is 4x7, V is 4x7, W is 4x7
    # U rows: (i,j) for i in 0..1, j in 0..1 -> indices 0,1,2,3
    # V rows: (j,k) for j in 0..1, k in 0..1 -> indices 0,1,2,3
    # W rows: (k,i) for k in 0..1, i in 0..1 -> indices 0,1,2,3
    # W index: k*2 + i

    # M1: (A11+A22)(B11+B22) -> U: A11=1,A22=1; V: B11=1,B22=1; W: C11=1,C22=1
    # M2: (A21+A22)B11 -> U: A21=1,A22=1; V: B11=1; W: C21=1
    # M3: A11(B21-B22) -> U: A11=1; V: B21=1,B22=-1; W: C12=1
    # M4: A22(B12-B11) -> U: A22=1; V: B12=1,B11=-1; W: C11=1,C21=1
    # M5: (A11+A12)B22 -> U: A11=1,A12=1; V: B22=1; W: C11=1
    # M6: (A21-A11)(B11+B12) -> U: A21=1,A11=-1; V: B11=1,B12=1; W: C22=1
    # M7: (A12-A22)(B21+B22) -> U: A12=1,A22=-1; V: B21=1,B22=1; W: C11=1

    # U index: i*2+j -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # V index: j*2+k -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # W index: k*2+i -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3

    strassen_U = np.array([
        #      M1 M2 M3 M4 M5 M6 M7
        [  1,  0,  1,  0,  1, -1,  0],  # A11
        [  0,  0,  0,  0,  1,  0,  1],  # A12
        [  0,  1,  0,  0,  0,  1,  0],  # A21
        [  1,  1,  0,  1,  0,  0, -1],  # A22
    ], dtype=float)

    strassen_V = np.array([
        #      M1 M2 M3 M4 M5 M6 M7
        [  1,  1,  0, -1,  0,  1,  0],  # B11
        [  1,  0,  0,  1,  0,  1,  0],  # B12
        [  0,  0,  1,  0,  0,  0,  1],  # B21
        [  1,  0, -1,  0,  1,  0,  1],  # B22
    ], dtype=float)

    strassen_W = np.array([
        #      M1 M2 M3 M4 M5 M6 M7
        [  1,  0,  0,  1, -1,  0,  1],  # C11
        [  0,  0,  1,  0,  0,  0,  0],  # C12
        [  0,  1,  0,  1,  0,  0,  0],  # C21
        [  1,  0,  0,  0,  0,  1,  0],  # C22
    ], dtype=float)

    # Verify Strassen: C = sum_c U[:,c] * V[:,c] * W[:,c] (outer product summed)
    # Check: for each (i,j,k), sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] should be 1
    # if (i,j,k) is a valid triple, else 0. For 2x2x2, all 8 triples should give 1.
    # Let's verify with numpy:
    C_check = np.zeros((4, 4, 4))
    for c in range(7):
        C_check += np.einsum('a,b,c->abc', strassen_U[:,c], strassen_V[:,c], strassen_W[:,c])
    # C_check[i,j,k] should be 1 for all i,j,k in {0,1}
    # But wait: the tensor for 2x2x2 matrix multiplication has T[i,j,k]=1 for all i,j,k
    # Actually no. The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k in the 2x2x2 case.
    # Let me verify:
    expected = np.ones((4, 4, 4))
    # Hmm, actually the tensor is T[i,j,k] = 1 for all combinations. Let me check C_check:
    # C_check[a,b,c] where a = i*2+j, b = j*2+k, c = k*2+i
    # This is not simply T[i,j,k]=1. Let me re-examine.
    # Actually the tensor T for matmul is: T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So the 3-index tensor has T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor of size 2x2x2)
    # When flattened: T_flat[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    # For m=p=2, n=2: a=i*2+j, b=j*2+k, c=k*2+i
    # The tensor value at (a,b,c) is 1 if there exists i,j,k with a=i*2+j, b=j*2+k, c=k*2+i
    # For 2x2x2, every (a,b,c) in [0,4)x[0,4)x[0,4) maps to valid (i,j,k)?
    # a in 0..3 -> i=a//2, j=a%2
    # b in 0..3 -> j=b//2, k=b%2  -- but j must match! j from a must equal j from b.
    # So not all 64 entries are 1. Only those where j is consistent.
    # This is getting complicated. Let me just verify numerically.

    # Actually, let me just build the full decomposition and verify at the end.

    # Build block decomposition
    # For each pair of n-part and m-part and p-part, create a block decomposition
    # If both n-part and m-part and p-part are >= 2, we could use Strassen, but
    # the blocks have different sizes. We'll use schoolbook for rectangular blocks
    # and Strassen only for 2x2x2 blocks.

    U_list = []
    V_list = []
    W_list = []

    for ni, ns in enumerate(n_parts):
        for mj, ms in enumerate(m_parts):
            for pk, ps in enumerate(p_parts):
                if ns == 2 and ms == 2 and ps == 2:
                    # Use Strassen
                    # U: rows i in [0,2), j in [0,2) -> 4 rows, 7 cols
                    # V: rows j in [0,2), k in [0,2) -> 4 rows, 7 cols
                    # W: rows k in [0,2), i in [0,2) -> 4 rows, 7 cols
                    U_block = strassen_U.copy()
                    V_block = strassen_V.copy()
                    W_block = strassen_W.copy()
                else:
                    # Schoolbook: rank = ns*ms*ps
                    r = ns * ms * ps
                    U_block = np.zeros((ns * ms, r))
                    V_block = np.zeros((ms * ps, r))
                    W_block = np.zeros((ps * ns, r))
                    for ii in range(ns):
                        for jj in range(ms):
                            for kk in range(ps):
                                col = ii * ms * ps + jj * ps + kk
                                U_block[ii * ms + jj, col] = 1
                                V_block[jj * ps + kk, col] = 1
                                W_block[kk * ns + ii, col] = 1
                U_list.append(U_block)
                V_list.append(V_block)
                W_list.append(W_block)

    U_full = np.hstack(U_list)
    V_full = np.hstack(V_list)
    W_full = np.hstack(W_list)

    rank = U_full.shape[1]

    # Verify: reconstruct tensor and check
    T_reconstructed = np.zeros((U_rows, V_rows, W_rows))
    for c in range(rank):
        T_reconstructed += np.einsum('a,b,c->abc', U_full[:, c], V_full[:, c], W_full[:, c])

    # Check against expected tensor
    T_expected = np.zeros((U_rows, V_rows, W_rows))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_expected[a, b, c] = 1

    if not np.allclose(T_reconstructed, T_expected):
        # Fallback to schoolbook
        rank = n * m * p
        U_full = np.zeros((U_rows, rank))
        V_full = np.zeros((V_rows, rank))
        W_full = np.zeros((W_rows, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_full[i * m + j, col] = 1
                    V_full[j * p + k, col] = 1
                    W_full[k * n + i, col] = 1
                    col += 1

    # Convert to list of lists with integer values
    U_out = U_full.tolist()
    V_out = V_full.tolist()
    W_out = W_full.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
