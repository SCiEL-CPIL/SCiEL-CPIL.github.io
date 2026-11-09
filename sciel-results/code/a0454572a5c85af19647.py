"""Matrix multiplication 4x4 * 4x9: tensor decomposition.

Strategy: Build a block-decomposition using Laderman's rank-7 algorithm for
4x4 matrix multiplication. The output 4x9 matrix is partitioned into three
sub-blocks of 4x3 columns each. Each sub-block requires a 4x4 * 4x3 product.
Since 4x3 <= 4x4, we can use the Laderman algorithm with zero-padded inputs,
giving rank 7 per block and total rank 21.

Additionally, we explore a structured split: split the 9 columns into 4+5,
where 4x4 uses Laderman (rank 7) and 4x5 uses a 4x4 Laderman + 4x1 schoolbook
(1 term per row), giving rank 7 + (7+4) = 18.

We verify each candidate exactly and return the best valid one.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
DEADLINE = 130.0  # seconds, safety margin from 150s limit


def _time_ok(start):
    return (time.time() - start) < DEADLINE


def _verify(U, V, W, n, m, p):
    """Verify decomposition T[i,j,k]=1 using numpy float with tight tolerance."""
    rank = U.shape[1]
    Uf = np.asarray(U, dtype=np.float64)
    Vf = np.asarray(V, dtype=np.float64)
    Wf = np.asarray(W, dtype=np.float64)

    # Reshape to (n, m, r), (m, p, r), (p, n, r)
    Ur = Uf.reshape(n, m, rank)
    Vr = Vf.reshape(m, p, rank)
    Wr = Wf.reshape(p, n, rank)

    # T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)

    return np.allclose(T, 1.0, atol=1e-9)


def _laderman_4x4():
    """Return (U, V, W) each of shape (16, 7) for 4x4 * 4x4 multiplication.

    Laderman's algorithm:
    U = [A11, A12+A22, A12, A21+A22, A22, A21, A11+A22]  (7 vectors of 4x4)
    V = [B11, B21, B11+B21, B22, B12, B22+B12, B12+B22]
    W = [C11, C12, C21, C22, C12+C21, C22+C11, C11+C22]

    Indices: U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
    """
    n = m = p = 4
    r = 7
    U = np.zeros((16, r), dtype=np.float64)
    V = np.zeros((16, r), dtype=np.float64)
    W = np.zeros((16, r), dtype=np.float64)

    # U: combinations of A entries
    # A is 4x4, indexed [i, j] -> U[i*4+j, c]
    U[0 * 4 + 0, 0] = 1   # A11
    U[0 * 4 + 1, 1] = 1   # A12
    U[0 * 4 + 1, 2] = 1
    U[1 * 4 + 0, 1] = 1   # A21
    U[1 * 4 + 1, 2] = 1   # A22
    U[0 * 4 + 0, 3] = 1   # A11 + A22
    U[1 * 4 + 1, 3] = 1
    U[0 * 4 + 1, 4] = 1   # A12 + A21
    U[1 * 4 + 0, 4] = 1
    U[0 * 4 + 0, 5] = 1   # A11 + A12
    U[0 * 4 + 1, 5] = 1
    U[1 * 4 + 0, 6] = 1   # A21 + A22
    U[1 * 4 + 1, 6] = 1

    # V: combinations of B entries, B[j,k] -> V[j*4+k, c]
    V[0 * 4 + 0, 0] = 1   # B11
    V[1 * 4 + 0, 1] = 1   # B21
    V[0 * 4 + 0, 2] = 1   # B11 + B21
    V[1 * 4 + 0, 2] = 1
    V[1 * 4 + 1, 3] = 1   # B22
    V[0 * 4 + 1, 4] = 1   # B12
    V[0 * 4 + 1, 5] = 1   # B12 + B22
    V[1 * 4 + 1, 5] = 1
    V[0 * 4 + 0, 6] = 1   # B11 + B12
    V[0 * 4 + 1, 6] = 1

    # W: combinations of C entries, C[k,i] -> W[k*4+i, c]
    W[0 * 4 + 0, 0] = 1   # C11
    W[0 * 4 + 1, 1] = 1   # C12
    W[1 * 4 + 0, 2] = 1   # C21
    W[1 * 4 + 1, 3] = 1   # C22
    W[0 * 4 + 1, 4] = 1   # C12 + C21
    W[1 * 4 + 0, 4] = 1
    W[1 * 4 + 1, 5] = 1   # C22 + C11
    W[0 * 4 + 0, 5] = 1
    W[0 * 4 + 0, 6] = 1   # C11 + C22
    W[1 * 4 + 1, 6] = 1

    return U, V, W


def _laderman_4x3():
    """Compute 4x4 * 4x3 using Laderman with zero-padded 4x4 B.

    Pad B to 4x4 by adding a zero column. The C output is 4x4, we only use
    the first 3 columns.

    Returns U (16, r), V (12, r), W (12, r) for n=4, m=4, p=3.
    """
    U_44, V_44, W_44 = _laderman_4x4()
    r = U_44.shape[1]  # 7

    # U stays the same: (16, 7)
    U = U_44.copy()

    # V: for 4x3, V[j*3+k, c] for j in 0..3, k in 0..2
    # In the 4x4 version, V[j*4+k, c]. We take k=0,1,2 (drop k=3).
    V = np.zeros((12, r), dtype=np.float64)
    for j in range(4):
        for k in range(3):
            V[j * 3 + k, :] = V_44[j * 4 + k, :]

    # W: for 4x3, W[k*4+i, c] for k in 0..2, i in 0..3
    # In the 4x4 version, W[k*4+i, c]. We take k=0,1,2 (drop k=3).
    W = np.zeros((12, r), dtype=np.float64)
    for k in range(3):
        for i in range(4):
            W[k * 4 + i, :] = W_44[k * 4 + i, :]

    return U, V, W


def _laderman_4x1():
    """Compute 4x4 * 4x1. This is just 4 scalar products, rank 4.

    But we can do better: each output C[i] = sum_j A[i,j]*B[j].
    Schoolbook rank = 4*4*1 = 16? No, it's 4 outputs, each needs 4 multiplications,
    but the tensor is rank 4 (one per output row, since B is a vector).

    Actually for 4x4 * 4x1: U[i*4+j, c], V[j*1+k, c] (k=0), W[k*4+i, c] (k=0).
    T[i, j, 0] = 1 for all i, j.
    Schoolbook: rank = 4*4 = 16.
    But we can do rank 4: for each i, one rank-1 term where U picks row i of A,
    V picks all of B, W picks output i.

    Actually the standard way: U[i*4+j, i] = 1, V[j, i] = B[j]... no, that's
    not a constant tensor. The tensor T[i,j,0]=1 for all i,j.

    Rank-1 terms: u * v * w where u is (16,), v is (4,), w is (4,).
    We need sum_c u_c[i,j] * v_c[j] * w_c[i] = 1 for all i,j.

    Simplest: for each (i,j), one term: u[i*4+j]=1, v[j]=1, w[i]=1. Rank 16.

    But can we do rank 4? Let u_c[i,j] = delta(i,c) (row c of A), v_c[j] = 1 (all of B),
    w_c[i] = delta(i,c) (output c). Then sum_c delta(i,c)*1*delta(i,c) = 1 if i=c.
    That gives T[i,j,0] = 1 for all j when i=c, but 0 otherwise. Not correct.

    The minimal rank for 4x4 * 4x1 is actually 4 (it's equivalent to computing
    4 dot products of length 4, which has rank 4 in the tensor sense... actually
    no, the tensor T[i,j,0]=1 is a 4x4x1 tensor which has rank 4).

    Let's use the trivial rank-16 schoolbook for simplicity, or find a rank-4.
    
    Actually, T[i,j,0] = 1 for all i in 0..3, j in 0..3.
    This is the all-ones 4x4 matrix (times the singleton 1). The rank of a
    matrix as a tensor in the "matrix multiplication" sense... 

    For the tensor T[i,j,k] with k=0 only, T[i,j,0]=1. This is a rank-4 tensor:
    Term c (c=0..3): U_c[i,j] = 1 if i==c else 0 (row selector), V_c[j] = 1 (all ones),
    W_c[i] = 1 if i==c else 0 (output selector).
    Then sum_c U_c[i,j]*V_c[j]*W_c[i] = sum_c [i==c] * 1 * [i==c] = 1 (since i=c for exactly one c).
    Yes! Rank 4.
    """
    n, m, p = 4, 4, 1
    r = 4
    U = np.zeros((16, r), dtype=np.float64)
    V = np.zeros((4, r), dtype=np.float64)
    W = np.zeros((4, r), dtype=np.float64)

    for c in range(4):
        for j in range(4):
            U[c * 4 + j, c] = 1  # Row c of A
        V[j, c] = 1              # All of B (B is a column vector of length 4)
        W[0 * 4 + c, c] = 1      # Output c

    return U, V, W


def _construct_block_decomposition():
    """Partition 4x9 into three 4x3 blocks, each using Laderman 4x4 with padding.

    Total rank: 3 * 7 = 21.
    """
    n, m, p = 4, 4, 9
    U_43, V_43, W_43 = _laderman_4x3()
    r_block = U_43.shape[1]  # 7
    total_r = 3 * r_block  # 21

    U = np.zeros((n * m, total_r), dtype=np.float64)
    V = np.zeros((m * p, total_r), dtype=np.float64)
    W = np.zeros((p * n, total_r), dtype=np.float64)

    for block in range(3):
        off = block * r_block
        # U is the same for each block (A doesn't change)
        U[:, off:off + r_block] = U_43

        # V: for block `block`, the columns k are block*3 .. block*3+2
        # V[j*9+k, off+c] = V_43[j*3+k_local, c]
        for j in range(m):
            for k_local in range(3):
                k = block * 3 + k_local
                V[j * p + k, off:off + r_block] = V_43[j * 3 + k_local, :]

        # W: for block `block`, the output rows k are block*3 .. block*3+2
        # W[k*4+i, off+c] = W_43[k_local*4+i, c]
        for k_local in range(3):
            k = block * 3 + k_local
            for i in range(n):
                W[k * n + i, off:off + r_block] = W_43[k_local * 4 + i, :]

    return U, V, W


def _construct_44_45_split():
    """Split 4x9 into 4x4 + 4x5. Use Laderman for 4x4 (rank 7) and
    Laderman-4x4 + schoolbook-4x1 for 4x5. Total rank 7 + 7 + 4 = 18.

    Actually, 4x5 = 4x4 * (first 4 cols) + 4x1 * (last col).
    But 4x4 * 4x4 is rank 7 (Laderman).
    4x4 * 4x1 is rank 4.
    So 4x5 can be done in rank 7 + 4 = 11? No, that's not right either.

    Better: 4x5 = use Laderman for a 4x4 sub-product (columns 0-3) and
    schoolbook for column 4. The 4x4 part has rank 7, the 4x1 part has rank 4.
    Total for 4x5: 7 + 4 = 11.
    Total for 4x9: 7 (first 4 cols) + 11 (next 5 cols) = 18.
    
    Wait, that double-counts. Let me think again.
    
    4x9 = columns 0-3 (4x4) + columns 4-8 (4x5).
    4x4: Laderman rank 7.
    4x5: can be decomposed as 4x4 (cols 4-7) + 4x1 (col 8).
         4x4 part: Laderman rank 7.
         4x1 part: rank 4.
    Total: 7 + 7 + 4 = 18.
    """
    n, m, p = 4, 4, 9
    U_44, V_44, W_44 = _laderman_4x4()
    U_41, V_41, W_41 = _laderman_4x1()

    r44 = U_44.shape[1]  # 7
    r41 = U_41.shape[1]  # 4
    total_r = r44 + r44 + r41  # 18

    U = np.zeros((n * m, total_r), dtype=np.float64)
    V = np.zeros((m * p, total_r), dtype=np.float64)
    W = np.zeros((p * n, total_r), dtype=np.float64)

    # Block 1: columns 0-3 (4x4)
    off1 = 0
    U[:, off1:off1 + r44] = U_44
    for j in range(m):
        for k in range(4):
            V[j * p + k, off1:off1 + r44] = V_44[j * 4 + k, :]
    W[0:16, off1:off1 + r44] = W_44  # k=0..3, i=0..3

    # Block 2: columns 4-7 (4x4)
    off2 = r44
    U[:, off2:off2 + r44] = U_44
    for j in range(m):
        for k in range(4):
            V[j * p + (k + 4), off2:off2 + r44] = V_44[j * 4 + k, :]
    W[16:32, off2:off2 + r44] = W_44  # k=4..7, i=0..3

    # Block 3: column 8 (4x1)
    off3 = r44 + r44
    U[:, off3:off3 + r41] = U_41
    for j in range(m):
        V[j * p + 8, off3:off3 + r41] = V_41[j, :]
    W[32:36, off3:off3 + r41] = W_41  # k=8, i=0..3

    return U, V, W


def _construct():
    start = time.time()
    n, m, p = DIMENSIONS

    # Try the 4x4 + 4x5 split (rank 18)
    U, V, W = _construct_44_45_split()
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try the three 4x3 blocks (rank 21)
    U, V, W = _construct_block_decomposition()
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook rank 144
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def construct():
    """Entry point: return {'U', 'V', 'W'} for the 4x4 * 4x9 matrix multiplication tensor."""
    return _construct()


if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Rank: {U.shape[1]}")
    print(f"Valid: {_verify(U, V, W, 4, 4, 9)}")
