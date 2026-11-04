import time
import numpy as np

START = time.time()
DEADLINE = START + 130.0


def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    # M6 = (A10-A00)(B00+B01)
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    return U, V, W


def _kron_tensors(A, B, C):
    """Kronecker-product a rank-r1 tensor (A1,B1,C1) with a rank-r2 tensor (A2,B2,C2)."""
    r1 = A.shape[1]
    r2 = B.shape[1]
    R = r1 * r2
    n1, m1, p1 = A.shape[0], B.shape[0], C.shape[0]
    n2, m2, p2 = B.shape[0], A.shape[0], C.shape[0]
    # The shapes: A is (n*m, r), B is (m*p, r), C is (p*n, r)
    # For Kronecker: A_out shape (n1*m1 * n2*m2, R)
    n1m = n1 * m1
    m1p = m1 * p1
    p1n = p1 * n1
    n2m = n2 * m2
    m2p = m2 * p2
    p2n = p2 * n2

    Aout = np.zeros((n1m * n2m, R), dtype=np.int16)
    Bout = np.zeros((m1p * m2p, R), dtype=np.int16)
    Cout = np.zeros((p1n * p2n, R), dtype=np.int16)

    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            Aout[(i1 * m1 + j1) * n2m + (i2 * m2 + j2), c] = \
                                A[i1 * m1 + j1, c1] * A[i2 * m2 + j2, c2]
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            Bout[(j1 * p1 + k1) * m2p + (j2 * p2 + k2), c] = \
                                B[j1 * p1 + k1, c1] * B[j2 * p2 + k2, c2]
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            Cout[(k1 * n1 + i1) * p2n + (k2 * n2 + i2), c] = \
                                C[k1 * n1 + i1, c1] * C[k2 * n2 + i2, c2]
    return Aout, Bout, Cout


def _build_4x4x4_tensors():
    """Build 4x4x4 via Kronecker product of two 2x2x2 Strassen."""
    U2, V2, W2 = _strassen_2x2x2()
    return _kron_tensors(U2, V2, W2)


def _build_rank98():
    """
    Build a rank-98 decomposition for 4x4 * 4x8.
    Uses two 4x4x4 blocks.
    """
    U4, V4, W4 = _build_4x4x4_tensors()

    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)

    for block in range(2):
        col_start = block * 49
        col_end = (block + 1) * 49
        U[:, col_start:col_end] = U4
        for j in range(4):
            for k_local in range(4):
                k_global = block * 4 + k_local
                V[j * 8 + k_global, col_start:col_end] = V4[j * 4 + k_local, :]
        for k_local in range(4):
            k_global = block * 4 + k_local
            for i in range(4):
                W[k_global * 4 + i, col_start:col_end] = W4[k_local * 4 + i, :]

    # Verify
    n, m, p = 4, 4, 8
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(total_rank):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(total - 1.0) > 1e-9:
                    return None

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _schoolbook(n=4, m=4, p=8):
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


def _try_rank96():
    """
    Attempt to find a rank-96 decomposition by exploiting the structure of
    4x4*4x8 = two 4x4*4x4 blocks, and checking if we can merge columns
    where one block has zeros.
    
    The key insight: in the 4x4x4 Strassen decomposition, many entries are
    zero. If we can find two columns (one from each block) that have
    complementary support (disjoint nonzero positions in U, V, or W), we
    might be able to merge them.
    """
    if time.time() > DEADLINE - 30:
        return None

    U4, V4, W4 = _build_4x4x4_tensors()
    
    # Try to see if we can reduce rank by sharing columns between the two blocks
    # where the supports don't conflict.
    
    # For a valid rank-96 solution, we'd need to remove 2 columns from the
    # rank-98 construction. This means finding pairs of columns (one from
    # each block) that can be "absorbed" or merged.
    
    # Actually, let's try a different approach: use the 3x3x3 Strassen variant
    # or other known decompositions for 4x4x4 with rank < 49.
    
    # Known: 4x4x4 matrix multiplication tensor has rank at most 49 (Strassen).
    # The best known upper bound is 49, and it's not known if it can be lower.
    
    # Let's try a hybrid approach: use rank-49 for one 4x4x4 block and
    # rank-47 for the other (if such exists), giving total 96.
    
    # Since we don't have a known rank-47 4x4x4 decomposition, this won't work
    # directly. Instead, let's try to find if the two 4x4x4 blocks can share
    # some tensor factors.
    
    # Alternative: use a 4x2x8 decomposition followed by 2x4x4, but that
    # doesn't help directly.
    
    # Let's just verify the rank-98 solution is correct and return it.
    # The rank-96 attempt is exploratory and unlikely to succeed with known
    # decompositions.
    
    return None


def construct():
    # Try rank-96 first (exploratory)
    if time.time() < DEADLINE - 30:
        try:
            res = _try_rank96()
            if res is not None:
                return res
        except Exception:
            pass

    # Fall back to rank-98
    if time.time() < DEADLINE - 5:
        try:
            res = _build_rank98()
            if res is not None:
                return res
        except Exception:
            pass

    return _schoolbook()


if __name__ == "__main__":
    res = construct()
    rank = len(res['U'][0])
    print(f"Rank: {rank}")
    n, m, p = 4, 4, 8
    U, V, W = res['U'], res['V'], res['W']
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    print(f"Valid: {valid}")
