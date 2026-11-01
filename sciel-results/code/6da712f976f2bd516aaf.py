import numpy as np
import time

DIMENSIONS = [2, 2, 9]
_START_TIME = time.time()
_TIMEOUT = 135  # safety margin before 150s kill


def _build_rank_32(n, m, p):
    """
    Build a rank-32 decomposition using Strassen for 4 blocks of 2x2x2
    and schoolbook for 1 block of 2x2x1.

    For n=2, m=2, p=9:
    - U shape: (n*m, rank) = (4, rank)
    - V shape: (m*p, rank) = (18, rank)
    - W shape: (p*n, rank) = (18, rank)

    U index: i*m + j for i in {0,1}, j in {0,1}
      U[0] = A00, U[1] = A01, U[2] = A10, U[3] = A11

    V index: j*p + k for j in {0,1}, k in 0..8
      V[j*9 + k] = B[j, k]

    W index: k*n + i for k in 0..8, i in {0,1}
      W[k*2 + i] = C[i, k]  (row i, column k of output)
    """
    rank = 32
    U = [[0] * rank for _ in range(n * m)]      # 4 x 32
    V = [[0] * rank for _ in range(m * p)]      # 18 x 32
    W = [[0] * rank for _ in range(p * n)]      # 18 x 32

    # Strassen terms for a 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1})
    # Each term: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (U_index, coeff)
    # v_coeffs: list of (v_offset, coeff) where v_offset is 0 or 1 (for j=0 or j=1)
    # w_coeffs: list of (w_offset, coeff) where w_offset is 0 or 1 (for i=0 or i=1)
    strassen_terms = [
        # P0 = (A00 + A11) * (B00 + B11)
        ([(0, 1), (3, 1)], [(0, 1), (1, 1)], [(0, 1), (1, 1)]),
        # P1 = (A10 + A11) * B00
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1)]),
        # P2 = A00 * (B10 - B00)
        ([(0, 1)], [(0, -1), (1, 1)], [(0, -1), (1, 1)]),
        # P3 = A11 * (B01 - B11)
        ([(3, 1)], [(0, 1), (1, -1)], [(0, 1), (1, -1)]),
        # P4 = (A00 + A01) * B11
        ([(0, 1), (1, 1)], [(1, 1)], [(0, 1)]),
        # P5 = (A10 - A00) * (B00 + B01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(1, -1)]),
        # P6 = (A01 - A11) * (B10 + B11)
        ([(1, 1), (3, -1)], [(0, 1), (1, 1)], [(0, 1)]),
    ]

    # Strassen decomposition of 2x2x2:
    # C00 = P0 - P3 + P6
    # C01 = P3 + P4
    # C10 = P1 + P2
    # C11 = P0 - P2 - P5 + P6
    # But let me re-derive to be sure.
    # Actually the standard Strassen:
    # P1 = (A11 - A00)(B00 + B01)
    # P2 = (A00 + A01) B11
    # P3 = (A10 + A11) B00
    # P4 = (A00 - A10)(B10 + B11)
    # P5 = (A01 + A11)(B01 - B11)
    # P6 = (A00 + A11)(B00 - B11)
    # P7 = (A01 + A10)(B11 - B10)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P2 + P5
    # C10 = P3 + P6
    # C11 = P1 - P3 + P4 + P6
    #
    # Let me just use a simpler approach: verify by direct computation.
    # Actually, let me just use the schoolbook for each 2x2x1 column
    # combined with Strassen for pairs.
    
    # Simpler approach: For each pair of output columns (k0, k0+1),
    # use Strassen on the 2x2 block B[:, k0:k0+2].
    # For the last column (k=8), use schoolbook.
    
    # Reset and redo
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    # 4 Strassen blocks for k-pairs (0,1), (2,3), (4,5), (6,7)
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
            for u_idx, coeff in u_coeffs:
                U[u_idx][col] += coeff
            for v_off, coeff in v_coeffs:
                # v_off=0 -> j=0, v_off=1 -> j=1
                # For k0: V[0*p + k0] = V[k0]
                # For k1: V[1*p + k1] = V[9 + k1]
                # But Strassen uses B00, B01, B10, B11 where
                # B00 = B[0,k0], B01 = B[0,k1], B10 = B[1,k0], B11 = B[1,k1]
                # v_off=0 means first j (j=0), v_off=1 means second j (j=1)
                # But we need to map to both k0 and k1
                # This is getting confusing. Let me use explicit indices.
                pass
            col += 1

    # This is getting complicated. Let me just use the correct explicit approach.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1
        # Strassen for B[:, k0:k1+1] which is a 2x2 matrix:
        # B00 = B[0,k0], B01 = B[0,k1], B10 = B[1,k0], B11 = B[1,k1]
        # V indices: B[0,k0]=V[k0], B[0,k1]=V[k1], B[1,k0]=V[9+k0], B[1,k1]=V[9+k1]
        # W indices: C[0,k0]=W[2*k0], C[0,k1]=W[2*k1], C[1,k0]=W[2*k0+1], C[1,k1]=W[2*k1+1]
        
        # Standard Strassen (7 products):
        # P0 = (A00+A11)*(B00+B11)
        # P1 = (A10+A11)*B00
        # P2 = A00*(B10-B00)
        # P3 = A11*(B01-B11)
        # P4 = (A00+A01)*B11
        # P5 = (A10-A00)*(B00+B01)
        # P6 = (A01-A11)*(B10+B11)
        # C00 = P0 - P3 + P6
        # C01 = P3 + P4
        # C10 = P1 + P2
        # C11 = P0 - P2 - P5 + P6
        
        terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # P0
            ([(0,1),(3,1)], [(k0,1),(9+k1,1)], [(2*k0,1),(2*k1,1),(2*k0+1,-1),(2*k1+1,1)]),
            # P1
            ([(2,1),(3,1)], [(k0,1)], [(2*k0+1,1),(2*k1+1,-1)]),
            # P2
            ([(0,1)], [(9+k0,1),(k0,-1)], [(2*k0,-1),(2*k0+1,1)]),
            # P3
            ([(3,1)], [(k1,1),(9+k1,-1)], [(2*k0,1),(2*k0+1,-1)]),
            # P4
            ([(0,1),(1,1)], [(9+k1,1)], [(2*k1,1)]),
            # P5
            ([(2,1),(0,-1)], [(k0,1),(k1,1)], [(2*k1+1,-1)]),
            # P6
            ([(1,1),(3,-1)], [(9+k0,1),(9+k1,1)], [(2*k0,1)]),
        ]
        
        for t in range(7):
            u_c, v_c, w_c = terms[t]
            for u_idx, coeff in u_c:
                U[u_idx][col] += coeff
            for v_idx, coeff in v_c:
                V[v_idx][col] += coeff
            for w_idx, coeff in w_c:
                W[w_idx][col] += coeff
            col += 1

    # Schoolbook for k=8 (single column)
    # C[0,8] = A00*B[0,8] + A01*B[1,8]
    # C[1,8] = A10*B[0,8] + A11*B[1,8]
    k = 8
    # C[0,k] = W[2*k] = W[16]
    # C[1,k] = W[2*k+1] = W[17]
    # B[0,k] = V[k] = V[8]
    # B[1,k] = V[9+k] = V[17]
    
    U[0][col] = 1
    V[8][col] = 1
    W[16][col] = 1
    col += 1

    U[1][col] = 1
    V[17][col] = 1
    W[16][col] = 1
    col += 1

    U[2][col] = 1
    V[8][col] = 1
    W[17][col] = 1
    col += 1

    U[3][col] = 1
    V[17][col] = 1
    W[17][col] = 1

    return U, V, W


def _validate(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes the multiplication tensor."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T_recon = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
    return np.allclose(T, T_recon, atol=1e-10)


def _try_remove_terms(U, V, W, n, m, p):
    """Try to remove terms from the decomposition if they are linearly dependent."""
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT - 10:
        return None

    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Try removing each single term
    for r in range(rank):
        if time.time() - _START_TIME > _TIMEOUT - 5:
            return None
        mask = [i for i in range(rank) if i != r]
        U = Ua[:, mask]
        V = Va[:, mask]
        W = Wa[:, mask]
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.allclose(T, T_recon, atol=1e-10):
            new_rank = rank - 1
            return (
                [[U[i][j] for j in range(new_rank)] for i in range(n * m)],
                [[V[i][j] for j in range(new_rank)] for i in range(m * p)],
                [[W[i][j] for j in range(new_rank)] for i in range(p * n)],
            )
    return None


def construct():
    n, m, p = DIMENSIONS

    # Build rank-32 decomposition
    U, V, W = _build_rank_32(n, m, p)

    # Validate
    if _validate(U, V, W, n, m, p):
        # Try to reduce rank
        reduced = _try_remove_terms(U, V, W, n, m, p)
        if reduced is not None:
            U2, V2, W2 = reduced
            if _validate(U2, V2, W2, n, m, p):
                return {"U": U2, "V": V2, "W": W2}
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 36
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
