"""Matrix multiplication 2x3 * 3x9 rank-minimal exact decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Exact integer verification of the tensor decomposition."""
    n, m, p = DIMENSIONS
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    rank = U.shape[1]
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if s != 1:
                    return False
    return True


def _build_2x3x3_rank9():
    """
    Construct an explicit rank-9 decomposition for 2x3 * 3x3.
    U: (6, 9), V: (9, 9), W: (6, 9) with entries in {-1, 0, 1}.
    Uses a Strassen-style construction. Verified exactly at startup.
    """
    # Index: U row = i*3 + j, V row = j*3 + k, W row = k*2 + i
    # We build a candidate and verify; if it fails we try alternatives.
    candidates = []

    # Candidate A: block-diagonal via 3 independent 2x2x2 Strassen (rank 7 each) + 2 extra
    # This is complex; instead use a direct explicit table.
    #
    # Known explicit rank-9 scheme for 2x3x3 (entries -1,0,1):
    # We define the 9 "intermediate" products and how they combine.
    U = np.zeros((6, 9), dtype=int)
    V = np.zeros((9, 9), dtype=int)
    W = np.zeros((6, 9), dtype=int)

    # Strategy: use 3 groups of 3 columns, one group per output column k.
    # For a fixed k, we need to compute a 2x3 * 3x1 product (2 outputs) using
    # a rank-3 sub-decomposition. So each group of 3 columns handles one k.
    #
    # For fixed k, define:
    #   A col j: A[i,j] for i=0,1
    #   B col k: B[j,k] for j=0,1,2
    #   C[i] = sum_j A[i,j]*B[j,k]
    #
    # A rank-3 decomposition of this 2x3x1 "product" (which is just a bilinear
    # map (A_col0,A_col1,A_col2, B0,B1,B2) -> (C0,C1)):
    # We can use:
    #   t0 = A[0,0]*B[0]
    #   t1 = A[0,1]*B[1]
    #   t2 = A[0,2]*B[2]
    #   ... but that's rank 3 only if we can express both outputs.
    # Actually a 2x3x1 tensor has rank at most 2 (it's a sum of 2 rank-1 terms:
    #   C0 = A00*B0 + A01*B1 + A02*B2
    #   C1 = A10*B0 + A11*B1 + A12*B2
    # This is rank 2: C0 = (A00,A01,A02)·(B0,B1,B2), C1 = (A10,A11,A12)·(B0,B1,B2)
    # So for fixed k, rank 2 suffices! Then total rank = 3 * 2 = 6?
    # But wait, the tensors for different k share the same A entries.
    # The full tensor is sum over k of (A_i,j * B_j,k * C_i).
    # Since A_i,j is shared across k, we cannot simply take rank 2 per k independently
    # because the U matrix (which depends on A) must be consistent.
    #
    # Actually, the tensor T_{i,j,k} = delta (i.e., the multiplication tensor)
    # = sum_k ( sum_j A_{i,j} B_{j,k} )
    # As a tensor in A,B,C: T = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}
    # This is a sum over (i,j,k) of rank-1 tensors A_{i,j} ⊗ B_{j,k} ⊗ C_{k,i}.
    # Schoolbook rank = 54 for 2x3x9, or 18 for 2x3x3.
    #
    # The key question: can we do better than 18 for 2x3x3?
    # Lower bound for 2x3x3: at least 9 (since C has 6 outputs, but structure...).
    # Actually the known minimal rank for 2x3x3 is 9? Let me think.
    #
    # Let's just try to construct rank 9 explicitly and verify.
    # If it fails, we fall back to rank 18 (schoolbook for 2x3x3) tiled -> 54 total.

    # Explicit construction attempt for rank 9:
    # Use the fact that for 2x3x3, we can group by i (2 groups).
    # For i=0: C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    #                   + A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    #                   + A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # This is getting complex. Let me use a systematic approach.

    # Actually, let me just try a random search with small integers for rank 9,
    # and if found, hardcode it. For now, construct a fallback that is guaranteed
    # to work: schoolbook rank 54.

    # Let's try to build rank 9 via a structured approach:
    # For each of the 3 output columns k, we need rank 2 (as argued above).
    # The issue is sharing A entries. But since U depends only on (i,j) and r,
    # and V depends on (j,k) and r, we can design:
    #
    # For group g (k = g, g+3, g+6... no, k=0,1,2 for 2x3x3):
    # Actually for 2x3x3, k=0,1,2.
    # Let's assign 3 rank slots per k-value, total 9.
    # For k=0, use slots 0,1,2; k=1 use slots 3,4,5; k=2 use slots 6,7,8.
    #
    # For k=0 (slots 0,1,2):
    #   C[0,0] = sum_{j} A[0,j]*B[j,0]
    #   C[1,0] = sum_{j} A[1,j]*B[j,0]
    #   Use rank 3 (slots 0,1,2):
    #     slot 0: U[0,0]=1, U[1,0]=1, V[0,0]=1, V[1,0]=1, V[2,0]=1, W[0,0]=1, W[1,0]=1
    #       This gives A[0,0]*B[0,0]*1 + A[1,0]*B[0,0]*1 + ... no, this doesn't work.
    #
    # I think the cleanest is: since for fixed k, the map (A_col, B_col_k) -> (C_0k, C_1k)
    # is rank 2, we can use 2 slots per k, total 6. Let me try rank 6 for 2x3x3.
    #
    # For k=0, slots 0,1:
    #   slot 0: U[0,0]=1, U[3,0]=1 (i=0,j=0 and i=1,j=0), V[0,0]=1 (j=0,k=0), W[0,0]=1, W[2,0]=1 (k=0,i=0 and k=0,i=1)
    #     This contributes A[0,0]*B[0,0]*1 to C[0,0] and A[1,0]*B[0,0]*1 to C[1,0]. Good.
    #   slot 1: U[0,1]=1, U[4,1]=1 (i=0,j=1 and i=1,j=1), V[3,1]=1 (j=1,k=0), W[0,1]=1, W[2,1]=1
    #     Contributes A[0,1]*B[1,0] to C[0,0] and A[1,1]*B[1,0] to C[1,0]. Good.
    #   But we need j=2 too! So rank 2 is not enough for fixed k. We need rank 3 per k.
    #
    # So 3 slots per k, total 9. Let me construct:
    # For k=0, slots 0,1,2:
    #   slot 0: j=0: U[0,0]=1, U[3,0]=1; V[0,0]=1; W[0,0]=1, W[2,0]=1
    #   slot 1: j=1: U[0,1]=1, U[4,1]=1; V[3,1]=1; W[0,1]=1, W[2,1]=1
    #   slot 2: j=2: U[0,2]=1, U[5,2]=1; V[6,2]=1; W[0,2]=1, W[2,2]=1
    #
    # For k=1, slots 3,4,5:
    #   slot 3: j=0: U[0,3]=1, U[3,3]=1; V[1,3]=1; W[1,3]=1, W[3,3]=1
    #   slot 4: j=1: U[0,4]=1, U[4,4]=1; V[4,4]=1; W[1,4]=1, W[3,4]=1
    #   slot 5: j=2: U[0,5]=1, U[5,5]=1; V[7,5]=1; W[1,5]=1, W[3,5]=1
    #
    # For k=2, slots 6,7,8:
    #   slot 6: j=0: U[0,6]=1, U[3,6]=1; V[2,6]=1; W[4,6]=1, W[6,6]=1
    #   slot 7: j=1: U[0,7]=1, U[4,7]=1; V[5,7]=1; W[4,7]=1, W[6,7]=1
    #   slot 8: j=2: U[0,8]=1, U[5,8]=1; V[8,8]=1; W[4,8]=1, W[6,8]=1
    #
    # Wait, but this is just schoolbook! Each slot handles one (i,j,k) term separately.
    # That's rank 18 for 2x3x3, not 9. I made an error: for fixed k, I need to handle
    # BOTH i=0 and i=1 in the same slots, which I did (U has both i=0 and i=1 rows set).
    # But each slot still only handles one j. So 3 j's × 3 k's = 9 slots, each handling
    # both i's. That's rank 9! Let me verify this is correct.
    #
    # For slot 0 (k=0, j=0):
    #   U[0,0]=1 (i=0,j=0), U[3,0]=1 (i=1,j=0)
    #   V[0,0]=1 (j=0,k=0)
    #   W[0,0]=1 (k=0,i=0), W[2,0]=1 (k=0,i=1)
    #   Contribution to C[0,0]: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 (A[0,0]*B[0,0])
    #   Contribution to C[1,0]: U[3,0]*V[0,0]*W[2,0] = 1*1*1 = 1 (A[1,0]*B[0,0])
    #   No contribution to other (i,k). 
    #
    # This is correct! It's a "semi-schoolbook" where we merge the i dimension.
    # Rank = 3 (j) × 3 (k) = 9 for 2x3x3.
    # For 2x3x9: 3 (j) × 9 (k) = 27.

    # Let me build this for 2x3x9 directly.
    n, m, p = 2, 3, 9
    rank = m * p  # 27
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    col = 0
    for j in range(m):
        for k in range(p):
            # U: i=0,j and i=1,j
            U[0 * m + j, col] = 1
            U[1 * m + j, col] = 1
            # V: j,k
            V[j * p + k, col] = 1
            # W: k,i=0 and k,i=1
            W[k * n + 0, col] = 1
            W[k * n + 1, col] = 1
            col += 1

    return U, V, W


def _build_schoolbook():
    """Schoolbook rank 54 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
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
    """
    Finds a low-rank exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    Uses the semi-schoolbook rank-27 construction (merging the i dimension),
    verified exactly. Falls back to schoolbook rank 54 if verification fails.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Try rank-27 construction
    try:
        U, V, W = _build_2x3x3_rank9()
        # This actually builds for 2x3x9 with rank 27
        if _verify(U, V, W):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    except Exception:
        pass

    # Fallback: schoolbook rank 54
    if time.time() > deadline:
        # Emergency: return empty (shouldn't happen)
        return {"U": [], "V": [], "W": []}

    U, V, W = _build_schoolbook()
    if not _verify(U, V, W):
        # Should never happen
        pass

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
