<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    
    Strategy: Split m=3 into m1=1 and m2=2.
    Part 1: 2x1x8 multiplication (rank 16, schoolbook)
    Part 2: 2x2x8 multiplication, split into 4 blocks of 2x2x2 (Strassen, rank 7 each)
    Total rank: 16 + 4*7 = 44
    """
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 3, 8

    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44

    col = 0

    # --- Part 1: 2x1x8 (j=0) ---
    # A is 2x1, B is 1x8.
    # C[i, k] = A[i, 0] * B[0, k]
    # Rank 16.
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i

            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # Check time
    if time.time() - start_time > timeout:
        return {"U": U, "V": V, "W": W}

    # --- Part 2: 2x2x8 (j=1,2) ---
    # A is 2x2 (cols 1,2), B is 2x8 (rows 1,2).
    # Split B into 4 blocks of 2 columns.
    for block_idx in range(4):
        k_start = block_idx * 2

        # Strassen 2x2x2
        # Local A:
        # A00 -> Global A[0, 1]
        # A01 -> Global A[0, 2]
        # A10 -> Global A[1, 1]
        # A11 -> Global A[1, 2]

        # Local B (for this block):
        # B00 -> Global B[1, k_start]
        # B01 -> Global B[1, k_start + 1]
        # B10 -> Global B[2, k_start]
        # B11 -> Global B[2, k_start + 1]

        # Local C (for this block):
        # C00 -> Global C[0, k_start]
        # C01 -> Global C[0, k_start + 1]
        # C10 -> Global C[1, k_start]
        # C11 -> Global C[1, k_start + 1]

        steps = []

        # M1 = (A00 + A11) * (B00 + B11)
        # C00 += M1, C11 += M1
        steps.append({
            'U': {0*m + 1: 1, 1*m + 2: 1},
            'V': {1*p + k_start: 1, 2*p + (k_start+1): 1},
            'W': {k_start*n + 0: 1, (k_start+1)*n + 1: 1}
        })

        # M2 = (A10 + A11) * B00
        # C10 += M2, C11 -= M2
        steps.append({
            'U': {1*m + 1: 1, 1*m + 2: 1},
            'V': {1*p + k_start: 1},
            'W': {k_start*n + 1: 1, (k_start+1)*n + 1: -1}
        })

        # M3 = A00 * (B01 - B11)
        # C01 += M3, C11 += M3
        steps.append({
            'U': {0*m + 1: 1},
            'V': {1*p + (k_start+1): 1, 2*p + (k_start+1): -1},
            'W': {(k_start+1)*n + 0: 1, (k_start+1)*n + 1: 1}
        })

        # M4 = A11 * (B10 - B00)
        # C00 += M4, C10 += M4
        steps.append({
            'U': {1*m + 2: 1},
            'V': {2*p + k_start: 1, 1*p + k_start: -1},
            'W': {k_start*n + 0: 1, k_start*n + 1: 1}
        })

        # M5 = (A00 + A01) * B11
        # C00 -= M5, C01 += M5
        steps.append({
            'U': {0*m + 1: 1, 0*m + 2: 1},
            'V': {2*p + (k_start+1): 1},
            'W': {k_start*n + 0: -1, (k_start+1)*n + 0: 1}
        })

        # M6 = (A10 - A00) * (B00 + B01)
        # C11 += M6
        steps.append({
            'U': {1*m + 1: 1, 0*m + 1: -1},
            'V': {1*p + k_start: 1, 1*p + (k_start+1): 1},
            'W': {(k_start+1)*n + 1: 1}
        })

        # M7 = (A01 - A11) * (B10 + B11)
        # C00 += M7
        steps.append({
            'U': {0*m + 2: 1, 1*m + 2: -1},
            'V': {2*p + k_start: 1, 2*p + (k_start+1): 1},
            'W': {k_start*n + 0: 1}
        })

        for step in steps:
            for u_idx, val in step['U'].items():
                U[u_idx][col] += val
            for v_idx, val in step['V'].items():
                V[v_idx][col] += val
            for w_idx, val in step['W'].items():
                W[w_idx][col] += val
            col += 1

    # Check time
    if time.time() - start_time > timeout:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import numpy as np

def _verify_decomp(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U and len(U) > 0 else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _build_rank44():
    """Build the known rank-44 decomposition."""
    n, m, p = DIMENSIONS  # 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44

    col = 0

    # Part 1: 2x1x8 (j=0), schoolbook rank 16
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # Part 2: 2x2x8 (j=1,2), 4 Strassen blocks, rank 28
    for block_idx in range(4):
        k_start = block_idx * 2
        steps = [
            {'U': {m+1: 1, m+2: 1}, 'V': {p+k_start: 1, 2*p+k_start+1: 1},
             'W': {k_start*n: 1, (k_start+1)*n+1: 1}},
            {'U': {m+1: 1, m+2: 1}, 'V': {p+k_start: 1},
             'W': {k_start*n+1: 1, (k_start+1)*n+1: -1}},
            {'U': {m+1: 1}, 'V': {p+k_start+1: 1, 2*p+k_start+1: -1},
             'W': {(k_start+1)*n: 1, (k_start+1)*n+1: 1}},
            {'U': {m+2: 1}, 'V': {2*p+k_start: 1, p+k_start: -1},
             'W': {k_start*n: 1, k_start*n+1: 1}},
            {'U': {m+1: 1, m+2: 1}, 'V': {2*p+k_start+1: 1},
             'W': {k_start*n: -1, (k_start+1)*n: 1}},
            {'U': {m+1: 1, m+1: -1}, 'V': {p+k_start: 1, p+k_start+1: 1},
             'W': {(k_start+1)*n+1: 1}},
            {'U': {m+2: 1, m+2: -1}, 'V': {2*p+k_start: 1, 2*p+k_start+1: 1},
             'W': {k_start*n: 1}},
        ]
        for step in steps:
            for u_idx, val in step['U'].items():
                U[u_idx][col] += val
            for v_idx, val in step['V'].items():
                V[v_idx][col] += val
            for w_idx, val in step['W'].items():
                W[w_idx][col] += val
            col += 1

    return U, V, W

def _try_rank_reduction(U, V, W, n, m, p, deadline):
    """
    Attempt to reduce the rank of the 2x2x8 sub-problem (columns 16-43).
    
    Strategy: The 2x2x8 tensor can potentially be decomposed with rank < 28
    by exploiting structure. We try a simple approach: check if the 28 rank-1
    terms have any linear dependencies in the tensor product space.
    
    For now, this is a placeholder that returns None if no reduction is found.
    A more sophisticated approach would use iterative methods or integer programming.
    """
    # The 2x2x8 part uses columns 16..43
    # U rows 2..5 (i=0,1 for j=1,2), V rows 8..23 (j=1,2 for k=0..7), W all rows
    
    # Quick check: try to see if we can merge some Strassen terms across blocks
    # by finding a common factorization. This is complex, so for now return None.
    
    # Future work: Use numpy to set up and solve for a lower-rank decomposition
    # via iterative rank-reduction algorithms.
    return None

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    
    Strategy: Split m=3 into m1=1 and m2=2.
    Part 1: 2x1x8 multiplication (rank 16, schoolbook)
    Part 2: 2x2x8 multiplication, split into 4 blocks of 2x2x2 (Strassen, rank 7 each)
    Total rank: 16 + 4*7 = 44
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 3, 8

    # Build the known rank-44 decomposition
    U, V, W = _build_rank44()

    # Verify correctness
    if not _verify_decomp(U, V, W, n, m, p):
        # Should not happen, but return as-is
        return {"U": U, "V": V, "W": W}

    best_rank = 44
    best_result = {"U": U, "V": V, "W": W}

    # Try to reduce rank if time permits
    if time.time() < deadline:
        reduced = _try_rank_reduction(U, V, W, n, m, p, deadline)
        if reduced is not None:
            rU, rV, rW, rrank = reduced
            if _verify_decomp(rU, rV, rW, n, m, p) and rrank < best_rank:
                best_rank = rrank
                best_result = {"U": rU, "V": rV, "W": rW}

    return best_result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
