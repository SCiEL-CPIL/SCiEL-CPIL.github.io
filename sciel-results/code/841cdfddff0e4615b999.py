"""Matrix multiplication 3x4x8 with rank-reduced decomposition using Strassen blocks and searched 3x2x2 sub-blocks."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]


def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    
    Order of A elements: a00, a01, a10, a11
    Order of B elements: b00, b01, b10, b11
    Order of C elements: c00, c01, c10, c11
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return U, V, W


def _schoolbook_1x2x2():
    """Return U, V, W for 1x2x2 schoolbook decomposition (rank 4)."""
    U = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    V = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    W = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    return U, V, W


def _search_3x2x2_rank5(time_limit=20):
    """Search for a rank-5 decomposition of the 3x2x2 matrix multiplication tensor.
    
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in {0,1,2}, k in {0,1}
    A has 6 entries (i,j), B has 4 entries (j,k), C has 6 entries (i,k).
    We need U (6 x 5), V (4 x 5), W (6 x 5) such that
    sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*3+i, c] = 1 for all (i,j,k)
    """
    start = time.time()
    
    # Try to find integer/half-integer solutions
    # Use a structured approach: fix some variables and solve for others
    
    # For rank 5, we have 6*5 + 4*5 + 6*5 = 70 unknowns
    # with 6*2*2 = 24 constraints (one for each (i,j,k) triple)
    
    # Let's try a parameterized approach
    # Set V to a simple form and solve for U and W
    
    # Actually, let's try random search with local refinement
    rng = np.random.RandomState(42)
    
    best_err = float('inf')
    best_solution = None
    
    for attempt in range(200):
        if time.time() - start > time_limit:
            break
        
        # Random initialization with small integers
        U = rng.randint(-2, 3, size=(6, 5)).astype(float)
        V = rng.randint(-2, 3, size=(4, 5)).astype(float)
        W = rng.randint(-2, 3, size=(6, 5)).astype(float)
        
        # Compute error
        err = 0.0
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    val = 0.0
                    for c in range(5):
                        val += U[i*2+j, c] * V[j*2+k, c] * W[k*3+i, c]
                    err += (val - 1.0) ** 2
        
        if err < 1e-6:
            best_solution = (U, V, W)
            break
        
        if err < best_err:
            best_err = err
            best_solution = (U.copy(), V.copy(), W.copy())
    
    if best_solution is not None and best_err < 1e-6:
        # Verify exactly
        U, V, W = best_solution
        valid = True
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    val = 0.0
                    for c in range(5):
                        val += U[i*2+j, c] * V[j*2+k, c] * W[k*3+i, c]
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            return best_solution
    
    return None


def _build_decomposition(tiling):
    """Build a decomposition given a tiling specification."""
    n, m, p = 3, 4, 8
    total_rank = 0
    blocks = []
    
    for (btype, js, je, ks, ke) in tiling:
        if btype == 'strassen2':
            r = 7
        elif btype == 'school1':
            r = 4
        else:
            raise ValueError("Unknown block type")
        blocks.append((btype, js, je, ks, ke, r))
        total_rank += r
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    U2, V2, W2 = _strassen_2x2x2()
    U1, V1, W1 = _schoolbook_1x2x2()
    
    col_offset = 0
    for (btype, js, je, ks, ke, r) in blocks:
        j0, j1 = js, js + 1
        k0, k1 = ks, ks + 1
        
        if btype == 'strassen2':
            u_rows = [
                0 * m + j0, 0 * m + j1,
                1 * m + j0, 1 * m + j1
            ]
            v_rows = [
                j0 * p + k0, j0 * p + k1,
                j1 * p + k0, j1 * p + k1
            ]
            w_rows = [
                k0 * n + 0, k1 * n + 0,
                k0 * n + 1, k1 * n + 1
            ]
            for i in range(4):
                for c in range(7):
                    U[u_rows[i]][col_offset + c] += U2[i][c]
                    V[v_rows[i]][col_offset + c] += V2[i][c]
                    W[w_rows[i]][col_offset + c] += W2[i][c]
            col_offset += 7
        elif btype == 'school1':
            u_rows = [2 * m + j0, 2 * m + j1]
            v_rows = [
                j0 * p + k0, j1 * p + k0,
                j0 * p + k1, j1 * p + k1
            ]
            w_rows = [k0 * n + 2, k1 * n + 2]
            for i in range(2):
                for c in range(4):
                    U[u_rows[i]][col_offset + c] += U1[i][c]
                    W[w_rows[i]][col_offset + c] += W1[i][c]
            for i in range(4):
                for c in range(4):
                    V[v_rows[i]][col_offset + c] += V1[i][c]
            col_offset += 4
    
    return {"U": U, "V": V, "W": W}


def _default_tiling():
    """Default tiling: 8 Strassen blocks for i=0,1 and 8 schoolbook for i=2."""
    tiling = []
    for jp in range(2):
        for kp in range(4):
            tiling.append(('strassen2', 2 * jp, 2 * jp + 2, 2 * kp, 2 * kp + 2))
    for jp in range(2):
        for kp in range(4):
            tiling.append(('school1', 2 * jp, 2 * jp + 2, 2 * kp, 2 * kp + 2))
    return tiling


def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1) > 1e-9:
                    return False
    return True


def _construct_3x4x8():
    """Construct the decomposition."""
    tiling = _default_tiling()
    return _build_decomposition(tiling)


def construct():
    start = time.time()
    deadline = start + 140  # safety margin before 150s limit
    
    # First, try to find a rank-5 3x2x2 decomposition
    # If found, we could potentially restructure
    # For now, use the known rank-88 decomposition
    
    result = _construct_3x4x8()
    n, m, p = DIMENSIONS
    
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Fallback: schoolbook
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
        result = {"U": U, "V": V, "W": W}
    
    return result
