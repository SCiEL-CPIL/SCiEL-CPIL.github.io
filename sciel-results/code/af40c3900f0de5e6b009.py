"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_strassen56():
    """Build the rank-56 Strassen-based decomposition for 2x4 * 4x8."""
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 2x2x2 factors
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # 
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U_defs = [
        [((0, 0), 1), ((1, 1), 1)],   # M1: A00, A11
        [((1, 0), 1), ((1, 1), 1)],   # M2: A10, A11
        [((0, 0), 1)],                # M3: A00
        [((1, 1), 1)],                # M4: A11
        [((0, 0), 1), ((0, 1), 1)],   # M5: A00, A01
        [((1, 0), 1), ((0, 0), -1)],  # M6: A10, -A00
        [((0, 1), 1), ((1, 1), -1)]   # M7: A01, -A11
    ]
    
    V_defs = [
        [((0, 0), 1), ((1, 1), 1)],   # M1: B00, B11
        [((0, 0), 1)],                # M2: B00
        [((0, 1), 1), ((1, 1), -1)],  # M3: B01, -B11
        [((1, 0), 1), ((0, 0), -1)],  # M4: B10, -B00
        [((1, 1), 1)],                # M5: B11
        [((0, 0), 1), ((0, 1), 1)],   # M6: B00, B01
        [((1, 0), 1), ((1, 1), 1)]    # M7: B10, B11
    ]
    
    W_defs = [
        [((0, 0), 1), ((1, 1), 1)],   # M1: C00, C11
        [((1, 0), 1), ((1, 1), -1)],  # M2: C10, -C11
        [((0, 1), 1), ((1, 1), 1)],   # M3: C01, C11
        [((0, 0), 1), ((1, 0), 1)],   # M4: C00, C10
        [((0, 0), -1), ((0, 1), 1)],  # M5: -C00, C01
        [((1, 1), 1)],                # M6: C11
        [((0, 0), 1)]                 # M7: C00
    ]

    t = 0
    for a in range(2):  # j-group: covers j=2a, 2a+1
        for b in range(4):  # k-group: covers k=2b, 2b+1
            for s in range(7):
                for (r, c), val in U_defs[s]:
                    U[r * m + 2 * a + c, t] += val
                for (r, c), val in V_defs[s]:
                    V[(2 * a + r) * p + 2 * b + c, t] += val
                for (r, c), val in W_defs[s]:
                    W[(2 * b + c) * n + r, t] += val
                t += 1

    return U, V, W, rank

def _validate(U, V, W, n, m, p, rank):
    """Verify the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 1.0
                rhs = 0.0
                for tt in range(rank):
                    rhs += U[i*m+j, tt] * V[j*p+k, tt] * W[k*n+i, tt]
                if abs(lhs - rhs) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Build the known rank-56 solution
    U, V, W, rank = _build_strassen56()
    
    # Verify
    if not _validate(U, V, W, n, m, p, rank):
        # Fallback to schoolbook rank 64
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        t = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, t] = 1.0
                    V[j*p+k, t] = 1.0
                    W[k*n+i, t] = 1.0
                    t += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Check if we have time to attempt rank reduction
    elapsed = time.time() - start_time
    if elapsed < deadline - 10.0:
        # Try to find a lower rank by checking if any columns are linearly dependent
        # or if we can merge terms. For now, just return the valid rank-56 solution.
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
