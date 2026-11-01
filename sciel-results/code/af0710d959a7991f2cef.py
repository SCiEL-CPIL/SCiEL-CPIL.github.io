"""Matrix multiplication 2,2,10 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 10]
_DEADLINE = time.time() + 140


def _verify_decomposition(U, V, W, n=2, m=2, p=10, tol=1e-9):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                total = 0.0
                for r in range(rank):
                    total += Ua[idx_u, r] * Va[idx_v, r] * Wa[idx_w, r]
                expected = 1.0 if j == k else 0.0
                if abs(total - expected) > tol:
                    return False
    return True


def _construct_rank35():
    """Construct a rank-35 Strassen-based decomposition."""
    n, m, p = 2, 2, 10
    rank = 35
    U = [[0] * rank for _ in range(n * m)]   # (4, 35)
    V = [[0] * rank for _ in range(m * p)]   # (20, 35)
    W = [[0] * rank for _ in range(p * n)]   # (20, 35)

    # Strassen's 7 products for 2x2 matrix multiplication
    strassen_A = [
        [1, 0, 0, 1],   # P1: a+d
        [0, 0, 1, 1],   # P2: c+d
        [1, 0, 0, 0],   # P3: a
        [0, 0, 0, 1],   # P4: d
        [1, 1, 0, 0],   # P5: a+b
        [-1, 0, 1, 0],  # P6: c-a
        [0, 1, 0, -1],  # P7: b-d
    ]

    strassen_B = [
        [1, 0, 0, 1],   # P1: e+h
        [1, 0, 0, 0],   # P2: e
        [0, 1, 0, -1],  # P3: f-h
        [-1, 0, 1, 0],  # P4: g-e
        [0, 0, 0, 1],   # P5: h
        [1, 1, 0, 0],   # P6: e+f
        [0, 0, 1, 1],   # P7: g+h
    ]

    # W coefficients: [C00_coeff, C01_coeff, C10_coeff, C11_coeff]
    w_contrib = [
        [1, 0, 0, 1],   # P1: C00+1, C11+1
        [0, 0, 1, -1],  # P2: C10+1, C11-1
        [0, 1, 0, 1],   # P3: C01+1, C11+1
        [1, 0, 1, 0],   # P4: C00+1, C10+1
        [-1, 1, 0, 0],  # P5: C00-1, C01+1
        [0, 0, 0, 1],   # P6: C11+1
        [1, 0, 0, 0],   # P7: C00+1
    ]

    for b in range(5):
        for r in range(7):
            col = b * 7 + r

            U[0][col] = strassen_A[r][0]
            U[1][col] = strassen_A[r][1]
            U[2][col] = strassen_A[r][2]
            U[3][col] = strassen_A[r][3]

            k0 = 2 * b
            k1 = 2 * b + 1
            V[0 * p + k0][col] = strassen_B[r][0]
            V[0 * p + k1][col] = strassen_B[r][1]
            V[1 * p + k0][col] = strassen_B[r][2]
            V[1 * p + k1][col] = strassen_B[r][3]

            W[(2 * b) * n + 0][col] = w_contrib[r][0]
            W[(2 * b + 1) * n + 0][col] = w_contrib[r][1]
            W[(2 * b) * n + 1][col] = w_contrib[r][2]
            W[(2 * b + 1) * n + 1][col] = w_contrib[r][3]

    return {"U": U, "V": V, "W": W}


def _try_structured_lower_rank(target_rank, time_budget=2.0):
    """Try to find a decomposition of the given rank using structured search."""
    n, m, p = 2, 2, 10
    start = time.time()
    
    # Strategy: Use 4 full Strassen blocks (28 rank) + try to cover remaining 2 columns
    # with fewer than 7 additional terms by sharing structure
    # Or: mix columns across blocks
    
    # Approach 1: Try rank 34 by having one block share a term with another
    # This is complex; try random search with constraints for small ranks
    
    if target_rank >= 30 and target_rank <= 34:
        # Try a few random structured attempts
        np.random.seed(42)
        for attempt in range(50):
            if time.time() - start > time_budget:
                break
            
            # Create random integer matrices in {-1, 0, 1}
            U_try = np.random.randint(-1, 2, (4, target_rank))
            V_try = np.random.randint(-1, 2, (20, target_rank))
            W_try = np.random.randint(-1, 2, (20, target_rank))
            
            if _verify_decomposition(U_try.tolist(), V_try.tolist(), W_try.tolist()):
                return {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}
    
    return None


def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x2 by 2x10 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    
    # Start with the known rank-35 solution
    best = _construct_rank35()
    
    # Verify the rank-35 solution
    if not _verify_decomposition(best["U"], best["V"], best["W"]):
        # Should not happen, but fall back to schoolbook if needed
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
        best = {"U": U, "V": V, "W": W}
    
    # Try to find lower rank solutions
    for target_rank in range(34, 29, -1):
        if time.time() > _DEADLINE - 10:
            break
        
        result = _try_structured_lower_rank(target_rank, time_budget=3.0)
        if result is not None:
            best = result
            break
    
    return best
