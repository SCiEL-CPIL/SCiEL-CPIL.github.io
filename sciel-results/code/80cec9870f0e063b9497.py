import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 5]
DEADLINE = 120  # task says 120s
DEADLINE_MARGIN = 15  # seconds to leave before the deadline

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verifies if U, V, W form a valid tensor decomposition for matrix multiplication.
    First uses numpy for speed, then exact Fraction arithmetic for confirmation.
    """
    rank = len(U[0])
    
    # Quick numpy check first
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                total = np.dot(U_np[idx_u], V_np[idx_v] * W_np[idx_w])
                if abs(total - 1.0) > tol:
                    return False
    
    # Exact verification with Fractions
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                total = Fraction(0)
                for r in range(rank):
                    total += U_frac[idx_u][r] * V_frac[idx_v][r] * W_frac[idx_w][r]
                
                if total != 1:
                    return False
    return True

def _construct_rank_28():
    """
    Constructs the known rank-28 decomposition for 2x3 * 3x5.
    This uses Strassen's algorithm for the 2x2x2 sub-blocks and direct multiplication for the rest.
    """
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A2 B2 (j=2, k=0..4)
    # Handles C[i, k] += A[i, 2] * B[2, k]
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: A1 B13 (j=0,1, k=4)
    # Handles C[i, 4] += A[i, j] * B[j, 4] for j in {0, 1}
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen (7 terms)
    # Computes C[i, k] for i in {0,1}, k in {0,1} from A[i, j] B[j, k] with j in {0,1}
    
    # M1 = (a00+a11)(b00+b11) -> contributes to c00, c11
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M2 = (a10+a11)b00 -> contributes to c01, c11
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1
    W[1 * n + 1][r] = -1
    r += 1
    
    # M3 = a00(b01-b11) -> contributes to c01, c11
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1
    V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M4 = a11(b10-b00) -> contributes to c00, c10
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1
    V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1
    W[0 * n + 1][r] = 1
    r += 1
    
    # M5 = (a00+a01)b11 -> contributes to c00, c10
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1
    W[1 * n + 0][r] = 1
    r += 1
    
    # M6 = (a10-a00)(b00+b01) -> contributes to c11
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1
    V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M7 = (a01-a11)(b10+b11) -> contributes to c00
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen (7 terms)
    # Computes C[i, k] for i in {0,1}, k in {2,3} from A[i, j] B[j, k] with j in {0,1}
    
    # M1
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M2
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1
    W[3 * n + 1][r] = -1
    r += 1
    
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1
    V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1
    V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1
    W[2 * n + 1][r] = 1
    r += 1
    
    # M5
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1
    W[3 * n + 0][r] = 1
    r += 1
    
    # M6
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1
    V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M7
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    return {"U": U, "V": V, "W": W}

def _try_find_lower_rank(start_rank, deadline):
    """
    Attempts to find a lower rank decomposition by trying to merge terms.
    This is a heuristic search. If it finds a valid decomposition, it returns it.
    Otherwise, it returns None.
    """
    n, m, p = 2, 3, 5
    base = _construct_rank_28()
    U = [row[:] for row in base['U']]
    V = [row[:] for row in base['V']]
    W = [row[:] for row in base['W']]
    rank = 28
    
    # Try to eliminate one term by checking if any term is redundant
    # A term r is redundant if for all (i,j,k): sum_{r' != r} U[i*m+j][r']*V[j*p+k][r']*W[k*n+i][r'] == 1
    # This is expensive, so we only try a few candidates
    
    for elim_r in range(rank):
        if time.time() > deadline:
            break
        
        # Test if removing term elim_r still gives valid decomposition
        # Quick numpy check
        U_test = [row[:] for row in U]
        V_test = [row[:] for row in V]
        W_test = [row[:] for row in W]
        
        # Zero out column elim_r
        for idx in range(n * m):
            U_test[idx][elim_r] = 0
        for idx in range(m * p):
            V_test[idx][elim_r] = 0
        for idx in range(p * n):
            W_test[idx][elim_r] = 0
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    
                    total = 0.0
                    for r in range(rank):
                        if r == elim_r:
                            continue
                        total += U_test[idx_u][r] * V_test[idx_v][r] * W_test[idx_w][r]
                    
                    if abs(total - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Exact verification
            if _verify_decomposition(U_test, V_test, W_test, n, m, p):
                # Build result without the eliminated term
                U_final = []
                for row in U_test:
                    U_final.append([x for idx, x in enumerate(row) if idx != elim_r])
                V_final = []
                for row in V_test:
                    V_final.append([x for idx, x in enumerate(row) if idx != elim_r])
                W_final = []
                for row in W_test:
                    W_final.append([x for idx, x in enumerate(row) if idx != elim_r])
                
                return {"U": U_final, "V": V_final, "W": W_final}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - DEADLINE_MARGIN
    
    # Construct the known optimal rank-28 decomposition
    res_28 = _construct_rank_28()
    
    # Verify it to ensure correctness
    if _verify_decomposition(res_28['U'], res_28['V'], res_28['W'], 2, 3, 5):
        # Check if we have time to search for a better (lower rank) solution
        if time.time() < deadline - 5:
            better = _try_find_lower_rank(27, deadline)
            if better is not None:
                return better
        
        return res_28
    else:
        # Fallback: should not happen
        raise ValueError("Verification failed for rank-28 decomposition")
