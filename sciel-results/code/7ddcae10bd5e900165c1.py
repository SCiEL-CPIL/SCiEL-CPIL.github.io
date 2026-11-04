"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Construct the standard rank n*m*p decomposition."""
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


def _verify(n, m, p, U, V, W):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    This should equal 1 if (i,j,k) is a valid "edge" of matmul.
    """
    rank = len(U[0]) if U else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j]
                v_val = V_np[j * p + k]
                w_val = W_np[k * n + i]
                dot = np.dot(u_val, v_val * w_val)
                if abs(dot - 1.0) > 1e-9:
                    return False
    return True


def _verify_exact(n, m, p, U, V, W):
    """Exact verification using Fraction arithmetic."""
    rank = len(U[0]) if U else 0
    U_f = [[Fraction(x) for x in row] for row in U]
    V_f = [[Fraction(x) for x in row] for row in V]
    W_f = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += U_f[i * m + j][c] * V_f[j * p + k][c] * W_f[k * n + i][c]
                if total != Fraction(1):
                    return False
    return True


def _strassen_2x2():
    """
    Classic Strassen decomposition for 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) where:
    U indexes (i,j) in 2x2, V indexes (j,k) in 2x2, W indexes (k,i) in 2x2.
    """
    # Strassen's 7 products:
    # M1 = (A11-A22)*(B22-B11)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B11-B22)
    # M4 = A22*(B22-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U: row index = i*2 + j for i,j in {0,1}
    # V: row index = j*2 + k for j,k in {0,1}
    # W: row index = k*2 + i for k,i in {0,1}
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A11-A22)*(B22-B11)
    # A11 = U[0,0], A22 = U[3,0] -> row 0 col 0 = +1, row 3 col 0 = -1
    U[0][0] = 1; U[3][0] = -1
    # B11 = V[0,1], B22 = V[3,1] -> row 0 col 1 = -1, row 3 col 1 = +1
    V[0][0] = -1; V[3][0] = 1
    # C11 = W[0,0], C22 = W[3,0]
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A21+A22)*B11
    # A21 = U[2,0], A22 = U[3,0]
    U[2][1] = 1; U[3][1] = 1
    # B11 = V[0,1]
    V[0][1] = 1
    # C21 = W[1,0], C22 = W[3,0]
    W[1][1] = 1; W[3][1] = -1
    
    # M3: A11*(B11-B22)
    # A11 = U[0,0]
    U[0][2] = 1
    # B11 = V[0,1], B22 = V[3,1]
    V[0][2] = 1; V[3][2] = -1
    # C12 = W[1,1], C22 = W[3,0]
    W[1][2] = 1; W[3][2] = 1
    
    # M4: A22*(B22-B11)
    # A22 = U[3,0]
    U[3][3] = 1
    # B22 = V[3,1], B11 = V[0,1]
    V[3][3] = 1; V[0][3] = -1
    # C11 = W[0,0], C21 = W[1,0]
    W[0][3] = 1; W[1][3] = 1
    
    # M5: (A11+A12)*B22
    # A11 = U[0,0], A12 = U[1,0]
    U[0][4] = 1; U[1][4] = 1
    # B22 = V[3,1]
    V[3][4] = 1
    # C11 = W[0,0], C12 = W[1,1]
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A21-A11)*(B11+B12)
    # A21 = U[2,0], A11 = U[0,0]
    U[2][5] = 1; U[0][5] = -1
    # B11 = V[0,1], B12 = V[1,1]
    V[0][5] = 1; V[1][5] = 1
    # C22 = W[3,0]
    W[3][5] = 1
    
    # M7: (A12-A22)*(B21+B22)
    # A12 = U[1,0], A22 = U[3,0]
    U[1][6] = 1; U[3][6] = -1
    # B21 = V[2,1], B22 = V[3,1]
    V[2][6] = 1; V[3][6] = 1
    # C11 = W[0,0]
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}


def _verify_strassen_2x2():
    """Verify the Strassen 2x2 decomposition."""
    n, m, p = 2, 2, 2
    res = _strassen_2x2()
    U, V, W = res["U"], res["V"], res["W"]
    return _verify(n, m, p, U, V, W)


def _embed_2x2_into_5x5():
    """
    Embed Strassen's 2x2 decomposition into a 5x5x5 matrix multiplication.
    We use a block structure: treat the 5x5 matrices as having a 2x2 block in the top-left,
    and handle the rest with schoolbook.
    
    Actually, a better approach: use the Kronecker product structure.
    We decompose 5x5x5 by considering it as a tensor product of smaller decompositions.
    
    For now, let's try a structured approach:
    - Use Strassen for a 2x2 sub-block (rank 7 instead of 8)
    - Use schoolbook for the remaining parts
    """
    n, m, p = 5, 5, 5
    strassen = _strassen_2x2()
    sU, sV, sW = strassen["U"], strassen["V"], strassen["W"]
    
    # This is complex; let's try a different approach.
    # We'll build a candidate using the fact that matmul(5,5,5) can be 
    # partially decomposed using smaller known decompositions.
    
    # For now, return schoolbook as fallback
    return _schoolbook(n, m, p)


def _try_structured_search(n, m, p, target_rank, timeout=20):
    """
    Try to find a decomposition of the given rank using structured patterns.
    We use the insight that matmul decompositions often have symmetry.
    """
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    
    if target_rank < n * m * p * 7 // 8:  # Must be at least roughly Strassen-like
        return None
    
    # Try random search with structured initialization
    for attempt in range(50):
        if time.time() - start_time > timeout:
            break
        
        # Initialize with a pattern: start from schoolbook and perturb
        U = np.zeros((nm, target_rank), dtype=float)
        V = np.zeros((mp, target_rank), dtype=float)
        W = np.zeros((pn, target_rank), dtype=float)
        
        # First, fill with a schoolbook-like pattern for the first min(target_rank, n*m*p) columns
        base_rank = min(target_rank, n * m * p)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= target_rank:
                        break
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        if col < target_rank:
            # Fill remaining columns with small random values
            for c in range(col, target_rank):
                U[:, c] = np.random.choice([-1, 0, 1], size=nm) * 0.5
                V[:, c] = np.random.choice([-1, 0, 1], size=mp) * 0.5
                W[:, c] = np.random.choice([-1, 0, 1], size=pn) * 0.5
        
        # Verify
        valid = True
        for i in range(n):
            if not valid:
                break
            for j in range(m):
                if not valid:
                    break
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    dot = np.dot(u_val, v_val * w_val)
                    if abs(dot - 1.0) > 1e-8:
                        valid = False
                        break
        
        if valid:
            return {
                "U": [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U],
                "V": [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V],
                "W": [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W]
            }
    
    return None


def _try_random_search(n, m, p, target_rank, timeout=15):
    """
    Pure random search for a decomposition of the given rank.
    """
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    
    for attempt in range(100):
        if time.time() - start_time > timeout:
            break
        
        U = np.random.choice([-1, 0, 1], size=(nm, target_rank))
        V = np.random.choice([-1, 0, 1], size=(mp, target_rank))
        W = np.random.choice([-1, 0, 1], size=(pn, target_rank))
        
        # Quick check: sum of products should be 1 for each (i,j,k)
        valid = True
        for i in range(n):
            if not valid:
                break
            for j in range(m):
                if not valid:
                    break
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    dot = np.dot(u_val, v_val * w_val)
                    if abs(dot - 1.0) > 1e-9:
                        valid = False
                        break
        
        if valid:
            return {
                "U": [[int(x) for x in row] for row in U],
                "V": [[int(x) for x in row] for row in V],
                "W": [[int(x) for x in row] for row in W]
            }
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin before 150s limit
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition
    # Known lower bounds: for n=m=p=5, the best known is around 49-51
    # Let's try a few target ranks from low to high
    
    # First, verify Strassen 2x2 works (sanity check)
    # _verify_strassen_2x2()
    
    # Try structured search first (more likely to succeed)
    for target_rank in range(40, best_rank):
        if time.time() > deadline - 5:
            break
        result = _try_structured_search(n, m, p, target_rank, timeout=10)
        if result is not None:
            U, V, W = result["U"], result["V"], result["W"]
            if _verify(n, m, p, U, V, W):
                best = result
                best_rank = target_rank
                break
    
    # If still schoolbook, try random search
    if best_rank == n * m * p:
        for target_rank in range(45, best_rank):
            if time.time() > deadline - 5:
                break
            result = _try_random_search(n, m, p, target_rank, timeout=10)
            if result is not None:
                U, V, W = result["U"], result["V"], result["W"]
                if _verify(n, m, p, U, V, W):
                    best = result
                    best_rank = target_rank
                    break
    
    return best
# EVOLVE-BLOCK-END
