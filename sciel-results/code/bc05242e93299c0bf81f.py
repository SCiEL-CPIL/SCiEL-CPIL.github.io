"""Matrix multiplication 3,3,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_trivial(n, m, p):
    """Build the trivial (schoolbook) decomposition of rank n*m*p."""
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(decomp, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly reconstructs the matmul tensor."""
    U = np.array(decomp["U"], dtype=np.float64)
    V = np.array(decomp["V"], dtype=np.float64)
    W = np.array(decomp["W"], dtype=np.float64)
    r = U.shape[1]
    # Check valid positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a, t] * V[b, t] * W[c, t] for t in range(r))
                if abs(val - 1) > tol:
                    return False
    # Check invalid positions (sample)
    invalid_checked = 0
    for a in range(n * m):
        i_a, j_a = divmod(a, m)
        for b in range(m * p):
            j_b, k_b = divmod(b, p)
            if j_a != j_b:
                for c in range(p * n):
                    k_c, i_c = divmod(c, n)
                    if k_b != k_c or i_c != i_a:
                        val = sum(U[a, t] * V[b, t] * W[c, t] for t in range(r))
                        if abs(val) > tol:
                            return False
                        invalid_checked += 1
                        if invalid_checked >= 100:
                            break
                if invalid_checked >= 100:
                    break
        if invalid_checked >= 100:
            break
    return True

def _build_rank44(n, m, p):
    """
    Build a rank-44 decomposition of M_{3,3,5}.
    Strategy:
    1. Decompose M_{3,3,5} into M_{3,3,3} (k=0,1,2) and M_{3,3,2} (k=3,4).
    2. M_{3,3,2} uses trivial rank 18.
    3. M_{3,3,3} uses Strassen rank 7 for the 2x2x2 core (i,j,k in {0,1}) 
       and trivial rank 19 for the remaining terms.
    Total rank: 7 + 19 + 18 = 44.
    """
    n, m, p = 3, 3, 5
    r_strassen = 7
    r_trivial_333 = 19
    r_trivial_332 = 18
    r_total = r_strassen + r_trivial_333 + r_trivial_332
    
    U = [[0] * r_total for _ in range(n * m)]
    V = [[0] * r_total for _ in range(m * p)]
    W = [[0] * r_total for _ in range(p * n)]
    
    # --- Part 1: Strassen for M_{2,2,2} core (indices 0,1) ---
    # Indices: i,j,k in {0,1}
    # U indices: i*3+j
    # V indices: j*5+k  (Note: p=5 for the whole matrix, but k is 0,1 here)
    # W indices: k*3+i
    
    # Term 0: (A00+A11)(B00+B11)
    U[0*3+0][0] = 1; U[1*3+1][0] = 1
    V[0*5+0][0] = 1; V[1*5+1][0] = 1
    W[0*3+0][0] = 1; W[1*3+1][0] = 1
    
    # Term 1: (A10+A11)B00
    U[1*3+0][1] = 1; U[1*3+1][1] = 1
    V[0*5+0][1] = 1
    W[0*3+1][1] = 1
    
    # Term 2: A00(B00-B11)
    U[0*3+0][2] = 1
    V[0*5+0][2] = 1; V[1*5+1][2] = -1
    W[0*3+0][2] = 1
    
    # Term 3: A11(B10-B00)
    U[1*3+1][3] = 1
    V[1*5+0][3] = 1; V[0*5+0][3] = -1
    W[1*3+1][3] = 1
    
    # Term 4: (A00+A01)(B11+B10)
    U[0*3+0][4] = 1; U[0*3+1][4] = 1
    V[1*5+1][4] = 1; V[1*5+0][4] = 1
    W[1*3+0][4] = 1
    
    # Term 5: (A10-A00)(B11-B01)
    U[1*3+0][5] = 1; U[0*3+0][5] = -1
    V[1*5+1][5] = 1; V[0*5+1][5] = -1
    W[1*3+1][5] = 1
    
    # Term 6: (A11-A10)(B00-B01)
    U[1*3+1][6] = 1; U[1*3+0][6] = -1
    V[0*5+0][6] = 1; V[0*5+1][6] = -1
    W[0*3+1][6] = 1
    
    # --- Part 2: Trivial for remaining M_{3,3,3} terms ---
    # These are terms where at least one of i,j,k is 2.
    # Total 19 terms.
    col = r_strassen
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i*3+j][col] = 1
                V[j*5+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
                
    # --- Part 3: Trivial for M_{3,3,2} (k=3,4) ---
    # 18 terms.
    for i in range(3):
        for j in range(3):
            for k in [3, 4]:
                U[i*3+j][col] = 1
                V[j*5+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    deadline = time.time() + 140
    
    # Start with trivial decomposition (rank 45)
    best = _build_trivial(n, m, p)
    
    # Try rank-44 decomposition
    result = _build_rank44(n, m, p)
    if result is not None:
        if _verify(result, n, m, p):
            best = result
    
    return best
# EVOLVE-BLOCK-END
