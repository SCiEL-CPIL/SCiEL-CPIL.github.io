"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction as Fr

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _validate(U, V, W, n, m, p):
    rank = len(W[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = Fr(0)
                for r in range(rank):
                    val += Fr(U[a][r]) * Fr(V[b][r]) * Fr(W[c][r])
                if val != 1:
                    return False
    return True

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2 matrix multiplication (rank 7)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen's 7 products for 2x2 matmul
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)h, M6 = (c-d)(f+h), M7 = (b-d)(e+g)
    # Result: [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # U indexed by (i,j) where i=0,1 row of A, j=0,1 col of A
    # V indexed by (j,k) where j=0,1 row of B, k=0,1 col of B  
    # W indexed by (k,i) where k=0,1 col of result, i=0,1 row of result
    
    # Term 0: M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1  # a, d
    V[0][0] = 1; V[3][0] = 1  # e, h
    W[0][0] = 1; W[3][0] = 1  # contributes to (0,0) and (1,1)
    
    # Term 1: M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1  # c, d
    V[0][1] = 1  # e
    W[2][1] = 1  # contributes to (1,0)
    
    # Term 2: M3 = a(f-h)
    U[0][2] = 1  # a
    V[1][2] = 1; V[3][2] = -1  # f, -h
    W[1][2] = 1  # contributes to (0,1)
    
    # Term 3: M4 = d(g-e)
    U[3][3] = 1  # d
    V[2][3] = 1; V[0][3] = -1  # g, -e
    W[0][3] = 1; W[2][3] = 1  # contributes to (0,0) and (1,0)
    
    # Term 4: M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1  # a, b
    V[3][4] = 1  # h
    W[0][4] = -1; W[1][4] = 1  # contributes to (0,1) with +, (0,0) with -
    
    # Term 5: M6 = (c-d)(f+h)
    U[2][5] = 1; U[3][5] = -1  # c, -d
    V[1][5] = 1; V[3][5] = 1  # f, h
    W[1][5] = 1  # contributes to (1,1)
    
    # Term 6: M7 = (b-d)(e+g)
    U[1][6] = 1; U[3][6] = -1  # b, -d
    V[0][6] = 1; V[2][6] = 1  # e, g
    W[0][6] = 1  # contributes to (0,0)
    
    return {"U": U, "V": V, "W": W}

def _combine_with_schoolbook(strassen, n, m, p, n_s=2, m_s=2, p_s=2):
    """Combine a Strassen decomposition for n_s x m_s x p_s with schoolbook for the rest."""
    # This is a simple approach: use Strassen for the top-left n_s x m_s x p_s block
    # and schoolbook for everything else
    rank_s = 7
    rank_rest = n*m*p - n_s*m_s*p_s
    total_rank = rank_s + rank_rest
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    # Fill in Strassen part
    for i in range(n_s):
        for j in range(m_s):
            for r in range(rank_s):
                U[i*m+j][r] = strassen["U"][i*m_s+j][r]
    for j in range(m_s):
        for k in range(p_s):
            for r in range(rank_s):
                V[j*p+k][r] = strassen["V"][j*p_s+k][r]
    for k in range(p_s):
        for i in range(n_s):
            for r in range(rank_s):
                W[k*n+i][r] = strassen["W"][k*n_s+i][r]
    
    # Fill in schoolbook part for the rest
    c = rank_s
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if this is in the Strassen block
                if i < n_s and j < m_s and k < p_s:
                    continue
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    
    return {"U": U, "V": V, "W": W}

def _try_finite_field_support(n, m, p, test_rank, deadline):
    """
    Try to find a valid support pattern over GF(2) for rank test_rank,
    then lift to rationals.
    
    Strategy: For each rank position r, we need to choose which (i,j) pairs
    have U[i*m+j][r] != 0, which (j,k) pairs have V[j*p+k][r] != 0,
    and which (k',i) pairs have W[k'*n+i][r] != 0.
    
    Over GF(2), the constraint is that for each (i,j,k):
    sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1 (mod 2)
    And for all other (a,b,c) not of the form (i*m+j, j*p+k, k*n+i):
    sum_r U[a][r] * V[b][r] * W[c][r] = 0 (mod 2)
    
    This is a cubic constraint, which is hard. Instead, let's use a simpler
    heuristic: try sparse patterns with small support per rank position.
    """
    if time.time() > deadline:
        return None
    
    # For each rank position, try to assign a "pattern" of non-zero entries
    # A simple pattern: for rank position r, pick one (i,j), one (j,k), one (k',i)
    # such that the tensor equation is satisfied
    
    # This is equivalent to finding a set of rank-1 tensors that sum to the matmul tensor
    # Let's try a greedy approach: for each (i,j,k), try to cover it with existing
    # rank positions or add a new one
    
    # Actually, let's try a different approach: use the structure of known decompositions
    # For [4,5,5], we can try to use a [4,4,4] decomposition (if known) plus extra terms
    
    # For now, let's try a randomized local search over the space of integer-valued
    # factor matrices with small entries
    
    random.seed(42 + test_rank)
    
    # Initialize with small random integers
    U = [[random.choice([-1, 0, 1]) for _ in range(test_rank)] for _ in range(n*m)]
    V = [[random.choice([-1, 0, 1]) for _ in range(test_rank)] for _ in range(m*p)]
    W = [[random.choice([-1, 0, 1]) for _ in range(test_rank)] for _ in range(p*n)]
    
    # Local search: try to fix violations
    for iteration in range(50):
        if time.time() > deadline:
            return None
        
        # Check which (i,j,k) are violated
        violations = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b = j*p+k
                    c = k*n+i
                    val = 0
                    for r in range(test_rank):
                        val += U[a][r] * V[b][r] * W[c][r]
                    if val != 1:
                        violations.append((i, j, k, val))
        
        if not violations:
            return {"U": U, "V": V, "W": W}
        
        # Pick a random violation and try to fix it
        i, j, k, val = random.choice(violations)
        a = i*m+j
        b = j*p+k
        c = k*n+i
        
        # Try flipping a random entry in U, V, or W
        if random.random() < 0.33:
            r = random.randint(0, test_rank-1)
            U[a][r] = random.choice([-1, 0, 1])
        elif random.random() < 0.5:
            r = random.randint(0, test_rank-1)
            V[b][r] = random.choice([-1, 0, 1])
        else:
            r = random.randint(0, test_rank-1)
            W[c][r] = random.choice([-1, 0, 1])
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook = _build_schoolbook(n, m, p)
    
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try combining Strassen with schoolbook
    strassen = _strassen_2x2x2()
    combined = _combine_with_schoolbook(strassen, n, m, p, n_s=2, m_s=2, p_s=2)
    combined_rank = len(combined["W"][0])
    
    if _validate(combined["U"], combined["V"], combined["W"], n, m, p):
        if combined_rank < best_rank:
            best_result = combined
            best_rank = combined_rank
    
    # Try finite field support search for lower ranks
    min_rank = max(n*m, m*p, p*n)
    
    for test_rank in range(min_rank, min(best_rank, min_rank + 15)):
        if time.time() > deadline - 5:
            break
        
        result = _try_finite_field_support(n, m, p, test_rank, deadline - 5)
        if result is not None:
            if _validate(result["U"], result["V"], result["W"], n, m, p):
                best_result = result
                best_rank = test_rank
                break
    
    return best_result
# EVOLVE-BLOCK-END
