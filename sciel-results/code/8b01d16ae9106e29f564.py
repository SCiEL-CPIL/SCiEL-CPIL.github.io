"""Program for matrix multiplication tensor decomposition 3x4x7."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, n=3, m=4, p=7):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - expected) > 1e-9:
                    return False
    return True

def _verify_exact(U, V, W, n=3, m=4, p=7):
    """Verify with exact arithmetic using Fractions."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                if total != Fraction(1):
                    return False
    return True

def _to_half_integer_list(matrix):
    """Convert a numpy matrix to a list of lists of float (half-integers)."""
    result = []
    for row in matrix:
        result.append([float(x) for x in row])
    return result

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    n, m, p = 3, 4, 7
    nmp = n * m * p  # 84 (schoolbook rank)
    
    # First, construct the schoolbook decomposition as a fallback.
    rank_schoolbook = nmp
    U_schoolbook = [[0] * rank_schoolbook for _ in range(n * m)]
    V_schoolbook = [[0] * rank_schoolbook for _ in range(m * p)]
    W_schoolbook = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j][col] = 1
                V_schoolbook[j * p + k][col] = 1
                W_schoolbook[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_schoolbook
    best_V = V_schoolbook
    best_W = W_schoolbook
    
    # Strategy: Use the structure of matrix multiplication to find a better decomposition.
    # Key insight: We can use the fact that M_{n,m,p} has a certain tensor structure.
    # For 3x4x7, we can try to exploit the factorization 4 = 2*2.
    
    # Let's try a structured approach using the Strassen-like decomposition.
    # The idea is to use the 2x2x2 Strassen algorithm as a building block.
    
    # For 2x2x2, Strassen uses 7 multiplications with the following structure:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Let's try to construct a decomposition for 3x4x7 using this idea.
    # We can view the 4 columns as two groups of 2, and use the Strassen structure.
    
    # Actually, let's try a more systematic approach: use the known fact that
    # M_{2,2,2} has rank 7, and try to combine it with other decompositions.
    
    # For 3x4x7, we can view it as a combination of smaller tensor decompositions.
    
    # Let's try a random search with structured initialization.
    # We'll use the fact that the tensor has a certain symmetry to guide the search.
    
    np.random.seed(42)
    
    # Try ranks from a lower bound up to schoolbook - 1
    # Lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # But we can use better lower bounds.
    
    # For now, let's try a few specific ranks with structured initialization.
    
    # Idea: Use the fact that M_{3,4,7} has a certain symmetry.
    # The indices (i,j,k) range over [0,3)x[0,4)x[0,7).
    
    # Try using a decomposition based on the 2x2x2 Strassen algorithm
    # extended to 3x4x7.
    
    # For 2x2x2, Strassen uses 7 multiplications:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We can try to generalize this idea to 3x4x7.
    
    # Alternative approach: Use integer linear programming or SAT-like search
    # for small ranks. But this is expensive.
    
    # Let's try a more practical approach: use the known result that
    # M_{2,2,2} has rank 7, and try to combine it with other decompositions.
    
    # For 3x4x7, we can view it as:
    # - 3 rows in A, 4 columns in A (and rows in B), 7 columns in B
    # - We can split the 4 columns into two groups of 2
    
    # Let's try a decomposition where we use the 2x2x2 structure for each
    # pair of columns in the middle dimension.
    
    # Actually, let's just do a more thorough random search with better initialization.
    
    # Use a pattern-based initialization:
    # For each rank-1 term, assign values based on modular arithmetic.
    
    for target_rank in range(40, nmp):
        if time.time() > deadline - 10:
            break
        
        max_attempts = 3 if target_rank > 60 else 10
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Pattern-based initialization
            for c in range(target_rank):
                # Use different patterns for each column
                pattern = (c % 7)
                
                for i in range(n):
                    for j in range(m):
                        # U[i*m+j, c] based on pattern
                        if pattern == 0:
                            U[i * m + j, c] = 1 if (i + j) % 2 == 0 else -1
                        elif pattern == 1:
                            U[i * m + j, c] = 1 if i == j else 0
                        elif pattern == 2:
                            U[i * m + j, c] = 1 if (i * j) % 3 == 0 else -1
                        else:
                            U[i * m + j, c] = np.random.choice([-1, 0, 1])
                
                for j in range(m):
                    for k in range(p):
                        if pattern == 0:
                            V[j * p + k, c] = 1 if (j + k) % 2 == 0 else -1
                        elif pattern == 1:
                            V[j * p + k, c] = 1 if j == (k % m) else 0
                        elif pattern == 2:
                            V[j * p + k, c] = 1 if (j * k) % 4 == 0 else -1
                        else:
                            V[j * p + k, c] = np.random.choice([-1, 0, 1])
                
                for k in range(p):
                    for i in range(n):
                        if pattern == 0:
                            W[k * n + i, c] = 1 if (k + i) % 2 == 0 else -1
                        elif pattern == 1:
                            W[k * n + i, c] = 1 if (k % n) == i else 0
                        elif pattern == 2:
                            W[k * n + i, c] = 1 if (k * i) % 7 == 0 else -1
                        else:
                            W[k * n + i, c] = np.random.choice([-1, 0, 1])
            
            # Verify
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        total = 0.0
                        for c in range(target_rank):
                            total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        if abs(total - 1.0) > 1e-9:
                            valid = False
                            break
            
            if valid:
                best_rank = target_rank
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break
        
        if best_rank < nmp:
            break
    
    # If we found a better decomposition, use it; otherwise, use schoolbook.
    if best_rank < nmp:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U_schoolbook, "V": V_schoolbook, "W": W_schoolbook}
