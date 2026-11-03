"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 4, 4, 7
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try to find a better decomposition using structured approach
    # Strategy: Use Strassen-like recursion for 4x4x4 part, schoolbook for 4x4x3 part
    
    # Part 1: 4x4x4 multiplication using Strassen (Rank 49)
    # Part 2: 4x4x3 multiplication using Schoolbook (Rank 48)
    # Total: 97
    
    # Let's implement Strassen 4x4x4 properly
    # A is 4x4, B is 4x4, C is 4x4
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # Strassen 2x2x2 terms:
    # S1 = (A11+A22)(B11+B22)
    # S2 = (A21+A22)B11
    # S3 = A11(B12-B22)
    # S4 = A22(B21-B11)
    # S5 = (A11+A12)B22
    # S6 = (A21-A11)(B11+B12)
    # S7 = (A12-A22)(B21+B22)
    
    # C11 = S1 + S4 - S5 + S7
    # C12 = S3 + S5
    # C21 = S2 + S4
    # C22 = S1 - S2 + S3 + S6
    
    # Each S is a 2x2x2 multiplication, which itself can be done with Strassen rank 7
    # So 7 * 7 = 49 terms for 4x4x4
    
    # Let's build the 49 terms for 4x4x4
    U_444 = []
    V_444 = []
    W_444 = []
    
    # Strassen 2x2x2 coefficients
    # For each of the 7 S terms, we have 7 sub-terms
    # S1: (a+d)(e+h) -> Strassen terms for 2x2x2
    # We need to map block indices to global indices
    
    # A blocks in 4x4:
    # A11: rows 0,1 cols 0,1 -> indices 0,1,4,5
    # A12: rows 0,1 cols 2,3 -> indices 2,3,6,7
    # A21: rows 2,3 cols 0,1 -> indices 8,9,12,13
    # A22: rows 2,3 cols 2,3 -> indices 10,11,14,15
    
    # B blocks in 4x4:
    # B11: rows 0,1 cols 0,1 -> indices 0,1,4,5
    # B12: rows 0,1 cols 2,3 -> indices 2,3,6,7
    # B21: rows 2,3 cols 0,1 -> indices 8,9,12,13
    # B22: rows 2,3 cols 2,3 -> indices 10,11,14,15
    
    # C blocks in 4x4:
    # C11: rows 0,1 cols 0,1 -> indices 0,1,4,5 (k*n+i where k=0,1; i=0,1)
    # C12: rows 0,1 cols 2,3 -> indices 2,3,6,7
    # C21: rows 2,3 cols 0,1 -> indices 8,9,12,13
    # C22: rows 2,3 cols 2,3 -> indices 10,11,14,15
    
    # Strassen 2x2x2:
    # Input: a,b,c,d (2x2 matrix A) and e,f,g,h (2x2 matrix B)
    # Output: 7 terms S1-S7
    
    # For each S term, we need to express it as sum of products
    # S1 = (a+d)(e+h)
    #   = a*e + a*h + d*e + d*h
    #   But in tensor decomposition, we need rank 7, not 4
    #   Strassen 2x2x2 uses:
    #   T1 = (a+d)(e+h)
    #   T2 = (c+d)e
    #   T3 = a(f-h)
    #   T4 = d(g-e)
    #   T5 = (a+b)h
    #   T6 = (c-a)(e+f)
    #   T7 = (b-d)(g+h)
    
    # So for S1 in 4x4x4, we actually use all 7 T terms
    # This is getting complex. Let me use a different approach.
    
    # Alternative: Just use the known fact that 4x4x4 has rank 49
    # and 4x4x3 has rank 48 (schoolbook)
    # Total rank 97
    
    # For now, let's try a simpler structured approach
    # Use block decomposition:
    # A (4x4) * B (4x7) = C (4x7)
    # Split B into B1 (4x4) and B2 (4x3)
    # C = A*B1 + A*B2
    
    # A*B1: 4x4x4, use Strassen rank 49
    # A*B2: 4x4x3, use schoolbook rank 48
    # Total: 97
    
    # But implementing Strassen 4x4x4 is complex.
    # Let's try a different strategy: random search with structure
    
    # Try rank 97 with structured initialization
    for target_rank in [97, 98, 99, 100]:
        if time.time() > deadline - 15:
            break
        
        # Structured initialization:
        # First 49 terms: for 4x4x4 part (columns 0-3 of B)
        # Last 48 terms: for 4x4x3 part (columns 4-6 of B)
        
        U = np.zeros((n * m, target_rank), dtype=float)
        V = np.zeros((m * p, target_rank), dtype=float)
        W = np.zeros((p * n, target_rank), dtype=float)
        
        # Part 1: 4x4x4 (first 49 terms)
        # Use Strassen-like structure
        for c in range(49):
            # Random but structured
            for i in range(n):
                for j in range(4):  # Only first 4 columns of B
                    U[i * m + j, c] = np.random.uniform(-1, 1)
            for j in range(4):
                for k in range(4):  # Only first 4 rows of B
                    V[j * p + k, c] = np.random.uniform(-1, 1)
            for i in range(n):
                for k in range(4):  # Only first 4 rows of C
                    W[k * n + i, c] = np.random.uniform(-1, 1)
        
        # Part 2: 4x4x3 (last 48 terms)
        for c in range(49, target_rank):
            for i in range(n):
                for j in range(m):
                    U[i * m + j, c] = np.random.uniform(-1, 1)
            for j in range(m):
                for k in range(4, 7):  # Last 3 rows of B
                    V[j * p + k, c] = np.random.uniform(-1, 1)
            for i in range(n):
                for k in range(4, 7):  # Last 3 rows of C
                    W[k * n + i, c] = np.random.uniform(-1, 1)
        
        # Check if this is a valid decomposition
        valid = True
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for l in range(m):
                        s = 0.0
                        for c in range(target_rank):
                            s += W[k * n + i, c] * U[i * m + j, c] * V[l * p + k, c]
                        expected = 1.0 if j == l else 0.0
                        if abs(s - expected) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Convert to list format with integer/half-integer entries
            U_list = [[round(val * 2) / 2 for val in U[r, :]] for r in range(n * m)]
            V_list = [[round(val * 2) / 2 for val in V[r, :]] for r in range(m * p)]
            W_list = [[round(val * 2) / 2 for val in W[r, :]] for r in range(p * n)]
            
            # Verify with rounded values
            valid_rounded = True
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for l in range(m):
                            s = 0.0
                            for c in range(target_rank):
                                s += W_list[k * n + i][c] * U_list[i * m + j][c] * V_list[l * p + k][c]
                            expected = 1.0 if j == l else 0.0
                            if abs(s - expected) > 1e-9:
                                valid_rounded = False
                                break
                        if not valid_rounded:
                            break
                    if not valid_rounded:
                        break
                if not valid_rounded:
                    break
            
            if valid_rounded:
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break
    
    return best
# EVOLVE-BLOCK-END
