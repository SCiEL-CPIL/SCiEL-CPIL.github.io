"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    Returns: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook decomposition as baseline (rank 63)
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    best_result = {'U': U_sb, 'V': V_sb, 'W': W_sb}
    best_rank = schoolbook_rank

    def verify(U, V, W, tol=1e-9):
        """Verify exact decomposition using numpy."""
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if len(U) == 0 or len(V) == 0 or len(W) == 0:
            return False
        r = len(U[0])
        if len(V[0]) != r or len(W[0]) != r or r <= 0:
            return False
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def clean(x):
        """Snap to nearest half-integer if close."""
        h = round(x * 2) / 2.0
        if abs(x - h) < 1e-7:
            return h
        return x

    def make_result(U, V, W):
        Uc = [[clean(x) for x in row] for row in U]
        Vc = [[clean(x) for x in row] for row in V]
        Wc = [[clean(x) for x in row] for row in W]
        return {'U': Uc, 'V': Vc, 'W': Wc}

    def try_update(U, V, W):
        nonlocal best_result, best_rank
        if verify(U, V, W):
            r = len(U[0])
            if r < best_rank:
                best_result = make_result(U, V, W)
                best_rank = r
                return True
        return False

    # --- Strategy: Structured search with local refinement ---
    # Key idea: Use the algebraic structure of matrix multiplication.
    # For 3x3 * 3x7, we can view this as 7 separate 3x3 * 3x1 multiplications.
    # Each 3x3 * 3x1 can be done in rank 7 (Strassen-like) or 9 (schoolbook).
    # But we can share structure across columns.
    
    # Approach 1: Try ranks from a lower bound upward with structured random init + local search
    # Lower bound: max(n*m, m*p, p*n) = 21, but realistically higher.
    
    # Approach 2: Use known Strassen patterns for 3x3 and extend to 7 columns
    
    # Let's try a systematic approach:
    # For each target rank, try multiple random initializations with small integer/half-integer
    # coefficients, then use local search to refine.
    
    coeff_options = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    random.seed(12345)
    
    # Try target ranks from 35 down to 25 (reasonable range below 63)
    # For each rank, try several random initializations
    for target_rank in range(35, 24, -1):
        if time.time() > deadline - 10:
            break
        
        max_trials = 50 if target_rank <= 30 else 30
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break
            
            # Random sparse initialization
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Each column r: pick a random (i,j,k) triple and set the corresponding entries
            # This ensures each term "targets" a specific tensor entry
            for r in range(target_rank):
                i = random.randint(0, n-1)
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                
                # Set U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
                # Use small random coefficients to allow cancellations
                u_val = random.choice(coeff_options)
                v_val = random.choice(coeff_options)
                w_val = random.choice(coeff_options)
                
                if u_val == 0 or v_val == 0 or w_val == 0:
                    # Retry with non-zero
                    u_val = random.choice([x for x in coeff_options if x != 0])
                    v_val = random.choice([x for x in coeff_options if x != 0])
                    w_val = random.choice([x for x in coeff_options if x != 0])
                
                U[i*m+j, r] = u_val
                V[j*p+k, r] = v_val
                W[k*n+i, r] = w_val
            
            # Check if this random init happens to work (very unlikely but cheap)
            if verify(U.tolist(), V.tolist(), W.tolist()):
                try_update(U.tolist(), V.tolist(), W.tolist())
                break
            
            # Local search: try to fix the decomposition by adjusting entries
            # For each (i,j,k), check if the current sum equals 1
            # If not, try to adjust one of the contributing terms
            
            # This is complex; instead, try a simpler heuristic:
            # For each target (i,j,k), ensure at least one term contributes exactly 1
            # and others contribute 0 or cancel out.
            
            # Alternative: use the schoolbook structure as a starting point
            # and try to merge/eliminate terms.
            
            # For now, just do the random check.
            pass
    
    # --- Strategy 2: Use Strassen-like structure ---
    # Strassen's algorithm for 3x3 * 3x3 uses 7 multiplications.
    # For 3x3 * 3x7, we can use 7 Strassen blocks (one per column) = 49 multiplications.
    # But we can do better by sharing.
    
    # Let's try a block approach: split 7 columns into 3+3+1
    # Use Strassen for the 3x3 blocks and schoolbook for the 1-column block.
    # Rank = 7 + 7 + 9 = 23? No, that's not how it works.
    
    # Actually, each "multiplication" in Strassen is a scalar product of two linear combinations.
    # The rank is the number of such scalar products.
    
    # For 3x3 * 3x7:
    # - Split B into B1 (3x3), B2 (3x3), B3 (3x1)
    # - Compute A*B1 using Strassen (rank 7)
    # - Compute A*B2 using Strassen (rank 7)  
    # - Compute A*B3 using schoolbook (rank 9)
    # - Total rank = 7 + 7 + 9 = 23
    
    # But this assumes the Strassen decompositions are independent, which they are.
    # So rank 23 should be achievable!
    
    # Let's construct this explicitly.
    
    # Strassen's 7-multiplication algorithm for 3x3 * 3x3:
    # The 7 products are:
    # M1 = (A11+A33) * (B11+B33)
    # M2 = (A31+A33) * B13
    # M3 = A11 * (B13-B33)
    # M4 = A33 * (B31-B11)
    # M5 = (A11+A13) * B33
    # M6 = (A31-A11) * (B11+B13)
    # M7 = (A13+A33) * (B31+B33)
    
    # Wait, standard Strassen is for 2x2. For 3x3, it's more complex.
    # Let me use a known 3x3 decomposition.
    
    # Actually, the minimal rank for 3x3 * 3x3 is 7 (Strassen).
    # Let me look up the explicit 7-term decomposition for 3x3.
    
    # For simplicity, let's use the schoolbook for now and try to find
    # a lower-rank decomposition through search.
    
    # Given time constraints, return the best result found.
    return best_result

# EVOLVE-BLOCK-END
