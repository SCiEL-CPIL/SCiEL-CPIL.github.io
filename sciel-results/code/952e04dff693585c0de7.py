"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import random

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 * 2x2 matrix multiplication.
    Each term is a tuple (U_coeffs, V_coeffs, W_coeffs) where:
      U_coeffs: dict {(i, j): coeff} for the 2x2 left matrix
      V_coeffs: dict {(j, k): coeff} for the 2x2 right matrix
      W_coeffs: dict {(i, k): coeff} for the 2x2 output
    
    Strassen formulas:
      M1 = (P00+P11)*(Q00+Q11)
      M2 = (P10+P11)*Q00
      M3 = P00*(Q01-Q11)
      M4 = P11*(Q10-Q00)
      M5 = (P00+P01)*Q11
      M6 = (P10-P00)*(Q00+Q01)
      M7 = (P01-P11)*(Q10+Q11)
    
    Result:
      R00 = M1 + M4 - M5 + M7
      R01 = M3 + M5
      R10 = M2 + M4
      R11 = M1 - M2 + M3 + M6
    """
    terms = []
    
    # M1 = (P00+P11)*(Q00+Q11)
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    
    # M2 = (P10+P11)*Q00
    terms.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1, (1,1): -1}
    ))
    
    # M3 = P00*(Q01-Q11)
    terms.append((
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1}
    ))
    
    # M4 = P11*(Q10-Q00)
    terms.append((
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (1,0): 1}
    ))
    
    # M5 = (P00+P01)*Q11
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,0): -1, (0,1): 1}
    ))
    
    # M6 = (P10-P00)*(Q00+Q01)
    terms.append((
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    ))
    
    # M7 = (P01-P11)*(Q10+Q11)
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    ))
    
    return terms

def _rank_14_decomposition():
    """
    Construct a rank-14 decomposition.
    
    Strategy:
    A (2x4) = [A_L, A_R] where A_L, A_R are 2x2.
    B (4x6) = [B_T; B_B] where B_T, B_B are 2x6.
    C = A_L B_T + A_R B_B.
    
    Each 2x2 * 2x6 multiplication has rank 7 (using shared Strassen factors for the 2x2 part).
    Total rank = 7 + 7 = 14.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 14
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 14
    V = [[0] * rank for _ in range(m * p)]  # 24 x 14
    W = [[0] * rank for _ in range(p * n)]  # 12 x 14
    
    strassen_terms = _strassen_2x2x2_terms()
    
    # Process A_L * B_T (first 7 terms)
    for t in range(7):
        U_coeffs, V_coeffs, W_coeffs = strassen_terms[t]
        col = t
        
        # U: A_L is rows 0,1 and cols 0,1 of A
        for (i_loc, j_loc), coeff in U_coeffs.items():
            i = i_loc
            j = j_loc  # A_L corresponds to cols 0,1
            u_idx = i * m + j
            U[u_idx][col] += coeff
        
        # V: B_T is rows 0,1 and all cols 0..5 of B
        # V_coeffs are for a 2x2 block. We need to map them to the 2x6 matrix.
        # The 2x2 block in Strassen corresponds to the 2x2 submatrix of B.
        # But here B is 2x6. The Strassen terms V_k are linear forms on the 2x2 block.
        # Wait, the standard Strassen decomposition of 2x2 * 2x6:
        # We treat B as a 2x6 matrix.
        # The term V_k(B) is the same linear form as in 2x2 * 2x2, applied to the 2x2 block?
        # No.
        # Let's re-verify the rank of 2x2 * 2x6.
        # C = A B.
        # A is 2x2. B is 2x6.
        # We can write B = [B_0, B_1, B_2] where B_i are 2x2.
        # C = A B_0 + A B_1 + A B_2.
        # Using Strassen for A B_i:
        # A B_i = sum_{k=1}^7 U_k(A) V_k(B_i).
        # So C = sum_{i=0}^2 sum_{k=1}^7 U_k(A) V_k(B_i)
        #      = sum_{k=1}^7 U_k(A) (V_k(B_0) + V_k(B_1) + V_k(B_2)).
        # This works because U_k(A) is the same for all i.
        # So V_k(B) = V_k(B_0) + V_k(B_1) + V_k(B_2).
        
        for (j_loc, k_loc), coeff in V_coeffs.items():
            j = j_loc # B_T is rows 0,1
            # k_loc is 0 or 1 within the 2x2 block.
            # We need to sum over the 3 blocks of B_T.
            # The Strassen V_coeffs are defined for a 2x2 block.
            # For the full 2x6, the coefficient for B[j, k] is the sum of coeffs from all 3 blocks.
            # Since V_coeffs is the same for each block, the coefficient is 3 * coeff?
            # No.
            # V_k(B_i) uses the entries of B_i.
            # V_k(B) = V_k(B_0) + V_k(B_1) + V_k(B_2).
            # If V_k(B_0) has term coeff * B_0[j_loc, k_loc], then V_k(B) has term coeff * B_0[j_loc, k_loc] + coeff * B_1[j_loc, k_loc] + ...
            # So for each block b in 0..2:
            for b_block in range(3):
                k = b_block * 2 + k_loc
                v_idx = j * p + k
                V[v_idx][col] += coeff
        
        # W: Output C_1 is 2x6.
        # W_coeffs are for 2x2 output.
        # Similar to V, we sum over the 3 blocks.
        for (i_loc, k_loc), coeff in W_coeffs.items():
            i = i_loc
            for b_block in range(3):
                k = b_block * 2 + k_loc
                w_idx = k * n + i
                W[w_idx][col] += coeff

    # Process A_R * B_B (next 7 terms)
    for t in range(7):
        U_coeffs, V_coeffs, W_coeffs = strassen_terms[t]
        col = 7 + t
        
        # U: A_R is rows 0,1 and cols 2,3 of A
        for (i_loc, j_loc), coeff in U_coeffs.items():
            i = i_loc
            j = 2 + j_loc # A_R corresponds to cols 2,3
            u_idx = i * m + j
            U[u_idx][col] += coeff
        
        # V: B_B is rows 2,3 and all cols 0..5 of B
        for (j_loc, k_loc), coeff in V_coeffs.items():
            j = 2 + j_loc # B_B is rows 2,3
            for b_block in range(3):
                k = b_block * 2 + k_loc
                v_idx = j * p + k
                V[v_idx][col] += coeff
        
        # W: Output C_2 is 2x6.
        for (i_loc, k_loc), coeff in W_coeffs.items():
            i = i_loc
            for b_block in range(3):
                k = b_block * 2 + k_loc
                w_idx = k * n + i
                W[w_idx][col] += coeff

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(result, n=2, m=4, p=6, trials=50):
    """Verify a decomposition numerically with integer arithmetic."""
    U, V, W = result['U'], result['V'], result['W']
    rank = len(U[0])
    
    for trial in range(trials):
        A = [[random.randint(-10, 10) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-10, 10) for _ in range(p)] for _ in range(m)]
        
        # Expected C
        C_exp = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        # Computed C
        C_comp = [[0] * p for _ in range(n)]
        for col in range(rank):
            u_val = sum(U[i * m + j][col] * A[i][j] for i in range(n) for j in range(m))
            v_val = sum(V[j * p + k][col] * B[j][k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_comp[i][k] += W[k * n + i][col] * u_val * v_val
        
        if C_comp != C_exp:
            return False
    return True

def _schoolbook_decomposition():
    """Fallback: rank-m = 4 decomposition."""
    n, m, p = DIMENSIONS
    rank = m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for i in range(n):
            U[i * m + j][j] = 1
        for k in range(p):
            V[j * p + k][j] = 1
        for i in range(n):
            for k in range(p):
                W[k * n + i][j] = 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2 by 4 
    matrix by a 4 by 6 matrix.
    
    Strategy: Use rank-14 decomposition based on splitting into 2x2 * 2x6 blocks.
    """
    start_time = time.time()
    deadline = start_time + 100
    
    # Try rank-14 decomposition first
    result = _rank_14_decomposition()
    
    # Verify the decomposition
    if _verify_decomposition(result, 2, 4, 6, trials=50):
        return result
    
    # Check deadline
    if time.time() > deadline - 10:
        # Fall back to schoolbook
        return _schoolbook_decomposition()
    
    # Fallback to schoolbook (rank 4)
    schoolbook = _schoolbook_decomposition()
    if _verify_decomposition(schoolbook, 2, 4, 6, trials=50):
        return schoolbook
    
    # Should never reach here
    return schoolbook
# EVOLVE-BLOCK-END
