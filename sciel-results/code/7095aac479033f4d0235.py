"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def _find_decomposition(n, m, p, max_rank):
    """Try to find a rank-r decomposition for n×m * m×p matrix multiplication."""
    nm, mp, pn = n * m, m * p, p * n
    
    for rank in range(max_rank, 0, -1):
        # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
        # This is a bilinear system. Let's try a structured approach.
        
        # Strategy: fix U and W to simple values, solve for V, or use iterative approach.
        # For small cases, we can try to find solutions with entries in {0, 1/2, 1, -1/2, -1}
        
        # Let's try a block-based approach exploiting the rectangular structure
        # Split into blocks and use Strassen-like recursion
        
        # For 2x4 * 4x8:
        # A (2x4) * B (4x8) = C (2x8)
        # Write B as [B1; B2] where B1, B2 are 2x8
        # C = A1*B1 + A2*B2 where A1, A2 are 2x2 (top/bottom halves of A)
        
        if n == 2 and m == 4 and p == 8:
            # Try to construct using block structure
            # A = [A_top; A_bottom] where each is 1x4? No, A is 2x4.
            # Let's split columns of A: A = [A_left | A_right] where each is 2x2
            # B = [B_top; B_bottom] where each is 2x8
            # C = A_left * B_top + A_right * B_bottom
            # Each 2x2 * 2x8 multiplication has naive rank 2*2*8 = 32
            # Total naive: 64, same as before.
            
            # Better: use the fact that 2x2*2x2 has rank 7 (Strassen)
            # Split B into 4 blocks of 2x2: B = [B1 B2; B3 B4]
            # A = [A1 A2] where A1, A2 are 2x2
            # C = A1*[B1 B2] + A2*[B3 B4]
            # = [A1*B1 + A2*B3, A1*B2 + A2*B4]
            # Each 2x2*2x2 product: rank 7
            # We have 4 such products, but they share A1, A2, B1..B4
            # Total rank: 4*7 = 28? Let's check.
            
            # C_left (2x2) = A1*B1 + A2*B3
            # C_right (2x2) = A1*B2 + A2*B4
            # Wait, that's not right. Let me redo.
            
            # A (2x4) = [A1 | A2] where A1, A2 are 2x2
            # B (4x8) = [B11 B12; B21 B22] where each Bii is 2x2
            # C (2x8) = [C1 | C2] where C1, C2 are 2x2
            # C1 = A1*B11 + A2*B21 (2x2)
            # C2 = A1*B12 + A2*B22 (2x2)
            
            # So we need to compute 4 2x2*2x2 products: A1*B11, A2*B21, A1*B12, A2*B22
            # Each has Strassen rank 7, total 28.
            # But we can potentially share some computations.
            
            # Actually, the total rank for computing all 4 products might be less than 28
            # if we can share intermediate results.
            
            # Let me just try to solve the system for rank=28, 24, 20, etc.
            
            # For now, let's use the naive approach but try to find if we can do better
            # by solving the linear system.
            
            # The system: for each (i,j,k), sum_r U[i*4+j,r]*V[j*8+k,r]*W[k*2+i,r] = 1
            # This is a system of 64 equations in 56*rank variables.
            
            # Let's try to find a solution with small entries by setting up the problem
            # as a matrix equation.
            
            # For fixed U and W, V is determined linearly. So let's fix U and W to 
            # simple patterns and solve for V.
            
            # Try: U[i*4+j, r] = 1 if r = i*4+j else 0 (identity-like)
            # Then V[j*8+k, r] * W[k*2+i, r] = 1 for r = i*4+j
            # This gives V[j*8+k, i*4+j] * W[k*2+i, i*4+j] = 1
            # Many solutions exist.
            
            # Let me try a different approach: use the known result that 
            # 2x4 * 4x8 might have rank 32 or less.
            
            # Let's just return the naive solution for now and note that
            # finding the optimal rank requires solving a non-linear system.
            pass
    
    # Fallback: naive decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}, rank

def construct():
    n, m, p = DIMENSIONS
    result, rank = _find_decomposition(n, m, p, n * m * p)
    return result
# EVOLVE-BLOCK-END
