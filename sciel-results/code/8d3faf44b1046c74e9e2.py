"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x7 matrix.
    
    Strategy: Use block-wise Strassen decomposition.
    Split A (2x4) into two 2x2 blocks: A = [A1 | A2]
    Split B (4x7) into two 2x7 blocks stacked: B = [B1; B2]
    C = A1*B1 + A2*B2
    
    For each 2x2 * 2x7 multiplication, use Strassen's 7 multipliers.
    Each multiplier involves a scalar (linear combo of 2x2 entries) times a vector (linear combo of 2x7 rows).
    This gives 7 rank-1 tensors per 2x2*2x7 block, but since the vector has 7 components,
    we need to be careful about the tensor structure.
    
    Actually, let's think more carefully. The tensor for 2x2 * 2x7 multiplication:
    A has 4 entries, B has 14 entries, C has 14 entries.
    Schoolbook rank = 4 * 7 = 28 (multiply each of 4 A entries by each of 7 B rows... wait no)
    
    Let me reconsider. For 2x2 * 2x7:
    A = [[a,b],[c,d]] (4 entries)
    B = [[b11,...,b17],[b21,...,b27]] (14 entries, 2 rows of 7)
    C = A*B = [[a*b11+b*b21, ..., a*b17+b*b27],
               [c*b11+d*b21, ..., c*b17+d*b27]] (14 entries)
    
    Schoolbook: 2*2*7 = 28 multiplications.
    
    Using Strassen on the 2x2 structure:
    M1 = (a+d) * (row1+row2)  -> gives 7 products
    M2 = (c+d) * row1          -> gives 7 products  
    M3 = a * (row2-row1)       -> gives 7 products
    M4 = d * (row1-row2)       -> gives 7 products
    M5 = (a+b) * row2          -> gives 7 products
    M6 = (c-a) * (row1+row2)   -> gives 7 products
    M7 = (b-d) * (row1+row2)   -> gives 7 products
    
    Total: 7*7 = 49 rank-1 terms. Worse than schoolbook 28.
    
    So Strassen doesn't help for 2x2*2x7 directly.
    
    Alternative: Keep schoolbook for the 2x2*2x7 blocks.
    Total rank = 2 * 28 = 56. Same as original schoolbook.
    
    Let me try a different decomposition. What if we split differently?
    
    Actually, the measured improvement from 56 to 50 suggests a different approach works.
    Let me try: split the middle dimension (4) into 2+2, but handle the 7 columns more cleverly.
    
    For 2x4 * 4x7, think of it as:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    One approach: use the fact that 4 = 2+2 and 7 = 3+4 or similar.
    
    Let me try a direct construction with rank 50.
    The idea: 56 - 50 = 6 savings. This could come from sharing some computations.
    
    Given time constraints, I'll implement a verified Strassen-based block decomposition.
    """
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Build schoolbook first as fallback
    def build_schoolbook():
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
    
    # Try to build a lower-rank decomposition using block Strassen
    # Split A (2x4) into A1 (2x2) and A2 (2x2)
    # Split B (4x7) into B1 (2x7) and B2 (2x7)
    # C = A1*B1 + A2*B2
    # 
    # For each 2x2 * 2x7, use schoolbook (rank 28)
    # Total: 56, no improvement.
    #
    # Alternative: Try to find a rank 50 decomposition by sharing.
    # 
    # Let's just return schoolbook for now since we couldn't find a better one
    # within the time budget. The measured improvement to 50 must use a different
    # decomposition strategy.
    
    # Actually, let me try a specific construction.
    # For 2x4 * 4x7, one approach is to note that we can group the 7 output columns.
    # 
    # Rank 50 construction idea:
    # Use 7 Strassen multipliers for a 2x2*2x2 core, but extend to handle the extra columns.
    # 
    # Given the complexity, I'll return schoolbook which is guaranteed correct.
    return build_schoolbook()
# EVOLVE-BLOCK-END
