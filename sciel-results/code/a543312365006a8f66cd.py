"""Strassen-based rank-55 decomposition for 2x3x10 matrix multiplication."""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-55 exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix.
    
    Strategy:
    - Split the 3x10 matrix B into three 1x10 row blocks: B1, B2, B3.
    - The product C[0,k] = A[0,0]*B1[k] + A[0,1]*B2[k] + A[0,2]*B3[k]
    - The product C[1,k] = A[1,0]*B1[k] + A[1,1]*B2[k] + A[1,2]*B3[k]
    
    For each k (0..9), we have a 2x3 * 3x1 multiplication.
    
    We use a rank-5 decomposition for each 2x3x1 multiplication, giving total rank 5*10 = 50.
    But let's be careful: we need to verify the rank-5 construction works.
    
    Actually, let's use a simpler approach based on Strassen's 2x2x2 (rank 7) idea.
    
    Alternative: Split B into B_top (2x10) and B_bottom (1x10).
    A = [A1 | A2] where A1 is 2x2, A2 is 2x1.
    C = A1 * B_top + A2 * B_bottom
    
    A1 * B_top: 2x2 * 2x10. This can be done with rank 7*10/2... no, that's not right.
    
    Let me think more carefully. The standard approach:
    
    For 2x3x10, we can view it as computing C[i,k] = sum_j A[i,j]*B[j,k] for i=0,1 and k=0..9.
    
    One decomposition strategy:
    - For j=0,1,2 and each k, we have terms A[i,j]*B[j,k].
    - Total terms: 2*3*10 = 60 (schoolbook).
    
    To reduce below 60, we need to combine terms.
    
    Let's try: group by k. For each k, we compute a 2x1 vector from 2x3 * 3x1.
    The 2x3x1 tensor has rank at most 6, but we can do better.
    
    Actually, for 2x3x1:
    C[0] = A[0,0]*B[0] + A[0,1]*B[1] + A[0,2]*B[2]
    C[1] = A[1,0]*B[0] + A[1,1]*B[1] + A[1,2]*B[2]
    
    We need U[i,j,r], V[j,r], W[i,r] such that sum_r U[i,j,r]*V[j,r]*W[i,r] = delta_{ii'} (with i' fixed to 0 or 1).
    
    Wait, the tensor is T[i,j,k] = delta_{kk'} where k' is the "output" index. For 2x3x1, the output is just 2 values.
    
    Let me use the known result: rank(2x3x1) = 5.
    
    Construction for rank 5:
    We need 5 terms. Let me try:
    Term 1: U[0,0]=1, U[1,0]=1, others 0; V[0]=1, V[1]=0, V[2]=0; W[0]=1, W[1]=-1
      Contributes: (delta_{i,0}+delta_{i,1})*delta_{j,0}*(delta_{i',0}-delta_{i',1})
      = delta_{j,0}*(1 - delta_{i',1})... this is getting messy.
    
    Let me just use the schoolbook rank 60 to be safe, or try rank 55 as mentioned in the knowledge.
    
    Actually, the knowledge says rank 55 Strassen-based works. Let me construct it.
    
    One approach: Use 2x2x2 Strassen (rank 7) and pad.
    Pad A to 2x4, B to 4x10. Then 2x4x10.
    Split into two 2x2x10 blocks? No, that doesn't work cleanly.
    
    Let me try a different approach:
    Split B into B1 (1x10), B2 (1x10), B3 (1x10).
    C[0] = A[0,0]*B1 + A[0,1]*B2 + A[0,2]*B3
    C[1] = A[1,0]*B1 + A[1,1]*B2 + A[1,2]*B3
    
    For each k, this is a 2x3 * 3x1 multiplication.
    
    I'll use rank 6 per k (schoolbook for 2x3x1), giving total rank 60.
    But the knowledge says 55 works. Let me try to find a rank-5 construction for 2x3x1.
    
    After some thought, I'll just return the schoolbook rank 60 to ensure correctness.
    """
    n, m, p = 2, 3, 10
    rank = n * m * p  # 60
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
