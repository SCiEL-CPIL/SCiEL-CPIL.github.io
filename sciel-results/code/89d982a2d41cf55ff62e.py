"""Rewritten program for 2x2 matrix multiplication rank minimization."""
import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    Rank 6 is impossible for 2x2 matrix multiplication over any field.
    The Strassen algorithm provides a rank-7 decomposition using integer coefficients.
    
    The interface requires:
    U: shape (n*m, rank) = (4, 7)
    V: shape (m*p, rank) = (4, 7)
    W: shape (p*n, rank) = (4, 7)
    
    The decomposition satisfies:
    C[i, k] = sum_r W[k*n + i, r] * (sum_a U[i*m + a, r] * A[i, a]) * (sum_b V[b*p + k, r] * B[b, k])
    Note: The standard Strassen form usually groups terms differently, but this
    bilinear form representation is equivalent if the indices are mapped correctly.
    
    Let's verify the index mapping for the standard Strassen products:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 - M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    We need to map these to U, V, W such that:
    C[i, k] = sum_r W_{k,i,r} * P_{i,r} * Q_{k,r}
    where P_{i,r} = sum_a U_{i,a,r} A_{i,a} and Q_{k,r} = sum_b V_{b,k,r} B_{b,k}.
    
    This implies that for each rank-1 term r, we have a product of a linear form in A and a linear form in B.
    Let $L_A^{(r)}$ be the linear form in A, and $L_B^{(r)}$ be the linear form in B.
    Then $M_r = L_A^{(r)} L_B^{(r)}$.
    And $C = \sum_r W_{i,k,r} M_r$.
    
    The vectors U, V, W encode these linear forms and coefficients.
    U[i, a, r] is the coefficient of A[i, a] in $L_A^{(r)}$.
    V[b, k, r] is the coefficient of B[b, k] in $L_B^{(r)}$.
    W[k, i, r] is the coefficient of $M_r$ in $C_{i,k}$.
    
    Let's define the 7 products:
    
    r=0: M1 = (A00+A11)(B00+B11)
    L_A: A00 + A11 -> U[0,0,0]=1, U[3,0,0]=1 (indices 0,3 for A00, A11)
    L_B: B00 + B11 -> V[0,0,0]=1, V[3,0,0]=1 (indices 0,3 for B00, B11)
    Coeffs for C:
    C00 += 1 -> W[0,0,0]=1 (index 0*2+0=0)
    C11 += 1 -> W[3,0,0]=1 (index 1*2+1=3)
    
    r=1: M2 = (A10+A11)B00
    L_A: A10 + A11 -> U[2,0,1]=1, U[3,0,1]=1
    L_B: B00 -> V[0,0,1]=1
    Coeffs for C:
    C10 += 1 -> W[1,0,1]=1 (index 0*2+1=1)
    C11 -= 1 -> W[3,0,1]=-1
    
    r=2: M3 = A00(B01-B11)
    L_A: A00 -> U[0,0,2]=1
    L_B: B01 - B11 -> V[1,0,2]=1, V[3,0,2]=-1
    Coeffs for C:
    C01 += 1 -> W[2,0,2]=1 (index 1*2+0=2)
    C11 += 1 -> W[3,0,2]=1
    
    r=3: M4 = A11(B10-B00)
    L_A: A11 -> U[3,0,3]=1
    L_B: B10 - B00 -> V[2,0,3]=1, V[0,0,3]=-1
    Coeffs for C:
    C00 += 1 -> W[0,0,3]=1
    C10 += 1 -> W[1,0,3]=1
    
    r=4: M5 = (A00+A01)B11
    L_A: A00 + A01 -> U[0,0,4]=1, U[1,0,4]=1
    L_B: B11 -> V[3,0,4]=1
    Coeffs for C:
    C00 -= 1 -> W[0,0,4]=-1
    C01 += 1 -> W[2,0,4]=1
    
    r=5: M6 = (A10-A00)(B00+B01)
    L_A: A10 - A00 -> U[2,0,5]=1, U[0,0,5]=-1
    L_B: B00 + B01 -> V[0,0,5]=1, V[1,0,5]=1
    Coeffs for C:
    C11 += 1 -> W[3,0,5]=1
    
    r=6: M7 = (A01-A11)(B10+B11)
    L_A: A01 - A11 -> U[1,0,6]=1, U[3,0,6]=-1
    L_B: B10 + B11 -> V[2,0,6]=1, V[3,0,6]=1
    Coeffs for C:
    C00 += 1 -> W[0,0,6]=1
    
    Now we construct the matrices U, V, W of shape (4, 7).
    Indices for U (A): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    Indices for V (B): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    Indices for W (C): 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)  <-- Note: W[k*n+i, r]
    W index mapping:
    C(0,0) -> k=0, i=0 -> 0*2+0 = 0
    C(1,0) -> k=0, i=1 -> 0*2+1 = 1
    C(0,1) -> k=1, i=0 -> 1*2+0 = 2
    C(1,1) -> k=1, i=1 -> 1*2+1 = 3
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # r=0
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # r=2
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # r=3
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # r=4
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # r=5
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # r=6
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
