"""Program for matrix multiplication tensor decomposition: 2x3 * 3x10."""
import time
import sys

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    # Use Strassen for the 2x2 part of m (rows 0,1) and schoolbook for row 2
    # Transpose problem: M_{2,3,10} ~ M_{10,3,2} via commutativity
    # Actually let's work directly with M_{2,3,10}
    # U: shape (n*m, rank) = (6, rank)
    # V: shape (m*p, rank) = (30, rank)
    # W: shape (p*n, rank) = (20, rank)
    
    # The tensor T[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c
    
    # Strategy: Split the middle dimension m=3 into a 2-part and a 1-part.
    # For j=0,1 (the first 2 rows of A and first 2 rows of B's columns... wait)
    # A is n x m = 2 x 3, B is m x p = 3 x 10, C is n x p = 2 x 10
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The schoolbook has rank n*m*p = 2*3*10 = 60
    # 
    # Block approach: Consider the sub-problem where we only use j=0,1.
    # This is a 2x2x10 multiplication: A_sub is 2x2, B_sub is 2x10, C_sub is 2x10
    # We can group the p=10 columns into 5 groups of 2.
    # For each group of 2 columns, we have a 2x2x2 block (A_sub 2x2, B_group 2x2, C_group 2x2)
    # Each 2x2x2 can be done with Strassen rank 7.
    # So for 5 groups: 5 * 7 = 35 for the j=0,1 part.
    # For j=2: A[0,2]*B[2,k] + A[1,2]*B[2,k] for each k.
    # This is a 2x1x10 multiplication, which needs rank 2*1*10 = 20 (schoolbook is minimal here).
    # Total rank: 35 + 20 = 55.
    
    # Let's construct this explicitly.
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]   # 6 x 55
    V = [[0] * rank for _ in range(m * p)]   # 30 x 55
    W = [[0] * rank for _ in range(p * n)]   # 20 x 55
    
    col = 0
    
    # Part 1: Strassen for 5 blocks of 2x2x2 (j=0,1; k in groups of 2)
    for g in range(5):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # A_sub[i,j] for i in {0,1}, j in {0,1}
        # B_sub[j,k] for j in {0,1}, k in {k0,k1}
        # C_sub[i,k] for i in {0,1}, k in {k0,k1}
        
        # Strassen 7 multiplications for 2x2*2x2:
        # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
        # Here A is 2x2 (rows i=0,1; cols j=0,1)
        # B is 2x2 (rows j=0,1; cols k=k0,k1)
        
        # P1 = (A11 - A10) * (B00 + B01)
        # P2 = (A10 + A11) * (B11 - B01)  
        # P3 = (A00 + A11) * (B00 + B11)
        # P4 = (A01 - A00) * (B10 + B11)
        # P5 = (A00 + A01) * (B01 - B11)
        # P6 = (A10 - A00) * (B00 + B10)
        # P7 = (A01 + A11) * (B01 + B11)
        
        # C00 = P3 + P5 - P2 + P6
        # C01 = P1 + P2
        # C10 = P3 + P4
        # C11 = P5 - P1 + P4 + P7
        
        # For each P_r, we need U entries, V entries, W entries.
        # The factor matrices encode the linear combinations.
        
        # P1: U has (A11 - A10) = A[1,1] - A[1,0]
        #   U[i*m+j, col] for j=0,1; i=0,1
        #   We need: coefficient for A[1,1] is +1, for A[1,0] is -1
        #   U[1*3+1][col] = 1  (i=1, j=1)
        #   U[1*3+0][col] = -1 (i=1, j=0)
        #   V[j*p+k, col]: B00 + B01 = B[0,k0] + B[0,k1]
        #   V[0*p+k0][col] = 1
        #   V[0*p+k1][col] = 1
        #   W[k*n+i, col]: C01 = P1 + P2, so P1 contributes +1 to C[0,k1]
        #   W[k1*n+0][col] = 1
        
        # P2: U: A10 + A11 = A[1,0] + A[1,1]
        #   U[1*3+0][col] = 1
        #   U[1*3+1][col] = 1
        #   V: B11 - B01 = B[1,k1] - B[0,k1]
        #   V[1*p+k1][col] = 1
        #   V[0*p+k1][col] = -1
        #   W: C01 = P1 + P2, so P2 contributes +1 to C[0,k1]
        #   W[k1*n+0][col] = 1
        
        # P3: U: A00 + A11 = A[0,0] + A[1,1]
        #   U[0*3+0][col] = 1
        #   U[1*3+1][col] = 1
        #   V: B00 + B11 = B[0,k0] + B[1,k1]
        #   V[0*p+k0][col] = 1
        #   V[1*p+k1][col] = 1
        #   W: C00 = P3 + P5 - P2 + P6, C10 = P3 + P4
        #   So P3 contributes +1 to C[0,k0] and +1 to C[1,k0]
        #   W[k0*n+0][col] = 1
        #   W[k0*n+1][col] = 1
        
        # P4: U: A01 - A00 = A[0,1] - A[0,0]
        #   U[0*3+1][col] = 1
        #   U[0*3+0][col] = -1
        #   V: B10 + B11 = B[1,k0] + B[1,k1]
        #   V[1*p+k0][col] = 1
        #   V[1*p+k1][col] = 1
        #   W: C10 = P3 + P4, so P4 contributes +1 to C[1,k0]
        #   W[k0*n+1][col] = 1
        
        # P5: U: A00 + A01 = A[0,0] + A[0,1]
        #   U[0*3+0][col] = 1
        #   U[0*3+1][col] = 1
        #   V: B01 - B11 = B[0,k1] - B[1,k1]
        #   V[0*p+k1][col] = 1
        #   V[1*p+k1][col] = -1
        #   W: C00 = P3 + P5 - P2 + P6, C11 = P5 - P1 + P4 + P7
        #   So P5 contributes +1 to C[0,k0] and +1 to C[1,k1]
        #   W[k0*n+0][col] = 1
        #   W[k1*n+1][col] = 1
        
        # P6: U: A10 - A00 = A[1,0] - A[0,0]
        #   U[1*3+0][col] = 1
        #   U[0*3+0][col] = -1
        #   V: B00 + B10 = B[0,k0] + B[1,k0]
        #   V[0*p+k0][col] = 1
        #   V[1*p+k0][col] = 1
        #   W: C00 = P3 + P5 - P2 + P6, so P6 contributes +1 to C[0,k0]
        #   W[k0*n+0][col] = 1
        
        # P7: U: A01 + A11 = A[0,1] + A[1,1]
        #   U[0*3+1][col] = 1
        #   U[1*3+1][col] = 1
        #   V: B01 + B11 = B[0,k1] + B[1,k1]
        #   V[0*p+k1][col] = 1
        #   V[1*p+k1][col] = 1
        #   W: C11 = P5 - P1 + P4 + P7, so P7 contributes +1 to C[1,k1]
        #   W[k1*n+1][col] = 1
        
        # Now let's code each P:
        
        # P1
        U[1*3 + 1][col] += 1
        U[1*3 + 0][col] += -1
        V[0*p + k0][col] += 1
        V[0*p + k1][col] += 1
        W[k1*n + 0][col] += 1
        col += 1
        
        # P2
        U[1*3 + 0][col] += 1
        U[1*3 + 1][col] += 1
        V[1*p + k1][col] += 1
        V[0*p + k1][col] += -1
        W[k1*n + 0][col] += 1
        col += 1
        
        # P3
        U[0*3 + 0][col] += 1
        U[1*3 + 1][col] += 1
        V[0*p + k0][col] += 1
        V[1*p + k1][col] += 1
        W[k0*n + 0][col] += 1
        W[k0*n + 1][col] += 1
        col += 1
        
        # P4
        U[0*3 + 1][col] += 1
        U[0*3 + 0][col] += -1
        V[1*p + k0][col] += 1
        V[1*p + k1][col] += 1
        W[k0*n + 1][col] += 1
        col += 1
        
        # P5
        U[0*3 + 0][col] += 1
        U[0*3 + 1][col] += 1
        V[0*p + k1][col] += 1
        V[1*p + k1][col] += -1
        W[k0*n + 0][col] += 1
        W[k1*n + 1][col] += 1
        col += 1
        
        # P6
        U[1*3 + 0][col] += 1
        U[0*3 + 0][col] += -1
        V[0*p + k0][col] += 1
        V[1*p + k0][col] += 1
        W[k0*n + 0][col] += 1
        col += 1
        
        # P7
        U[0*3 + 1][col] += 1
        U[1*3 + 1][col] += 1
        V[0*p + k1][col] += 1
        V[1*p + k1][col] += 1
        W[k1*n + 1][col] += 1
        col += 1
    
    # Part 2: Schoolbook for j=2 (2x1x10), rank 20
    for k in range(p):
        # C[i,k] += A[i,2] * B[2,k]
        # U: A[i,2], so U[i*3+2, col] = 1 for i=0,1
        # V: B[2,k], so V[2*p+k, col] = 1
        # W: C[i,k] = W[k*n+i, col] = 1 for i=0,1
        
        # i=0
        U[0*3 + 2][col] += 1
        V[2*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        # i=1
        U[1*3 + 2][col] += 1
        V[2*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
    
    # col should be 55
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
