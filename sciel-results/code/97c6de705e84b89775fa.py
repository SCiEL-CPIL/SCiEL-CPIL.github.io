"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START

def _strassen_2x2_decomposition():
    """
    Return the rank-7 decomposition for 2x2 matrix multiplication.
    Indices:
      U: (n*m) x rank = 4 x 7
      V: (m*p) x rank = 4 x 7
      W: (p*n) x rank = 4 x 7
    Convention:
      A is 2x2, B is 2x2, C = A*B is 2x2.
      U[i*m + j, c] corresponds to A[i,j]
      V[j*p + k, c] corresponds to B[j,k]
      W[k*n + i, c] corresponds to C[i,k]  (trace convention: W[k*n+i, r])
      Here n=2, m=2, p=2.
      So U index: i*2 + j
      V index: j*2 + k
      W index: k*2 + i
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the 7 bilinear forms. Each is a pair of linear forms in A and B.
    # For each product c, we have:
    #   U contribution: coefficients for A[i,j]
    #   V contribution: coefficients for B[j,k]
    #   W contribution: coefficient for C[i,k]
    
    # We'll define each of the 7 terms explicitly.
    # Term 0: M1 = (A00 + A11)(B00 + B11)
    # A part: A00=1, A11=1 -> U[0*2+0]=U[0]=1, U[1*2+1]=U[3]=1
    # B part: B00=1, B11=1 -> V[0*2+0]=V[0]=1, V[1*2+1]=V[3]=1
    # C part: C00 += 1, C11 += 1 -> W[0*2+0]=W[0]=1, W[1*2+1]=W[3]=1
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # Term 1: M2 = (A10 + A11)B00
    # A part: A10=1, A11=1 -> U[1*2+0]=U[2]=1, U[3]=1
    # B part: B00=1 -> V[0]=1
    # C part: C10 += 1, C11 += -1 -> W[0*2+1]=W[1]=-1, W[1*2+1]=W[3]=-1
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[1][1] += -1; W[3][1] += -1
    
    # Term 2: M3 = A00(B10 - B11)
    # A part: A00=1 -> U[0]=1
    # B part: B10=1, B11=-1 -> V[1*2+0]=V[2]=1, V[3]=-1
    # C part: C01 += 1, C11 += 1 -> W[0*2+1]=W[1]=1, W[3]=1
    U[0][2] += 1
    V[2][2] += 1; V[3][2] += -1
    W[1][2] += 1; W[3][2] += 1
    
    # Term 3: M4 = A11(B01 - B00)
    # A part: A11=1 -> U[3]=1
    # B part: B01=1, B00=-1 -> V[0*2+1]=V[1]=1, V[0]=-1
    # C part: C00 += 1, C10 += 1 -> W[0]=1, W[1]=1
    U[3][3] += 1
    V[1][3] += 1; V[0][3] += -1
    W[0][3] += 1; W[1][3] += 1
    
    # Term 4: M5 = (A00 + A01)B11
    # A part: A00=1, A01=1 -> U[0]=1, U[1]=1
    # B part: B11=1 -> V[3]=1
    # C part: C00 += -1, C01 += 1 -> W[0]=-1, W[1]=1
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] += -1; W[1][4] += 1
    
    # Term 5: M6 = (A10 - A00)(B00 + B01)
    # A part: A10=1, A00=-1 -> U[2]=1, U[0]=-1
    # B part: B00=1, B01=1 -> V[0]=1, V[1]=1
    # C part: C11 += 1 -> W[3]=1
    U[2][5] += 1; U[0][5] += -1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # Term 6: M7 = (A01 - A11)(B10 + B11)
    # A part: A01=1, A11=-1 -> U[1]=1, U[3]=-1
    # B part: B10=1, B11=1 -> V[2]=1, V[3]=1
    # C part: C00 += 1 -> W[0]=1
    U[1][6] += 1; U[3][6] += -1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1
    
    return U, V, W


def _verify_2x2(U, V, W):
    """Verify the 2x2 Strassen decomposition is correct."""
    n, m, p = 2, 2, 2
    rank = len(U[0])
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] == W[k*n+i, c] for all i,j,k
    # Actually the condition is: for all i,j,k:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c]... no.
    # The tensor decomposition condition: 
    #   T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to C[i,k]
    # For matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k (each term A[i,j]*B[j,k] contributes to C[i,k])
    # Wait, no. The multiplication tensor T has T[i,j,k] = 1 if A[i,j]*B[j,k] is a term in C[i,k].
    # For standard matrix multiplication, every (i,j,k) triple contributes exactly once.
    # So T[i,j,k] = 1 for all i,j,k.
    
    # Check: for all i, j, k:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    We use a block-based approach. The 2x4 * 4x8 multiplication can be decomposed
    by viewing the matrices in blocks. 
    
    Strategy: Use the fact that we can partition the 4x4 middle dimension into blocks.
    A = 2x4, B = 4x8, C = 2x8.
    
    We can write B as a 2x2 block matrix of 4x4 blocks... but that doesn't directly help.
    
    Alternative: Use a Kronecker-like product of smaller decompositions.
    
    Actually, let's think about this differently. The tensor for 2x4*4x8 has:
    - n=2 (rows of A, cols of C)
    - m=4 (cols of A, rows of B)  
    - p=8 (cols of B, cols of C)
    - Schoolbook rank = 2*4*8 = 64
    
    We can try to use Strassen on block structures.
    
    One approach: partition m=4 into two blocks of size 2, and p=8 into four blocks of size 2.
    Then A is 2x4 = 2x(2+2), B is 4x8 = (2+2)x(2+2+2+2).
    
    C[i, k] = sum_{j=0}^{3} A[i,j] B[j,k]
    
    Let's partition j into j0 (0,1) and j1 (2,3), and k into k0(0,1), k1(2,3), k2(4,5), k3(6,7).
    
    C[i, k_b] = A[i, 0:2] * B[0:2, k_b] + A[i, 2:4] * B[2:4, k_b]
    
    Each term is a 2x2 * 2x2 multiplication, which has rank 7.
    There are 2 (j-blocks) * 4 (k-blocks) = 8 such products.
    Total rank = 8 * 7 = 56.
    
    But we can do better by sharing. Let's use Strassen across the j-blocks.
    
    For each k-block b, we have:
    C[:, k_b] = A[:, 0:2] * B[0:2, k_b] + A[:, 2:4] * B[2:4, k_b]
    
    This is a sum of two 2x2 * 2x2 products. We can use Strassen-like combination.
    
    Actually, let's just implement the 8 independent 2x2*2x2 products first (rank 56),
    then try to optimize.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    n, m, p = 2, 4, 8
    
    # Block decomposition:
    # A: 2x4, split cols into two 2-blocks: A0 = A[:, 0:2], A1 = A[:, 2:4]
    # B: 4x8, split rows into two 2-blocks and cols into four 2-blocks
    #   B[0:2, k_b] and B[2:4, k_b] for b=0,1,2,3
    # C: 2x8, split cols into four 2-blocks: C[:, k_b]
    
    # For each k-block b (0..3):
    #   C[:, k_b] = A0 * B_top_b + A1 * B_bot_b
    # where A0, A1 are 2x2, B_top_b, B_bot_b are 2x2
    
    # Each 2x2*2x2 product has rank 7 (Strassen).
    # We have 8 such products (2 A-blocks x 4 B-col-blocks).
    # Naive: rank = 8 * 7 = 56
    
    # But we can potentially share across k-blocks.
    
    # Let's first build the rank-56 decomposition.
    U_2x2, V_2x2, W_2x2 = _strassen_2x2_decomposition()
    
    # Now build the full decomposition.
    # For each (a_block, k_block, strassen_term) where a_block in {0,1}, k_block in {0,1,2,3}:
    #   a_block=0: A rows 0-1, cols 0-1 -> A[i, a_block*2 + j]
    #   a_block=1: A rows 0-1, cols 2-3 -> A[i, 2 + j]
    #   k_block b: B cols b*2 to b*2+1, B rows depend on a_block
    #     a_block=0: B rows 0-1
    #     a_block=1: B rows 2-3
    
    rank = 56
    U = [[Fraction(0)] * rank for _ in range(n * m)]  # 8 x 56
    V = [[Fraction(0)] * rank for _ in range(m * p)]  # 32 x 56
    W = [[Fraction(0)] * rank for _ in range(p * n)]  # 16 x 56
    
    col = 0
    for a_block in range(2):
        for k_block in range(4):
            for c in range(7):
                # U: A[i, a_block*2 + j] for i in {0,1}, j in {0,1}
                for i in range(2):
                    for j in range(2):
                        U[i * 4 + (a_block * 2 + j)][col] += U_2x2[i * 2 + j][c]
                
                # V: B[j, k_block*2 + k] for j in {a_block*2, a_block*2+1}, k in {0,1}
                for j_local in range(2):
                    for k_local in range(2):
                        j_global = a_block * 2 + j_local
                        k_global = k_block * 2 + k_local
                        V[j_global * 8 + k_global][col] += V_2x2[j_local * 2 + k_local][c]
                
                # W: C[i, k_block*2 + k] for i in {0,1}, k in {0,1}
                for i in range(2):
                    for k_local in range(2):
                        k_global = k_block * 2 + k_local
                        W[k_global * 2 + i][col] += W_2x2[k_local * 2 + i][c]
                
                col += 1
    
    # Verify correctness
    # Check: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # This is expensive, so let's do a spot check
    # Actually, let's verify a few entries
    
    # Convert to numpy for faster verification
    import numpy as np
    U_np = np.array([[float(x) for x in row] for row in U])
    V_np = np.array([[float(x) for x in row] for row in V])
    W_np = np.array([[float(x) for x in row] for row in W])
    
    # Verify: T[i,j,k] = sum_c U[i*4+j, c] * V[j*8+k, c] * W[k*2+i, c] should be 1
    # Use tensor contraction
    # T[i,j,k] = sum_c U[i*4+j, c] * V[j*8+k, c] * W[k*2+i, c]
    # Let's verify all entries
    correct = True
    for i in range(2):
        for j in range(4):
            for k in range(8):
                s = 0.0
                for c in range(rank):
                    s += U_np[i*4+j, c] * V_np[j*8+k, c] * W_np[k*2+i, c]
                if abs(s - 1.0) > 1e-10:
                    correct = False
                    break
            if not correct:
                break
        if not correct:
            break
    
    if not correct:
        # Fallback to schoolbook
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
        return {"U": U, "V": V, "W": W}
    
    # Convert to int if possible
    U_int = [[int(x) if x == int(x) else float(x) for x in row] for row in U]
    V_int = [[int(x) if x == int(x) else float(x) for x in row] for row in V]
    W_int = [[int(x) if x == int(x) else float(x) for x in row] for row in W]
    
    return {"U": U_int, "V": V_int, "W": W_int}

# EVOLVE-BLOCK-END
