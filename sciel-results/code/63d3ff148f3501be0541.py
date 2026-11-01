"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal tensor decomposition for M(2,2,7).
    
    A is 2x2, B is 2x7, C = A*B is 2x7.
    
    The current program uses the schoolbook decomposition with rank 28.
    The inspirations suggest that the tensor M(2,2,7) can be viewed as a 
    tensor product structure. However, the rank-21 construction in 
    inspiration 3 is invalid because it incorrectly assumes that 
    M(2,2,7) = M(2,2,1) x M(2,1,7), which is not a valid tensor product 
    decomposition for matrix multiplication.
    
    Let's analyze the correct structure:
    M(2,2,7) has entries T[i,j,k] where i in {0,1} (row of A), 
    j in {0,1} (col of A / row of B), k in {0..6} (col of B).
    The multiplication is C[i,k] = sum_j A[i,j] * B[j,k].
    
    The schoolbook rank is 2*2*7 = 28.
    
    Can we do better? Let's think about the tensor rank.
    The tensor is M(2,2,7). This is not a direct sum of 7 independent 
    M(2,2,1) tensors in the usual sense because the A matrix is shared.
    
    Actually, M(2,2,7) can be seen as the tensor product of M(2,2,1) 
    (which is just the 4-term sum for 2x2 times 2x1) and M(1,1,7) 
    (which is trivial rank 1 per term). But that's not quite right.
    
    Let's use a different approach. The tensor M(n,m,p) for matrix 
    multiplication has known bounds. For M(2,2,7), the rank is known 
    to be 28 (schoolbook) in general, but there might be structured 
    decompositions.
    
    However, looking at the problem more carefully:
    - A is 2x2 (4 entries)
    - B is 2x7 (14 entries)  
    - C is 2x7 (14 entries)
    
    The bilinear form has 4*14 = 56 input pairs producing 14 outputs.
    
    The key insight: For each column k of B, we compute a 2-vector 
    C[:,k] = A * B[:,k]. This is a 2x2 matrix times a 2x1 vector.
    
    The rank of the 2x2 times 2x1 multiplication (M(2,2,1)) is 4 
    (schoolbook, since each output C[i] = sum_j A[i,j]*B[j], and 
    there's no way to reduce below 4 for exact computation in general).
    
    Since the 7 columns are independent (B has 7 independent columns), 
    and A is shared, the total rank should be 7 * 4 = 28.
    
    Wait, but can we share terms across columns? The A entries are the 
    same for all columns, but the B entries are different. In a tensor 
    decomposition, each rank-1 term is a product u_r * v_r * w_r where 
    u_r is in the A-space, v_r in the B-space, w_r in the C-space.
    
    For the schoolbook, each term corresponds to a unique (i,j,k) triple.
    To share, we would need to combine terms with the same (i,j) but 
    different k, or same (j,k) but different i, etc.
    
    Actually, the tensor M(2,2,7) is isomorphic to M(2,2,1) tensor 
    M(1,1,7) in a certain sense, but the rank of a tensor product 
    is the product of ranks. M(2,2,1) has rank 4, M(1,1,7) has rank 7 
    (since it's just 7 independent 1x1 multiplications). So the 
    tensor product would have rank 4*7 = 28.
    
    This confirms that 28 is the natural rank.
    
    However, let's verify if there's any structure we can exploit.
    The problem asks for rank-minimal, so we should try to find if 
    rank < 28 is possible.
    
    According to known results, the rank of M(2,2,p) for p >= 2 is 
    actually 2p + something? Let me think...
    
    For M(2,2,2), the rank is 7 (Strassen). For M(2,2,1), the rank is 4.
    For M(2,2,7), it's not immediately clear.
    
    Actually, M(2,2,p) can be decomposed using the fact that it's 
    equivalent to p copies of M(2,2,1) but with shared A. The rank 
    might be less than 4p.
    
    Let's try a construction with rank < 28.
    
    Idea: Use Strassen-like tricks. For each pair of columns, we can 
    use 7 terms instead of 8 (4*2). But 7 columns is odd, so we can 
    group 6 columns into 3 pairs (3*7=21 terms) and handle the last 
    column separately (4 terms), total 25 terms.
    
    Wait, Strassen for 2x2*2x2 uses 7 terms. If we group columns 
    (k1, k2), we can use 7 terms to compute C[:,k1] and C[:,k2] 
    together.
    
    Let me construct this:
    
    For columns k and k+1 (pair), we want to compute:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    C[0,k+1] = A[0,0]*B[0,k+1] + A[0,1]*B[1,k+1]
    C[1,k+1] = A[1,0]*B[0,k+1] + A[1,1]*B[1,k+1]
    
    Using Strassen-like products:
    M1 = (A[0,0]+A[1,1]) * (B[0,k]+B[0,k+1])
    M2 = (A[0,0]+A[1,1]) * (B[1,k]+B[1,k+1])
    M3 = (A[0,1]+A[1,1]) * (B[1,k]-B[0,k])
    M4 = (A[0,0]-A[1,0]) * (B[0,k]+B[1,k])
    M5 = (A[0,0]-A[1,0]) * (B[0,k+1]+B[1,k+1])
    M6 = (A[1,0]+A[0,0]) * (B[1,k]-B[0,k])  # Wait, this is getting messy
    
    Actually, the standard Strassen decomposition for 2x2*2x2 is:
    C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M6
    
    Where:
    M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    M2 = (A[1,0]+A[1,1])(B[0,0]+B[0,1])
    M3 = (A[0,0]+A[0,1])(B[1,0]+B[1,1])
    M4 = (A[1,1]-A[1,0])(B[0,1]+B[1,1])
    M5 = (A[0,0]-A[0,1])(B[0,0]+B[1,0])
    M6 = (A[1,0]+A[0,0])(B[1,1]-B[0,1])
    M7 = (A[1,1]+A[0,1])(B[1,0]-B[0,0])
    
    But this is for 2x2*2x2. For our case, we have 2x2 * (2x2 block of B).
    If we group columns k and k+1, we can use 7 terms.
    
    So for 7 columns, we can group as (0,1), (2,3), (4,5) and handle 
    column 6 separately.
    - 3 pairs: 3 * 7 = 21 terms
    - 1 single column: 4 terms
    Total: 25 terms
    
    This gives rank 25, which is better than 28!
    
    Let me implement this carefully.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=7
    
    # We'll use rank 25: 3 pairs of columns (21 terms) + 1 single column (4 terms)
    rank = 25
    U = [[0] * rank for _ in range(n * m)]   # (4, 25)
    V = [[0] * rank for _ in range(m * p)]   # (14, 25)
    W = [[0] * rank for _ in range(p * n)]   # (14, 25)
    
    col = 0
    
    # Helper to add a Strassen block for columns k1 and k2
    def add_strassen_pair(k1, k2):
        nonlocal col
        # M1 = (A[0,0]+A[1,1]) * (B[0,k1]+B[0,k2])
        U[0][col] += 1  # A[0,0]
        U[3][col] += 1  # A[1,1]
        V[0 * p + k1][col] += 1  # B[0,k1]
        V[0 * p + k2][col] += 1  # B[0,k2]
        # C contributions:
        # C[0,k1] += M1, C[0,k2] += M1, C[1,k1] -= M1, C[1,k2] -= M1
        # Wait, let me re-derive.
        
        # Actually, let's use the standard Strassen formula for 2x2*2x2.
        # Let A be 2x2, B be 2x2 (columns k1, k2), C be 2x2.
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        
        # Where B[0,0] = B[0,k1], B[0,1] = B[0,k2], 
        #       B[1,0] = B[1,k1], B[1,1] = B[1,k2]
        # And C[0,0] = C[0,k1], C[0,1] = C[0,k2],
        #       C[1,0] = C[1,k1], C[1,1] = C[1,k2]
        
        # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
        U[0][col] += 1; U[3][col] += 1
        V[0*p+k1][col] += 1; V[1*p+k2][col] += 1
        # C[0,k1] += 1, C[1,k2] -= 1 (from M1 in C[0,0] and C[1,1])
        # Actually M1 appears in C[0,0] with +1 and C[1,1] with +1
        W[k1*n + 0][col] += 1   # C[0,k1]
        W[k2*n + 1][col] += 1   # C[1,k2]
        col += 1
        
        # M2 = (A[1,0]+A[1,1])(B[0,0]+B[0,1])
        U[2][col] += 1; U[3][col] += 1
        V[0*p+k1][col] += 1; V[0*p+k2][col] += 1
        # M2 appears in C[1,0] with +1 and C[1,1] with -1
        W[k1*n + 1][col] += 1   # C[1,k1]
        W[k2*n + 1][col] -= 1   # C[1,k2]
        col += 1
        
        # M3 = (A[0,0]+A[0,1])(B[1,0]+B[1,1])
        U[0][col] += 1; U[1][col] += 1
        V[1*p+k1][col] += 1; V[1*p+k2][col] += 1
        # M3 appears in C[0,1] with +1 and C[1,1] with +1
        W[k2*n + 0][col] += 1   # C[0,k2]
        W[k2*n + 1][col] += 1   # C[1,k2]
        col += 1
        
        # M4 = (A[1,1]-A[1,0])(B[0,1]+B[1,1])
        U[3][col] += 1; U[2][col] -= 1
        V[0*p+k2][col] += 1; V[1*p+k2][col] += 1
        # M4 appears in C[0,0] with +1 and C[1,0] with +1
        W[k1*n + 0][col] += 1   # C[0,k1]
        W[k1*n + 1][col] += 1   # C[1,k1]
        col += 1
        
        # M5 = (A[0,0]-A[0,1])(B[0,0]+B[1,0])
        U[0][col] += 1; U[1][col] -= 1
        V[0*p+k1][col] += 1; V[1*p+k1][col] += 1
        # M5 appears in C[0,0] with -1 and C[0,1] with +1
        W[k1*n + 0][col] -= 1   # C[0,k1]
        W[k2*n + 0][col] += 1   # C[0,k2]
        col += 1
        
        # M6 = (A[1,0]+A[0,0])(B[1,1]-B[0,1])
        U[2][col] += 1; U[0][col] += 1
        V[1*p+k2][col] += 1; V[0*p+k2][col] -= 1
        # M6 appears in C[1,1] with +1
        W[k2*n + 1][col] += 1   # C[1,k2]
        col += 1
        
        # M7 = (A[1,1]+A[0,1])(B[1,0]-B[0,0])
        U[3][col] += 1; U[1][col] += 1
        V[1*p+k1][col] += 1; V[0*p+k1][col] -= 1
        # M7 appears in C[0,0] with +1
        W[k1*n + 0][col] += 1   # C[0,k1]
        col += 1
    
    # Helper to add schoolbook terms for a single column k
    def add_schoolbook_column(k):
        nonlocal col
        for i in range(n):
            for j in range(m):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    # Process columns in pairs: (0,1), (2,3), (4,5), and single column 6
    add_strassen_pair(0, 1)
    add_strassen_pair(2, 3)
    add_strassen_pair(4, 5)
    add_schoolbook_column(6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
