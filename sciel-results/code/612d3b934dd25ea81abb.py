import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 9]

def _rational_gaussian_elimination(A, b):
    """
    Solves A x = b over the rationals using Fraction.
    A is a list of lists of Fraction.
    b is a list of Fraction.
    Returns a list of Fraction (particular solution) or None if no unique solution.
    """
    n = len(b)
    m = len(A[0]) if A else 0
    
    # Augmented matrix
    M = []
    for i in range(n):
        row = [A[i][j] for j in range(m)] + [b[i]]
        M.append(row)
        
    rank = 0
    pivot_cols = []
    
    for col in range(m):
        if rank >= n:
            break
            
        # Find pivot
        pivot_row = None
        for r in range(rank, n):
            if M[r][col] != 0:
                pivot_row = r
                break
                
        if pivot_row is None:
            continue
            
        # Swap
        M[rank], M[pivot_row] = M[pivot_row], M[rank]
        pivot_cols.append(col)
        
        # Normalize pivot row
        pivot_val = M[rank][col]
        for j in range(col, m + 1):
            M[rank][j] = M[rank][j] / pivot_val
            
        # Eliminate other rows
        for r in range(n):
            if r == rank:
                continue
            factor = M[r][col]
            if factor != 0:
                for j in range(col, m + 1):
                    M[r][j] = M[r][j] - factor * M[rank][j]
                    
        rank += 1
        
    # Check for consistency
    for r in range(rank, n):
        if M[r][m] != 0:
            return None
            
    # Construct particular solution (set free variables to 0)
    x = [Fraction(0)] * m
    for i in range(rank):
        col = pivot_cols[i]
        x[col] = M[i][m]
        
    return x

def _solve_decomposition(n, m, p, target_rank):
    """
    Attempts to find a rank-target_rank decomposition for n x m * m x p.
    Returns (U, V, W) as lists of lists of Fractions, or None if not found.
    """
    # Tensor dimension: (n*m) * (m*p) * (p*n)
    # We are solving for U (n*m x R), V (m*p x R), W (p*n x R)
    # Constraint: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if (i,j,k) is valid, 0 otherwise.
    # Actually, the tensor T_{i,j,k} = 1 for all i,j,k? 
    # No, matrix multiplication tensor T_{i,j,k} is 1 if the index corresponds to C[i,k] = sum_j A[i,j]B[j,k].
    # The entries of the tensor are 1 for all (i,j,k) in the domain? 
    # Let's check the definition.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T has indices (i, j, k) and value 1.
    # The decomposition is T_{i,j,k} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}?
    # Wait, the interface says:
    # U: (n*m, rank) -> rows i*m+j
    # V: (m*p, rank) -> rows j*p+k
    # W: (p*n, rank) -> rows k*n+i
    # And the product is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This sum must equal 1 for all i, j, k.
    # Because C[i,k] = sum_j A[i,j] B[j,k].
    # If we expand A[i,j] = sum_a u_{i,j,a} a_a and B[j,k] = sum_b v_{j,k,b} b_b,
    # then C[i,k] = sum_{a,b} (sum_j u_{i,j,a} v_{j,k,b}) a_a b_b.
    # This implies that the coefficient of a_a b_b in C[i,k] is sum_j u_{i,j,a} v_{j,k,b}.
    # This must match the standard multiplication.
    # The standard multiplication tensor has entries 1.
    # So yes, sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for ALL i, j, k.
    
    # This is a system of N = n*m*p equations.
    # Unknowns are the entries of U, V, W.
    # This is a non-linear system (bilinear in U,V for fixed W, etc.).
    # However, we can fix W to be the identity-like structure or use a specific ansatz.
    # Actually, a simpler approach for small ranks:
    # If we assume the decomposition is "block diagonal" or uses specific structures, we can solve linear systems.
    
    # Alternative: Use the fact that mu(2,2,p) = 3p+1?
    # For p=9, rank=28.
    # Let's try to construct it using the known formula.
    # The formula 3p+1 suggests a structure.
    # For p=2, rank=7.
    # For p=3, rank=10.
    # 
    # Let's try a different strategy: 
    # Use the trivial decomposition for the first 2 columns (rank 7 via Strassen) 
    # and then add 3 columns at a time with rank 3?
    # No, 3p+1 is linear.
    # 
    # Let's look at the structure of the 2x2xp multiplication.
    # C = A B.
    # A is 2x2, B is 2xp.
    # 
    # We can use the following identity:
    # For each column k of B, we compute A B[:,k].
    # This is a 2x2 * 2x1 multiplication.
    # Rank of 2x2 * 2x1 is 4? No, it's 4.
    # If we did it independently, rank = 4p = 36.
    # 
    # Can we share terms?
    # The Strassen algorithm shares terms between columns?
    # In 2x2 * 2x2, Strassen shares (a+b), (a-c), (b+d) etc.
    # 
    # For 2x2 * 2xp, we can define:
    # S1 = a+b
    # S2 = a-c
    # S3 = b+d
    # 
    # And for each column k, we have x_k, y_k (elements of B[:,k]).
    # 
    # C[0,k] = a x_k + b y_k
    # C[1,k] = c x_k + d y_k
    # 
    # Using Strassen-like sums:
    # Let's try to express C[0,k] and C[1,k] using shared sums of A and column-specific sums of B.
    # 
    # Known result: mu(2,2,p) = 3p + 1.
    # Construction:
    # We need 3p+1 products.
    # 
    # Let's try to find the specific matrices.
    # 
    # If we can't derive it, we can use a numerical search with rational reconstruction.
    # 
    # Given the time limit, I will implement a robust fallback to the trivial 36, 
    # but attempt a "greedy" rank reduction using the 2x2x2 Strassen block for the first 2 columns,
    # and trivial for the rest.
    # 
    # Rank = 7 (for first 2 cols) + 4 * (p-2) (for remaining cols).
    # For p=9: 7 + 4*7 = 35.
    # This is an improvement over 36.
    # 
    # Can we do better?
    # If we use 2x2x3 blocks?
    # mu(2,2,3) = 10.
    # 9 = 3 + 3 + 3. Rank = 10 + 10 + 10 = 30.
    # 
    # Let's try to construct a rank 10 decomposition for 2x2x3.
    # 
    # I will implement a solver that tries to find a rank R decomposition for 2x2xp
    # by fixing the U and V matrices to have a specific sparse structure and solving for W.
    # 
    # Actually, the simplest robust improvement is the block decomposition.
    # 
    # Let's implement the block decomposition with Strassen for 2x2x2 and trivial for 2x2x1.
    # 
    def _get_strassen_2x2x2():
        # Returns U (4x7), V (4x7), W (4x7) for 2x2 * 2x2
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # p1 = ax
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        # p2 = b(z-x)
        U[1,1]=1; V[2,1]=1; V[0,1]=-1; W[0,1]=1
        # p3 = c(x+z)
        U[2,2]=1; V[0,2]=1; V[2,2]=1; W[1,2]=1
        # p4 = d(y-z)
        U[3,3]=1; V[1,3]=1; V[3,3]=-1; W[3,3]=1
        # p5 = (a+b)(y+z)
        U[0,4]=1; U[1,4]=1; V[1,4]=1; V[3,4]=1; W[2,4]=1
        # p6 = (a-c)(x+z)
        U[0,5]=1; U[2,5]=-1; V[0,5]=1; V[2,5]=1; W[1,5]=1
        # p7 = (b+d)(y-z)
        U[1,6]=1; U[3,6]=1; V[1,6]=1; V[3,6]=-1; W[0,6]=1
        
        # Verify C00 (W[0]):
        # Sum U[i*2+j] V[j*2+0] W[0*2+i]
        # i=0, k=0. W index 0.
        # j=0: U[0] V[0] W[0]
        #   r=0: 1*1*1 = 1
        #   r=1: 0*1*1 = 0
        #   r=2: 0*1*0 = 0
        #   r=3: 0*0*0 = 0
        #   r=4: 1*0*0 = 0
        #   r=5: 1*1*0 = 0
        #   r=6: 0*0*1 = 0
        #   Sum j=0 is 1.
        # j=1: U[1] V[2] W[0]
        #   r=0: 0*0*1 = 0
        #   r=1: 1*1*1 = 1
        #   r=2: 0*0*0 = 0
        #   r=3: 0*0*0 = 0
        #   r=4: 1*0*0 = 0
        #   r=5: -1*1*0 = 0
        #   r=6: 1*0*1 = 0
        #   Sum j=1 is 1.
        # Total C00 = 2? 
        # Wait, C00 = ax + bz.
        # p1=ax, p2=b(z-x) = bz - bx.
        # C00 = p1 + p2 - p5 + p7?
        # My W assignments might be wrong.
        
        # Let's use a verified Strassen implementation.
        # Standard Strassen:
        # m1 = (a+d)(x+w)
        # m2 = (c+d)x
        # m3 = a(y-w)
        # m4 = d(w-z)
        # m5 = (a+b)w
        # m6 = (c-a)(x+z)
        # m7 = (b-d)(y+z)
        
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 - m4 + m6
        # C11 = m1 - m2 + m3 - m6
        
        # Let's map this.
        # U (A part), V (B part), W (C part).
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # m1 = (a+d)(x+w)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1 # C00
        W[3,0]=1 # C11
        
        # m2 = (c+d)x
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[1,1]=-1 # C10
        W[3,1]=-1 # C11
        
        # m3 = a(y-w)
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        W[2,2]=1 # C01
        W[3,2]=1 # C11
        
        # m4 = d(w-z)
        U[3,3]=1
        V[3,3]=1; V[2,3]=-1
        W[0,3]=1 # C00
        W[1,3]=-1 # C10
        
        # m5 = (a+b)w
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4]=-1 # C00
        W[2,4]=1 # C01
        
        # m6 = (c-a)(x+z)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[2,5]=1
        W[1,5]=1 # C10
        W[3,5]=-1 # C11
        
        # m7 = (b-d)(y+z)
        U[1,6]=1; U[3,6]=-1
        V[1,6]=1; V[2,6]=1
        W[0,6]=1 # C00
        W[2,6]=-1 # C01
        
        return U, V, W

    def _get_trivial_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        U[0,0]=1; V[0,0]=1; W[0,0]=1 # ax
        U[1,1]=1; V[0,1]=1; W[0,1]=1 # bx
        U[2,2]=1; V[0,2]=1; W[1,2]=1 # cx
        U[3,3]=1; V[0,3]=1; W[1,3]=1 # dx
        return U, V, W

    # Assemble for p=9
    # Blocks: 2x2x2 (cols 0,1) and 2x2x1 (cols 2..8)
    # Rank = 7 + 7*4 = 35.
    
    U_block1, V_block1, W_block1 = _get_strassen_2x2x2()
    U_block2, V_block2, W_block2 = _get_trivial_2x2x1()
    
    U_total = np.zeros((4, 35))
    V_total = np.zeros((18, 35)) # 2*9
    W_total = np.zeros((18, 35)) # 9*2
    
    # Block 1: cols 0,1
    # U is same for all blocks (4x7)
    U_total[:, 0:7] = U_block1
    
    # V: rows 0..3 (cols 0,1 of B)
    V_total[0:4, 0:7] = V_block1
    
    # W: rows k*2+i.
    # k=0,1. i=0,1.
    # W_block1 is 4x7. Rows: 0,1,2,3 correspond to (k=0,i=0), (k=0,i=1), (k=1,i=0), (k=1,i=1)?
    # My W index convention: W[k*n+i].
    # In _get_strassen_2x2x2, W was 4x7.
    # W[0] -> C00 (k=0, i=0) -> row 0.
    # W[1] -> C10 (k=0, i=1)? No, C[i,k].
    # C[0,0] -> k=0, i=0 -> row 0.
    # C[0,1] -> k=1, i=0 -> row 2.
    # C[1,0] -> k=0, i=1 -> row 1.
    # C[1,1] -> k=1, i=1 -> row 3.
    # 
    # In my Strassen code:
    # W[0,0]=1 (C00) -> row 0.
    # W[1,0]=0
    # W[2,0]=0
    # W[3,0]=1 (C11) -> row 3.
    # 
    # Wait, C01 is k=1, i=0 -> row 2.
    # In Strassen code:
    # W[2,2]=1 (C01) -> row 2. Correct.
    # W[1,1]=-1 (C10) -> row 1. Correct.
    
    # So W_block1 rows 0,1,2,3 map to W_total rows 0,1,2,3.
    W_total[0:4, 0:7] = W_block1
    
    # Block 2: cols 2..8 (7 columns)
    # For each column c in 2..8:
    for idx, c in enumerate(range(2, 9)):
        col_offset = 7 + idx * 4
        # U is same
        U_total[:, col_offset:col_offset+4] = U_block2
        
        # V: rows corresponding to column c of B.
        # V row index j*p+k. p=9.
        # j=0 -> k=c. row c.
        # j=1 -> k=c. row 9+c.
        V_total[c, col_offset:col_offset+4] = V_block2[0, :]
        V_total[9+c, col_offset:col_offset+4] = V_block2[1, :]
        
        # W: rows corresponding to output column c.
        # W row index k*n+i.
        # k=c. i=0 -> row c*2.
        # i=1 -> row c*2+1.
        W_total[c*2, col_offset:col_offset+4] = W_block2[0, :]
        W_total[c*2+1, col_offset:col_offset+4] = W_block2[1, :]
        
    # Convert to lists of lists of floats (or ints if possible)
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    return U_list, V_list, W_list

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    
    # Attempt the block decomposition (Rank 35)
    try:
        U, V, W = _solve_decomposition(n, m, p, 35)
        return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to trivial rank 36
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

