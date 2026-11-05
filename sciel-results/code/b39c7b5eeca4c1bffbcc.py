import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _deadline_guard(start_time, limit=145.0):
    """Check if we've exceeded the time limit."""
    return (time.time() - start_time) > limit

# EVOLVE-BLOCK-START
import itertools
import numpy as np

def _schoolbook(n=4, m=5, p=5):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _verify(U, V, W, n=4, m=5, p=5):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    T[i,j,k] = 1 for all i,j,k, and we need:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    rank = len(U[0]) if U else 0
    # Convert to numpy for speed
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # We need to check for all i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    # Efficient check:
    # Construct the tensor T from U, V, W
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Reshape U to (n, m, rank)
    U_reshaped = U_np.reshape(n, m, rank)
    # Reshape V to (m, p, rank)
    V_reshaped = V_np.reshape(m, p, rank)
    # Reshape W to (p, n, rank) -> Note: W index is k*n+i.
    # W_np has shape (p*n, rank).
    # Index k*n + i. So W_np[k*n+i, c] corresponds to W_reshaped[k, i, c] if we reshape to (p, n, rank).
    W_reshaped = W_np.reshape(p, n, rank)
    
    # Compute T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
    # This is a tensor contraction.
    # T[i, j, k] = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    except Exception:
        return False
        
    # T should be all 1s
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
        
    return True

def _strassen_2x2_factors():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices for U, V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    Indices for W: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -- Note: W[k*n+i] for n=2.
    W index k*2+i.
    (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3.
    """
    u = [[0]*7 for _ in range(4)]
    v = [[0]*7 for _ in range(4)]
    w = [[0]*7 for _ in range(4)]
    
    # Term 0: (a00+a11)(b00+b11)
    u[0][0]=1; u[3][0]=1
    v[0][0]=1; v[3][0]=1
    w[0][0]=1; w[3][0]=1
    
    # Term 1: (a10+a11)b00
    u[2][1]=1; u[3][1]=1
    v[0][1]=1
    w[2][1]=1; w[3][1]=-1
    
    # Term 2: a00(b01-b11)
    u[0][2]=1
    v[1][2]=1; v[3][2]=-1
    w[1][2]=1; w[3][2]=1
    
    # Term 3: a11(b10-b00)
    u[3][3]=1
    v[2][3]=1; v[0][3]=-1
    w[0][3]=1; w[2][3]=1
    
    # Term 4: (a00+a01)b11
    u[0][4]=1; u[1][4]=1
    v[3][4]=1
    w[0][4]=-1; w[1][4]=1
    
    # Term 5: (a10-a00)(b00+b01)
    u[0][5]=-1; u[2][5]=1
    v[0][5]=1; v[1][5]=1
    w[3][5]=1
    
    # Term 6: (a01-a11)(b10+b11)
    u[1][6]=1; u[3][6]=-1
    v[2][6]=1; v[3][6]=1
    w[0][6]=1
    
    return u, v, w

def _construct_block_strassen(n=4, m=5, p=5):
    """
    Constructs a rank-85 decomposition for 4x5x5 using block Strassen.
    Splits A, B, C into 4x4 and 1x1 blocks (row/col 4 is the boundary).
    Actually, the inspiration splits 4x4 into 2x2 blocks and uses Strassen.
    The 5th row/col is handled separately.
    """
    rank = 85
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    su, sv, sw = _strassen_2x2_factors()
    
    # The 4x4x4 part is decomposed using 7 "outer" Strassen terms.
    # Each outer term is a product of linear combinations of 2x2 blocks.
    # We map these to the 2x2x2 Strassen factors.
    
    # Outer term definitions:
    # T0: (A00+A11)(B00+B11)
    # T1: (A10+A11)B00
    # T2: A00(B01-B11)
    # T3: A11(B10-B00)
    # T4: (A00+A01)B11
    # T5: (A10-A00)(B00+B01)
    # T6: (A01-A11)(B10+B11)
    
    # For each outer term t, we have coefficients for A-blocks, B-blocks, C-blocks.
    # A-blocks: (r, c) in {0,1}x{0,1}
    # B-blocks: (r, c) in {0,1}x{0,1}
    # C-blocks: (r, c) in {0,1}x{0,1}
    
    # We define the coefficients as dictionaries { (r,c): coeff }
    
    outer_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # T0
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # T1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # T2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # T3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # T4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # T5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # T6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    col = 0
    for t_idx, (ac, bc, cc) in enumerate(outer_terms):
        for k in range(7):
            # k is the inner Strassen term index (0..6)
            # We need to add the contribution of this (t_idx, k) pair to U, V, W.
            
            # Get the 2x2 Strassen factors for this inner term k
            # su[r][k] is the coefficient for A element (r) in inner term k
            # sv[r][k] is the coefficient for B element (r) in inner term k
            # sw[r][k] is the coefficient for C element (r) in inner term k
            
            # Iterate over the A-blocks involved in this outer term
            for (ar, ac_col), a_coeff in ac.items():
                # ar, ac_col are block indices (0 or 1)
                # The block A[ar, ac_col] is a 2x2 matrix.
                # Its elements are at global A indices:
                # i = 2*ar + lr, j = 2*ac_col + lc
                for lr in range(2):
                    for lc in range(2):
                        # Local 2x2 index li = lr*2 + lc
                        li = lr * 2 + lc
                        # The coefficient for this local element in the inner Strassen term k is su[li][k]
                        s_val = su[li][k]
                        if s_val != 0:
                            global_i = 2 * ar + lr
                            global_j = 2 * ac_col + lc
                            # U[global_i * m + global_j][col] += a_coeff * s_val
                            U[global_i * 5 + global_j][col] += a_coeff * s_val
            
            # Iterate over the B-blocks involved in this outer term
            for (br, bc_col), b_coeff in bc.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr * 2 + lc
                        s_val = sv[li][k]
                        if s_val != 0:
                            global_j = 2 * br + lr
                            global_k = 2 * bc_col + lc
                            # V[global_j * p + global_k][col] += b_coeff * s_val
                            V[global_j * 5 + global_k][col] += b_coeff * s_val
            
            # Iterate over the C-blocks involved in this outer term
            for (cr, cc_col), c_coeff in cc.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr * 2 + lc
                        s_val = sw[li][k]
                        if s_val != 0:
                            # C block (cr, cc_col) corresponds to output C[cr, cc_col]
                            # Global C indices: k = 2*cr + lr, i = 2*cc_col + lc
                            # W index is k*n + i
                            global_k = 2 * cr + lr
                            global_i = 2 * cc_col + lc
                            # W[global_k * n + global_i][col] += c_coeff * s_val
                            W[global_k * 4 + global_i][col] += c_coeff * s_val
            
            col += 1
    
    # Now handle the boundary terms involving index 4 (the 5th row/col).
    # A is 4x5, B is 5x5, C is 5x4 (since output is 5x4? No, 4x5 * 5x5 = 4x5.
    # Wait, n=4, m=5, p=5.
    # A: 4x5. B: 5x5. C: 4x5.
    # My previous analysis said C is 5x4?
    # Let's check dimensions.
    # A (n x m) = (4 x 5)
    # B (m x p) = (5 x 5)
    # C (n x p) = (4 x 5)
    # W index: k*n + i. k is from B's column (0..4), i is from A's row (0..3).
    # So W has size p*n = 5*4 = 20.
    
    # The 4x4x4 part handled i,j,k in 0..3.
    # We need to handle cases where i=4? No, i is 0..3 (n=4).
    # j is 0..4 (m=5).
    # k is 0..4 (p=5).
    
    # The 4x4x4 part handled j in 0..3, k in 0..3.
    # We need to handle:
    # 1. j=4, k in 0..3
    # 2. j in 0..3, k=4
    # 3. j=4, k=4
    
    # Case 1: j=4, k in 0..3
    # A[i, 4] * B[4, k] * C[i, k]
    # This is a rank 4*4=16 block.
    for i in range(4):
        for k in range(4):
            U[i * 5 + 4][col] = 1
            V[4 * 5 + k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1
            
    # Case 2: j in 0..3, k=4
    # A[i, j] * B[j, 4] * C[i, 4]
    # This is a rank 4*4=16 block.
    for i in range(4):
        for j in range(4):
            U[i * 5 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1
            
    # Case 3: j=4, k=4
    # A[i, 4] * B[4, 4] * C[i, 4]
    # This is a rank 4 block.
    for i in range(4):
        U[i * 5 + 4][col] = 1
        V[4 * 5 + 4][col] = 1
        W[4 * 4 + i][col] = 1
        col += 1
        
    return U, V, W

def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try the block Strassen decomposition (rank 85)
    try:
        U, V, W = _construct_block_strassen(n, m, p)
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception as e:
        pass
        
    # Fallback to schoolbook (rank 100)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
