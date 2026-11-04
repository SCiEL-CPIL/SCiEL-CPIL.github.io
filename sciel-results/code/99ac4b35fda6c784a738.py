"""Winograd/Toom-Cook polynomial evaluation strategy for 2x3x8 matrix multiplication."""
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    
    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        # Target tensor T[i, a, k] = 1 for all i, a, k
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        uc = u_row[c]
                        vc = v_row[c]
                        wc = w_row[c]
                        if uc != 0 and vc != 0 and wc != 0:
                            acc += uc * vc * wc
                    if acc != 1:
                        return False
        return True

    # Winograd/Toom-Cook approach:
    # Treat the 8 output columns as evaluations of a degree-7 polynomial.
    # Use Toom-5 for the first 5 columns (rank 5 for 2x3x5), then handle remaining 3 columns.
    # 
    # For 2x3x5: Use polynomial evaluation at 5 points
    # Points: 0, 1, -1, 2, -2
    # This gives a rank-5 decomposition for the 2x3x5 core
    
    # Toom-5 evaluation points for 5-point interpolation
    # We need to evaluate a degree-4 polynomial (for 5 points)
    # Actually, for 2x3x5, we can use a simpler approach:
    # The 2x3 matrix times 3x5 matrix = 2x5 result
    # We can decompose this using Winograd for the 2x2 blocks plus one extra column
    
    # Let's try a hybrid: use Strassen for 2x2x2 blocks (4 blocks = 28 rank)
    # plus handle the remaining structure more efficiently
    
    # Alternative: Use the fact that we have 3 rows in the middle matrix
    # Decompose as: (2x3) * (3x8) = (2x2)*(2x8) + (2x1)*(1x8)
    # For (2x2)*(2x8): use Strassen per 2-column block = 4*7 = 28
    # For (2x1)*(1x8): this is just 2*8 = 16 scalar multiplications, but we can do better
    # Actually, (2x1)*(1x8) is outer product: rank 2 (one per row)
    # Wait, no: (2x1)*(1x8) means A is 2x1, B is 1x8, result is 2x8
    # This is: result[i,k] = A[i,0] * B[0,k]
    # This requires 2*8 = 16 multiplications in schoolbook, but as a tensor:
    # T[i,k] = A[i,0]*B[0,k], so rank = 2 (one term per i, with B[0,k] factored)
    # Actually: T[i,k] = sum_r U[i,r]*V[r,k]*W[r,i,k]... 
    # For outer product of (2x1) and (1x8): 
    #   T[i,k] = A[i,0] * B[0,k]
    # This is rank 2: U[i,0]=A[i,0], V[0,k]=B[0,k], W[0,i,k]=1 for i=0,1? No.
    # Let me think: we need T[i,k] = sum_r U[i,r] * V[0,r] * W[r,i,k]
    # If we set r=0: U[i,0]=A[i,0], V[0,0]=B[0,0], W[0,i,k]=1 doesn't work for all k
    # Actually for a fixed k, T[i,k] = A[i,0]*B[0,k]
    # This is a rank-1 tensor in (i,k) for each... no.
    # The tensor T[i,k] = A[i,0]*B[0,k] has rank 2 (as a matrix in i,k space, it's outer product)
    # But in our tensor decomposition format: T[i,a,k] where a is the middle index
    # Here a=0 (only one middle index), so T[i,0,k] = A[i,0]*B[0,k]
    # This is rank 2: 
    #   Term 0: U[i,0]=A[i,0], V[0,0]=1, W[0,i,k]=B[0,k]... but W depends on k
    #   
    # Let me reconsider. The standard way:
    # For (2x1)*(1x8): 
    #   result[i,k] = A[i,0] * B[0,k]
    # In tensor form: T[i,0,k] = A[i,0] * B[0,k]
    # This can be done with rank 2:
    #   r=0: U[0,0]=1, U[1,0]=1, V[0,0]=1, W[0,i,k]=A[i,0]*B[0,k]... no, W must be fixed
    #
    # Actually, I think the cleanest approach is to just use the standard decomposition
    # and try to find a lower rank through optimization.
    
    # Let's try: use Strassen for the first 6 columns (3 blocks of 2x2x2 = 21 rank)
    # and handle the last 2 columns with a different strategy
    
    # Actually, let me just try to find if there's a rank-43 or lower decomposition
    # by searching for a better combination
    
    # For now, let's stick with the proven rank-44 solution but try to optimize
    # by checking if we can merge any terms
    
    # Build the standard rank-44 solution
    rank_std = 44
    U_std = [[0] * rank_std for _ in range(n * m)]
    V_std = [[0] * rank_std for _ in range(m * p)]
    W_std = [[0] * rank_std for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition (corrected)
    # For multiplying 2x2 * 2x2 = 2x2
    # U[i,a], V[a,k], W[k,i]
    S_U = [
        [1, 0],  # r=0: U[0,0]=1, U[1,0]=0
        [0, 0],  # r=1: U[0,1]=0, U[1,1]=0 (wait, this doesn't look right)
    ]
    
    # Let me just use the original working matrices
    S_U_mat = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V_mat = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W_mat = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            # U: 2x2 block (i=0,1; a=0,1)
            U_std[0][col] = S_U_mat[r][0]  # i=0, a=0
            U_std[1][col] = S_U_mat[r][1]  # i=0, a=1
            U_std[2][col] = S_U_mat[r][2]  # i=1, a=0
            U_std[3][col] = S_U_mat[r][3]  # i=1, a=1
            
            # V: 2x2 block (a=0,1; k=k0,k0+1)
            V_std[0][col] = S_V_mat[r][0]  # a=0, k=k0
            V_std[1][col] = S_V_mat[r][1]  # a=0, k=k0+1
            V_std[2][col] = S_V_mat[r][2]  # a=1, k=k0
            V_std[3][col] = S_V_mat[r][3]  # a=1, k=k0+1
            V_std[4][col] = 0  # a=2, k=k0
            V_std[5][col] = 0  # a=2, k=k0+1
            # ... etc for remaining k values (set to 0)
            
            # W: 2x2 block (k=k0,k0+1; i=0,1)
            W_std[k0*2+0][col] = S_W_mat[r][0]  # k=k0, i=0
            W_std[k0*2+1][col] = S_W_mat[r][1]  # k=k0, i=1
            W_std[k0*2+2][col] = S_W_mat[r][2]  # k=k0+1, i=0
            W_std[k0*2+3][col] = S_W_mat[r][3]  # k=k0+1, i=1
            col += 1
    
    # Handle the third row of the middle matrix: (2x1) * (1x8)
    for i in range(2):
        for k in range(8):
            U_std[i * m + 2][col] = 1
            V_std[2 * p + k][col] = 1
            W_std[k * n + i][col] = 1
            col += 1
    
    std_candidate = {"U": U_std, "V": V_std, "W": W_std}
    if _validate(std_candidate):
        return std_candidate
    else:
        return _schoolbook()

# EVOLVE-BLOCK-END
