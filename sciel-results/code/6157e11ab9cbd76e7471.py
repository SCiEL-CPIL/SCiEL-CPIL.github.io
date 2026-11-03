"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
_START_TIME = time.time()
TIME_LIMIT = 110.0  # safety margin before 120s kill

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    """
    Construct a rank-reduced tensor decomposition for 4x4 * 4x6 matrix multiplication
    using Strassen's algorithm on 2x2 blocks.
    
    We split A (4x4) into 2x2 blocks: A11, A12, A21, A22
    We split B (4x6) into 2x2 blocks: B11, B12, B13, B21, B22, B23
    Output C (4x6) has 2x2 blocks: C11, C12, C13, C21, C22, C23
    
    For each output block C[i,j] (i in {1,2}, j in {1,2,3}):
    C[i,j] = A[i,1]*B[1,j] + A[i,2]*B[2,j]
    
    Each term is a 2x2 * 2x2 product. We use Strassen to reduce multiplications.
    """
    n, m, p = 4, 4, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_term(offset_i, offset_j, offset_k):
        """
        Add Strassen decomposition of X_block * Y_block where:
        X_block is a 2x2 block of A at (offset_i, offset_j) 
        Y_block is a 2x2 block of B at (offset_j, offset_k)
        offset_i: starting row index in A (0 or 2)
        offset_j: starting col index in A / row index in B (0 or 2)
        offset_k: starting col index in B (0, 2, or 4)
        """
        # Strassen multiplications for X*Y where X is 2x2, Y is 2x2:
        # x00 = A[offset_i, offset_j], x01 = A[offset_i, offset_j+1]
        # x10 = A[offset_i+1, offset_j], x11 = A[offset_i+1, offset_j+1]
        # y00 = B[offset_j, offset_k], y01 = B[offset_j, offset_k+1]
        # y10 = B[offset_j+1, offset_k], y11 = B[offset_j+1, offset_k+1]
        # C00 = C[offset_i, offset_k], C01 = C[offset_i, offset_k+1]
        # C10 = C[offset_i+1, offset_k], C11 = C[offset_i+1, offset_k+1]
        
        mults = [
            # m1 = (x00+x11)(y00+y11)
            ({(offset_i, offset_j): 1, (offset_i+1, offset_j+1): 1},
             {(offset_j, offset_k): 1, (offset_j+1, offset_k+1): 1},
             {(offset_i, offset_k): 1, (offset_i+1, offset_k+1): 1}),
            # m2 = (x10+x11)y00
            ({(offset_i+1, offset_j): 1, (offset_i+1, offset_j+1): 1},
             {(offset_j, offset_k): 1},
             {(offset_i+1, offset_k): 1, (offset_i+1, offset_k+1): -1}),
            # m3 = x00(y01-y11)
            ({(offset_i, offset_j): 1},
             {(offset_j, offset_k+1): 1, (offset_j+1, offset_k+1): -1},
             {(offset_i, offset_k+1): 1, (offset_i+1, offset_k+1): 1}),
            # m4 = x11(y10-y00)
            ({(offset_i+1, offset_j+1): 1},
             {(offset_j+1, offset_k): 1, (offset_j, offset_k): -1},
             {(offset_i, offset_k): 1, (offset_i+1, offset_k): 1}),
            # m5 = (x00+x01)y11
            ({(offset_i, offset_j): 1, (offset_i, offset_j+1): 1},
             {(offset_j+1, offset_k+1): 1},
             {(offset_i, offset_k): -1, (offset_i, offset_k+1): 1}),
            # m6 = (x10-x00)(y00+y01)
            ({(offset_i+1, offset_j): 1, (offset_i, offset_j): -1},
             {(offset_j, offset_k): 1, (offset_j, offset_k+1): 1},
             {(offset_i+1, offset_k+1): 1}),
            # m7 = (x01-x11)(y10+y11)
            ({(offset_i, offset_j+1): 1, (offset_i+1, offset_j+1): -1},
             {(offset_j+1, offset_k): 1, (offset_j+1, offset_k+1): 1},
             {(offset_i, offset_k): 1}),
        ]
        
        for U_coeffs, V_coeffs, W_coeffs in mults:
            U_col = [0] * (n * m)
            V_col = [0] * (m * p)
            W_col = [0] * (p * n)
            
            for (i, j), coeff in U_coeffs.items():
                U_col[i * m + j] = coeff
            
            for (j, k), coeff in V_coeffs.items():
                V_col[j * p + k] = coeff
            
            for (i, k), coeff in W_coeffs.items():
                W_col[k * n + i] = coeff
            
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)
    
    # For each output block (rb, cb):
    # C[rb, cb] = A[rb, 0]*B[0, cb] + A[rb, 1]*B[1, cb]
    # Use Strassen for A[rb,0]*B[0,cb] and A[rb,1]*B[1,cb]
    
    for rb in range(2):  # row block of A/C
        for cb in range(3):  # col block of B/C
            if _time_remaining() < 5.0:
                break
            # Term 1: A[rb, 0] * B[0, cb]
            add_strassen_term(rb*2, 0, cb*2)
            
            # Term 2: A[rb, 1] * B[1, cb]
            add_strassen_term(rb*2, 2, cb*2)
    
    rank = len(U_cols)
    
    # Transpose to get U as (n*m, rank), V as (m*p, rank), W as (p*n, rank)
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
