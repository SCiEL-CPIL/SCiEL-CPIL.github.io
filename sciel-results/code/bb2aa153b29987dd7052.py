import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    Uses a block decomposition 5 = 2 + 3.
    Uses Strassen (rank 7) for the 2x2 block product A11*B11.
    Uses Schoolbook for all other block products.
    Total rank = 7 (Strassen) + 12 + 12 + 18 + 12 + 18 + 18 + 27 (Schoolbook) = 124.
    """
    start_time = time.time()
    DEADLINE = 140  # seconds
    
    n, m, p = DIMENSIONS
    
    # Helper to create a zero matrix
    def zeros(rows, cols):
        return [[0] * cols for _ in range(rows)]
    
    # We will build lists of non-zero entries for U, V, W
    # U: (25, Rank), V: (25, Rank), W: (25, Rank)
    # We use sparse representation: list of (row, col, val)
    # But since Rank is unknown until we sum up, we can just build the matrices directly.
    # Max rank estimate: 125.
    # Let's just build the matrices with a fixed size and then trim?
    # No, let's build lists of columns.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_indices, v_indices, w_indices):
        """
        u_indices: list of (local_i, local_j, val) for A
        v_indices: list of (local_j, local_k, val) for B
        w_indices: list of (local_k, local_i, val) for C
        """
        # We need to map local indices to global indices.
        # This function assumes we are inside a specific block context.
        # So it's better to pass the offset or use closures.
        pass

    # Define block offsets
    # A: row i, col j
    # B: row j, col k
    # C: row k, col i (Trace convention: W[k*n+i])
    
    # Block 11: A[0:2, 0:2], B[0:2, 0:2], C[0:2, 0:2]
    # Block 12: A[0:2, 2:5], B[0:2, 2:5], C[2:5, 0:2]
    # Block 21: A[2:5, 0:2], B[2:5, 0:2], C[0:2, 2:5]
    # Block 22: A[2:5, 2:5], B[2:5, 2:5], C[2:5, 2:5]
    
    def process_block(a_row_start, a_col_start, a_rows, a_cols, 
                      b_row_start, b_col_start, b_rows, b_cols, 
                      c_row_start, c_col_start, c_rows, c_cols, 
                      decomposition_func, 
                      a_local_to_global, b_local_to_global, c_local_to_global):
        """
        decomposition_func returns (U_local, V_local, W_local) 
        where U_local is (a_rows*a_cols, rank), etc.
        """
        U_loc, V_loc, W_loc = decomposition_func()
        rank = U_loc.shape[1]
        
        for t in range(rank):
            # Build global U column
            u_col = [0] * 25
            v_col = [0] * 25
            w_col = [0] * 25
            
            # Map U
            for i in range(a_rows):
                for j in range(a_cols):
                    val = U_loc[i * a_cols + j, t]
                    if val != 0:
                        gi, gj = a_local_to_global(i, j)
                        u_col[gi * 5 + gj] = val
                        
            # Map V
            for j in range(b_rows):
                for k in range(b_cols):
                    val = V_loc[j * b_cols + k, t]
                    if val != 0:
                        gj, gk = b_local_to_global(j, k)
                        v_col[gj * 5 + gk] = val
                        
            # Map W
            for k in range(c_rows):
                for i in range(c_cols):
                    val = W_loc[k * c_cols + i, t]
                    if val != 0:
                        gk, gi = c_local_to_global(k, i)
                        w_col[gk * 5 + gi] = val
                        
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)

    # --- Decomposition Functions ---
    
    def strassen_2x2():
        """
        Returns (U, V, W) for 2x2x2 with rank 7.
        U: (4, 7), V: (4, 7), W: (4, 7)
        Local indices:
        U[i, j] -> A[i, j]
        V[j, k] -> B[j, k]
        W[k, i] -> C[i, k] (Trace convention: W[k*n+i] where n=2)
        So W index is k*2 + i.
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
        # A indices: 00=0, 01=1, 10=2, 11=3
        # B indices: 00=0, 01=1, 10=2, 11=3
        # C indices: 00=0, 01=1, 10=2, 11=3 (Wait, trace convention W[k*n+i])
        # C[i,k] is at W[k*2+i].
        # C00 -> W[0*2+0] = W[0]
        # C11 -> W[1*2+1] = W[3]
        
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00 -> C10+=M2
        # C10 -> W[0*2+1] = W[1]
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        # M3 = A00(B10-B11) -> C01+=M3, C11+=M3
        # C01 -> W[1*2+0] = W[2]
        # C11 -> W[1*2+1] = W[3]
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1
        
        # M4 = A11(B01-B10) -> C00+=M4, C10+=M4
        # C00 -> W[0]
        # C10 -> W[1]
        U[3, 3] = 1
        V[1, 3] = 1; V[2, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # M5 = (A00+A01)B11 -> C01+=-M5, C11+=M5
        # C01 -> W[2]
        # C11 -> W[3]
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[2, 4] = -1; W[3, 4] = 1
        
        # M6 = (A10-A00)(B00+B10) -> C11+=M6
        # C11 -> W[3]
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B01+B11) -> C00+=M7
        # C00 -> W[0]
        U[1, 6] = 1; U[3, 6] = -1
        V[1, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def schoolbook(n_a, n_b, n_c):
        """
        Returns (U, V, W) for n_a x n_b x n_c schoolbook.
        U: (n_a*n_b, rank), V: (n_b*n_c, rank), W: (n_c*n_a, rank)
        Rank = n_a * n_b * n_c
        """
        rank = n_a * n_b * n_c
        U = np.zeros((n_a * n_b, rank))
        V = np.zeros((n_b * n_c, rank))
        W = np.zeros((n_c * n_a, rank))
        col = 0
        for i in range(n_a):
            for j in range(n_b):
                for k in range(n_c):
                    U[i * n_b + j, col] = 1
                    V[j * n_c + k, col] = 1
                    W[k * n_a + i, col] = 1
                    col += 1
        return U, V, W

    # --- Process Blocks ---
    
    # Block 11: 2x2
    # A: i in 0,1; j in 0,1
    # B: j in 0,1; k in 0,1
    # C: k in 0,1; i in 0,1
    def map_a_11(i, j): return (i, j)
    def map_b_11(j, k): return (j, k)
    def map_c_11(k, i): return (k, i)
    process_block(0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, strassen_2x2, map_a_11, map_b_11, map_c_11)
    
    # Block 12: 2x3
    # A: i in 0,1; j in 2,3,4 -> Local j' in 0,1,2
    # B: j in 0,1; k in 2,3,4 -> Local k' in 0,1,2
    # C: k in 2,3,4; i in 0,1 -> Local k' in 0,1,2, i' in 0,1
    # Note: C is 3x2 in local coords (k' x i'), but global is 5x5.
    # Schoolbook for 2x3 by 2x3? No.
    # A12 is 2x3. B12 is 2x3.
    # Product A11 * B12? No, C12 = A11*B12 + A12*B22.
    # We are processing C12 = A11*B12 + A12*B22.
    # This block decomposition splits the sum.
    # Term 1: A11 (2x2) * B12 (2x3) -> C_part1 (2x3)
    # Term 2: A12 (2x3) * B22 (3x3) -> C_part2 (2x3)
    
    # Let's process Term 1: A11 * B12
    # A11: 2x2. B12: 2x3. Output: 2x3.
    # n_a=2, n_b=2, n_c=3.
    def map_a_t1(i, j): return (i, j) # A11 local
    def map_b_t1(j, k): return (j, 2+k) # B12 local j->global j, k->global 2+k
    def map_c_t1(k, i): return (2+k, i) # C local k->global 2+k, i->global i
    process_block(0, 0, 2, 2, 0, 2, 2, 3, 2, 0, 3, 2, lambda: schoolbook(2, 2, 3), map_a_t1, map_b_t1, map_c_t1)
    
    # Term 2: A12 * B22
    # A12: 2x3. B22: 3x3. Output: 2x3.
    # n_a=2, n_b=3, n_c=3.
    def map_a_t2(i, j): return (i, 2+j) # A12 local j->global 2+j
    def map_b_t2(j, k): return (2+j, 2+k) # B22 local
    def map_c_t2(k, i): return (2+k, i) # C local
    process_block(0, 2, 2, 3, 2, 2, 3, 3, 2, 0, 3, 2, lambda: schoolbook(2, 3, 3), map_a_t2, map_b_t2, map_c_t2)
    
    # Block 21: 3x2
    # C21 = A21*B11 + A22*B21
    # Term 1: A21 (3x2) * B11 (2x2) -> C_part1 (3x2)
    # n_a=3, n_b=2, n_c=2.
    def map_a_t3(i, j): return (2+i, j) # A21 local
    def map_b_t3(j, k): return (j, k) # B11 local
    def map_c_t3(k, i): return (k, 2+i) # C local k->global k, i->global 2+i
    process_block(2, 0, 3, 2, 0, 0, 2, 2, 0, 2, 2, 3, lambda: schoolbook(3, 2, 2), map_a_t3, map_b_t3, map_c_t3)
    
    # Term 2: A22 (3x3) * B21 (3x2) -> C_part2 (3x2)
    # n_a=3, n_b=3, n_c=2.
    def map_a_t4(i, j): return (2+i, 2+j) # A22 local
    def map_b_t4(j, k): return (2+j, k) # B21 local
    def map_c_t4(k, i): return (k, 2+i) # C local
    process_block(2, 2, 3, 3, 2, 0, 3, 2, 0, 2, 2, 3, lambda: schoolbook(3, 3, 2), map_a_t4, map_b_t4, map_c_t4)
    
    # Block 22: 3x3
    # C22 = A22*B22
    # n_a=3, n_b=3, n_c=3.
    def map_a_t5(i, j): return (2+i, 2+j) # A22 local
    def map_b_t5(j, k): return (2+j, 2+k) # B22 local
    def map_c_t5(k, i): return (2+k, 2+i) # C local
    process_block(2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, lambda: schoolbook(3, 3, 3), map_a_t5, map_b_t5, map_c_t5)

    # Assemble matrices
    rank = len(U_cols)
    U = [[0] * rank for _ in range(25)]
    V = [[0] * rank for _ in range(25)]
    W = [[0] * rank for _ in range(25)]
    
    for t in range(rank):
        for i in range(25):
            U[i][t] = U_cols[t][i]
            V[i][t] = V_cols[t][i]
            W[i][t] = W_cols[t][i]
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
