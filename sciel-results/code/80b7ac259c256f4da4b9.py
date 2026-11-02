"""Matrix multiplication 2,3,7 tensor decomposition via Strassen blocks."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or low-rank) tensor decomposition for multiplying
    a 2x3 matrix by a 3x7 matrix.
    
    Strategy: Split the 2x3 matrix A into two parts:
      - A1: first 2 columns (2x2)
      - A2: last 1 column (2x1)
    
    Correspondingly split B (3x7) into:
      - B1: first 2 rows (2x7)
      - B2: last 1 row (1x7)
    
    Then C = A*B = A1*B1 + A2*B2
    
    For A1*B1 (2x2 times 2x7): Use Strassen decomposition.
    We split B1 into three parts: B1a (2x2), B1b (2x2), B1c (2x3).
    Use Strassen for 2x2*2x2 blocks and trivial for 2x2*2x3.
    
    Actually, let's use a cleaner approach: 
    For 2x2 * 2x7, we can treat it as three 2x2*2x2 blocks plus one 2x2*2x1 block.
    But 7 = 2+2+2+1.
    
    Let me use Strassen for the 2x2*2x2 parts (rank 7 each) and trivial for 2x2*2x1 (rank 4).
    Total for A1*B1: 3*7 + 4 = 25
    
    For A2*B2 (2x1 * 1x7): This is an outer product structure.
    C2[i][k] = A2[i][0] * B2[0][k]
    This is rank 14 (each of the 14 entries needs its own term in general, 
    but actually since it's rank-1 in the bilinear sense, we can decompose it with rank 14).
    
    Wait, let me reconsider. The tensor for 2x1*1x7 has dimensions:
    U: 2*1 = 2, V: 1*7 = 7, W: 7*2 = 14
    The bilinear form is C[i][k] = A[i][0] * B[0][k]
    This is a rank-1 tensor in the sense that it's a single product, but in the 
    tensor decomposition sense (sum of rank-1 tensors), it takes rank 14 because
    each (i,k) pair is independent.
    
    Actually no. Let me think again. The tensor T has T[i*1+0, 0*7+k, k*2+i] = 1 
    for all i in {0,1} and k in {0..6}. This is a 2x7x14 tensor where
    T[a, b, c] is nonzero only when a=i*1+0, b=0*7+k, c=k*2+i.
    
    For a fixed (i,k), we need U[i, r] * V[k, r] * W[k*2+i, r] to contribute to C[i][k].
    Since the bilinear form is C[i][k] = sum_r U[i,r]*V[k,r]*W[k*2+i,r] = A[i]*B[k],
    this is a rank-1 tensor in the bilinear form sense. But in the tensor decomposition 
    sense (where each term is a product of three vectors), we need at least 14 terms 
    because the supports on the W dimension are disjoint for different (i,k) pairs? 
    
    No wait. Let me be more careful. The matrix multiplication tensor for m1 x m2 * m2 x m3 
    has U of size m1*m2, V of size m2*m3, W of size m3*m1.
    
    For 2x1 * 1x7: m1=2, m2=1, m3=7.
    U: 2*1 = 2, V: 1*7 = 7, W: 7*2 = 14.
    The tensor is T[a,b,c] = 1 if a = i*1+0, b = 0*7+k, c = k*2+i for some i,k.
    
    So T[i, k, k*2+i] = 1 for i in {0,1}, k in {0,...,6}.
    
    This tensor has rank 14 because each of the 14 entries is in a different position 
    and they don't overlap in a way that allows sharing.
    
    Actually, I can see this more simply: the bilinear form is 
    C[i,k] = A[i] * B[k] where A has 2 components and B has 7 components.
    This is a rank-1 bilinear form, meaning it can be written as a single product 
    of two linear forms. But in the tensor decomposition framework, we need:
    
    C[i,k] = sum_r U[i,r] * V[k,r] * W[k*2+i, r]
    
    For this to equal A[i]*B[k] for all A,B, we need the W matrix to separate the 
    (i,k) pairs. Since W has rows indexed by k*2+i (0..13), and each (i,k) maps to 
    a unique row, we can set:
    - For each r from 0 to 13, assign it to a specific (i_r, k_r).
    - U[i, r] = 1 if i == i_r else 0
    - V[k, r] = 1 if k == k_r else 0
    - W[k*2+i, r] = 1 if (i,k) == (i_r, k_r) else 0
    
    This gives rank 14 for the 2x1*1x7 part.
    
    Total rank: 25 + 14 = 39.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=7
    
    # We'll build U, V, W as lists of lists.
    # U has shape (n*m, rank) = (6, rank)
    # V has shape (m*p, rank) = (21, rank)
    # W has shape (p*n, rank) = (14, rank)
    
    # Start with empty lists, we'll append columns.
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Part 1: A1 * B1 where A1 is 2x2 (rows 0,1; cols 0,1 of A) and B1 is 2x7 (rows 0,1 of B)
    # A1 indices in U: i*m + j for i in {0,1}, j in {0,1} => indices 0,1,2,3
    # B1 indices in V: j*p + k for j in {0,1}, k in {0..6} => indices 0..13
    
    # Split B1 into B1a (cols 0,1), B1b (cols 2,3), B1c (cols 4,5), B1d (col 6)
    # For B1a, B1b: use Strassen (2x2 * 2x2), rank 7 each
    # For B1c: use Strassen-like or trivial? 2x2 * 2x2, use Strassen, rank 7
    # For B1d: 2x2 * 2x1, use trivial, rank 4
    
    # Actually, let me just use trivial for all to be safe, then optimize.
    # Trivial 2x2 * 2x7: rank 14. Plus 2x1 * 1x7: rank 14. Total 28.
    # That's still better than 42.
    
    # Let me use Strassen for the 2x2*2x2 blocks.
    
    def strassen_2x2_2x2(U_offset, V_offset, W_offset, col_offset):
        """
        Add Strassen decomposition for 2x2 * 2x2.
        U_offset: starting index in U for the 2x2 block (4 consecutive entries)
        V_offset: starting index in V for the 2x2 block (4 consecutive entries)
        W_offset: starting index in W for the 2x2 block (4 consecutive entries)
        col_offset: starting column index
        Returns: number of columns added
        """
        # Strassen formulas for C = A*B where A,B,C are 2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # A indices: A11=U_offset+0, A12=U_offset+1, A21=U_offset+2, A22=U_offset+3
        # B indices: B11=V_offset+0, B12=V_offset+1, B21=V_offset+2, B22=V_offset+3
        # C indices: C11=W_offset+0, C12=W_offset+1, C21=W_offset+2, C22=W_offset+3
        
        # Each Strassen product r has:
        # U coefficients for A11,A12,A21,A22
        # V coefficients for B11,B12,B21,B22
        # W coefficients for C11,C12,C21,C22
        
        products = [
            # (u_coeffs, v_coeffs, w_coeffs)
            # M1 = (A11+A22)(B11+B22)
            ([1,0,0,1], [1,0,0,1], [1,0,0,0]),  # contributes to C11
            # M2 = (A21+A22)B11
            ([0,0,1,1], [1,0,0,0], [0,0,1,0]),  # contributes to C21
            # M3 = A11(B12-B22)
            ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),  # contributes to C12
            # M4 = A22(B21-B11)
            ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),  # contributes to C11, C21
            # M5 = (A11+A12)B22
            ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),  # contributes to C11, C12
            # M6 = (A21-A11)(B11+B12)
            ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),  # contributes to C22
            # M7 = (A12-A22)(B21+B22)
            ([0,1,0,-1], [0,0,1,1], [1,0,0,1]),  # contributes to C11, C22
        ]
        
        for u_c, v_c, w_c in products:
            # Create U column: 6 entries (n*m = 2*3 = 6)
            u_col = [0.0] * 6
            # A11 at index 0, A12 at index 1, A21 at index 2, A22 at index 3
            u_col[U_offset + 0] = u_c[0]
            u_col[U_offset + 1] = u_c[1]
            u_col[U_offset + 2] = u_c[2]
            u_col[U_offset + 3] = u_c[3]
            
            # Create V column: 21 entries (m*p = 3*7 = 21)
            v_col = [0.0] * 21
            # B11 at index V_offset+0, B12 at V_offset+1, B21 at V_offset+2, B22 at V_offset+3
            v_col[V_offset + 0] = v_c[0]
            v_col[V_offset + 1] = v_c[1]
            v_col[V_offset + 2] = v_c[2]
            v_col[V_offset + 3] = v_c[3]
            
            # Create W column: 14 entries (p*n = 7*2 = 14)
            w_col = [0.0] * 14
            # C11 at index W_offset+0, C12 at W_offset+1, C21 at W_offset+2, C22 at W_offset+3
            w_col[W_offset + 0] = w_c[0]
            w_col[W_offset + 1] = w_c[1]
            w_col[W_offset + 2] = w_c[2]
            w_col[W_offset + 3] = w_c[3]
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
        
        return 7
    
    # Strassen for A1 * B1a (cols 0,1 of B1)
    # A1: U indices 0,1,2,3 (i=0,1; j=0,1)
    # B1a: V indices 0,1,2,3 (j=0,1; k=0,1) => V[j*7+k] = j*7+k
    #   j=0,k=0 => 0; j=0,k=1 => 1; j=1,k=0 => 7; j=1,k=1 => 8
    # C1a: W indices for i=0,1; k=0,1 => W[k*2+i]
    #   i=0,k=0 => 0; i=1,k=0 => 1; i=0,k=1 => 2; i=1,k=1 => 3
    strassen_2x2_2x2(
        U_offset=0,  # A1: A11=0, A12=1, A21=2, A22=3
        V_offset=0,  # B1a: B11=0, B12=1, B21=7, B22=8
        W_offset=0,  # C1a: C11=0, C12=2, C21=1, C22=3
        col_offset=0
    )
    
    # Wait, I need to fix the W indices. W is indexed by k*n+i = k*2+i.
    # For the 2x2 output block with i in {0,1}, k in {0,1}:
    # C11: i=0,k=0 => W[0]
    # C12: i=0,k=1 => W[2]
    # C21: i=1,k=0 => W[1]
    # C22: i=1,k=1 => W[3]
    
    # So the mapping is: C11->0, C12->2, C21->1, C22->3
    # In my Strassen function, I assumed C11->0, C12->1, C21->2, C22->3
    # I need to remap. Let me fix the W_offset usage.
    
    # Actually, let me redo this more carefully. Let me clear and restart.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_2x2_2x2(A_indices, B_indices, C_indices):
        """
        Add Strassen decomposition.
        A_indices: [A11, A12, A21, A22] - indices in U
        B_indices: [B11, B12, B21, B22] - indices in V
        C_indices: [C11, C12, C21, C22] - indices in W
        """
        products = [
            # (u_coeffs for A11,A12,A21,A22, v_coeffs for B11,B12,B21,B22, w_coeffs for C11,C12,C21,C22)
            ([1,0,0,1], [1,0,0,1], [1,0,0,0]),
            ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
            ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
            ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),
            ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
            ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
            ([0,1,0,-1], [0,0,1,1], [1,0,0,1]),
        ]
        
        for u_c, v_c, w_c in products:
            u_col = [0.0] * 6
            v_col = [0.0] * 21
            w_col = [0.0] * 14
            
            u_col[A_indices[0]] = u_c[0]
            u_col[A_indices[1]] = u_c[1]
            u_col[A_indices[2]] = u_c[2]
            u_col[A_indices[3]] = u_c[3]
            
            v_col[B_indices[0]] = v_c[0]
            v_col[B_indices[1]] = v_c[1]
            v_col[B_indices[2]] = v_c[2]
            v_col[B_indices[3]] = v_c[3]
            
            w_col[C_indices[0]] = w_c[0]
            w_col[C_indices[1]] = w_c[1]
            w_col[C_indices[2]] = w_c[2]
            w_col[C_indices[3]] = w_c[3]
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
    
    def add_trivial_2x2_2x1(A_indices, B_indices, C_indices):
        """
        Add trivial decomposition for 2x2 * 2x1.
        A_indices: [A11, A12, A21, A22]
        B_indices: [B1, B2]
        C_indices: [C1, C2]
        """
        # C1 = A11*B1 + A12*B2
        # C2 = A21*B1 + A22*B2
        products = [
            # (u_coeffs, v_coeffs, w_coeffs)
            ([1,0,0,0], [1,0], [1,0]),   # A11*B1 -> C1
            ([0,1,0,0], [0,1], [1,0]),   # A12*B2 -> C1
            ([0,0,1,0], [1,0], [0,1]),   # A21*B1 -> C2
            ([0,0,0,1], [0,1], [0,1]),   # A22*B2 -> C2
        ]
        
        for u_c, v_c, w_c in products:
            u_col = [0.0] * 6
            v_col = [0.0] * 21
            w_col = [0.0] * 14
            
            u_col[A_indices[0]] = u_c[0]
            u_col[A_indices[1]] = u_c[1]
            u_col[A_indices[2]] = u_c[2]
            u_col[A_indices[3]] = u_c[3]
            
            v_col[B_indices[0]] = v_c[0]
            v_col[B_indices[1]] = v_c[1]
            
            w_col[C_indices[0]] = w_c[0]
            w_col[C_indices[1]] = w_c[1]
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
    
    # Part 1: A1 * B1 where A1 is 2x2, B1 is 2x7
    # A1: U indices for i=0,1; j=0,1
    #   A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[2], A[1,1]=U[3]
    # B1: V indices for j=0,1; k=0..6
    #   B[j,k] = V[j*7+k]
    # C1: W indices for i=0,1; k=0..6
    #   C[i,k] = W[k*2+i]
    
    # Split B1 into:
    # B1a: j=0,1; k=0,1 => V[0],V[1],V[7],V[8]
    # B1b: j=0,1; k=2,3 => V[2],V[3],V[9],V[10]
    # B1c: j=0,1; k=4,5 => V[4],V[5],V[11],V[12]
    # B1d: j=0,1; k=6 => V[6],V[13]
    
    # C1a: i=0,1; k=0,1 => W[0],W[2],W[1],W[3]
    #   C11(i=0,k=0)=W[0], C12(i=0,k=1)=W[2], C21(i=1,k=0)=W[1], C22(i=1,k=1)=W[3]
    # C1b: i=0,1; k=2,3 => W[4],W[6],W[5],W[7]
    # C1c: i=0,1; k=4,5 => W[8],W[10],W[9],W[11]
    # C1d: i=0,1; k=6 => W[12],W[13]
    
    # A1 indices: A11=0, A12=1, A21=2, A22=3
    
    # Strassen for A1 * B1a
    add_strassen_2x2_2x2(
        A_indices=[0, 1, 2, 3],
        B_indices=[0, 1, 7, 8],
        C_indices=[0, 2, 1, 3]
    )
    
    # Strassen for A1 * B1b
    add_strassen_2x2_2x2(
        A_indices=[0, 1, 2, 3],
        B_indices=[2, 3, 9, 10],
        C_indices=[4, 6, 5, 7]
    )
    
    # Strassen for A1 * B1c
    add_strassen_2x2_2x2(
        A_indices=[0, 1, 2, 3],
        B_indices=[4, 5, 11, 12],
        C_indices=[8, 10, 9, 11]
    )
    
    # Trivial for A1 * B1d
    add_trivial_2x2_2x1(
        A_indices=[0, 1, 2, 3],
        B_indices=[6, 13],
        C_indices=[12, 13]
    )
    
    # Part 2: A2 * B2 where A2 is 2x1, B2 is 1x7
    # A2: U indices for i=0,1; j=2
    #   A[0,2]=U[4], A[1,2]=U[5]
    # B2: V indices for j=2; k=0..6
    #   B[2,k] = V[2*7+k] = V[14+k]
    # C2: W indices for i=0,1; k=0..6
    #   C[i,k] = W[k*2+i]
    
    # For each i in {0,1} and k in {0..6}:
    # C[i,k] += A[i,2] * B[2,k]
    # U[4] * V[14+k] * W[k*2+i]
    
    for i in range(2):
        for k in range(7):
            u_col = [0.0] * 6
            v_col = [0.0] * 21
            w_col = [0.0] * 14
            
            u_col[4 + i] = 1.0  # A[i,2]
            v_col[14 + k] = 1.0  # B[2,k]
            w_col[k * 2 + i] = 1.0  # C[i,k]
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
    
    # Convert to row-major format: U has shape (6, rank), so transpose.
    rank = len(U_cols)
    U = [[U_cols[r][i] for r in range(rank)] for i in range(6)]
    V = [[V_cols[r][i] for r in range(rank)] for i in range(21)]
    W = [[W_cols[r][i] for r in range(rank)] for i in range(14)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
