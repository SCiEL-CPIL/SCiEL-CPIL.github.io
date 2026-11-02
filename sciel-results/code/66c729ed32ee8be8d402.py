<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Build a rank-minimal exact tensor decomposition for 3x4x8 matrix multiplication
    using Strassen's algorithm on 2x2x2 blocks where possible.
    
    The key insight: for each 2x2x2 sub-block, we can use Strassen's 7-multiplication
    algorithm instead of 8. We partition the 3x4x8 multiplication into blocks and
    apply Strassen where dimensions allow.
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # We'll build the decomposition by considering the structure.
    # For a 3x4x8 multiplication, we can think of it as:
    # - The 4x8 part can be seen as two 2x4 blocks in the middle dimension
    # - Or we can partition differently to maximize Strassen applications
    
    # Strategy: Use Strassen on 2x2x2 blocks.
    # Partition: n=3 (1+2), m=4 (2+2), p=8 (4+4)
    # This gives us blocks where we can apply Strassen.
    
    # Actually, let's use a simpler approach: directly construct using
    # Strassen's formulas for each 2x2x2 subproblem.
    
    # For 3x4x8, we can view this as:
    # A (3x4) * B (4x8) = C (3x8)
    # 
    # Partition A as [A11(2x2) A12(2x2); A21(1x2) A22(1x2)]
    # Partition B as [B11(2x4) B12(2x4); B21(2x4) B22(2x4)]
    # 
    # This is getting complex. Let's use a different strategy:
    # Build the full decomposition by enumerating all (i,j,k) triples
    # and then try to merge terms using Strassen identities.
    
    # Simpler approach: Use the known fact that for 2x2x2, rank is 7.
    # For 3x4x8, we can partition into blocks and apply Strassen.
    
    # Let's use a direct construction with Strassen on 2x2x2 blocks:
    # We'll create U, V, W matrices and fill them in.
    
    # Start with schoolbook rank = 3*4*8 = 96
    # With Strassen on appropriate 2x2x2 blocks, we can reduce this.
    
    # For now, let's implement a clean Strassen-based approach:
    # Partition the multiplication into 2x2x2 blocks where possible.
    
    # Actually, let me think about this more carefully.
    # The tensor T_{ijk} for matrix mult has T_{ijk} = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # 
    # For 3x4x8: i in [0,1,2], j in [0,1,2,3], k in [0,...,7]
    # 
    # We can group j into pairs: (0,1) and (2,3)
    # We can group k into pairs: (0,1), (2,3), (4,5), (6,7)
    # For i, we have 3 rows, so we can't fully group into 2x2, but we can do (0,1) and {2}
    
    # Let's build the decomposition by handling each combination of blocks:
    # For block (i_block, j_block, k_block):
    #   - If i_block has size 2, j_block has size 2, k_block has size 2: use Strassen (7 mults)
    #   - Otherwise: use schoolbook (size_i * size_j * size_k mults)
    
    U_cols = []  # List of columns for U
    V_cols = []  # List of columns for V
    W_cols = []  # List of columns for W
    
    def add_column(u_vals, v_vals, w_vals):
        """Add a column to U, V, W."""
        U_cols.append(u_vals)
        V_cols.append(v_vals)
        W_cols.append(w_vals)
    
    def schoolbook_block(i_indices, j_indices, k_indices):
        """Add schoolbook terms for a block."""
        for i in i_indices:
            for j in j_indices:
                for k in k_indices:
                    u_vals = [0] * (n * m)
                    v_vals = [0] * (m * p)
                    w_vals = [0] * (p * n)
                    u_vals[i * m + j] = 1
                    v_vals[j * p + k] = 1
                    w_vals[k * n + i] = 1
                    add_column(u_vals, v_vals, w_vals)
    
    def strassen_block(i_indices, j_indices, k_indices):
        """
        Add Strassen terms for a 2x2x2 block.
        i_indices = [i0, i1], j_indices = [j0, j1], k_indices = [k0, k1]
        
        Strassen's 7 multiplications for 2x2x2:
        M1 = (A00 + A11) * (B00 + B11)
        M2 = (A10 + A11) * B00
        M3 = A00 * (B10 - B00)
        M4 = A11 * (B11 - B01)
        M5 = (A00 + A01) * B11
        M6 = (A10 - A00) * (B00 + B01)
        M7 = (A01 - A11) * (B10 + B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        i0, i1 = i_indices
        j0, j1 = j_indices
        k0, k1 = k_indices
        
        # M1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        u1 = [0] * (n * m)
        v1 = [0] * (m * p)
        w1 = [0] * (p * n)
        u1[i0 * m + j0] = 1
        u1[i1 * m + j1] = 1
        v1[j0 * p + k0] = 1
        v1[j1 * p + k1] = 1
        # C00 += M1, C11 += M1
        w1[k0 * n + i0] = 1
        w1[k1 * n + i1] = 1
        add_column(u1, v1, w1)
        
        # M2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        u2 = [0] * (n * m)
        v2 = [0] * (m * p)
        w2 = [0] * (p * n)
        u2[i1 * m + j0] = 1
        u2[i1 * m + j1] = 1
        v2[j0 * p + k0] = 1
        # C10 += M2, C11 -= M2
        w2[k0 * n + i1] = 1
        w2[k1 * n + i1] = -1
        add_column(u2, v2, w2)
        
        # M3 = A[i0,j0] * (B[j1,k0] - B[j0,k0])
        u3 = [0] * (n * m)
        v3 = [0] * (m * p)
        w3 = [0] * (p * n)
        u3[i0 * m + j0] = 1
        v3[j1 * p + k0] = 1
        v3[j0 * p + k0] = -1
        # C00 += M3, C01 += M3
        w3[k0 * n + i0] = 1
        w3[k1 * n + i0] = 1
        add_column(u3, v3, w3)
        
        # M4 = A[i1,j1] * (B[j1,k1] - B[j0,k1])
        u4 = [0] * (n * m)
        v4 = [0] * (m * p)
        w4 = [0] * (p * n)
        u4[i1 * m + j1] = 1
        v4[j1 * p + k1] = 1
        v4[j0 * p + k1] = -1
        # C00 += M4, C10 += M4
        w4[k1 * n + i0] = 1
        w4[k0 * n + i1] = 1
        add_column(u4, v4, w4)
        
        # M5 = (A[i0,j0] + A[i0,j1]) * B[j1,k1]
        u5 = [0] * (n * m)
        v5 = [0] * (m * p)
        w5 = [0] * (p * n)
        u5[i0 * m + j0] = 1
        u5[i0 * m + j1] = 1
        v5[j1 * p + k1] = 1
        # C00 -= M5, C01 += M5
        w5[k1 * n + i0] = -1
        w5[k1 * n + i0] = 1  # Wait, this is wrong, let me fix
        # Actually: C00 = M1 + M4 - M5 + M7, C01 = M3 + M5
        # So for M5: C00 gets -1, C01 gets +1
        w5 = [0] * (p * n)
        w5[k1 * n + i0] = -1  # C00 contribution
        # But we also need C01 contribution... this is getting complex with single columns
        # Let me reconsider the approach.
        
        # Actually, each "multiplication" in Strassen corresponds to one column in the decomposition.
        # The W matrix entries can be -1, 0, or 1 to handle the additions/subtractions.
        # But each column represents one scalar multiplication, so we need to be careful.
        
        # Let me restart with a cleaner approach.
        pass
    
    # Let me use a cleaner implementation:
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_entry, v_entry, w_entry):
        """Add one rank-1 term to the decomposition."""
        u_vals = [0] * (n * m)
        v_vals = [0] * (m * p)
        w_vals = [0] * (p * n)
        for (idx, val) in u_entry.items():
            u_vals[idx] = val
        for (idx, val) in v_entry.items():
            v_vals[idx] = val
        for (idx, val) in w_entry.items():
            w_vals[idx] = val
        U_cols.append(u_vals)
        V_cols.append(v_vals)
        W_cols.append(w_vals)
    
    # Partition: 
    # i: [0,1] (size 2), [2] (size 1)
    # j: [0,1] (size 2), [2,3] (size 2)
    # k: [0,1], [2,3], [4,5], [6,7] (each size 2)
    
    i_blocks = [[0, 1], [2]]
    j_blocks = [[0, 1], [2, 3]]
    k_blocks = [[0, 1], [2, 3], [4, 5], [6, 7]]
    
    for ib in i_blocks:
        for jb in j_blocks:
            for kb in k_blocks:
                if len(ib) == 2 and len(jb) == 2 and len(kb) == 2:
                    # Use Strassen: 7 multiplications
                    i0, i1 = ib
                    j0, j1 = jb
                    k0, k1 = kb
                    
                    # M1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
                    add_term(
                        {i0*m+j0: 1, i1*m+j1: 1},
                        {j0*p+k0: 1, j1*p+k1: 1},
                        {k0*n+i0: 1, k1*n+i1: 1}
                    )
                    
                    # M2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
                    add_term(
                        {i1*m+j0: 1, i1*m+j1: 1},
                        {j0*p+k0: 1},
                        {k0*n+i1: 1, k1*n+i1: -1}
                    )
                    
                    # M3 = A[i0,j0] * (B[j1,k0] - B[j0,k0])
                    add_term(
                        {i0*m+j0: 1},
                        {j1*p+k0: 1, j0*p+k0: -1},
                        {k0*n+i0: 1, k1*n+i0: 1}
                    )
                    
                    # M4 = A[i1,j1] * (B[j1,k1] - B[j0,k1])
                    add_term(
                        {i1*m+j1: 1},
                        {j1*p+k1: 1, j0*p+k1: -1},
                        {k1*n+i0: 1, k0*n+i1: 1}
                    )
                    
                    # M5 = (A[i0,j0] + A[i0,j1]) * B[j1,k1]
                    add_term(
                        {i0*m+j0: 1, i0*m+j1: 1},
                        {j1*p+k1: 1},
                        {k1*n+i0: -1, k1*n+i0: 1}  # C00: -1, C01: +1 -> but same index!
                    )
                    # Wait, this is problematic. Let me fix the Strassen formulas.
                    # 
                    # Correct Strassen for 2x2x2:
                    # C00 = M1 + M4 - M5 + M7
                    # C01 = M3 + M5
                    # C10 = M2 + M4
                    # C11 = M1 - M2 + M3 + M6
                    #
                    # Where:
                    # M1 = (A00+A11)(B00+B11)
                    # M2 = (A10+A11)B00
                    # M3 = A00(B10-B00)
                    # M4 = A11(B11-B01)
                    # M5 = (A00+A01)B11
                    # M6 = (A10-A00)(B00+B01)
                    # M7 = (A01-A11)(B10+B11)
                    #
                    # Let me redo this properly.
                    pass
                else:
                    # Schoolbook for this block
                    for i in ib:
                        for j in jb:
                            for k in kb:
                                add_term(
                                    {i*m+j: 1},
                                    {j*p+k: 1},
                                    {k*n+i: 1}
                                )
    
    # The above has issues with the Strassen implementation. Let me use a verified approach.
    # For now, return the schoolbook decomposition which is correct.
    
    U = U_cols
    V = V_cols
    W = W_cols
    
    if not U:
        # Fallback to pure schoolbook
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
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

I realize my Strassen implementation has issues with the W matrix entries (multiple contributions to the same index need to be combined). Let me provide a corrected version:

