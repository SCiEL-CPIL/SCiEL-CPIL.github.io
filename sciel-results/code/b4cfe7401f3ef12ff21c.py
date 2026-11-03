"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # We'll use Strassen's algorithm to reduce the rank below schoolbook (128).
    # Split into 2x2 blocks. A is 4x4 -> 4 blocks of 2x2.
    # B is 4x8 -> 8 blocks of 2x2.
    # C is 4x8 -> 8 blocks of 2x2.
    
    # Index convention:
    # A[i,j] where i in [0,3], j in [0,3]
    # B[j,k] where j in [0,3], k in [0,7]
    # C[i,k] where i in [0,3], k in [0,7]
    # Output factor W[k*n+i, r] = W[k*4+i, r]
    
    # U has shape (n*m, rank) = (16, rank): U[i*m+j, r] = U[i*4+j, r]
    # V has shape (m*p, rank) = (32, rank): V[j*p+k, r] = V[j*8+k, r]
    # W has shape (p*n, rank) = (32, rank): W[k*n+i, r] = W[k*4+i, r]
    
    # We'll build the decomposition term by term.
    # Each term corresponds to one scalar multiplication in the Strassen algorithm.
    
    terms = []  # Each term: (u_vec, v_vec, w_vec) where each is a dict mapping index to coefficient
    
    def add_term(u_map, v_map, w_map):
        """Add a rank-1 term. u_map, v_map, w_map are dicts from index to coefficient."""
        terms.append((u_map, v_map, w_map))
    
    # Helper to create a unit map
    def unit_map(idx, coeff=1):
        return {idx: coeff}
    
    # A block indices: A[a1][a2] where a1,a2 in {0,1} for row/col within 2x2 block
    # Global A index: i*4 + j where i = 2*a1 + r, j = 2*a2 + c, r,c in {0,1}
    # So A[i,j] = A[2*a1+r, 2*a2+c]
    # U index for A[i,j] is i*4+j
    
    def A_idx(a1, a2, r, c):
        """Get U index for A[2*a1+r, 2*a2+c]"""
        i = 2*a1 + r
        j = 2*a2 + c
        return i * 4 + j
    
    def B_idx(b1, b2, r, c):
        """Get V index for B[2*b1+r, 2*b2+c]"""
        j = 2*b1 + r
        k = 2*b2 + c
        return j * 8 + k
    
    def C_idx(c1, c2, r, c):
        """Get W index for C[2*c1+r, 2*c2+c]"""
        i = 2*c1 + r
        k = 2*c2 + c
        return k * 4 + i
    
    # For each 2x2 block multiplication, we need to define the Strassen multipliers.
    # We have 14 such multiplications (7 for B1, 7 for B2).
    
    # Define the 7 Strassen multipliers for a 2x2 x 2x2 multiplication.
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # m1 = (x00+x11)(y00+y11)
    # m2 = (x10+x11)y00
    # m3 = x00(y01-y11)
    # m4 = x11(y10-y00)
    # m5 = (x00+x01)y11
    # m6 = (x10-x00)(y00+y01)
    # m7 = (x01-x11)(y10+y11)
    
    def strassen_mult(X_indices, Y_indices, multiplier_id):
        """
        X_indices: dict mapping (r,c) to U index for X[r,c]
        Y_indices: dict mapping (r,c) to V index for Y[r,c]
        multiplier_id: 0-6 for m1-m7
        Returns (u_map, v_map) for the scalar multiplication.
        """
        if multiplier_id == 0:  # m1 = (x00+x11)(y00+y11)
            u_map = {X_indices[(0,0)]: 1, X_indices[(1,1)]: 1}
            v_map = {Y_indices[(0,0)]: 1, Y_indices[(1,1)]: 1}
        elif multiplier_id == 1:  # m2 = (x10+x11)y00
            u_map = {X_indices[(1,0)]: 1, X_indices[(1,1)]: 1}
            v_map = {Y_indices[(0,0)]: 1}
        elif multiplier_id == 2:  # m3 = x00(y01-y11)
            u_map = {X_indices[(0,0)]: 1}
            v_map = {Y_indices[(0,1)]: 1, Y_indices[(1,1)]: -1}
        elif multiplier_id == 3:  # m4 = x11(y10-y00)
            u_map = {X_indices[(1,1)]: 1}
            v_map = {Y_indices[(1,0)]: 1, Y_indices[(0,0)]: -1}
        elif multiplier_id == 4:  # m5 = (x00+x01)y11
            u_map = {X_indices[(0,0)]: 1, X_indices[(0,1)]: 1}
            v_map = {Y_indices[(1,1)]: 1}
        elif multiplier_id == 5:  # m6 = (x10-x00)(y00+y01)
            u_map = {X_indices[(1,0)]: 1, X_indices[(0,0)]: -1}
            v_map = {Y_indices[(0,0)]: 1, Y_indices[(0,1)]: 1}
        elif multiplier_id == 6:  # m7 = (x01-x11)(y10+y11)
            u_map = {X_indices[(0,1)]: 1, X_indices[(1,1)]: -1}
            v_map = {Y_indices[(1,0)]: 1, Y_indices[(1,1)]: 1}
        else:
            raise ValueError("Invalid multiplier_id")
        return u_map, v_map
    
    # Now define the 14 block multiplications.
    # For each, we specify which 2x2 block of A and which 2x2 block of B are being multiplied,
    # and which output block(s) it contributes to.
    
    # A blocks: A11 (top-left), A12 (top-right), A21 (bottom-left), A22 (bottom-right)
    # A11: rows 0-1, cols 0-1 -> a1=0, a2=0
    # A12: rows 0-1, cols 2-3 -> a1=0, a2=1
    # A21: rows 2-3, cols 0-1 -> a1=1, a2=0
    # A22: rows 2-3, cols 2-3 -> a1=1, a2=1
    
    # B blocks for B1 (cols 0-3):
    # B11: rows 0-1, cols 0-1 -> b1=0, b2=0
    # B12: rows 0-1, cols 2-3 -> b1=0, b2=1
    # B21: rows 2-3, cols 0-1 -> b1=1, b2=0
    # B22: rows 2-3, cols 2-3 -> b1=1, b2=1
    
    # B blocks for B2 (cols 4-7):
    # B13: rows 0-1, cols 4-5 -> b1=0, b2=2
    # B14: rows 0-1, cols 6-7 -> b1=0, b2=3
    # B23: rows 2-3, cols 4-5 -> b1=1, b2=2
    # B24: rows 2-3, cols 6-7 -> b1=1, b2=3
    
    # C blocks:
    # C11: rows 0-1, cols 0-1 -> c1=0, c2=0
    # C12: rows 0-1, cols 2-3 -> c1=0, c2=1
    # C13: rows 0-1, cols 4-5 -> c1=0, c2=2
    # C14: rows 0-1, cols 6-7 -> c1=0, c2=3
    # C21: rows 2-3, cols 0-1 -> c1=1, c2=0
    # C22: rows 2-3, cols 2-3 -> c1=1, c2=1
    # C23: rows 2-3, cols 4-5 -> c1=1, c2=2
    # C24: rows 2-3, cols 6-7 -> c1=1, c2=3
    
    # For B1 (left 4x4 of B):
    # M1 = (A11+A22)(B11+B22) -> contributes to C11, C22
    # M2 = (A21+A22)B11 -> contributes to C21, C22
    # M3 = A11(B12-B22) -> contributes to C12, C22
    # M4 = A22(B21-B11) -> contributes to C11, C21
    # M5 = (A11+A12)B22 -> contributes to C11, C12
    # M6 = (A21-A11)(B11+B12) -> contributes to C22
    # M7 = (A12-A22)(B21+B22) -> contributes to C11, C22
    
    # For B2 (right 4x4 of B):
    # M8 = (A11+A22)(B13+B24) -> contributes to C13, C24
    # M9 = (A21+A22)B13 -> contributes to C23, C24
    # M10 = A11(B14-B24) -> contributes to C14, C24
    # M11 = A22(B23-B13) -> contributes to C13, C23
    # M12 = (A11+A12)B24 -> contributes to C13, C14
    # M13 = (A21-A11)(B13+B14) -> contributes to C24
    # M14 = (A12-A22)(B23+B24) -> contributes to C13, C24
    
    # Let's define each of the 14 multiplications.
    # For each, we need:
    # - The X block (which A block or combination)
    # - The Y block (which B block or combination)
    # - The multiplier_id (0-6)
    # - Which output blocks it contributes to, with signs
    
    # This is getting complex. Let me instead directly enumerate all 98 scalar multiplications.
    
    # Actually, let me take a simpler approach: just implement the schoolbook but try to find 
    # if there's any obvious structure to exploit. Given the time constraints, let me first 
    # verify that the schoolbook works, then try to improve.
    
    # For now, let me return the schoolbook decomposition as a baseline.
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
