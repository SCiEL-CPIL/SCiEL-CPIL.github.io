import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create schoolbook decomposition for n x m x p
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return U, V, W

    # Helper to create Strassen decomposition for 2x2x2
    def strassen_222():
        # Returns U, V, W of shape (4, 7), (4, 7), (4, 7)
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = -1
        
        # M2 = (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3 = A11(B12-B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A22(B21-B11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1; W[0, 3] = 1
        
        # M5 = (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1; W[3, 4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1; W[0, 5] = -1
        
        # M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = -1; W[3, 6] = 1
        
        return U, V, W

    # Recursive construction using additive splits
    def build_decomp(n, m, p, offset_i, offset_j, offset_k):
        # Returns list of (U_sub, V_sub, W_sub, i_off, j_off, k_off)
        # But it's easier to just accumulate into global arrays
        
        # Base cases
        if n == 2 and m == 2 and p == 2:
            U, V, W = strassen_222()
            return [(U, V, W, offset_i, offset_j, offset_k)]
        
        # If any dimension is 1, use schoolbook for that part
        # But we can just use schoolbook for small non-2x2x2
        # Actually, let's use the specific splits derived
        
        # Split p if p > 2
        # Split m if m > 2
        # Split n if n > 2
        
        # Heuristic: Split the largest dimension
        # Or use the specific structure for 3x5x7
        
        # For this specific task, we can hardcode the splits
        # 3x5x7 -> 3x5x4 + 3x5x3
        # 3x5x4 -> 3x5x2 + 3x5x2
        # 3x5x2 -> 3x4x2 + 3x1x2
        # 3x4x2 -> 3x2x2 + 3x2x2
        # 3x2x2 -> 2x2x2 + 1x2x2
        # 3x5x3 -> 3x5x2 + 3x5x1
        # 3x5x1 -> schoolbook
        # 3x1x2 -> schoolbook
        # 1x2x2 -> schoolbook
        
        # This is getting complex to generalize. I will just implement the specific splits.
        
        results = []
        
        # Split p
        if p > 2:
            p1 = p // 2
            p2 = p - p1
            # Recurse on p1 and p2
            # But we need to be careful with the specific optimizations
            # Let's just use schoolbook for anything not 2x2x2 or 1x1x1 etc?
            # No, we want to use the 87 rank.
            
        # I will implement a specific builder for the 3x5x7 case.
        
        return [] # Placeholder

    # I will write a direct construction for rank 87.
    
    # List of sub-problems: (n, m, p, i_off, j_off, k_off, decomp_type)
    # decomp_type: 'strassen' or 'schoolbook'
    
    subproblems = []
    
    # 3x5x7
    # Split p=7 into 4 and 3
    # 3x5x4 (k_off=0)
    # 3x5x3 (k_off=4)
    
    # 3x5x4 splits into two 3x5x2
    # 3x5x2 (k_off=0)
    # 3x5x2 (k_off=2)
    
    # 3x5x2 splits into 3x4x2 and 3x1x2
    # 3x4x2 (j_off=0)
    # 3x1x2 (j_off=4)
    
    # 3x4x2 splits into two 3x2x2
    # 3x2x2 (j_off=0)
    # 3x2x2 (j_off=2)
    
    # 3x2x2 splits into 2x2x2 and 1x2x2
    # 2x2x2 (i_off=0)
    # 1x2x2 (i_off=2)
    
    # 3x5x3 splits into 3x5x2 and 3x5x1
    # 3x5x2 (k_off=4) -> same as above but k_off=4
    # 3x5x1 (k_off=6)
    
    # Let's trace the leaves:
    # From 3x5x4:
    #  2 * (3x5x2)
    #     2 * (3x4x2 + 3x1x2)
    #        2 * (2 * 3x2x2 + 3x1x2)
    #           2 * (2 * (2x2x2 + 1x2x2) + 3x1x2)
    # Leaves from 3x5x4:
    #   4 * 2x2x2
    #   4 * 1x2x2
    #   4 * 3x1x2
    
    # From 3x5x3:
    #   3x5x2 (k_off=4)
    #      3x4x2 (j_off=0, k_off=4)
    #         2 * 3x2x2
    #            2 * (2x2x2 + 1x2x2)
    #      3x1x2 (j_off=4, k_off=4)
    #   3x5x1 (k_off=6)
    # Leaves from 3x5x3:
    #   2 * 2x2x2
    #   2 * 1x2x2
    #   1 * 3x1x2
    #   1 * 3x5x1
    
    # Total leaves:
    # 6 * 2x2x2 (Rank 6*7 = 42)
    # 6 * 1x2x2 (Rank 6*2 = 12)
    # 5 * 3x1x2 (Rank 5*6 = 30)
    # 1 * 3x5x1 (Rank 1*15 = 15)
    # Total Rank: 42 + 12 + 30 + 15 = 99?
    
    # Wait, my previous calculation was 87.
    # Let's re-evaluate.
    # R(3,5,4) = 48.
    # R(3,5,2) = 24.
    # R(3,4,2) = 18.
    # R(3,2,2) = 9.
    # 4 * R(3,2,2) = 36.
    # 4 * R(3,1,2) = 24.
    # Total 60? No.
    
    # 3x5x4 = 2 * 3x5x2.
    # 3x5x2 = 3x4x2 + 3x1x2.
    # 3x4x2 = 2 * 3x2x2.
    # 3x2x2 = 2x2x2 + 1x2x2.
    # So 3x4x2 = 2 * (2x2x2 + 1x2x2) = 2*7 + 2*2 = 18. Correct.
    # 3x5x2 = 18 + 6 = 24. Correct.
    # 3x5x4 = 48. Correct.
    
    # 3x5x3 = 3x5x2 + 3x5x1 = 24 + 15 = 39. Correct.
    # Total 87.
    
    # My leaf count:
    # From 3x5x4:
    #   2 * (3x4x2 + 3x1x2)
    #   3x4x2 has 2 * 3x2x2.
    #   3x2x2 has 1 * 2x2x2 + 1 * 1x2x2.
    #   So 3x4x2 has 2 * 2x2x2 + 2 * 1x2x2.
    #   So 3x5x2 has 2 * 2x2x2 + 2 * 1x2x2 + 1 * 3x1x2.
    #   So 3x5x4 has 4 * 2x2x2 + 4 * 1x2x2 + 2 * 3x1x2.
    #   Rank: 4*7 + 4*2 + 2*6 = 28 + 8 + 12 = 48. Correct.
    
    # From 3x5x3:
    #   3x5x2 + 3x5x1.
    #   3x5x2 has 2 * 2x2x2 + 2 * 1x2x2 + 1 * 3x1x2.
    #   3x5x1 is schoolbook.
    #   Rank: 2*7 + 2*2 + 1*6 + 15 = 14 + 4 + 6 + 15 = 39. Correct.
    
    # Total Rank: 48 + 39 = 87.
    
    # I will implement this.

    U_list = []
    V_list = []
    W_list = []
    
    def add_decomp(n, m, p, i_off, j_off, k_off, type):
        if type == 'strassen':
            U, V, W = strassen_222()
        else:
            U, V, W = schoolbook(n, m, p)
        
        # Map to global indices
        U_g = np.zeros((n*m, U.shape[1]))
        V_g = np.zeros((m*p, V.shape[1]))
        W_g = np.zeros((p*n, W.shape[1]))
        
        # Actually, I need to append to global lists
        # But global shapes are fixed.
        # I will collect all (U_sub, V_sub, W_sub, i_off, j_off, k_off) and then combine.
        
        U_list.append((U, i_off, j_off, n, m))
        V_list.append((V, j_off, k_off, m, p))
        W_list.append((W, k_off, i_off, p, n))

    # Add leaves
    # 4 * 2x2x2
    for _ in range(4):
        # Offsets for the 2x2x2 blocks
        # They come from 3x5x4 -> 3x5x2 (k=0,2) -> 3x4x2 (j=0,2) -> 3x2x2 (i=0)
        # The 4 blocks are at:
        # (i=0, j=0, k=0), (i=0, j=2, k=0), (i=0, j=0, k=2), (i=0, j=2, k=2)
        pass # I need to track offsets carefully.

    # This is getting tedious. I will write a recursive function that returns the matrices.
    
    def get_sub(n, m, p):
        if n == 2 and m == 2 and p == 2:
            return strassen_222()
        return schoolbook(n, m, p)

    # I will just hardcode the 87 rank construction by combining the parts.
    
    # Part 1: 3x5x4
    # 2 * 3x5x2
    # 3x5x2 = 3x4x2 + 3x1x2
    # 3x4x2 = 2 * 3x2x2
    # 3x2x2 = 2x2x2 + 1x2x2
    
    # I will create a function that takes offsets and returns the local matrices.
    
    def build_part(n, m, p):
        if n == 2 and m == 2 and p == 2:
            return strassen_222()
        return schoolbook(n, m, p)

    # I will assemble the final U, V, W by concatenating the parts.
    
    parts = []
    
    # 3x5x4
    # k_off = 0
    # 3x5x2 (k=0)
    #   3x4x2 (j=0)
    #     3x2x2 (j=0)
    #       2x2x2 (i=0)
    #       1x2x2 (i=2)
    #     3x2x2 (j=2)
    #       2x2x2 (i=0)
    #       1x2x2 (i=2)
    #   3x1x2 (j=4)
    # 3x5x2 (k=2)
    #   ... same with k=2
    
    # 3x5x3
    # k_off = 4
    # 3x5x2 (k=4)
    #   ...
    # 3x5x1 (k=6)
    
    # I will write a helper to map local indices to global.
    
    total_rank = 87
    U = np.zeros((15, total_rank))
    V = np.zeros((35, total_rank))
    W = np.zeros((21, total_rank))
    
    col = 0
    
    def add_block(n, m, p, i_off, j_off, k_off):
        nonlocal col
        if n == 2 and m == 2 and p == 2:
            Ub, Vb, Wb = strassen_222()
        else:
            Ub, Vb, Wb = schoolbook(n, m, p)
        
        for r in range(Ub.shape[1]):
            for i_l in range(n):
                for j_l in range(m):
                    val = Ub[i_l*m + j_l, r]
                    if val != 0:
                        U[(i_off+i_l)*m + (j_off+j_l), col] += val
            
            for j_l in range(m):
                for k_l in range(p):
                    val = Vb[j_l*p + k_l, r]
                    if val != 0:
                        V[(j_off+j_l)*p + (k_off+k_l), col] += val
            
            for k_l in range(p):
                for i_l in range(n):
                    val = Wb[k_l*n + i_l, r]
                    if val != 0:
                        W[(k_off+k_l)*n + (i_off+i_l), col] += val
                        
            col += 1

    # Add blocks for 3x5x4
    # 3x5x2 (k=0)
    # 3x4x2 (j=0)
    # 3x2x2 (j=0)
    add_block(2, 2, 2, 0, 0, 0)
    add_block(1, 2, 2, 2, 0, 0)
    # 3x2x2 (j=2)
    add_block(2, 2, 2, 0, 2, 0)
    add_block(1, 2, 2, 2, 2, 0)
    # 3x1x2 (j=4)
    add_block(3, 1, 2, 0, 4, 0)
    
    # 3x5x2 (k=2)
    add_block(2, 2, 2, 0, 0, 2)
    add_block(1, 2, 2, 2, 0, 2)
    add_block(2, 2, 2, 0, 2, 2)
    add_block(1, 2, 2, 2, 2, 2)
    add_block(3, 1, 2, 0, 4, 2)
    
    # Add blocks for 3x5x3
    # 3x5x2 (k=4)
    add_block(2, 2, 2, 0, 0, 4)
    add_block(1, 2, 2, 2, 0, 4)
    add_block(2, 2, 2, 0, 2, 4)
    add_block(1, 2, 2, 2, 2, 4)
    add_block(3, 1, 2, 0, 4, 4)
    
    # 3x5x1 (k=6)
    add_block(3, 5, 1, 0, 0, 6)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create schoolbook decomposition for n x m x p
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return U, V, W

    # Helper to create Strassen decomposition for 2x2x2
    def strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = -1
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1; W[0, 3] = 1
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1; W[3, 4] = 1
        
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1; W[0, 5] = -1
        
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = -1; W[3, 6] = 1
        
        return U, V, W

    total_rank = 87
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    
    def add_block(n_b, m_b, p_b, i_off, j_off, k_off):
        nonlocal col
        if n_b == 2 and m_b == 2 and p_b == 2:
            Ub, Vb, Wb = strassen_222()
        else:
            Ub, Vb, Wb = schoolbook(n_b, m_b, p_b)
        
        for r in range(Ub.shape[1]):
            for i_l in range(n_b):
                for j_l in range(m_b):
                    val = Ub[i_l*m_b + j_l, r]
                    if val != 0:
                        U[(i_off+i_l)*m + (j_off+j_l), col] += val
            
            for j_l in range(m_b):
                for k_l in range(p_b):
                    val = Vb[j_l*p_b + k_l, r]
                    if val != 0:
                        V[(j_off+j_l)*p + (k_off+k_l), col] += val
            
            for k_l in range(p_b):
                for i_l in range(n_b):
                    val = Wb[k_l*n_b + i_l, r]
                    if val != 0:
                        W[(k_off+k_l)*n + (i_off+i_l), col] += val
                        
            col += 1

    # 3x5x4
    add_block(2, 2, 2, 0, 0, 0)
    add_block(1, 2, 2, 2, 0, 0)
    add_block(2, 2, 2, 0, 2, 0)
    add_block(1, 2, 2, 2, 2, 0)
    add_block(3, 1, 2, 0, 4, 0)
    
    add_block(2, 2, 2, 0, 0, 2)
    add_block(1, 2, 2, 2, 0, 2)
    add_block(2, 2, 2, 0, 2, 2)
    add_block(1, 2, 2, 2, 2, 2)
    add_block(3, 1, 2, 0, 4, 2)
    
    # 3x5x3
    add_block(2, 2, 2, 0, 0, 4)
    add_block(1, 2, 2, 2, 0, 4)
    add_block(2, 2, 2, 0, 2, 4)
    add_block(1, 2, 2, 2, 2, 4)
    add_block(3, 1, 2, 0, 4, 4)
    
    add_block(3, 5, 1, 0, 0, 6)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
