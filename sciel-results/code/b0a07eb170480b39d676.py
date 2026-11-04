"""Tensor decomposition of 5x5x5 matrix multiplication."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    tmax = t0 + 105.0

    n = 5
    m = 5
    p = 5

    # Build sparse column representation: U_cols[r] = {row: val}, etc.
    U_cols = []
    V_cols = []
    W_cols = []

    # Part A: frame terms for i==4 or j==4, with each k
    for ii in range(n):
        for jj in range(m):
            if ii == 4 or jj == 4:
                for kk in range(p):
                    U_cols.append({ii * m + jj: 1})
                    V_cols.append({jj * p + kk: 1})
                    W_cols.append({kk * n + ii: 1})

    # Part B: 4x4x4 core via Strassen recursively (rank 49)
    # We implement Strassen for 4x4 by dividing into 2x2 blocks and
    # applying Strassen to each block product (which is 2x2x2x2, rank 7)
    
    def add_strassen_4x4(U_cols, V_cols, W_cols, i_off, j_off, k_off):
        """Add Strassen decomposition for 4x4 core at offsets (i_off, j_off, k_off)."""
        # 2x2 block offsets
        # A blocks: (a,b) -> rows i_off+2a..i_off+2a+1, cols j_off+2b..j_off+2b+1
        # B blocks: (b,c) -> rows j_off+2b..j_off+2b+1, cols k_off+2c..k_off+2c+1
        
        def get_A_block(a, b):
            """Return list of (i, j) for A block (a, b)."""
            result = []
            for di in range(2):
                for dj in range(2):
                    result.append((i_off + 2*a + di, j_off + 2*b + dj))
            return result
        
        def get_B_block(b, c):
            """Return list of (j, k) for B block (b, c)."""
            result = []
            for dj in range(2):
                for dk in range(2):
                    result.append((j_off + 2*b + dj, k_off + 2*c + dk))
            return result
        
        # Strassen 7 products for 4x4 (treating 2x2 blocks as "scalars")
        # Each product is a bilinear form on 2x2 blocks
        # We need to expand each into rank-1 terms for the 2x2x2x2 subproblem
        
        # For a product (sum_a A_a)(sum_b B_b), where A_a and B_b are 2x2 blocks,
        # the result is a 2x2 matrix. Each element of the result is a bilinear form
        # on the entries of the 2x2 blocks.
        
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # For each Strassen product, we have a set of (A_block, B_block, coeff) pairs
        # Each such pair represents a product of two 2x2 blocks
        # We decompose each 2x2x2x2 product using Strassen (rank 7)
        
        # Strassen for 2x2x2x2:
        # Given 2x2 matrices P and Q, compute PQ using 7 multiplications:
        # s1 = (P00+P11)(Q00+Q11)
        # s2 = (P10+P11)Q00
        # s3 = P00(Q01-Q11)
        # s4 = P11(Q10-Q00)
        # s5 = (P00+P01)Q11
        # s6 = (P10-P00)(Q00+Q01)
        # s7 = (P01-P11)(Q10+Q11)
        # Result:
        # R00 = s1+s4-s5+s7
        # R01 = s3+s5
        # R10 = s2+s4
        # R11 = s1-s2+s3+s6
        
        # For our purposes, we just need to add rank-1 terms
        # Each "multiplication" in Strassen is a product of two linear combinations
        # of entries, which gives one rank-1 term in the tensor decomposition
        
        def add_strassen_2x2(U_cols, V_cols, W_cols, 
                              A_entries, B_entries, coeff):
            """Add Strassen decomposition for 2x2x2x2.
            A_entries: dict mapping (di,dj) -> (i,j) for A block
            B_entries: dict mapping (dj,dk) -> (j,k) for B block
            coeff: overall coefficient
            """
            # A entries: A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1) in local coords
            # B entries: B00=(0,0), B01=(0,1), B10=(1,0), B11=(1,1) in local coords
            
            # s1 = (A00+A11)(B00+B11)
            # s2 = (A10+A11)B00
            # s3 = A00(B01-B11)
            # s4 = A11(B10-B00)
            # s5 = (A00+A01)B11
            # s6 = (A10-A00)(B00+B01)
            # s7 = (A01-A11)(B10+B11)
            
            strassen_2x2 = [
                # (A_terms, B_terms) where each is list of ((local_idx), coeff)
                ([((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)]),  # s1
                ([((1,0),1), ((1,1),1)], [((0,0),1)]),              # s2
                ([((0,0),1)], [((0,1),1), ((1,1),-1)]),              # s3
                ([((1,1),1)], [((1,0),1), ((0,0),-1)]),              # s4
                ([((0,0),1), ((0,1),1)], [((1,1),1)]),              # s5
                ([((1,0),1), ((0,0),-1)], [((0,0),1), ((0,1),1)]),  # s6
                ([((0,1),1), ((1,1),-1)], [((1,0),1), ((1,1),1)]),  # s7
            ]
            
            for A_terms, B_terms in strassen_2x2:
                u_acc = {}
                v_acc = {}
                w_acc = {}
                for (a_idx, a_coeff) in A_terms:
                    for (b_idx, b_coeff) in B_terms:
                        c_coeff = coeff * a_coeff * b_coeff
                        if c_coeff == 0:
                            continue
                        # Get global indices
                        i, j = A_entries[a_idx]
                        j2, k = B_entries[b_idx]
                        # j and j2 should be the same (middle index)
                        ui = i * m + j
                        vi = j * p + k
                        wi = k * n + i
                        u_acc[ui] = u_acc.get(ui, 0) + c_coeff
                        v_acc[vi] = v_acc.get(vi, 0) + c_coeff
                        w_acc[wi] = w_acc.get(wi, 0) + c_coeff
                u_acc = {k: v for k, v in u_acc.items() if v != 0}
                v_acc = {k: v for k, v in v_acc.items() if v != 0}
                w_acc = {k: v for k, v in w_acc.items() if v != 0}
                if u_acc and v_acc and w_acc:
                    U_cols.append(u_acc)
                    V_cols.append(v_acc)
                    W_cols.append(w_acc)
        
        # Define the 7 Strassen products for 4x4
        # Each is a list of (A_block, B_block, coeff)
        M1 = [((0,0),(0,0),1), ((0,0),(0,1),1), ((1,1),(0,0),1), ((1,1),(0,1),1)]
        M2 = [((1,0),(0,0),1), ((1,1),(0,0),1)]
        M3 = [((0,0),(0,1),1), ((0,0),(1,1),-1)]
        M4 = [((1,1),(1,0),1), ((1,1),(0,0),-1)]
        M5 = [((0,0),(1,1),1), ((0,1),(1,1),1)]
        M6 = [((1,0),(0,0),1), ((1,0),(0,1),1), ((0,0),(0,0),-1), ((0,0),(0,1),-1)]
        M7 = [((0,1),(1,0),1), ((0,1),(1,1),1), ((1,1),(1,0),-1), ((1,1),(1,1),-1)]
        
        for prod in [M1, M2, M3, M4, M5, M6, M7]:
            for (ab, bc, c_coeff) in prod:
                a_blk, b_blk = ab
                b_blk2, c_blk = bc
                # The middle index must match: b_blk == b_blk2
                if b_blk != b_blk2:
                    continue
                # Get the 2x2 blocks
                A_entries = {}
                for (i, j) in get_A_block(a_blk, b_blk):
                    di = i - (i_off + 2*a_blk)
                    dj = j - (j_off + 2*b_blk)
                    A_entries[(di, dj)] = (i, j)
                
                B_entries = {}
                for (j, k) in get_B_block(b_blk2, c_blk):
                    dj = j - (j_off + 2*b_blk2)
                    dk = k - (k_off + 2*c_blk)
                    B_entries[(dj, dk)] = (j, k)
                
                add_strassen_2x2(U_cols, V_cols, W_cols, A_entries, B_entries, c_coeff)
    
    # Add 4x4 core at offset (0,0,0)
    add_strassen_4x4(U_cols, V_cols, W_cols, 0, 0, 0)

    rank = len(U_cols)

    # Build dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        for idx, val in U_cols[r].items():
            U[idx][r] = val
        for idx, val in V_cols[r].items():
            V[idx][r] = val
        for idx, val in W_cols[r].items():
            W[idx][r] = val

    # Verify with random integer tests
    import random
    rng = random.Random(12345)
    for trial in range(200):
        if time.time() > tmax:
            break
        A = [[rng.randint(0, 5) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(0, 5) for _ in range(p)] for _ in range(m)]
        C_true = [[sum(A[ii][jj] * B[jj][kk] for jj in range(m)) for kk in range(p)] for ii in range(n)]
        C_test = [[0] * p for _ in range(n)]
        for ii in range(n):
            for kk in range(p):
                s = 0
                for r in range(rank):
                    for jj in range(m):
                        s += U[ii * m + jj][r] * V[jj * p + kk][r] * W[kk * n + ii][r]
                C_test[ii][kk] = s
        if C_test != C_true:
            return _schoolbook()

    return {"U": U, "V": V, "W": W}


def _schoolbook():
    """Fallback schoolbook decomposition, rank 125."""
    n, m, p = 5, 5, 5
    rk = n * m * p
    U = [[0] * rk for _ in range(n * m)]
    V = [[0] * rk for _ in range(m * p)]
    W = [[0] * rk for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
