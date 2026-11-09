"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36
    
    # Use Strassen-based decomposition for 4x4 * 4x9
    # Split A (4x4) into 2x2 blocks, B (4x9) into 2x9 blocks
    # Each 2x2 * 2x9 product uses Strassen with rank 7
    # Total: 4 * 7 = 28
    
    def strassen_2x2_2x9(U_offset, V_offset, W_offset, rank_offset, A_block_idx, B_block_idx):
        """
        Generate Strassen decomposition for 2x2 * 2x9 product.
        A_block_idx: which 2x2 block of A (0=A11, 1=A12, 2=A21, 3=A22)
        B_block_idx: which 2x9 block of B (0=B_top, 1=B_bot)
        Returns list of (U_entries, V_entries, W_entries) for 7 rank-1 terms.
        """
        terms = []
        
        # A block is 2x2, indexed by (i,j) where i=0,1 (row within block), j=0,1 (col within block)
        # B block is 2x9, indexed by (j,k) where j=0,1 (row within block), k=0..8 (col)
        # Output is 2x9, indexed by (i,k) where i=0,1 (row), k=0..8 (col)
        
        # Global indices:
        # A: i_global = block_row*2 + i_local, j_global = block_col*2 + j_local
        # B: j_global = block_row*2 + j_local, k_global = k_local
        # W: k*n + i_global where i_global is the output row
        
        # For A block (br, bc): A_local(i,j) -> A_global(br*2+i, bc*2+j)
        # For B block (br): B_local(j,k) -> B_global(br*2+j, k)
        # Output: C_local(i,k) -> C_global(br*2+i, k) -> W index = k*4 + (br*2+i)
        
        # Strassen 7 products for M(2x2) * N(2x9):
        # M = [[a, b], [c, d]] where a=M[0,0], b=M[0,1], c=M[1,0], d=M[1,1]
        # N has rows n1 (N[0,:]) and n2 (N[1,:]), each 1x9
        
        # P1 = (a+d) * (n1+n2)
        # P2 = (c+d) * (n1+n2)
        # P3 = (a-b) * (n2-n1)
        # P4 = (a+c) * (n1-n2)
        # P5 = (b+d) * (n2-n1)
        # P6 = (a-c) * (n1+n2)
        # P7 = (b-d) * (n2+n1)
        
        # Result row 0: P1 + P4 - P5 + P7
        # Result row 1: P2 + P6 - P4 + P3
        
        # Wait, let me verify with standard Strassen:
        # For square 2x2 * 2x2:
        # P1 = (a+d)(e+h)
        # P2 = (c+d)(e+f)
        # P3 = (a-b)(g+h)
        # P4 = (a+c)(e-g)
        # P5 = (b+d)(f-h)
        # P6 = (a-c)(e+g)
        # P7 = (b-d)(f+h)
        # C[0,0] = P1 + P4 - P5 + P7
        # C[0,1] = P3 + P5
        # C[1,0] = P2 + P4
        # C[1,1] = P1 - P2 + P3 + P6
        
        # For rectangular 2x2 * 2x9, N = [[n1_0, n1_1, ..., n1_8],
        #                               [n2_0, n2_1, ..., n2_8]]
        # This is like 9 separate 2x2 * 1x1 products, but we can factor.
        # The 7 products become:
        # P1 = (a+d) * (n1+n2)  -- 1x9 vector
        # P2 = (c+d) * (n1+n2)  -- wait, this doesn't match standard Strassen
        
        # Let me use the correct rectangular Strassen.
        # Actually, for 2x2 * 2x9, we can think of it as computing M * N where N is 2x9.
        # The standard Strassen decomposition for 2x2 * 2x2 gives 7 products.
        # For 2x2 * 2x9, we apply the same 7 bilinear forms but with N having 9 columns.
        
        # The 7 bilinear forms (scalar products that become scalar*vector products):
        # Let N_col_k = [n1_k, n2_k]^T for k=0..8
        # For each k, the 7 Strassen products give:
        # P1_k = (a+d)(n1_k + n2_k)
        # P2_k = (c+d)(n1_k + n2_k)  -- NO, this is wrong
        
        # I think I need to be more careful. Let me derive from scratch.
        # We want: C[0,k] = a*n1_k + b*n2_k
        #            C[1,k] = c*n1_k + d*n2_k
        
        # Using 7 bilinear forms:
        # P1 = (a+d) * (n1+n2)  where n1, n2 are 1x9 vectors
        # P2 = (c+d) * (n1+n2)
        # P3 = (a-b) * (n2-n1)
        # P4 = (a+c) * (n1-n2)
        # P5 = (b+d) * (n2-n1)
        # P6 = (a-c) * (n1+n2)
        # P7 = (b-d) * (n2+n1)
        
        # C[0,:] = P1 + P4 - P5 + P7
        # C[1,:] = P2 + P6 - P4 + P3
        
        # Verify C[0,k]:
        # = (a+d)(n1_k+n2_k) + (a+c)(n1_k-n2_k) - (b+d)(n2_k-n1_k) + (b-d)(n2_k+n1_k)
        # = (a+d)(n1_k+n2_k) + (a+c)(n1_k-n2_k) + (b+d)(n1_k-n2_k) + (b-d)(n1_k+n2_k)
        # n1_k coefficient: (a+d) + (a+c) + (b+d) + (b-d) = a+d+a+c+b+d+b-d = 2a+b+c+d
        # That's not a. So this is wrong.
        
        # Let me use the correct Strassen formulas for rectangular case.
        # The key is that for 2x2 * 2x9, we can use the same 7 products as 2x2*2x2
        # but applied column-wise, which means each product is a scalar times a 1x9 vector.
        
        # Standard Strassen for 2x2 * 2x2 with N = [[e,f],[g,h]]:
        # P1 = (a+d)(e+h)
        # P2 = (c+d)(e+f)
        # P3 = (a-b)(g+h)
        # P4 = (a+c)(e-g)
        # P5 = (b+d)(f-h)
        # P6 = (a-c)(e+g)
        # P7 = (b-d)(f+h)
        
        # For 2x2 * 2x9, N = [[n1_0, n1_1, ..., n1_8],
        #                       [n2_0, n2_1, ..., n2_8]]
        # We can write this as 9 separate 2x2 * 1x1 problems, but that's 63 products.
        # Instead, we factor: each Strassen product is a scalar times a 1x9 vector.
        
        # The 7 products for the rectangular case:
        # P1 = (a+d) * (n1 + n2)  where n1, n2 are 1x9
        # P2 = (c+d) * (n1 + n2)  -- NO
        
        # I think the issue is that standard Strassen doesn't directly extend this way.
        # Let me just use a different approach: use the known rank-7 decomposition.
        
        # Actually, I recall that for 2x2 * 2xn, the rank is 7 for n>=2.
        # The decomposition uses 7 bilinear forms. Let me look this up mentally.
        
        # For M = [[a,b],[c,d]] and N (2xn):
        # Let N1 = first row of N (1xn), N2 = second row of N (1xn)
        # 
        # 7 products:
        # P1 = (a+d)(N1+N2)
        # P2 = (c+d)(N1+N2)
        # P3 = (a-b)(N2-N1)
        # P4 = (a+c)(N1-N2)
        # P5 = (b+d)(N2-N1)
        # P6 = (a-c)(N1+N2)
        # P7 = (b-d)(N2+N1)
        #
        # C row 0 = P1 + P4 - P5 + P7
        # C row 1 = P2 + P6 - P4 + P3
        #
        # I already verified this doesn't work. Let me try a different set.
        #
        # Maybe:
        # P1 = (a+d)(N1+N2)
        # P2 = (c+d)N1
        # P3 = a(N2-N1)
        # P4 = d*N1
        # P5 = (a+b)N2
        # P6 = (c-a)(N1+N2)
        # P7 = (b-d)(N2+N1)
        #
        # C row 0 = P1 - P5 + P7
        # C row 1 = P4 + P5 - P1 + P6
        #
        # Verify C[0,k]:
        # = (a+d)(n1_k+n2_k) - (a+b)n2_k + (b-d)(n2_k+n1_k)
        # = (a+d)n1_k + (a+d)n2_k - (a+b)n2_k + (b-d)n2_k + (b-d)n1_k
        # n1_k: (a+d) + (b-d) = a+b
        # n2_k: (a+d) - (a+b) + (b-d) = a+d-a-b+b-d = 0
        # So C[0,k] = (a+b)n1_k, which is wrong.
        
        # I'm struggling with the exact Strassen formulas for rectangular case.
        # Let me just implement the schoolbook for now and verify it works,
        # then try to improve.
        
        # Actually, let me just use the known result and implement it correctly.
        # For 2x2 * 2xn, one valid rank-7 decomposition is:
        #
        # P1 = (a+d) * (N1 + N2)
        # P2 = (c+d) * (N1 + N2)  
        # P3 = (a-b) * (N2 - N1)
        # P4 = (a+c) * (N1 - N2)
        # P5 = (b+d) * (N2 - N1)
        # P6 = (a-c) * (N1 + N2)
        # P7 = (b-d) * (N2 + N1)
        #
        # Hmm, let me try yet another approach. I'll just verify computationally.
        
        return terms
    
    # Let me just implement schoolbook first to ensure correctness, then optimize
    rank = n * m * p  # 144
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
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
