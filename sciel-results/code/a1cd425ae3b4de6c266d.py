import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank calculation:
    # 6 blocks of C (2x3 grid of 2x2 blocks)
    # Each block has 2 Strassen terms (rank 7 each) and 1 schoolbook term (rank 4)
    # Total rank = 6 * (7 + 7 + 4) = 6 * 18 = 108
    rank = 108
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1

    # Helper for Strassen 2x2x2
    # local u_idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # local v_idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # local w_idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    def add_strassen(base_u, base_v, base_w):
        nonlocal col
        # M1: (A00+A11)(B00+B11)
        U[base_u+0][col] += 1; U[base_u+3][col] += 1
        V[base_v+0][col] += 1; V[base_v+3][col] += 1
        W[base_w+0][col] += 1; W[base_w+3][col] += 1
        col += 1
        # M2: (A10+A11)B00
        U[base_u+2][col] += 1; U[base_u+3][col] += 1
        V[base_v+0][col] += 1
        W[base_w+2][col] += 1; W[base_w+3][col] -= 1
        col += 1
        # M3: A00(B01-B11)
        U[base_u+0][col] += 1
        V[base_v+1][col] += 1; V[base_v+3][col] -= 1
        W[base_w+1][col] += 1; W[base_w+3][col] += 1
        col += 1
        # M4: A11(B10-B00)
        U[base_u+3][col] += 1
        V[base_v+2][col] += 1; V[base_v+0][col] -= 1
        W[base_w+0][col] += 1; W[base_w+2][col] += 1
        col += 1
        # M5: (A00+A01)B11
        U[base_u+0][col] += 1; U[base_u+1][col] += 1
        V[base_v+3][col] += 1
        W[base_w+0][col] -= 1; W[base_w+1][col] += 1
        col += 1
        # M6: (A10-A00)(B00+B01)
        U[base_u+2][col] += 1; U[base_u+0][col] -= 1
        V[base_v+0][col] += 1; V[base_v+1][col] += 1
        W[base_w+3][col] += 1
        col += 1
        # M7: (A01-A11)(B10+B11)
        U[base_u+1][col] += 1; U[base_u+3][col] -= 1
        V[base_v+2][col] += 1; V[base_v+3][col] += 1
        W[base_w+0][col] += 1
        col += 1

    # Helper for Schoolbook 2x1x2
    # A is 2x1 (rows 0,1; col 0)
    # B is 1x2 (row 0; cols 0,1)
    # C is 2x2
    def add_schoolbook_212(base_u, base_v, base_w):
        nonlocal col
        # C00 = A00*B00
        U[base_u+0][col] += 1
        V[base_v+0][col] += 1
        W[base_w+0][col] += 1
        col += 1
        # C01 = A00*B01
        U[base_u+0][col] += 1
        V[base_v+1][col] += 1
        W[base_w+1][col] += 1
        col += 1
        # C10 = A10*B00
        U[base_u+2][col] += 1
        V[base_v+0][col] += 1
        W[base_w+2][col] += 1
        col += 1
        # C11 = A10*B01
        U[base_u+2][col] += 1
        V[base_v+1][col] += 1
        W[base_w+3][col] += 1
        col += 1

    # Partitioning:
    # n=4: rows 0,1 (block 0), rows 2,3 (block 1)
    # m=5: cols 0,1 (block 0), cols 2,3 (block 1), col 4 (block 2)
    # p=6: cols 0,1 (block 0), cols 2,3 (block 1), cols 4,5 (block 2)
    
    # C blocks:
    # C[0,0] (rows 0,1; cols 0,1):
    #   A[0:2, 0:2] * B[0:2, 0:2]  (Strassen)
    #   A[0:2, 2:4] * B[2:4, 0:2]  (Strassen)
    #   A[0:2, 4:5] * B[4:5, 0:2]  (Schoolbook 2x1x2)
    
    # Bases for A (n*m = 20)
    # A row block 0 (rows 0,1): base 0
    # A row block 1 (rows 2,3): base 10 (2*5)
    # A col block 0 (cols 0,1): offset 0
    # A col block 1 (cols 2,3): offset 2
    # A col block 2 (col 4): offset 4
    
    # Bases for B (m*p = 30)
    # B row block 0 (rows 0,1): base 0
    # B row block 1 (rows 2,3): base 12 (2*6)
    # B row block 2 (row 4): base 24 (4*6)
    # B col block 0 (cols 0,1): offset 0
    # B col block 1 (cols 2,3): offset 2
    # B col block 2 (cols 4,5): offset 4
    
    # Bases for W (p*n = 24)
    # W index k*n + i.
    # C col block 0 (cols 0,1): k=0,1? No, k is the column of C.
    # C is 4x6.
    # C row block 0 (rows 0,1): i=0,1.
    # C row block 1 (rows 2,3): i=2,3.
    # C col block 0 (cols 0,1): k=0,1.
    # C col block 1 (cols 2,3): k=2,3.
    # C col block 2 (cols 4,5): k=4,5.
    
    # W base for C[row_block, col_block]:
    # w_idx = k*n + i
    # For C[0,0]: k in 0,1; i in 0,1.
    # w=0: k=0,i=0. w=1: k=0,i=1. w=2: k=1,i=0. w=3: k=1,i=1.
    # Base W for C[0,0] is 0.
    # For C[0,1]: k in 2,3; i in 0,1.
    # w=2*4+0=8. w=2*4+1=9. w=3*4+0=12. w=3*4+1=13.
    # Base W for C[0,1] is 8.
    # For C[0,2]: k in 4,5; i in 0,1.
    # w=4*4+0=16. w=17. w=20. w=21.
    # Base W for C[0,2] is 16.
    
    # For C[1,0]: k in 0,1; i in 2,3.
    # w=0*4+2=2. w=3. w=1*4+2=6. w=7.
    # Base W for C[1,0] is 2.
    
    # For C[1,1]: k in 2,3; i in 2,3.
    # w=2*4+2=10. w=11. w=14. w=15.
    # Base W for C[1,1] is 10.
    
    # For C[1,2]: k in 4,5; i in 2,3.
    # w=4*4+2=18. w=19. w=22. w=23.
    # Base W for C[1,2] is 18.

    # Loop over C blocks
    for ci in range(2): # C row block
        for cj in range(3): # C col block
            
            # Base indices
            # A row block ci: base_u_a = ci * 10
            base_u_a = ci * 10
            
            # B col block cj: offset_v_b = cj * 2
            offset_v_b = cj * 2
            
            # W base for C[ci, cj]
            # k_start = cj * 2
            # i_start = ci * 2
            # w_base = k_start * 4 + i_start
            w_base = (cj * 2) * 4 + (ci * 2)
            
            # Term 1: A[ci, 0] * B[0, cj] (2x2 * 2x2)
            # A col block 0: offset 0
            # B row block 0: base 0
            base_u = base_u_a + 0
            base_v = 0 + offset_v_b
            add_strassen(base_u, base_v, w_base)
            
            # Term 2: A[ci, 1] * B[1, cj] (2x2 * 2x2)
            # A col block 1: offset 2
            # B row block 1: base 12
            base_u = base_u_a + 2
            base_v = 12 + offset_v_b
            add_strassen(base_u, base_v, w_base)
            
            # Term 3: A[ci, 2] * B[2, cj] (2x1 * 1x2)
            # A col block 2: offset 4
            # B row block 2: base 24
            base_u = base_u_a + 4
            base_v = 24 + offset_v_b
            add_schoolbook_212(base_u, base_v, w_base)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
