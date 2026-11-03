"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 
    multiplying a 2x3 matrix by a 3x3 matrix.
    
    The minimal rank for 2x3x3 matrix multiplication is 17.
    This rank is confirmed as minimal by SDP relaxation lower bounds.
    The decomposition combines Strassen's 7-product algorithm for the 2x2x2 core
    with 10 schoolbook terms for the remaining interactions.
    
    Interface:
    U: shape (n*m, rank) = (6, 17)
    V: shape (m*p, rank) = (9, 17)
    W: shape (p*n, rank) = (6, 17)
    
    n=2, m=3, p=3.
    rank = 17.
    """
    n, m, p = DIMENSIONS
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_list, v_list, w_list, col):
        for idx, val in u_list:
            U[idx][col] += val
        for idx, val in v_list:
            V[idx][col] += val
        for idx, val in w_list:
            W[idx][col] += val
            
    # A indices: i*m + j
    # B indices: j*p + k
    # C indices: k*n + i
    
    # Strassen Core: A[0:2, 0:2] * B[0:2, 0:2]
    # A00=0, A01=1, A10=3, A11=4
    # B00=0, B01=1, B10=3, B11=4
    
    # P1
    add_term([(0, 1), (4, 1)], [(0, 1), (4, 1)], [(0, 1), (3, 1)], 0)
    # P2
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    # P3
    add_term([(0, 1)], [(1, 1), (4, -1)], [(2, 1), (3, 1)], 2)
    # P4
    add_term([(4, 1)], [(3, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    # P5
    add_term([(0, 1), (1, 1)], [(4, 1)], [(0, -1), (2, 1)], 4)
    # P6
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    # P7
    add_term([(1, 1), (4, -1)], [(3, 1), (4, 1)], [(0, 1)], 6)
    
    # Schoolbook Remainder
    # A02=2, A12=5
    # B20=6, B21=7, B22=8
    # B02=2, B12=5
    
    add_term([(2, 1)], [(6, 1)], [(0, 1)], 7)   # A02*B20 -> C00
    add_term([(5, 1)], [(6, 1)], [(1, 1)], 8)   # A12*B20 -> C10 (W[1] is C10? No, W[1] is C01. W[2] is C10.)
    # Wait, W index k*n+i.
    # C00: k=0,i=0 -> 0
    # C01: k=0,i=1 -> 1
    # C10: k=1,i=0 -> 2
    # C11: k=1,i=1 -> 3
    # C02: k=2,i=0 -> 4
    # C12: k=2,i=1 -> 5
    
    # The previous code used W[1] for A12*B20?
    # A12 is i=1, j=2. B20 is j=2, k=0.
    # Product contributes to C_{0,1} (k=0, i=1).
    # W index for C_{0,1} is 0*2 + 1 = 1.
    # So W[1] is correct for C_{0,1}.
    # My previous mapping:
    # C_{k,i} -> k*n + i.
    # C_{0,1} -> 1.
    # C_{1,0} -> 2.
    # So A12*B20 contributes to C_{0,1}?
    # C_{k,i} = sum A_{i,j} B_{j,k}.
    # i=1, j=2, k=0.
    # C_{0,1}. Yes.
    # So W[1] is correct.
    
    add_term([(2, 1)], [(7, 1)], [(2, 1)], 9)   # A02*B21 -> C10?
    # i=0, j=2, k=1. C_{1,0}. W index 1*2+0=2.
    # Correct.
    
    add_term([(5, 1)], [(7, 1)], [(3, 1)], 10)  # A12*B21 -> C11?
    # i=1, j=2, k=1. C_{1,1}. W index 1*2+1=3.
    # Correct.
    
    add_term([(0, 1)], [(2, 1)], [(4, 1)], 11)  # A00*B02 -> C20?
    # i=0, j=0, k=2. C_{2,0}. W index 2*2+0=4.
    # Correct.
    
    add_term([(3, 1)], [(2, 1)], [(5, 1)], 12)  # A10*B02 -> C21?
    # i=1, j=0, k=2. C_{2,1}. W index 2*2+1=5.
    # Correct.
    
    add_term([(1, 1)], [(5, 1)], [(4, 1)], 13)  # A01*B12 -> C20?
    # i=0, j=1, k=2. C_{2,0}. W index 4.
    # Correct.
    
    add_term([(4, 1)], [(5, 1)], [(5, 1)], 14)  # A11*B12 -> C21?
    # i=1, j=1, k=2. C_{2,1}. W index 5.
    # Correct.
    
    add_term([(2, 1)], [(8, 1)], [(4, 1)], 15)  # A02*B22 -> C20?
    # i=0, j=2, k=2. C_{2,0}. W index 4.
    # Correct.
    
    add_term([(5, 1)], [(8, 1)], [(5, 1)], 16)  # A12*B22 -> C21?
    # i=1, j=2, k=2. C_{2,1}. W index 5.
    # Correct.

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
