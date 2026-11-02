DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=7
    # We construct a decomposition of rank 56.
    # Split A into A1 (3x2), A2 (3x2)
    # Split B into B1 (2x7), B2 (2x7)
    # C = A1 B1 + A2 B2
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    # Process C1 = A1 B1
    # A1: i in 0..2, j in 0..1
    # B1: j in 0..1, k in 0..6
    
    # i=0 part: A1[0,0]*B1[0,k] + A1[0,1]*B1[1,k]
    for k in range(p):
        add_term(0 * m + 0, 0 * p + k, k * n + 0, 1) # A[0,0] B[0,k] -> C[0,k]
        add_term(0 * m + 1, 1 * p + k, k * n + 0, 1) # A[0,1] B[1,k] -> C[0,k]
        
    # i in 1,2 part: A1[1:3, :] is 2x2. B1 is 2x7.
    # Split B1 into 3 blocks of 2x2 and 1 block of 2x1
    # Block 1: k=0,1 (2x2)
    # Block 2: k=2,3 (2x2)
    # Block 3: k=4,5 (2x2)
    # Block 4: k=6   (2x1)
    
    # Helper for 2x2x2 schoolbook
    # A_local: i in 0,1; j in 0,1
    # B_local: j in 0,1; k in 0,1
    # C_local: i in 0,1; k in 0,1
    def add_2x2x2_schoolbook(i_base, j_base, k_base):
        nonlocal col
        # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
        add_term((i_base+0)*m + (j_base+0), (j_base+0)*p + (k_base+0), (k_base+0)*n + (i_base+0), 1)
        add_term((i_base+0)*m + (j_base+1), (j_base+1)*p + (k_base+0), (k_base+0)*n + (i_base+0), 1)
        # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
        add_term((i_base+0)*m + (j_base+0), (j_base+0)*p + (k_base+1), (k_base+1)*n + (i_base+0), 1)
        add_term((i_base+0)*m + (j_base+1), (j_base+1)*p + (k_base+1), (k_base+1)*n + (i_base+0), 1)
        # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
        add_term((i_base+1)*m + (j_base+0), (j_base+0)*p + (k_base+0), (k_base+0)*n + (i_base+1), 1)
        add_term((i_base+1)*m + (j_base+1), (j_base+1)*p + (k_base+0), (k_base+0)*n + (i_base+1), 1)
        # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
        add_term((i_base+1)*m + (j_base+0), (j_base+0)*p + (k_base+1), (k_base+1)*n + (i_base+1), 1)
        add_term((i_base+1)*m + (j_base+1), (j_base+1)*p + (k_base+1), (k_base+1)*n + (i_base+1), 1)

    # Helper for 2x2x1 schoolbook
    def add_2x2x1_schoolbook(i_base, j_base, k_base):
        nonlocal col
        # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
        add_term((i_base+0)*m + (j_base+0), (j_base+0)*p + (k_base+0), (k_base+0)*n + (i_base+0), 1)
        add_term((i_base+0)*m + (j_base+1), (j_base+1)*p + (k_base+0), (k_base+0)*n + (i_base+0), 1)
        # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
        add_term((i_base+1)*m + (j_base+0), (j_base+0)*p + (k_base+0), (k_base+0)*n + (i_base+1), 1)
        add_term((i_base+1)*m + (j_base+1), (j_base+1)*p + (k_base+0), (k_base+0)*n + (i_base+1), 1)

    # C1 blocks
    add_2x2x2_schoolbook(1, 0, 0) # i=1,2; j=0,1; k=0,1
    add_2x2x2_schoolbook(1, 0, 2) # i=1,2; j=0,1; k=2,3
    add_2x2x2_schoolbook(1, 0, 4) # i=1,2; j=0,1; k=4,5
    add_2x2x1_schoolbook(1, 0, 6) # i=1,2; j=0,1; k=6
    
    # Process C2 = A2 B2
    # A2: i in 0..2, j in 2..3
    # B2: j in 2..3, k in 0..6
    
    # i=0 part: A2[0,2]*B2[2,k] + A2[0,3]*B2[3,k]
    for k in range(p):
        add_term(0 * m + 2, 2 * p + k, k * n + 0, 1)
        add_term(0 * m + 3, 3 * p + k, k * n + 0, 1)
        
    # i in 1,2 part
    # j_base = 2
    add_2x2x2_schoolbook(1, 2, 0)
    add_2x2x2_schoolbook(1, 2, 2)
    add_2x2x2_schoolbook(1, 2, 4)
    add_2x2x1_schoolbook(1, 2, 6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
