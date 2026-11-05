"""Improved program for the suite task; uses block decomposition with Strassen for 2x2x2 blocks."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=5, p=7
    
    # Total rank calculation
    # Block 1: n=1, m=5, p=7 -> Rank 35
    # Block 2: n=2, m=5, p=7
    #   Sub 2a: n=2, m=2, p=7 -> Rank 25 (3x Strassen 7 + 1x Trivial 4)
    #   Sub 2b: n=2, m=2, p=7 -> Rank 25
    #   Sub 2c: n=2, m=1, p=7 -> Rank 14
    #   Total Block 2: 64
    # Total Rank: 99
    
    total_rank = 99
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a trivial block
    def add_trivial_block(i_start, j_start, k_start, n_b, m_b, p_b):
        nonlocal col
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    u_idx = (i_start + i) * m + (j_start + j)
                    v_idx = (j_start + j) * p + (k_start + k)
                    w_idx = (k_start + k) * n + (i_start + i)
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1

    # Helper to add a Strassen block (2x2x2)
    def add_strassen_block(i_start, j_start, k_start):
        nonlocal col
        # Local indices 0,1 for i,j,k
        # U indices: (i, j)
        # V indices: (j, k)
        # W indices: (k, i)
        
        # Term 1: M1
        # U: (0,0)=1, (1,1)=1
        # V: (0,0)=1, (1,1)=1
        # W: (0,0)=1, (1,1)=1
        U[(i_start+0)*m + (j_start+0)][col] = 1
        U[(i_start+1)*m + (j_start+1)][col] = 1
        V[(j_start+0)*p + (k_start+0)][col] = 1
        V[(j_start+1)*p + (k_start+1)][col] = 1
        W[(k_start+0)*n + (i_start+0)][col] = 1
        W[(k_start+1)*n + (i_start+1)][col] = 1
        col += 1
        
        # Term 2: M2
        # U: (1,0)=1, (1,1)=1
        # V: (0,0)=1
        # W: (0,1)=1
        U[(i_start+1)*m + (j_start+0)][col] = 1
        U[(i_start+1)*m + (j_start+1)][col] = 1
        V[(j_start+0)*p + (k_start+0)][col] = 1
        W[(k_start+0)*n + (i_start+1)][col] = 1
        col += 1
        
        # Term 3: M3
        # U: (0,0)=1
        # V: (0,1)=1, (1,1)=-1
        # W: (0,1)=1, (1,1)=-1
        U[(i_start+0)*m + (j_start+0)][col] = 1
        V[(j_start+0)*p + (k_start+1)][col] = 1
        V[(j_start+1)*p + (k_start+1)][col] = -1
        W[(k_start+0)*n + (i_start+1)][col] = 1
        W[(k_start+1)*n + (i_start+1)][col] = -1
        col += 1
        
        # Term 4: M4
        # U: (1,1)=1
        # V: (1,0)=1, (0,0)=-1
        # W: (0,0)=1, (1,1)=1
        U[(i_start+1)*m + (j_start+1)][col] = 1
        V[(j_start+1)*p + (k_start+0)][col] = 1
        V[(j_start+0)*p + (k_start+0)][col] = -1
        W[(k_start+0)*n + (i_start+0)][col] = 1
        W[(k_start+1)*n + (i_start+1)][col] = 1
        col += 1
        
        # Term 5: M5
        # U: (0,0)=1, (0,1)=1
        # V: (1,1)=1
        # W: (1,0)=1
        U[(i_start+0)*m + (j_start+0)][col] = 1
        U[(i_start+0)*m + (j_start+1)][col] = 1
        V[(j_start+1)*p + (k_start+1)][col] = 1
        W[(k_start+1)*n + (i_start+0)][col] = 1
        col += 1
        
        # Term 6: M6
        # U: (1,0)=1, (0,0)=-1
        # V: (0,0)=1, (0,1)=1
        # W: (1,0)=1, (0,0)=-1
        U[(i_start+1)*m + (j_start+0)][col] = 1
        U[(i_start+0)*m + (j_start+0)][col] = -1
        V[(j_start+0)*p + (k_start+0)][col] = 1
        V[(j_start+0)*p + (k_start+1)][col] = 1
        W[(k_start+1)*n + (i_start+0)][col] = 1
        W[(k_start+0)*n + (i_start+0)][col] = -1
        col += 1
        
        # Term 7: M7
        # U: (0,1)=1, (1,1)=-1
        # V: (1,0)=1, (1,1)=1
        # W: (1,1)=1, (0,0)=-1
        U[(i_start+0)*m + (j_start+1)][col] = 1
        U[(i_start+1)*m + (j_start+1)][col] = -1
        V[(j_start+1)*p + (k_start+0)][col] = 1
        V[(j_start+1)*p + (k_start+1)][col] = 1
        W[(k_start+1)*n + (i_start+1)][col] = 1
        W[(k_start+0)*n + (i_start+0)][col] = -1
        col += 1

    # Block 1: n=1, m=5, p=7 (i=0, j=0..4, k=0..6)
    add_trivial_block(0, 0, 0, 1, 5, 7)
    
    # Block 2: n=2, m=5, p=7 (i=1..2, j=0..4, k=0..6)
    # Sub 2a: n=2, m=2, p=7 (i=1..2, j=0..1, k=0..6)
    #   Split p=7 into 2,2,2,1
    #   p=0..1: Strassen
    add_strassen_block(1, 0, 0)
    #   p=2..3: Strassen
    add_strassen_block(1, 0, 2)
    #   p=4..5: Strassen
    add_strassen_block(1, 0, 4)
    #   p=6: Trivial 2x2x1
    add_trivial_block(1, 0, 6, 2, 2, 1)
    
    # Sub 2b: n=2, m=2, p=7 (i=1..2, j=2..3, k=0..6)
    #   Split p=7 into 2,2,2,1
    #   p=0..1: Strassen
    add_strassen_block(1, 2, 0)
    #   p=2..3: Strassen
    add_strassen_block(1, 2, 2)
    #   p=4..5: Strassen
    add_strassen_block(1, 2, 4)
    #   p=6: Trivial 2x2x1
    add_trivial_block(1, 2, 6, 2, 2, 1)
    
    # Sub 2c: n=2, m=1, p=7 (i=1..2, j=4, k=0..6)
    add_trivial_block(1, 4, 0, 2, 1, 7)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
