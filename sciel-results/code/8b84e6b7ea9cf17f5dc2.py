import numpy as np

def construct():
    n, m, p = 3, 4, 7
    
    max_rank = 100
    U = np.zeros((n * m, max_rank), dtype=int)
    V = np.zeros((m * p, max_rank), dtype=int)
    W = np.zeros((p * n, max_rank), dtype=int)
    
    rank = 0
    
    # Helper to add a term
    # u_idx: global index in U
    # v_idx: global index in V
    # w_idx: global index in W
    # u_val, v_val, w_val: coefficients
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal rank
        U[u_idx, rank] = u_val
        V[v_idx, rank] = v_val
        W[w_idx, rank] = w_val
        rank += 1

    # Helper to add a standard multiplication block
    # local_n, local_m, local_p
    # offset_i, offset_j, offset_k
    def add_standard(local_n, local_m, local_p, offset_i, offset_j, offset_k):
        for i in range(local_n):
            for j in range(local_m):
                for k in range(local_p):
                    u_idx = (offset_i + i) * m + (offset_j + j)
                    v_idx = (offset_j + j) * p + (offset_k + k)
                    w_idx = (offset_k + k) * n + (offset_i + i)
                    add_term(u_idx, v_idx, w_idx, 1, 1, 1)

    # Helper to add Strassen block 2x2x2
    # offset_i, offset_j, offset_k
    def add_strassen(offset_i, offset_j, offset_k):
        # 7 terms
        # M1
        # A: (0,0), (1,1)
        # B: (0,0), (1,1)
        # C: (0,0), (1,1), (1,0), (0,1)
        
        # Term 1
        U[(offset_i+0)*m + (offset_j+0), rank] = 1
        U[(offset_i+1)*m + (offset_j+1), rank] = 1
        V[(offset_j+0)*p + (offset_k+0), rank] = 1
        V[(offset_j+1)*p + (offset_k+1), rank] = 1
        W[(offset_k+0)*n + (offset_i+0), rank] = 1
        W[(offset_k+1)*n + (offset_i+1), rank] = 1
        W[(offset_k+1)*n + (offset_i+0), rank] = -1
        W[(offset_k+0)*n + (offset_i+1), rank] = 1
        rank += 1
        
        # Term 2
        # A: (1,0), (1,1)
        # B: (0,0)
        # C: (1,0)
        U[(offset_i+1)*m + (offset_j+0), rank] = 1
        U[(offset_i+1)*m + (offset_j+1), rank] = 1
        V[(offset_j+0)*p + (offset_k+0), rank] = 1
        W[(offset_k+0)*n + (offset_i+1), rank] = 1
        rank += 1
        
        # Term 3
        # A: (0,0)
        # B: (0,1), (1,1)
        # C: (0,1)
        U[(offset_i+0)*m + (offset_j+0), rank] = 1
        V[(offset_j+0)*p + (offset_k+1), rank] = 1
        V[(offset_j+1)*p + (offset_k+1), rank] = -1
        W[(offset_k+1)*n + (offset_i+0), rank] = 1
        rank += 1
        
        # Term 4
        # A: (1,1)
        # B: (1,0), (0,0)
        # C: (0,0)
        U[(offset_i+1)*m + (offset_j+1), rank] = 1
        V[(offset_j+1)*p + (offset_k+0), rank] = 1
        V[(offset_j+0)*p + (offset_k+0), rank] = -1
        W[(offset_k+0)*n + (offset_i+0), rank] = 1
        rank += 1
        
        # Term 5
        # A: (0,0), (0,1)
        # B: (1,1)
        # C: (1,1)
        U[(offset_i+0)*m + (offset_j+0), rank] = 1
        U[(offset_i+0)*m + (offset_j+1), rank] = 1
        V[(offset_j+1)*p + (offset_k+1), rank] = 1
        W[(offset_k+1)*n + (offset_i+1), rank] = 1
        rank += 1
        
        # Term 6
        # A: (1,0), (0,0)
        # B: (0,0), (0,1)
        # C: (1,1)
        U[(offset_i+1)*m + (offset_j+0), rank] = 1
        U[(offset_i+0)*m + (offset_j+0), rank] = -1
        V[(offset_j+0)*p + (offset_k+0), rank] = 1
        V[(offset_j+0)*p + (offset_k+1), rank] = 1
        W[(offset_k+1)*n + (offset_i+1), rank] = 1
        rank += 1
        
        # Term 7
        # A: (0,1), (1,1)
        # B: (1,0), (1,1)
        # C: (0,0)
        U[(offset_i+0)*m + (offset_j+1), rank] = 1
        U[(offset_i+1)*m + (offset_j+1), rank] = -1
        V[(offset_j+1)*p + (offset_k+0), rank] = 1
        V[(offset_j+1)*p + (offset_k+1), rank] = 1
        W[(offset_k+0)*n + (offset_i+0), rank] = 1
        rank += 1

    # Block 1: p in [0, 3)
    # Split m=4 into 2+2
    # Sub-block 1.1: n=3, m=2, p=3, offsets i=0, j=0, k=0
    #   Split n=3 into 2+1
    #   Sub-sub 1.1.1: n=2, m=2, p=3, offsets i=0, j=0, k=0
    #     Split p=3 into 2+1
    #     Sub-sub-sub 1.1.1.1: n=2, m=2, p=2, offsets i=0, j=0, k=0 -> Strassen
    #     Sub-sub-sub 1.1.1.2: n=2, m=2, p=1, offsets i=0, j=0, k=2 -> Standard
    #   Sub-sub 1.1.2: n=1, m=2, p=3, offsets i=2, j=0, k=0 -> Standard
    
    add_strassen(0, 0, 0)
    add_standard(2, 2, 1, 0, 0, 2)
    add_standard(1, 2, 3, 2, 0, 0)
    
    # Sub-block 1.2: n=3, m=2, p=3, offsets i=0, j=2, k=0
    #   Split n=3 into 2+1
    #   Sub-sub 1.2.1: n=2, m=2, p=3, offsets i=0, j=2, k=0
    #     Split p=3 into 2+1
    #     Sub-sub-sub 1.2.1.1: n=2, m=2, p=2, offsets i=0, j=2, k=0 -> Strassen
    #     Sub-sub-sub 1.2.1.2: n=2, m=2, p=1, offsets i=0, j=2, k=2 -> Standard
    #   Sub-sub 1.2.2: n=1, m=2, p=3, offsets i=2, j=2, k=0 -> Standard
    
    add_strassen(0, 2, 0)
    add_standard(2, 2, 1, 0, 2, 2)
    add_standard(1, 2, 3, 2, 2, 0)

    # Block 2: p in [3, 7)
    # Split n=3 into 2+1
    # Sub-block 2.1: n=2, m=4, p=4, offsets i=0, j=0, k=3
    #   Split m=4 into 2+2
    #   Sub-sub 2.1.1: n=2, m=2, p=4, offsets i=0, j=0, k=3
    #     Split p=4 into 2+2
    #     Sub-sub-sub 2.1.1.1: n=2, m=2, p=2, offsets i=0, j=0, k=3 -> Strassen
    #     Sub-sub-sub 2.1.1.2: n=2, m=2, p=2, offsets i=0, j=0, k=5 -> Strassen
    #   Sub-sub 2.1.2: n=2, m=2, p=4, offsets i=0, j=2, k=3
    #     Split p=4 into 2+2
    #     Sub-sub-sub 2.1.2.1: n=2, m=2, p=2, offsets i=0, j=2, k=3 -> Strassen
    #     Sub-sub-sub 2.1.2.2: n=2, m=2, p=2, offsets i=0, j=2, k=5 -> Strassen
    
    add_strassen(0, 0, 3)
    add_strassen(0, 0, 5)
    add_strassen(0, 2, 3)
    add_strassen(0, 2, 5)
    
    # Sub-block 2.2: n=1, m=4, p=4, offsets i=2, j=0, k=3
    # Standard
    
    add_standard(1, 4, 4, 2, 0, 3)

    # Trim
    U = U[:, :rank]
    V = V[:, :rank]
    W = W[:, :rank]

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
