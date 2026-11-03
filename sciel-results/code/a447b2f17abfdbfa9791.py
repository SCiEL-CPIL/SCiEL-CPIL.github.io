"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-84 tensor decomposition for 3x5x6 matrix multiplication.
    
    The approach uses a block decomposition:
    n=3 split into 2+1
    m=5 split into 2+2+1
    p=6 split into 2+2+2
    
    Sub-blocks:
    - (2,2,2) blocks: Rank 7 (Strassen)
    - Other blocks: Rank n*m*p (Schoolbook)
    
    Total Rank:
    6 blocks of (2,2,2) -> 6 * 7 = 42
    3 blocks of (2,1,2) -> 3 * 4 = 12
    6 blocks of (1,2,2) -> 6 * 4 = 24
    3 blocks of (1,1,2) -> 3 * 2 = 6
    Total: 42 + 12 + 24 + 6 = 84
    """
    import numpy as np

    n, m, p = DIMENSIONS  # 3, 5, 6

    # Define block splits
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]

    # Pre-calculate start indices for each block
    n_starts = []
    curr = 0
    for s in n_splits:
        n_starts.append(curr)
        curr += s
    m_starts = []
    curr = 0
    for s in m_splits:
        m_starts.append(curr)
        curr += s
    p_starts = []
    curr = 0
    for s in p_splits:
        p_starts.append(curr)
        curr += s

    U = []
    V = []
    W = []

    # Helper to add a rank-1 term
    def add_term(u_vec, v_vec, w_vec):
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)

    # Helper to add a schoolbook block
    def add_schoolbook_block(ni, mi, pi, n_loc, m_loc, p_loc):
        # ni, mi, pi are start indices in global n, m, p
        # n_loc, m_loc, p_loc are local dimensions
        for i in range(n_loc):
            for j in range(m_loc):
                for k in range(p_loc):
                    u = [0] * (n * m)
                    v = [0] * (m * p)
                    w = [0] * (p * n)
                    
                    # Global indices
                    gi = ni + i
                    gj = mi + j
                    gk = pi + k
                    
                    u[gi * m + gj] = 1
                    v[gj * p + gk] = 1
                    w[gk * n + gi] = 1
                    
                    add_term(u, v, w)

    # Helper to add a Strassen block (2x2x2)
    def add_strassen_block(ni, mi, pi):
        # ni, mi, pi are start indices
        # Local indices 0,1 for n, m, p
        
        # M1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
        u1 = [0]*(n*m); v1 = [0]*(m*p); w1 = [0]*(p*n)
        u1[ni*m + mi] = 1; u1[(ni+1)*m + (mi+1)] = 1
        v1[mi*p + pi] = 1; v1[(mi+1)*p + (pi+1)] = 1
        w1[pi*n + ni] = 1; w1[(pi+1)*n + (ni+1)] = 1
        add_term(u1, v1, w1)

        # M2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,-1]
        u2 = [0]*(n*m); v2 = [0]*(m*p); w2 = [0]*(p*n)
        u2[(ni+1)*m + mi] = 1; u2[(ni+1)*m + (mi+1)] = 1
        v2[mi*p + pi] = 1
        w2[(pi+1)*n + (ni+1)] = 1; w2[(pi+1)*n + (ni+1)] = -1 # Wait, w index for C22 is (pi+1)*n + (ni+1)
        # C21 is (pi)*n + (ni+1)
        w2[pi*n + (ni+1)] = 1
        w2[(pi+1)*n + (ni+1)] = -1
        add_term(u2, v2, w2)

        # M3: u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,1]
        u3 = [0]*(n*m); v3 = [0]*(m*p); w3 = [0]*(p*n)
        u3[ni*m + mi] = 1
        v3[mi*p + (pi+1)] = 1; v3[(mi+1)*p + (pi+1)] = -1
        w3[pi*n + (ni+1)] = 1 # C12 is (pi)*n + (ni+1)
        w3[(pi+1)*n + (ni+1)] = 1 # C22 is (pi+1)*n + (ni+1)
        add_term(u3, v3, w3)

        # M4: u=[0,0,0,1], v=[0,0,1,-1], w=[1,0,1,0]
        u4 = [0]*(n*m); v4 = [0]*(m*p); w4 = [0]*(p*n)
        u4[(ni+1)*m + (mi+1)] = 1
        v4[(mi+1)*p + pi] = 1; v4[(mi+1)*p + (pi+1)] = -1
        w4[pi*n + ni] = 1 # C11
        w4[pi*n + (ni+1)] = 1 # C21
        add_term(u4, v4, w4)

        # M5: u=[1,1,0,0], v=[0,0,0,1], w=[-1,1,0,0]
        u5 = [0]*(n*m); v5 = [0]*(m*p); w5 = [0]*(p*n)
        u5[ni*m + mi] = 1; u5[ni*m + (mi+1)] = 1
        v5[(mi+1)*p + (pi+1)] = 1
        w5[pi*n + ni] = -1 # C11
        w5[pi*n + (ni+1)] = 1 # C12
        add_term(u5, v5, w5)

        # M6: u=[0,0,1,-1], v=[1,1,0,0], w=[0,0,0,1]
        u6 = [0]*(n*m); v6 = [0]*(m*p); w6 = [0]*(p*n)
        u6[(ni+1)*m + mi] = 1; u6[(ni+1)*m + (mi+1)] = -1
        v6[mi*p + pi] = 1; v6[mi*p + (pi+1)] = 1
        w6[(pi+1)*n + (ni+1)] = 1 # C22
        add_term(u6, v6, w6)

        # M7: u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,0]
        u7 = [0]*(n*m); v7 = [0]*(m*p); w7 = [0]*(p*n)
        u7[ni*m + (mi+1)] = 1; u7[(ni+1)*m + (mi+1)] = -1
        v7[(mi+1)*p + pi] = 1; v7[(mi+1)*p + (pi+1)] = 1
        w7[pi*n + ni] = 1 # C11
        add_term(u7, v7, w7)

    # Iterate through all blocks
    for i, ni in enumerate(n_starts):
        n_loc = n_splits[i]
        for j, mj in enumerate(m_starts):
            m_loc = m_splits[j]
            for k, pk in enumerate(p_starts):
                p_loc = p_splits[k]
                
                if n_loc == 2 and m_loc == 2 and p_loc == 2:
                    add_strassen_block(ni, mj, pk)
                else:
                    add_schoolbook_block(ni, mj, pk, n_loc, m_loc, p_loc)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
