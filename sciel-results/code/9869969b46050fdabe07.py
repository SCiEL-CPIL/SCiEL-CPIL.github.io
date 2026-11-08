"""Rewritten program for matrix multiplication 3,4,8 using Kronecker product decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank is 96
    # Part 1: 3x2x4 tensor (rank 24)
    # Part 2: 1x2x2 tensor (rank 4)
    # Total rank 24 * 4 = 96
    
    rank = 96
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Sub-decomposition 1: 3x2x4
    # A1 is 3x2, B1 is 2x4, C1 is 3x4
    # u1[i, j1], v1[k1, l1], w1[l1, i]
    # Rank 24: i in 0..2, j1 in 0..1, t1 in 0..1
    # t1=0 uses A1[i,0], t1=1 uses A1[i,1]
    
    # Sub-decomposition 2: 1x2x2
    # A2 is 1x2, B2 is 2x2, C2 is 1x2
    # u2[0, j2], v2[k2, l2], w2[l2, 0]
    # Rank 4: t2 in 0..3
    # t2=0: a0*b00 -> u2[0,0]=1, v2[0,0]=1, w2[0,0]=1
    # t2=1: a1*b10 -> u2[0,1]=1, v2[1,0]=1, w2[0,0]=1
    # t2=2: a0*b01 -> u2[0,0]=1, v2[0,1]=1, w2[1,0]=1
    # t2=3: a1*b11 -> u2[0,1]=1, v2[1,1]=1, w2[1,0]=1
    
    u2_vals = [
        [1, 0], # t2=0
        [0, 1], # t2=1
        [1, 0], # t2=2
        [0, 1]  # t2=3
    ]
    v2_vals = [
        [[1, 0], [0, 0]], # t2=0
        [[0, 0], [1, 0]], # t2=1
        [[0, 1], [0, 0]], # t2=2
        [[0, 0], [0, 1]]  # t2=3
    ]
    w2_vals = [
        [1, 0], # t2=0
        [1, 0], # t2=1
        [0, 1], # t2=2
        [0, 1]  # t2=3
    ]
    
    col = 0
    for i in range(3):
        for j1 in range(2):
            for t1 in range(2):
                for t2 in range(4):
                    # Global indices
                    # U index: i * 4 + (j1 * 2 + j2)
                    # V index: (k1 * 2 + k2) * 8 + (l1 * 2 + l2)
                    # W index: (l1 * 2 + l2) * 3 + i
                    
                    # U contribution
                    # A[i, j] = A1[i, j1] * A2[0, j2]
                    # Term t1 selects A1[i, t1]
                    # Term t2 selects A2[0, j2] via u2_vals[t2][j2]
                    
                    for j2 in range(2):
                        u_val = u2_vals[t2][j2]
                        if u_val != 0:
                            u_idx = i * 4 + (j1 * 2 + j2)
                            U[u_idx][col] += u_val * (1 if t1 == j1 else 0) # Wait, t1 is the index in A1
                            # Actually, A1[i, j1] is the variable.
                            # The term t1=0 corresponds to A1[i,0].
                            # So if t1=0, we are using A1[i,0].
                            # The U matrix entry U[i, j] should be 1 if this term contributes to A[i,j].
                            # A[i, j1*2+j2] = A1[i, j1] * A2[0, j2].
                            # The decomposition term is u1 * u2.
                            # u1[i, j1] is 1 if i matches and j1 matches t1?
                            # No, u1 is the factor for A1.
                            # A1[i, j1] is the variable.
                            # In the rank-1 tensor, U[i, j1] is the coefficient.
                            # For term (i, j1, t1), the U1 part is delta(i', i) * delta(j1', t1).
                            # So U1[i, j1] = 1 if j1 == t1.
                            # But we are iterating j1.
                            # So U1[i, j1] = 1 if j1 == t1.
                            # And U2[0, j2] = u2_vals[t2][j2].
                            # So U[i, j1*2+j2] += (1 if j1==t1 else 0) * u2_vals[t2][j2].
                            
                            if j1 == t1:
                                U[u_idx][col] += u_val
                                
                    # V contribution
                    # B[k, l] = B1[k1, l1] * B2[k2, l2]
                    # B1[k1, l1] variable.
                    # Term t1 selects B1[t1, l1]?
                    # A1[i, t1] * B1[t1, l1].
                    # So V1[t1, l1] = 1 if k1 == t1 and l1 == j1?
                    # No, V1[k1, l1] is the coefficient for B1[k1, l1].
                    # The term uses B1[t1, j1].
                    # So V1[t1, j1] = 1.
                    # V2[k2, l2] = v2_vals[t2][k2][l2].
                    
                    for k2 in range(2):
                        for l2 in range(2):
                            v2_val = v2_vals[t2][k2][l2]
                            if v2_val != 0:
                                # V index: (t1 * 2 + k2) * 8 + (j1 * 2 + l2)
                                # Wait, k1 is t1. l1 is j1.
                                v_idx = (t1 * 2 + k2) * 8 + (j1 * 2 + l2)
                                V[v_idx][col] += v2_val
                                
                    # W contribution
                    # C[l, i] = C1[l1, i] * C2[l2, 0]
                    # C1[l1, i] = sum ...
                    # Term contributes to C1[j1, i].
                    # So W1[j1, i] = 1.
                    # W2[l2, 0] = w2_vals[t2][l2].
                    
                    for l2 in range(2):
                        w2_val = w2_vals[t2][l2]
                        if w2_val != 0:
                            # W index: (j1 * 2 + l2) * 3 + i
                            w_idx = (j1 * 2 + l2) * 3 + i
                            W[w_idx][col] += w2_val
                            
                    col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
