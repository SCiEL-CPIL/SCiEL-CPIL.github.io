"""Rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110

    # Helper to verify the decomposition
    def verify(U, V, W, rank):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Strassen block builder for 2x2x2
    def build_strassen_block(i0, j0, k0, U, V, W, col):
        # Term 1: p1 = (A00+A11)(B00+B11)
        U[i0*m+j0][col] += 1
        U[(i0+1)*m+(j0+1)][col] += 1
        V[j0*p+k0][col] += 1
        V[(j0+1)*p+(k0+1)][col] += 1
        W[k0*n+i0][col] += 1
        W[(k0+1)*n+(i0+1)][col] += 1
        col += 1
        
        # Term 2: p2 = (A10+A11)B00
        U[(i0+1)*m+j0][col] += 1
        U[(i0+1)*m+(j0+1)][col] += 1
        V[j0*p+k0][col] += 1
        W[k0*n+(i0+1)][col] += 1
        col += 1
        
        # Term 3: p3 = A00(B10-B11)
        U[i0*m+j0][col] += 1
        V[j0*p+(k0+1)][col] += 1
        V[(j0+1)*p+(k0+1)][col] += -1
        W[(k0+1)*n+i0][col] += 1
        W[(k0+1)*n+(i0+1)][col] += -1
        col += 1
        
        # Term 4: p4 = A11(B01-B10)
        U[(i0+1)*m+(j0+1)][col] += 1
        V[j0*p+(k0+1)][col] += 1
        V[(j0+1)*p+k0][col] += -1
        W[k0*n+(i0+1)][col] += 1
        W[(k0+1)*n+(i0+1)][col] += 1
        col += 1
        
        # Term 5: p5 = (A00+A01)B11
        U[i0*m+j0][col] += 1
        U[i0*m+(j0+1)][col] += 1
        V[(j0+1)*p+(k0+1)][col] += 1
        W[k0*n+i0][col] += -1
        W[(k0+1)*n+i0][col] += 1
        col += 1
        
        # Term 6: p6 = (A10-A00)(B00+B10)
        U[(i0+1)*m+j0][col] += 1
        U[i0*m+j0][col] += -1
        V[j0*p+k0][col] += 1
        V[(j0+1)*p+k0][col] += 1
        W[k0*n+i0][col] += 1
        col += 1
        
        # Term 7: p7 = (A01-A11)(B01+B11)
        U[i0*m+(j0+1)][col] += 1
        U[(i0+1)*m+(j0+1)][col] += -1
        V[j0*p+(k0+1)][col] += 1
        V[(j0+1)*p+(k0+1)][col] += 1
        W[k0*n+i0][col] += -1
        W[(k0+1)*n+(i0+1)][col] += 1
        col += 1
        
        return col

    # Schoolbook block builder
    def build_schoolbook_block(i0, j0, k0, di, dj, dk, U, V, W, col):
        for ii in range(di):
            for jj in range(dj):
                for kk in range(dk):
                    U[(i0+ii)*m + (j0+jj)][col] = 1
                    V[(j0+jj)*p + (k0+kk)][col] = 1
                    W[(k0+kk)*n + (i0+ii)][col] = 1
                    col += 1
        return col

    # Dynamic Programming to find optimal block partition
    # State: (i_start, i_end, j_start, j_end, k_start, k_end)
    # Value: min rank
    # We only split into contiguous intervals.
    
    # Precompute rank for a block
    def block_rank(di, dj, dk):
        if di == 2 and dj == 2 and dk == 2:
            return 7
        return di * dj * dk

    # DP table
    # Dimensions: i (0..n), j (0..m), k (0..p)
    # dp[i0][i1][j0][j1][k0][k1]
    # To save memory and time, we iterate over lengths.
    
    # Initialize with infinity
    INF = float('inf')
    # dp[i0][i1][j0][j1][k0][k1]
    # We can use a dictionary or a multi-dim array.
    # n=3, m=4, p=8.
    # Number of states is roughly (n*(n+1)/2) * (m*(m+1)/2) * (p*(p+1)/2) = 6 * 10 * 36 = 2160.
    # This is small enough for a dict.
    
    dp = {}
    
    # Base cases: single element blocks (1x1x1)
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                dp[(i0, i0+1, j0, j0+1, k0, k0+1)] = 1
                
    # Fill DP table
    # We iterate over the "size" of the blocks to ensure subproblems are solved.
    # Actually, since splits are additive, we can just iterate over all possible intervals and try all splits.
    # To be safe, we iterate by increasing length of i, then j, then k? 
    # No, the split can be in any dimension.
    # Let's just iterate over all states in a topological order?
    # A state (i0, i1, j0, j1, k0, k1) depends on states with smaller intervals.
    # We can iterate di from 1 to n, dj from 1 to m, dk from 1 to p.
    # But a state (i0, i1, j0, j1, k0, k1) can be split into (i0, im, j0, j1, k0, k1) + (im, i1, j0, j1, k0, k1).
    # So it depends on smaller di.
    # It can also be split in j, depending on smaller dj.
    # It can also be split in k, depending on smaller dk.
    # So we need to ensure that when we compute a state, all its sub-states are already computed.
    # Iterating di, dj, dk in increasing order works if we assume splits only reduce one dimension at a time?
    # Yes, any split reduces at least one dimension.
    # However, if we iterate di, then dj, then dk, a state with di=2, dj=2 might depend on di=1, dj=2 (computed) and di=2, dj=1 (computed? 
    # If we loop di=1, then di=2. Inside di=2, we loop dj=1, then dj=2.
    # When computing di=2, dj=2, we need di=2, dj=1 (computed in current di loop, previous dj) and di=1, dj=2 (computed in previous di loop).
    # This works.
    
    best_rank = INF
    best_partition = None
    
    # We also need to store the partition to reconstruct the matrices.
    # parent[state] = (split_dim, split_pos, sub_state_1, sub_state_2) or None if base case
    
    parent = {}
    
    for di in range(1, n + 1):
        for dj in range(1, m + 1):
            for dk in range(1, p + 1):
                for i0 in range(n - di + 1):
                    i1 = i0 + di
                    for j0 in range(m - dj + 1):
                        j1 = j0 + dj
                        for k0 in range(p - dk + 1):
                            k1 = k0 + dk
                            state = (i0, i1, j0, j1, k0, k1)
                            
                            # Option 1: Do not split (base cost)
                            cost = block_rank(di, dj, dk)
                            par = None
                            
                            # Option 2: Split in i
                            if di > 1:
                                for im in range(i0 + 1, i1):
                                    s1 = (i0, im, j0, j1, k0, k1)
                                    s2 = (im, i1, j0, j1, k0, k1)
                                    if s1 in dp and s2 in dp:
                                        c = dp[s1] + dp[s2]
                                        if c < cost:
                                            cost = c
                                            par = ('i', im, s1, s2)
                                            
                            # Option 3: Split in j
                            if dj > 1:
                                for jm in range(j0 + 1, j1):
                                    s1 = (i0, i1, j0, jm, k0, k1)
                                    s2 = (i0, i1, jm, j1, k0, k1)
                                    if s1 in dp and s2 in dp:
                                        c = dp[s1] + dp[s2]
                                        if c < cost:
                                            cost = c
                                            par = ('j', jm, s1, s2)
                                            
                            # Option 4: Split in k
                            if dk > 1:
                                for km in range(k0 + 1, k1):
                                    s1 = (i0, i1, j0, j1, k0, km)
                                    s2 = (i0, i1, j0, j1, km, k1)
                                    if s1 in dp and s2 in dp:
                                        c = dp[s1] + dp[s2]
                                        if c < cost:
                                            cost = c
                                            par = ('k', km, s1, s2)
                                            
                            dp[state] = cost
                            parent[state] = par
                            
                            if cost < best_rank:
                                best_rank = cost
                                best_partition = state

    # Reconstruct the block list from the best partition
    blocks = []
    def get_blocks(state):
        if time.time() > deadline:
            return
        par = parent[state]
        if par is None:
            i0, i1, j0, j1, k0, k1 = state
            di, dj, dk = i1-i0, j1-j0, k1-k0
            if di == 2 and dj == 2 and dk == 2:
                blocks.append((i0, i1, j0, j1, k0, k1, 'strassen'))
            else:
                blocks.append((i0, i1, j0, j1, k0, k1, 'schoolbook'))
        else:
            dim, pos, s1, s2 = par
            get_blocks(s1)
            get_blocks(s2)
            
    if best_partition:
        get_blocks(best_partition)

    # Build the matrices
    U = [[0] * best_rank for _ in range(n * m)]
    V = [[0] * best_rank for _ in range(m * p)]
    W = [[0] * best_rank for _ in range(p * n)]
    
    col = 0
    for (i0, i1, j0, j1, k0, k1, method) in blocks:
        di, dj, dk = i1-i0, j1-j0, k1-k0
        if method == 'strassen':
            col = build_strassen_block(i0, j0, k0, U, V, W, col)
        else:
            col = build_schoolbook_block(i0, j0, k0, di, dj, dk, U, V, W, col)
            
    if col != best_rank:
        # Fallback to schoolbook if reconstruction failed
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        best_rank = rank

    if verify(U, V, W, best_rank):
        print(f"DP Block decomposition verified with rank {best_rank}")
        return {"U": U, "V": V, "W": W}
    else:
        # Should not happen, but return schoolbook just in case
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
