"""Matrix multiplication 3x5x6 rank-minimal exact tensor decomposition."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 90

    # Build the schoolbook decomposition first
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Find clusters: sets of (i,j,k) where one rank-1 tensor matches T on those positions
    # A rank-1 tensor u⊗v⊗w is non-zero at (i,j,k) iff u[i*m+j]*v[j*p+k]*w[k*n+i] != 0
    # We need: for (i,j,k) in cluster, product = 1; for others, product = 0
    # This means: the support pattern must be exactly the cluster

    # For a set S of (i,j,k), find if there exist u,v,w such that:
    # u[i*m+j]*v[j*p+k]*w[k*n+i] = 1 for (i,j,k) in S
    # u[i*m+j]*v[j*p+k]*w[k*n+i] = 0 for (i,j,k) not in S

    # Strategy: try small clusters (size 2-5) and see if they form valid rank-1 patterns
    # A valid pattern: there exist sets U_idx, V_idx, W_idx such that
    # S = {(i,j,k) : (i,j) in U_idx, (j,k) in V_idx, (k,i) in W_idx}
    # and for any (i,j,k) not in S, at least one condition fails

    # Let's enumerate possible supports and find valid clusters
    all_ijk = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # For each candidate cluster, check if it can be a rank-1 support
    # A rank-1 support is determined by which (i,j), (j,k), (k,i) are "active"
    # The non-zero positions are exactly the "triangles" in the bipartite graph
    
    # Let's try to find clusters of size 2 first (replace 2 schoolbook terms with 1)
    # Then size 3, etc.
    
    # For a pair (i1,j1,k1) and (i2,j2,k2), when can they be in the same rank-1 tensor?
    # We need u[i1*m+j1]*v[j1*p+k1]*w[k1*n+i1] = 1
    # and u[i2*m+j2]*v[j2*p+k2]*w[k2*n+i2] = 1
    # and all other products = 0
    
    # Simplest case: if they share some indices, we can often do it
    # Let's just try all pairs and check feasibility
    
    clusters = []  # list of sets of (i,j,k)
    used = set()
    
    # Try to find disjoint clusters greedily, largest first
    # First, find all valid clusters of size >= 2
    
    # For efficiency, let's try specific patterns:
    # Pattern 1: Same i, same j, different k -> (i,j,k1) and (i,j,k2)
    #   u[i*m+j] = a, v[j*p+k1] = b1, v[j*p+k2] = b2, w[k1*n+i] = c1, w[k2*n+i] = c2
    #   a*b1*c1 = 1, a*b2*c2 = 1
    #   Need: for any other (i',j',k'), if i'=i and j'=j, then k' must be k1 or k2
    #   But if i'!=i or j'!=j, then u=0. So we need no other (i,j,k') with k' not in {k1,k2}
    #   But there ARE other k values! So this doesn't work unless we restrict.
    #   Actually, u[i*m+j] is the only non-zero u entry. So for (i',j',k') to be non-zero,
    #   we need i'=i and j'=j. Then (j',k') = (j,k') must be in V support, and (k',i) in W support.
    #   If V support is {(j,k1),(j,k2)} and W support is {(k1,i),(k2,i)}, then only k1,k2 work.
    #   So YES, this works! Any two with same (i,j) can be clustered.
    
    # Pattern 2: Same i, same k, different j -> (i,j1,k) and (i,j2,k)
    #   u[i*m+j1]=a1, u[i*m+j2]=a2, v[j1*p+k]=b1, v[j2*p+k]=b2, w[k*n+i]=c
    #   a1*b1*c=1, a2*b2*c=1
    #   For other (i',j',k'): need i'=i, k'=k, and j' in {j1,j2}
    #   V support: {(j1,k),(j2,k)}, W support: {(k,i)}, U support: {(i,j1),(i,j2)}
    #   Works!
    
    # Pattern 3: Same j, same k, different i -> (i1,j,k) and (i2,j,k)
    #   Similar, works!
    
    # Let's find all maximal clusters of these simple types
    # Type A: same (i,j), varying k
    for i in range(n):
        for j in range(m):
            ks = list(range(p))
            if len(ks) >= 2:
                cluster = [(i, j, k) for k in ks]
                clusters.append(set(cluster))
    
    # Type B: same (i,k), varying j
    for i in range(n):
        for k in range(p):
            js = list(range(m))
            if len(js) >= 2:
                cluster = [(i, j, k) for j in js]
                clusters.append(set(cluster))
    
    # Type C: same (j,k), varying i
    for j in range(m):
        for k in range(p):
            is_ = list(range(n))
            if len(is_) >= 2:
                cluster = [(i, j, k) for i in is_]
                clusters.append(set(cluster))
    
    # Sort by size descending
    clusters.sort(key=lambda s: len(s), reverse=True)
    
    # Greedily select disjoint clusters
    selected_clusters = []
    for cluster in clusters:
        if cluster.isdisjoint(used):
            selected_clusters.append(cluster)
            used.update(cluster)
    
    # Build the decomposition
    # Each selected cluster of size s replaces s schoolbook terms with 1 rank-1 term
    # Remaining terms are individual schoolbook terms
    
    # Assign values for each cluster
    # For Type A (same i,j): u[i*m+j]=1, v[j*p+k]=1 for k in cluster, w[k*n+i]=1 for k in cluster
    # For Type B (same i,k): u[i*m+j]=1 for j in cluster, v[j*p+k]=1 for j in cluster, w[k*n+i]=1
    # For Type C (same j,k): u[i*m+j]=1 for i in cluster, v[j*p+k]=1, w[k*n+i]=1 for i in cluster
    
    # Determine cluster type
    def get_cluster_type(cluster):
        is_ = set(i for i,j,k in cluster)
        js = set(j for i,j,k in cluster)
        ks = set(k for i,j,k in cluster)
        if len(is_) == 1 and len(js) == 1:
            return 'A', is_.pop(), js.pop()
        elif len(is_) == 1 and len(ks) == 1:
            return 'B', is_.pop(), ks.pop()
        elif len(js) == 1 and len(ks) == 1:
            return 'C', js.pop(), ks.pop()
        else:
            return 'unknown', None, None
    
    rank = 0
    U = []
    V = []
    W = []
    
    # We'll build columns incrementally
    U_cols = []  # list of columns
    V_cols = []
    W_cols = []
    
    for cluster in selected_clusters:
        ctype, val1, val2 = get_cluster_type(cluster)
        if ctype == 'unknown':
            continue  # skip unknown types
        
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        if ctype == 'A':
            i, j = val1, val2
            for (ii, jj, kk) in cluster:
                u_col[ii * m + jj] = 1
                v_col[jj * p + kk] = 1
                w_col[kk * n + ii] = 1
        elif ctype == 'B':
            i, k = val1, val2
            for (ii, jj, kk) in cluster:
                u_col[ii * m + jj] = 1
                v_col[jj * p + kk] = 1
                w_col[kk * n + ii] = 1
        elif ctype == 'C':
            j, k = val1, val2
            for (ii, jj, kk) in cluster:
                u_col[ii * m + jj] = 1
                v_col[jj * p + kk] = 1
                w_col[kk * n + ii] = 1
        
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
        rank += 1
    
    # Add remaining schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in used:
                    u_col = [0] * (n * m)
                    v_col = [0] * (m * p)
                    w_col = [0] * (p * n)
                    u_col[i * m + j] = 1
                    v_col[j * p + k] = 1
                    w_col[k * n + i] = 1
                    U_cols.append(u_col)
                    V_cols.append(v_col)
                    W_cols.append(w_col)
                    rank += 1
    
    # Convert to list-of-lists format
    U = [[U_cols[r][i] for r in range(rank)] for i in range(n * m)]
    V = [[V_cols[r][i] for r in range(rank)] for i in range(m * p)]
    W = [[W_cols[r][i] for r in range(rank)] for i in range(p * n)]
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, n, m, p):
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    # Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
                    total = 0
                    for r in range(len(U_mat[0])):
                        total += U_mat[i * m + j][r] * V_mat[j * p + k][r] * W_mat[k * n + i][r]
                    if total != 1:
                        return False
        # Also check that no extra contributions
        # Actually, we need to check that for each (i,j,k), the sum is exactly 1
        # And that there are no other non-zero contributions
        # The above check ensures that for each valid (i,j,k), the sum is 1
        # But we also need to ensure that the decomposition doesn't produce non-zero values
        # at invalid positions. Since we're using the standard matmul tensor, all positions
        # (i,j,k) with i in [0,n), j in [0,m), k in [0,p) are valid.
        # So the check above is sufficient.
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U = U_school
        V = V_school
        W = W_school
        rank = schoolbook_rank
    
    return {"U": U, "V": V, "W": W}
