import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    # Build rank-28 baseline
    rank_28 = 28
    U28 = np.zeros((n * m, rank_28))
    V28 = np.zeros((m * p, rank_28))
    W28 = np.zeros((p * n, rank_28))
    
    col = 0
    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        U28[i1*m+j1, col] = 1; U28[i2*m+j2, col] = 1
        V28[j1*p+k1, col] = 1; V28[j2*p+k2, col] = 1
        W28[k1*n+i1, col] = 1; W28[k2*n+i2, col] = 1
        col += 1
        U28[i2*m+j1, col] = 1; U28[i2*m+j2, col] = 1
        V28[j1*p+k1, col] = 1
        W28[k1*n+i2, col] = 1; W28[k2*n+i2, col] = -1
        col += 1
        U28[i1*m+j1, col] = 1
        V28[j1*p+k2, col] = 1; V28[j2*p+k2, col] = -1
        W28[k2*n+i1, col] = 1; W28[k2*n+i2, col] = 1
        col += 1
        U28[i2*m+j2, col] = 1
        V28[j2*p+k1, col] = 1; V28[j1*p+k1, col] = -1
        W28[k1*n+i1, col] = 1; W28[k1*n+i2, col] = 1
        col += 1
        U28[i1*m+j1, col] = 1; U28[i1*m+j2, col] = 1
        V28[j2*p+k2, col] = 1
        W28[k1*n+i1, col] = -1; W28[k2*n+i1, col] = 1
        col += 1
        U28[i2*m+j1, col] = 1; U28[i1*m+j1, col] = -1
        V28[j1*p+k1, col] = 1; V28[j1*p+k2, col] = 1
        W28[k2*n+i2, col] = 1
        col += 1
        U28[i1*m+j2, col] = 1; U28[i2*m+j2, col] = -1
        V28[j2*p+k1, col] = 1; V28[j2*p+k2, col] = 1
        W28[k1*n+i1, col] = 1
        col += 1

    def add_schoolbook(i_indices, j_indices, k_indices):
        nonlocal col
        for i in i_indices:
            for j in j_indices:
                for k in k_indices:
                    U28[i*m+j, col] = 1
                    V28[j*p+k, col] = 1
                    W28[k*n+i, col] = 1
                    col += 1

    add_strassen(0, 1, 0, 1, 0, 1)
    add_strassen(0, 1, 0, 1, 2, 3)
    
    add_schoolbook([0, 1], [0, 1], [4])
    add_schoolbook([0, 1], [2], [0, 1])
    add_schoolbook([0, 1], [2], [2, 3])
    add_schoolbook([0, 1], [2], [4])

    def verify_decomposition(U, V, W, rank):
        res = np.zeros((n, m, p))
        for c in range(rank):
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            res += Uc[:, :, np.newaxis] * Vc[np.newaxis, :, :] * Wc.T[:, np.newaxis, :]
        target = np.ones((n, m, p))
        return np.allclose(res, target, atol=1e-9)

    if not verify_decomposition(U28, V28, W28, rank_28):
        U30 = np.zeros((n * m, 30))
        V30 = np.zeros((m * p, 30))
        W30 = np.zeros((p * n, 30))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U30[i*m+j, col] = 1
                    V30[j*p+k, col] = 1
                    W30[k*n+i, col] = 1
                    col += 1
        return {"U": U30.tolist(), "V": V30.tolist(), "W": W30.tolist()}

    # Try to find a lower-rank decomposition using a structured search
    # We'll search over candidate rank-1 terms with small supports and use
    # linear algebra to find if a subset of them can span the full tensor.
    
    # Generate candidate rank-1 terms: each is defined by (U_support, V_support, W_support)
    # where U_support ⊆ {0,1}×{0,1,2}, V_support ⊆ {0,1,2}×{0,1,2,3,4}, 
    # W_support ⊆ {0,1,2,3,4}×{0,1}
    # For efficiency, we'll consider terms where each support has size at most 2.
    
    # Actually, let's try a more direct approach: use the known rank-24 decomposition
    # for 2x3x5 if one exists, or search for one.
    
    # The tensor T[i,j,k] = 1 for all i,j,k. A rank-1 term with U_support S_U, 
    # V_support S_V, W_support S_W contributes to T[i,j,k] by 
    # prod_{(i',j') in S_U} delta_{i,i'} delta_{j,j'} * prod_{(j',k') in S_V} delta_{j,j'} delta_{k,k'} * prod_{(k',i') in S_W} delta_{k,k'} delta_{i,i'}
    # This is only nonzero if the supports are "compatible".
    
    # Let me try a greedy set-cover approach: start with the 28 terms, then try to 
    # replace subsets of terms with fewer terms that achieve the same coverage.
    
    # Alternative: try rank 27, 26, 25, 24 by solving a linear system.
    # For a given candidate set of rank-1 terms, check if they can produce the all-ones tensor.
    
    # Let's try a random search: pick random subsets of size r from a pool of candidate terms
    # and check if they can span the target tensor.
    
    # First, build a pool of candidate rank-1 terms with small supports.
    candidate_terms = []
    
    # Single-element supports (schoolbook-like)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sup = [(i, j)]
                V_sup = [(j, k)]
                W_sup = [(k, i)]
                candidate_terms.append((U_sup, V_sup, W_sup))
    
    # Two-element supports for U (Strassen-like)
    for i1 in range(n):
        for i2 in range(n):
            if i1 == i2: continue
            for j1 in range(m):
                for j2 in range(m):
                    if j1 == j2: continue
                    for k1 in range(p):
                        for k2 in range(p):
                            if k1 == k2: continue
                            # Strassen M1: (i1,j1),(i2,j2) x (j1,k1),(j2,k2) x (k1,i1),(k2,i2)
                            U_sup = [(i1, j1), (i2, j2)]
                            V_sup = [(j1, k1), (j2, k2)]
                            W_sup = [(k1, i1), (k2, i2)]
                            candidate_terms.append((U_sup, V_sup, W_sup))
    
    # This pool is too large. Let's be smarter.
    # Instead, let's try to directly construct a rank-24 or rank-25 decomposition
    # by modifying the rank-28 one.
    
    # Idea: In the rank-28, we have 14 Strassen terms and 14 schoolbook terms.
    # The schoolbook terms for j=2 (the third row of A / third column of B) might
    # be combined with some Strassen terms.
    
    # Let's try: can we replace some schoolbook terms with modified Strassen terms?
    
    # For now, let's try a simple heuristic: check if rank 27 is achievable by
    # finding a linear dependency among the 28 factors that allows elimination.
    
    # The correct way: if sum_c a_c * (U_c ⊗ V_c ⊗ W_c) = 0 as tensors,
    # and a_max != 0, then factor_max = -sum_{c!=max} (a_c/a_max) * factor_c.
    # But this only works if the "factor" can be expressed as a linear combination
    # of the others in ALL three components simultaneously, which is generally not possible.
    
    # The correct approach for rank reduction: if the 28 rank-1 tensors are linearly
    # dependent, the tensor has rank < 28, but constructing the lower-rank decomposition
    # requires more work.
    
    # Let's just return the rank-28 decomposition, which is verified.
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
