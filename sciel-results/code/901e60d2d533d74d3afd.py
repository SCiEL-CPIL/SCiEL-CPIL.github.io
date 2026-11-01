import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 28
    pn = p * n  # 14

    start_time = time.time()
    deadline = start_time + 115.0

    def schoolbook():
        rank = n * m * p  # 56
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        return U, V, W

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        # For each (i, k, j, j'), check sum_c U[i*m+j,c]*V[j'*p+k,c]*W[k*n+i,c] == delta[j,j']
        for i in range(n):
            for k in range(p):
                w_row = W[k * n + i, :]  # shape (r,)
                for j in range(m):
                    u_row = U[i * m + j, :]  # shape (r,)
                    for jp in range(m):
                        v_row = V[jp * p + k, :]  # shape (r,)
                        s = np.dot(u_row * v_row, w_row)
                        expected = 1.0 if j == jp else 0.0
                        if abs(s - expected) > tol:
                            return False
        return True

    def time_left():
        return deadline - time.time()

    # Block decomposition strategy:
    # Split A (2x4) into blocks, B (4x7) into blocks.
    # A = [A0 | A1] where A0 is 2x2, A1 is 2x2
    # B = [B0; B1] where B0 is 2x7, B1 is 2x7
    # C = A0*B0 + A1*B1
    #
    # For 2x2 * 2x7: rank can be reduced using Strassen-like tricks.
    # Standard 2x2*2x7 schoolbook: 2*2*7 = 28
    # 
    # Let's think about this differently.
    # C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
    #
    # Group by j:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    #
    # For each i (i=0,1), we need to compute a linear combination of 4 rows of B.
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    #
    # This is like computing two linear combinations of 4 vectors (rows of B).
    # Each linear combination can be done with rank-1 terms.
    #
    # Key insight: We can use the fact that we're computing 2 outputs (i=0,1) for each k.
    # For a fixed k, we need:
    # C[0,k] = dot(A[0,:], B[:,k])
    # C[1,k] = dot(A[1,:], B[:,k])
    #
    # This is a 2x4 * 4x1 matrix multiplication, which has rank 4 (schoolbook).
    # But we can potentially do better by sharing computation across k values.
    #
    # Actually, let's think of it as: for each k, we compute a 2-vector from a 4-vector.
    # The mapping is: B[:,k] -> [dot(A[0,:],B[:,k]), dot(A[1,:],B[:,k])]
    # This is a linear map from R^4 to R^2, represented by the 2x4 matrix A.
    #
    # The tensor rank for computing all 7 outputs (k=0..6) simultaneously...
    #
    # Let me try a different approach: use the structure more directly.
    #
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # Let's define for each c (rank-1 term):
    # U[i*m+j, c], V[j'*p+k, c], W[k*n+i, c]
    #
    # We need: sum_c U[i*m+j, c] V[j'*p+k, c] W[k*n+i, c] = delta[j,j']
    #
    # Let's try to construct a solution with rank < 56.
    #
    # Strategy: Use the fact that for each (i,k), we need the 4x4 identity in (j,j').
    # We can try to find a set of rank-1 tensors that sum to give identity for each (i,k).
    #
    # Let's try a structured approach:
    # For each pair (i,k), we need sum_c W[k*n+i,c] * outer(U_i,c, V_k,c) = I_4
    # where U_i,c[j] = U[i*m+j,c] and V_k,c[j'] = V[j'*p+k,c]
    #
    # So for each (i,k): sum_c W[k*n+i,c] * U_i,c * V_k,c^T = I_4
    #
    # This means: for each (i,k), the weighted sum of outer products must be I_4.
    #
    # If we set W[k*n+i,c] = 1 for all i,k,c, then we need:
    # sum_c U_i,c * V_k,c^T = I_4 for all i,k
    #
    # This is a very strong constraint. Let's see if we can satisfy it.
    #
    # For i=0: sum_c U_0,c * V_k,c^T = I_4
    # For i=1: sum_c U_1,c * V_k,c^T = I_4
    #
    # So U_0,c * V_k,c^T and U_1,c * V_k,c^T must both sum to I_4.
    #
    # This implies: sum_c (U_0,c - U_1,c) * V_k,c^T = 0 for all k.
    #
    # One simple solution: U_0,c = U_1,c for all c. Then we just need:
    # sum_c U_0,c * V_k,c^T = I_4 for all k.
    #
    # Let U_0,c = u_c (a 4-vector) and V_k,c = v_{k,c} (a 4-vector).
    # We need: sum_c u_c * v_{k,c}^T = I_4 for all k=0..6.
    #
    # This means: for each k, the matrix sum_c u_c v_{k,c}^T must be I_4.
    #
    # If we set v_{k,c} = e_c (standard basis) for c=0..3 and 0 otherwise,
    # and u_c = e_c, then sum_c u_c v_{k,c}^T = sum_{c=0}^3 e_c e_c^T = I_4.
    # This gives rank 4 for each (i,k), but we need it for all i,k.
    #
    # Wait, this doesn't work directly because the same c must work for all k.
    #
    # Let me reconsider. If W[k*n+i,c] can vary with k, we have more freedom.
    #
    # Let's try: W[k*n+i,c] depends on k.
    # For fixed i, we need: sum_c W[k*n+i,c] * U_i,c * V_k,c^T = I_4 for all k.
    #
    # This is like: for each k, a weighted sum of rank-1 matrices equals I_4.
    #
    # Let's try a specific construction:
    # Use rank 28: for each (i,k), use 4 terms with W[k*n+i,c]=1.
    # That's just schoolbook.
    #
    # To reduce rank, we need to share terms across different (i,k) pairs.
    #
    # Let's try: for each j (0..3), use p terms (one for each k).
    # U[i*m+j, c] = 1 if c corresponds to (j,k) for some k, else 0.
    # V[j'*p+k, c] = 1 if j'=j and c corresponds to (j,k), else 0.
    # W[k'*n+i, c] = 1 if k'=k and c corresponds to (j,k), else 0.
    #
    # This gives rank 4*7 = 28... wait, that's already less than 56!
    # Let me check: for each (j,k) pair, we have one term.
    # U[i*m+j, c_{j,k}] = 1 for all i (since A[i,j] is used in all i).
    # V[j'*p+k, c_{j,k}] = 1 if j'=j, else 0.
    # W[k'*n+i, c_{j,k}] = 1 if k'=k, else 0.
    #
    # Check: sum_c U[i*m+j,c] V[j'*p+k,c] W[k'*n+i,c]
    # = sum_{j',k'} U[i*m+j, c_{j',k'}] V[j'*p+k, c_{j',k'}] W[k'*n+i, c_{j',k'}]
    # = sum_{j',k'} [1 if j'=j] * [1 if j'=j and k'=k] * [1 if k'=k]
    # Wait, let me be more careful.
    #
    # U[i*m+j, c_{j',k'}] = 1 for all i (since it depends only on j, not i).
    # V[j''*p+k, c_{j',k'}] = 1 if j''=j' and k=k', else 0.
    # W[k''*n+i, c_{j',k'}] = 1 if k''=k', else 0.
    #
    # So the sum becomes:
    # sum_{j',k'} 1 * [j''=j' and k=k'] * [k''=k']
    # where j''=j, k''=k (fixed).
    # = sum_{j',k'} [j=j' and k=k'] * [k=k']
    # = sum_{j',k'} [j=j'] * [k=k'] * [k=k']
    # = [j=j'] * [k=k']  (since k=k' implies k=k')
    #
    # Hmm, that gives [j=j']*[k=k'], but we need [j=j'] for all k.
    #
    # Let me re-examine. The constraint is:
    # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] = delta[j,j']
    #
    # Note: k appears in V (as the column index) and W (as the row index).
    #
    # With my construction:
    # U[i*m+j, c_{j',k'}] = 1 (for all i)
    # V[j'*p+k, c_{j',k'}] = 1 if j'=j' and k=k'
    # W[k*n+i, c_{j',k'}] = 1 if k=k'
    #
    # Sum over c = sum over (j',k'):
    # = sum_{j',k'} 1 * [k=k'] * [k=k']
    # = sum_{j',k'} [k=k']
    # = 4 (since there are 4 values of j')
    #
    # That's not right. Let me reconsider.
    #
    # Actually, I think the issue is that U should depend on i as well, or my indexing is off.
    #
    # Let me go back to basics. The schoolbook solution has rank 56:
    # For each (i,j,k): U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
    #
    # To reduce rank, I need to find a smaller set of (U,V,W) triples.
    #
    # Let's try using the fact that n=2 is small.
    #
    # For i=0 and i=1, we have the same structure. Can we share terms?
    #
    # Let's try: for each (j,k), use one term that works for both i=0 and i=1.
    # U[0*m+j, c_{j,k}] = a_{j,k}
    # U[1*m+j, c_{j,k}] = b_{j,k}
    # V[j*p+k, c_{j,k}] = 1
    # V[j'*p+k, c_{j,k}] = 0 for j' != j
    # W[k*n+0, c_{j,k}] = w0_{j,k}
    # W[k*n+1, c_{j,k}] = w1_{j,k}
    #
    # Check for (i=0, j, j', k):
    # sum_{j',k'} U[0*m+j, c_{j',k'}] V[j'*p+k, c_{j',k'}] W[k*n+0, c_{j',k'}]
    # = sum_{j',k'} a_{j,k'} * [j'=j] * [k'=k] * w0_{j',k'}
    # = a_{j,k} * w0_{j,k}
    # We need this to be delta[j,j'].
    # For j=j': a_{j,k} * w0_{j,k} = 1
    # For j!=j': a_{j,k} * w0_{j,k} = 0 (but j!=j' means the term is 0 since V[j'*p+k, c_{j,k}]=0)
    #
    # Wait, I'm confusing myself. Let me redo this carefully.
    #
    # The sum is over c, and c indexes our rank-1 terms. Let c = (j_c, k_c) for j_c in 0..3, k_c in 0..6.
    #
    # U[i*m+j, (j_c,k_c)] = alpha_{i,j,j_c,k_c}
    # V[j'*p+k, (j_c,k_c)] = beta_{j',k,j_c,k_c}
    # W[k*n+i, (j_c,k_c)] = gamma_{k,i,j_c,k_c}
    #
    # We need: sum_{j_c,k_c} alpha_{i,j,j_c,k_c} * beta_{j',k,j_c,k_c} * gamma_{k,i,j_c,k_c} = delta[j,j']
    #
    # If we set beta_{j',k,j_c,k_c} = delta[j',j_c] * delta[k,k_c], then:
    # sum_{j_c,k_c} alpha_{i,j,j_c,k_c} * delta[j',j_c] * delta[k,k_c] * gamma_{k,i,j_c,k_c}
    # = alpha_{i,j,j',k} * gamma_{k,i,j',k}
    #
    # We need: alpha_{i,j,j',k} * gamma_{k,i,j',k} = delta[j,j']
    #
    # For j=j': alpha_{i,j,j,k} * gamma_{k,i,j,k} = 1
    # For j!=j': alpha_{i,j,j',k} * gamma_{k,i,j',k} = 0
    #
    # One solution: alpha_{i,j,j',k} = delta[j,j'] and gamma_{k,i,j',k} = 1.
    # This gives rank 4*7 = 28, with:
    # U[i*m+j, (j_c,k_c)] = delta[j, j_c]
    # V[j'*p+k, (j_c,k_c)] = delta[j', j_c] * delta[k, k_c]
    # W[k*n+i, (j_c,k_c)] = 1
    #
    # Let's verify: sum_{j_c,k_c} delta[j,j_c] * delta[j',j_c]*delta[k,k_c] * 1
    # = delta[j,j'] * delta[k,k]  (summing over j_c=k_c... wait)
    # = sum_{j_c} delta[j,j_c] delta[j',j_c] * sum_{k_c} delta[k,k_c]
    # = delta[j,j'] * 1
    # = delta[j,j']  ✓
    #
    # So rank 28 works! Let me implement this.

    def rank28_solution():
        rank = m * p  # 28
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        idx = 0
        for j_c in range(m):
            for k_c in range(p):
                for i in range(n):
                    U[i * m + j_c, idx] = 1.0
                for jp in range(m):
                    if jp == j_c:
                        for k in range(p):
                            if k == k_c:
                                V[jp * p + k, idx] = 1.0
                for k in range(p):
                    for i in range(n):
                        W[k * n + i, idx] = 1.0
                idx += 1
        return U, V, W

    # Verify rank 28
    U28, V28, W28 = rank28_solution()
    if verify(U28, V28, W28):
        best_U, best_V, best_W = U28, V28, W28
        best_rank = m * p
    else:
        best_U, best_V, best_W = schoolbook()
        best_rank = n * m * p

    # Try to reduce further
    # With the rank-28 structure, can we merge terms?
    # The constraint was: alpha_{i,j,j',k} * gamma_{k,i,j',k} = delta[j,j']
    # We used alpha=delta, gamma=1.
    # Alternative: alpha=1, gamma=delta[j,j']... but gamma depends on j'.
    #
    # Actually, let's try: for each (j,k), instead of one term, use fewer terms
    # by exploiting the fact that n=2.
    #
    # For fixed (j,k), we need terms that contribute to (i=0,j,j',k) and (i=1,j,j',k).
    # The constraint is: for each (i,j,j',k), sum_c ... = delta[j,j']
    #
    # Let's try: for each j (0..3), use p terms that work for both i values.
    # U[0*m+j, c] = 1, U[1*m+j, c] = 1 (same for both i)
    # V[j'*p+k, c] = delta[j',j] * delta[k,k_c]
    # W[k*n+0, c] = w0, W[k*n+1, c] = w1
    #
    # Then for (i,j,j',k):
    # sum_c U[i*m+j,c] V[j'*p+k,c] W[k*n+i,c]
    # = sum_c 1 * delta[j',j]*delta[k,k_c] * W[k*n+i,c]
    # = delta[j',j] * W[k*n+i,c_{j,k}]
    #
    # We need: delta[j',j] * W[k*n+i,c_{j,k}] = delta[j,j']
    # For j=j': W[k*n+i,c_{j,k}] = 1
    # For j!=j': automatically 0.
    #
    # So we need W[k*n+0, c_{j,k}] = 1 and W[k*n+1, c_{j,k}] = 1 for all j,k.
    # This is the same as before, giving rank 28.
    #
    # To reduce rank, we need to share c indices across different (j,k) pairs.
    #
    # Let's try: use only m=4 terms total (one per j), with W depending on k.
    # c = j_c for j_c in 0..3.
    # U[i*m+j, j_c] = delta[j, j_c]
    # V[j'*p+k, j_c] = delta[j', j_c] * f(k, j_c)
    # W[k*n+i, j_c] = g(k, i, j_c)
    #
    # Sum: sum_{j_c} delta[j,j_c] * delta[j',j_c]*f(k,j_c) * g(k,i,j_c)
    # = delta[j,j'] * f(k,j) * g(k,i,j)
    #
    # We need: delta[j,j'] * f(k,j) * g(k,i,j) = delta[j,j']
    # So: f(k,j) * g(k,i,j) = 1 for all k,i,j.
    #
    # One solution: f(k,j)=1, g(k,i,j)=1. Rank=4.
    # But wait, let's check if this actually works.
    #
    # U[i*m+j, j_c] = delta[j, j_c]
    # V[j'*p+k, j_c] = delta[j', j_c]
    # W[k*n+i, j_c] = 1
    #
    # Sum over j_c: sum_{j_c} delta[j,j_c] * delta[j',j_c] * 1 = delta[j,j']
    #
    # This gives rank 4! Let me verify.

    def rank4_solution():
        rank = m  # 4
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for j_c in range(m):
            for i in range(n):
                U[i * m + j_c, j_c] = 1.0
            for jp in range(m):
                if jp == j_c:
                    for k in range(p):
                        V[jp * p + k, j_c] = 1.0
            for k in range(p):
                for i in range(n):
                    W[k * n + i, j_c] = 1.0
        return U, V, W

    U4, V4, W4 = rank4_solution()
    if verify(U4, V4, W4):
        best_U, best_V, best_W = U4, V4, W4
        best_rank = m
    else:
        # Fall back to rank 28 if rank 4 doesn't work
        if best_rank > m * p:
            best_U, best_V, best_W = U28, V28, W28
            best_rank = m * p

    # Try to go below rank 4
    # The constraint is: sum_c U[i*m+j,c] V[j'*p+k,c] W[k*n+i,c] = delta[j,j']
    #
    # For rank r < 4, we need to find r vectors u_c, v_c, w_c such that
    # the 4x4 matrix (in j,j') defined by sum_c u_c[j]*v_c[j']*w_c = I_4 for all (i,k).
    #
    # Since w_c can depend on (i,k), we have more freedom.
    #
    # Let's try rank 3 with a search.

    def search_small_rank(rank, max_trials=2000):
        if time_left() < 10.0:
            return None

        for trial in range(max_trials):
            if time_left() < 10.0:
                return None

            # Initialize with small integers
            U = np.random.randint(-2, 3, size=(nm, rank)).astype(float)
            V = np.random.randint(-2, 3, size=(mp, rank)).astype(float)
            W = np.random.randint(-2, 3, size=(pn, rank)).astype(float)

            if verify(U, V, W):
                return U, V, W

            # Local search: try flipping entries
            for _ in range(20):
                if time_left() < 10.0:
                    break

                # Pick a random matrix and entry to modify
                mat_choice = np.random.randint(3)
                if mat_choice == 0:
                    idx = np.random.randint(nm)
                    col = np.random.randint(rank)
                    old = U[idx, col]
                    U[idx, col] = np.random.randint(-3, 4)
                    if verify(U, V, W):
                        return U, V, W
                    U[idx, col] = old
                elif mat_choice == 1:
                    idx = np.random.randint(mp)
                    col = np.random.randint(rank)
                    old = V[idx, col]
                    V[idx, col] = np.random.randint(-3, 4)
                    if verify(U, V, W):
                        return U, V, W
                    V[idx, col] = old
                else:
                    idx = np.random.randint(pn)
                    col = np.random.randint(rank)
                    old = W[idx, col]
                    W[idx, col] = np.random.randint(-3, 4)
                    if verify(U, V, W):
                        return U, V, W
                    W[idx, col] = old

        return None

    # Try ranks from 3 down to 1
    for rank in range(min(3, best_rank - 1), 0, -1):
        if time_left() < 15.0:
            break
        result = search_small_rank(rank, max_trials=500)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = rank
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
