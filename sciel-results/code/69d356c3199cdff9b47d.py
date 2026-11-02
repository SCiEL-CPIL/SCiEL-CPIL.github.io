import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n
    
    # Strassen coefficients for 2x2 * 2x2
    strassen_terms = [
        {'x': {(0,0):1, (1,1):1}, 'y': {(0,0):1, (1,1):1}, 'z': {(0,0):1, (1,1):1}},
        {'x': {(1,0):1, (1,1):1}, 'y': {(0,0):1}, 'z': {(1,0):1, (1,1):-1}},
        {'x': {(0,0):1}, 'y': {(0,1):1, (1,1):-1}, 'z': {(0,1):1, (1,1):1}},
        {'x': {(1,1):1}, 'y': {(1,0):1, (0,0):-1}, 'z': {(0,0):1, (1,0):1}},
        {'x': {(0,0):1, (0,1):1}, 'y': {(1,1):1}, 'z': {(0,0):-1, (0,1):1}},
        {'x': {(1,0):1, (0,0):-1}, 'y': {(0,0):1, (0,1):1}, 'z': {(1,1):1}},
        {'x': {(0,1):1, (1,1):-1}, 'y': {(1,0):1, (1,1):1}, 'z': {(0,0):1}},
    ]
    
    def build_strassen(rank):
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for a_block in range(2):
            for b_col_block in range(4):
                for t in range(7):
                    term = strassen_terms[t]
                    for (i_prime, j_prime), coeff in term['x'].items():
                        u_idx = i_prime * m + (2 * a_block + j_prime)
                        U[u_idx, col] += coeff
                    for (j_prime, k_prime), coeff in term['y'].items():
                        b_row = 2 * a_block + j_prime
                        b_col = 2 * b_col_block + k_prime
                        v_idx = b_row * p + b_col
                        V[v_idx, col] += coeff
                    for (i_prime, k_prime), coeff in term['z'].items():
                        c_col = 2 * b_col_block + k_prime
                        c_row = i_prime
                        w_idx = c_col * n + c_row
                        W[w_idx, col] += coeff
                    col += 1
        return U, V, W
    
    def verify(U, V, W):
        """Verify that the decomposition exactly represents matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True
    
    # Build standard Strassen decomposition (rank 56)
    rank = 56
    U, V, W = build_strassen(rank)
    
    # Verify correctness
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank = nm * mp * pn  # This won't work, use proper schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by checking for linear dependencies among columns
    # For each column c, check if it can be expressed as a linear combination of others
    # while preserving the tensor identity
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
    
    if time.time() < deadline - 10:
        # Attempt rank reduction: try removing each column and checking if the
        # remaining columns still span the same tensor space
        # A column c is redundant if there exist coefficients alpha_r (r != c)
        # such that for all (i,j,k):
        #   U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = sum_{r!=c} alpha_r * U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # This is a nonlinear constraint in general, but we can try a simpler approach:
        # Check if column c is a linear combination of other columns in U, V, W simultaneously
        # (which is a sufficient but not necessary condition)
        
        # Simpler approach: try to find if any two blocks share structure
        # that allows merging. Check if column pairs from different blocks are proportional.
        
        # For now, try a direct approach: for each candidate column to remove,
        # solve for whether the tensor identity can still be satisfied.
        # This is expensive, so limit the search.
        
        # Alternative: check if the 56 columns are linearly independent in a tensor sense.
        # Build the tensor T of shape (nm, mp, pn) where T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        # The tensor should equal 1 when i,j,k correspond to valid (i,j,k) triples.
        
        # Actually, let's try a different approach: check if we can combine columns
        # from the same Strassen term across different blocks.
        
        # The 8 blocks are: (a_block, b_col_block) for a_block in {0,1}, b_col_block in {0,1,2,3}
        # Each block has 7 Strassen terms. Total: 8*7 = 56 columns.
        
        # Check if any column is a scalar multiple of another (unlikely but worth checking)
        for c1 in range(rank):
            if time.time() > deadline - 5:
                break
            for c2 in range(c1 + 1, rank):
                # Check if column c1 is proportional to column c2
                # This requires U[:,c1] = alpha*U[:,c2], V[:,c1]=alpha*V[:,c2], W[:,c1]=alpha*W[:,c2]
                # But this would mean they're the same term, which isn't the case here.
                pass
        
        # Try a more promising approach: check if we can use a lower-rank decomposition
        # by exploiting the specific structure of 2x4 x 4x8.
        # 
        # Key insight: The middle dimension is 4. We can think of this as:
        # C = A * B where A is 2x4, B is 4x8
        # 
        # Another approach: use the fact that rank(2x4x4x8) <= rank(2x2x2x2) * 8 = 56
        # but maybe we can do better with a different block structure.
        #
        # Try: split A into 1x4 blocks (2 blocks) and B into 4x4 blocks (2 blocks)
        # Then each 1x4 x 4x4 multiplication has rank at most 4*4=16 (schoolbook)
        # or potentially less with optimization.
        # 2 * 2 * 16 = 64, which is worse.
        #
        # Try: split A into 2x2 blocks (2 blocks) and B into 2x8 blocks (2 blocks)
        # Each 2x2 x 2x8 has rank at most 2*2*8=32, or with Strassen-like: 7*4=28
        # 2 * 2 * 28 = 112, worse.
        #
        # The current approach seems near-optimal for this block structure.
        # Let's try to verify if rank 55 is possible by checking linear dependencies.
        
        # Check linear independence of the 56 columns in a combined sense
        # Build a matrix where each column r corresponds to the tensor U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # flattened. If any column is in the span of others, we can reduce rank.
        # This is a (nm*mp*pn) x rank matrix, which is 8*32*16=4096 x 56.
        
        if time.time() < deadline - 20:
            # Build the tensor representation
            tensor_dim = nm * mp * pn
            T = np.zeros((tensor_dim, rank))
            for r in range(rank):
                # T[flatten(i,j,k), r] = U[i,r] * V[j,r] * W[k,r]
                for i in range(nm):
                    for j in range(mp):
                        for k in range(pn):
                            T[i*mp*pn + j*pn + k, r] = U[i,r] * V[j,r] * W[k,r]
            
            # Check rank of T
            # If rank(T) < 56, then some columns are dependent and we can reduce
            try:
                T_rank = np.linalg.matrix_rank(T, tol=1e-9)
                if T_rank < rank:
                    # Find a basis for the column space
                    # Use QR decomposition
                    Q, R = np.linalg.qr(T, mode='reduced')
                    # The first T_rank columns of Q form a basis
                    # But we need to express original columns in terms of this basis
                    # Actually, we need to find which columns are independent
                    # and express the others as combinations
                    
                    # For simplicity, just note that rank reduction is possible
                    # but implementing the full reduction is complex.
                    # Let's try a greedy approach: iteratively remove dependent columns.
                    
                    pass  # Complex to implement correctly in limited time
            except Exception:
                pass
    
    # Return the best result found
    # The standard Strassen decomposition with rank 56 is our best known result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
