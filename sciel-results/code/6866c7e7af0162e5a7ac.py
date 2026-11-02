import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 26, 26
    
    def A_idx(i, j): return i * m + j
    def B_idx(j, k): return j * p + k
    def C_idx(i, k): return k * n + i
    
    def build_strassen_solution():
        R = 46
        U = np.zeros((nm, R), dtype=float)
        V = np.zeros((mp, R), dtype=float)
        W = np.zeros((pn, R), dtype=float)
        
        def add_strassen_block(k0, col_offset):
            prods = [
                ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
                ({(1,0):1,(1,1):1}, {(0,0):1}),
                ({(0,0):1}, {(0,1):1,(1,1):-1}),
                ({(1,1):1}, {(1,0):1,(0,0):-1}),
                ({(0,0):1,(0,1):1}, {(1,1):1}),
                ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}),
                ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}),
            ]
            for t, (a_coeff, b_coeff) in enumerate(prods):
                col = col_offset + t
                for (i, j), c in a_coeff.items():
                    U[A_idx(i, j), col] += c
                for (j, kl), c in b_coeff.items():
                    V[B_idx(j, k0 + kl), col] += c
            c_combos = [
                (0, k0,     [(0,1),(3,1),(4,-1),(6,1)]),
                (0, k0+1,   [(2,1),(4,1)]),
                (1, k0,     [(1,1),(3,1)]),
                (1, k0+1,   [(0,1),(1,-1),(2,1),(5,1)]),
            ]
            for (i, k, terms) in c_combos:
                for (t, coeff) in terms:
                    W[C_idx(i, k), col_offset + t] += coeff
        
        def add_single_block(k, col_offset):
            prods = [
                ((0,0),(0,k)),
                ((0,1),(1,k)),
                ((1,0),(0,k)),
                ((1,1),(1,k)),
            ]
            for t, ((i,j),(jj,kk)) in enumerate(prods):
                col = col_offset + t
                U[A_idx(i, j), col] += 1
                V[B_idx(jj, kk), col] += 1
            W[C_idx(0, k), col_offset+0] += 1
            W[C_idx(0, k), col_offset+1] += 1
            W[C_idx(1, k), col_offset+2] += 1
            W[C_idx(1, k), col_offset+3] += 1
        
        col = 0
        for k0 in range(0, 12, 2):
            add_strassen_block(k0, col)
            col += 7
        add_single_block(12, col)
        col += 4
        assert col == R
        return U, V, W
    
    def verify(U, V, W, tol=1e-8):
        target_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target_indices.append((i * m + j, j * p + k, k * n + i))
        target_arr = np.array(target_indices)
        a_idx = target_arr[:, 0]
        b_idx = target_arr[:, 1]
        c_idx = target_arr[:, 2]
        U_t = U[a_idx, :]
        V_t = V[b_idx, :]
        W_t = W[c_idx, :]
        prod = U_t * V_t * W_t
        sums = np.sum(prod, axis=1)
        return np.all(np.abs(sums - 1.0) < tol)
    
    # Start with the known rank-46 solution
    U, V, W = build_strassen_solution()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 46
    
    if not verify(U, V, W):
        # Fallback to schoolbook
        R0 = n * m * p
        U0 = np.zeros((nm, R0), dtype=float)
        V0 = np.zeros((mp, R0), dtype=float)
        W0 = np.zeros((pn, R0), dtype=float)
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U0[i * m + j, col_idx] = 1.0
                    V0[j * p + k, col_idx] = 1.0
                    W0[k * n + i, col_idx] = 1.0
                    col_idx += 1
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    
    # Try to find a rank-45 solution by searching for a direction
    # that allows one column to be expressed as a linear combination
    # of the others, then eliminate it.
    
    # The idea: if the rank is 45, then there exists a linear dependence
    # among the 46 rank-1 tensors. We can try to find a vector (α, β, γ)
    # such that α⊗β⊗γ = Σ_t λ_t (u_t⊗v_t⊗w_t)
    
    # Alternative approach: try random perturbations and see if we can
    # reduce the rank by finding a null direction.
    
    # Let's try a different strategy: use the fact that the last column
    # (k=12) uses 4 separate multiplications. Maybe we can express one of
    # these as a combination of Strassen products.
    
    # For k=12: C[0,12] = A[0,0]*B[0,12] + A[0,1]*B[1,12]
    #            C[1,12] = A[1,0]*B[0,12] + A[1,1]*B[1,12]
    
    # The Strassen block for k0=10 computes:
    # M0 = (A[0,0]+A[1,1])*(B[0,10]+B[1,11])
    # M1 = (A[1,0]+A[1,1])*B[0,10]
    # M2 = A[0,0]*(B[0,11]-B[1,11])
    # M3 = A[1,1]*(B[1,10]-B[0,10])
    # M4 = (A[0,0]+A[0,1])*B[1,11]
    # M5 = (A[1,0]-A[0,0])*(B[0,10]+B[0,11])
    # M6 = (A[0,1]-A[1,1])*(B[1,10]+B[1,11])
    
    # C[0,10] = M0+M3-M4+M6
    # C[0,11] = M2+M4
    # C[1,10] = M1+M3
    # C[1,11] = M0-M1+M2+M5
    
    # For k=12, we need:
    # C[0,12] = A[0,0]*B[0,12] + A[0,1]*B[1,12]
    # C[1,12] = A[1,0]*B[0,12] + A[1,1]*B[1,12]
    
    # These don't naturally share with the Strassen block for k0=10.
    
    # Let me try a numerical search: can we find a rank-45 decomposition?
    # Strategy: start with rank-46, add a small random perturbation,
    # then try to project out one dimension.
    
    if time.time() < deadline - 60:
        # Try to find a rank-45 solution
        # Approach: the 46 rank-1 tensors span a 46-dim space.
        # If rank is 45, there's a linear dependence.
        
        # Build the tensor in flattened form and check its rank
        # T is a 4x26x26 tensor. Flatten to 4x(26*26) matrix.
        T = np.zeros((nm, mp * pn), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b * pn + c] = 1.0
        
        # The tensor rank is at least max of slice ranks
        # Slice along A: rank of T as 4 x (26*26) matrix
        slice_A_rank = np.linalg.matrix_rank(T)
        
        # Try to see if we can find a rank-45 decomposition by
        # optimizing over a parameter that creates a linear dependence.
        
        # Alternative: use the current solution and try to eliminate one column
        # by expressing it as a linear combination of others.
        
        # For each column r, check if u_r ⊗ v_r ⊗ w_r is in the span of
        # the other 45 rank-1 tensors. This is a 45-dimensional linear system.
        
        # The space of 4x26x26 tensors has dimension 4*26*26 = 2704.
        # 46 rank-1 tensors span at most 46 dimensions.
        
        # Let's check if the 46 tensors are linearly independent.
        # Flatten each rank-1 tensor and form a 2704 x 46 matrix.
        
        tensors = []
        for r in range(46):
            t = np.outer(U[:, r], np.outer(V[:, r], W[:, r]).flatten())
            tensors.append(t)
        tensor_mat = np.column_stack(tensors)  # 2704 x 46
        
        # Check rank
        rank_of_tensors = np.linalg.matrix_rank(tensor_mat)
        
        if rank_of_tensors < 46:
            # There's a linear dependence! We can reduce the rank.
            # Find the null space
            from scipy.linalg import null_space
            null = null_space(tensor_mat)
            if null.shape[1] > 0:
                # We have a linear dependence
                # The rank is rank_of_tensors
                # We need to find a basis for the span
                # and express the decomposition with fewer terms.
                
                # Use SVD to find a rank-reduced decomposition
                U_svd, S, Vt = np.linalg.svd(tensor_mat, full_matrices=False)
                # The top rank_of_tensors left singular vectors span the column space
                # We can use the first rank_of_tensors columns of Vt as a new basis
                
                new_rank = rank_of_tensors
                new_Vt = Vt[:new_rank, :]  # new_rank x 46
                
                # New factor matrices:
                # The original decomposition is: T = Σ_r u_r ⊗ v_r ⊗ w_r
                # = Σ_r (u_r ⊗ v_r) ⊗ w_r (reshaped)
                # In the new basis: T = Σ_s (Σ_r new_Vt[s,r] * u_r⊗v_r) ⊗ ... 
                # This doesn't directly give us the new U, V, W.
                
                # Actually, for a rank-1 tensor decomposition, if we have a linear
                # dependence Σ_r λ_r (u_r⊗v_r⊗w_r) = 0, we can't simply remove
                # one term unless the dependence is simple.
                
                # This approach is getting complicated. Let me try a different strategy.
                pass
        
        # Try a direct numerical search for rank 45
        # Use the fact that we can parameterize nearby decompositions
        
        # Actually, let me try a simpler approach: just verify and return the
        # rank-46 solution, since finding rank 45 is non-trivial.
    
    # Snap to half-integers for clean output
    U = np.round(best_U * 2) / 2
    V = np.round(best_V * 2) / 2
    W = np.round(best_W * 2) / 2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
