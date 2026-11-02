import time
import numpy as np

DIMENSIONS = [2, 2, 8]

def construct():
    start_time = time.time()
    deadline = 145.0

    n, m, p = DIMENSIONS  # 2, 2, 8

    # Rank 28 decomposition using Strassen's algorithm for 4 groups of 2 columns.
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Strassen's 7 products for 2x2 matrix multiplication.
    strassen_terms = [
        {'A': {(0,0): 1.0, (1,1): 1.0}, 
         'B': {(0,0): 1.0, (1,1): 1.0}, 
         'C': {(0,0): 1.0, (1,1): 1.0}},
        {'A': {(1,0): 1.0, (1,1): 1.0}, 
         'B': {(0,0): 1.0}, 
         'C': {(1,0): 1.0, (1,1): -1.0}},
        {'A': {(0,0): 1.0}, 
         'B': {(0,1): 1.0, (1,1): -1.0}, 
         'C': {(0,1): 1.0, (1,1): 1.0}},
        {'A': {(1,1): 1.0}, 
         'B': {(1,0): 1.0, (0,0): -1.0}, 
         'C': {(0,0): 1.0, (1,0): 1.0}},
        {'A': {(0,0): 1.0, (0,1): 1.0}, 
         'B': {(1,1): 1.0}, 
         'C': {(0,0): -1.0, (0,1): 1.0}},
        {'A': {(1,0): 1.0, (0,0): -1.0}, 
         'B': {(0,0): 1.0, (0,1): 1.0}, 
         'C': {(1,1): 1.0}},
        {'A': {(0,1): 1.0, (1,1): -1.0}, 
         'B': {(1,0): 1.0, (1,1): 1.0}, 
         'C': {(0,0): 1.0}},
    ]

    # Apply Strassen to each group of 2 columns
    for q in range(p // 2):
        k1 = 2 * q
        k2 = 2 * q + 1
        for r, term in enumerate(strassen_terms):
            t = q * 7 + r
            
            for (i, j), coeff in term['A'].items():
                U[i * m + j, t] = coeff
            
            for (j, k_sub), coeff in term['B'].items():
                k_global = k1 + k_sub
                V[j * p + k_global, t] = coeff
            
            for (i, k_sub), coeff in term['C'].items():
                k_global = k1 + k_sub
                W[k_global * n + i, t] = coeff

    # Try to find a linear dependency among the 28 Strassen terms
    # to reduce rank to 27 or lower
    if time.time() - start_time < deadline - 10:
        try:
            # Build the tensor as a matrix to check for dependencies
            # T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
            # We want to check if the 28 rank-1 tensors are linearly independent
            # Flatten the tensor: each rank-1 term contributes U_t ⊗ V_t ⊗ W_t
            # Check if any term is a linear combination of others
            
            # Build the 28 vectors in R^(nm * mp * np)
            # Actually, check linear independence of the rank-1 tensors
            # A simpler approach: check if the 28 terms span a space of dimension < 28
            
            # Represent each term as a vector
            terms_vecs = []
            for t in range(rank):
                u_t = U[:, t]
                v_t = V[:, t]
                w_t = W[:, t]
                # Kronecker product u_t ⊗ v_t ⊗ w_t
                # Flatten to a single vector
                vec = np.kron(np.kron(u_t, v_t), w_t)
                terms_vecs.append(vec)
            
            terms_matrix = np.column_stack(terms_vecs)
            
            # Check rank of this matrix
            # Use SVD to find numerical rank
            U_svd, s, Vt_svd = np.linalg.svd(terms_matrix, full_matrices=False)
            tol = 1e-10
            numerical_rank = np.sum(s > tol)
            
            if numerical_rank < rank:
                # There's a dependency - we can reduce rank
                # Find which terms are dependent
                # This is complex, so for now just note the rank
                pass
        except:
            pass

    # Verify the decomposition
    if time.time() - start_time < deadline - 5:
        all_correct = True
        for i in range(n):
            for j in range(m):
                u_vec = U[i*m + j, :]
                v_vecs = V[j*p:(j+1)*p, :]
                w_indices = [k*n + i for k in range(p)]
                w_vecs = W[w_indices, :]
                
                prod = u_vec[None, :] * v_vecs * w_vecs
                sums = np.sum(prod, axis=1)
                
                if not np.allclose(sums, 1.0, atol=1e-10):
                    all_correct = False
                    break
            if not all_correct:
                break
        
        if not all_correct:
            # Fallback to schoolbook if verification fails
            rank = 32
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        t = i * m * p + j * p + k
                        U[i * m + j, t] = 1.0
                        V[j * p + k, t] = 1.0
                        W[k * n + i, t] = 1.0

    if time.time() - start_time > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
