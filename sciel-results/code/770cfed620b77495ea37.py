"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 3, 10
    start_time = time.time()
    deadline = start_time + 145  # 145s deadline (150s budget, 5s safety margin)
    
    # Start with the known rank-55 Strassen-based decomposition
    rank = 55
    U_mat = np.zeros((n * m, rank), dtype=float)
    V_mat = np.zeros((m * p, rank), dtype=float)
    W_mat = np.zeros((p * n, rank), dtype=float)
    
    terms = []
    
    # Generate Strassen terms for 5 blocks of 2x2
    for l in range(5):
        k0 = 2 * l
        k1 = 2 * l + 1
        
        # A indices (0-5)
        a00, a01, a10, a11 = 0, 1, 3, 4
        
        # B indices (0-29)
        b0k0, b0k1 = 0 * 10 + k0, 0 * 10 + k1
        b1k0, b1k1 = 1 * 10 + k0, 1 * 10 + k1
        
        # W indices (0-19)
        c0k0, c0k1 = k0 * 2 + 0, k1 * 2 + 0
        c1k0, c1k1 = k0 * 2 + 1, k1 * 2 + 1
        
        # M1
        terms.append(([(a00, 1), (a11, 1)], [(b0k0, 1), (b1k1, 1)], [(c0k0, 1), (c1k1, 1)]))
        # M2
        terms.append(([(a10, 1), (a11, 1)], [(b0k0, 1)], [(c1k0, 1), (c1k1, -1)]))
        # M3
        terms.append(([(a00, 1)], [(b0k1, 1), (b1k1, -1)], [(c0k1, 1), (c1k1, 1)]))
        # M4
        terms.append(([(a11, 1)], [(b1k0, 1), (b0k0, -1)], [(c0k0, 1), (c1k0, 1)]))
        # M5
        terms.append(([(a00, 1), (a01, 1)], [(b1k1, 1)], [(c0k0, -1), (c0k1, 1)]))
        # M6
        terms.append(([(a10, 1), (a00, -1)], [(b0k0, 1), (b0k1, 1)], [(c1k1, 1)]))
        # M7
        terms.append(([(a01, 1), (a11, -1)], [(b1k0, 1), (b1k1, 1)], [(c0k0, 1)]))
    
    # Schoolbook for j=2
    for i in range(2):
        for k in range(10):
            u_idx = i * 3 + 2
            v_idx = 2 * 10 + k
            w_idx = k * 2 + i
            terms.append(([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)]))
    
    # Fill matrices
    for t_idx, (u_terms, v_terms, w_terms) in enumerate(terms):
        for idx, coeff in u_terms:
            U_mat[idx, t_idx] = coeff
        for idx, coeff in v_terms:
            V_mat[idx, t_idx] = coeff
        for idx, coeff in w_terms:
            W_mat[idx, t_idx] = coeff
    
    # Try to reduce rank by checking if we can eliminate redundant terms
    # Check if any term is a linear combination of others
    best_U, best_V, best_W = U_mat.copy(), V_mat.copy(), W_mat.copy()
    best_rank = rank
    
    # Time-limited search for rank reduction
    # Try removing terms and checking if the decomposition still works
    # This is expensive, so we only try a limited number of removals
    
    if time.time() < deadline:
        # Build the tensor and check if rank can be reduced
        # The tensor T[i,j,k] = delta_{i,k} (for matrix multiplication)
        # Actually, for matrix multiplication A @ B, the tensor is T[i,k,j] = A[i,j] * B[j,k]
        # In our indexing: A is (n x m), B is (m x p), C is (n x p)
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor has indices (i, j, k) where A[i,j], B[j,k], C[i,k]
        
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # U[i*m + j, t] = coefficient for A[i,j] in term t
        # V[j*p + k, t] = coefficient for B[j,k] in term t
        # W[k*n + i, t] = coefficient for C[i,k] in term t
        
        # The decomposition requires: for all (i,j,k), sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = delta_{i,k} * 1
        # Wait, that's not right. Let me re-examine.
        
        # Actually, for matrix multiplication, we want:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor decomposition is:
        # sum_t U[i,j,t] * V[j,k,t] * W[i,k,t] = A[i,j] * B[j,k] for all i,j,k
        # No, that's not the standard formulation either.
        
        # Standard formulation: The multiplication tensor T is defined by
        # T[i,j,k] = 1 if i,k are the output indices and j is the "middle" index
        # Actually, let me just verify the current decomposition works.
        
        # Verify: for random matrices, check if U @ V^T gives the right C
        def verify(U, V, W, n, m, p):
            # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
            # For each term t, we compute: (U[:,t] reshaped to (n,m)) @ (V[:,t] reshaped to (m,p))
            # and add it to C weighted by W[:,t]
            # Wait, that's not how it works either.
            
            # The decomposition is: C = sum_t (U_t outer V_t) * W_t
            # where U_t is (n,m), V_t is (m,p), W_t is (n,p)
            # But U[:,t] is a vector of length n*m, so U_t = U[:,t].reshape(n,m)
            # Similarly V_t = V[:,t].reshape(m,p), W_t = W[:,t].reshape(p,n).T
            
            # Actually, I think the convention is:
            # C[i,k] = sum_j sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
            
            # Let's test with a specific case
            n_test, m_test, p_test = 2, 3, 10
            A_test = np.random.rand(n_test, m_test)
            B_test = np.random.rand(m_test, p_test)
            C_expected = A_test @ B_test
            
            C_computed = np.zeros((n_test, p_test))
            rank_test = U.shape[1]
            for t in range(rank_test):
                U_t = U[:n_test*m_test, t].reshape(n_test, m_test)
                V_t = V[:m_test*p_test, t].reshape(m_test, p_test)
                W_t = W[:p_test*n_test, t].reshape(p_test, n_test).T
                C_computed += (U_t @ V_t) * W_t
            
            return np.allclose(C_computed, C_expected, atol=1e-10)
        
        # Verify the current decomposition
        if not verify(U_mat, V_mat, W_mat, n, m, p):
            # Something is wrong, but we still return the current result
            pass
        
        # Try to reduce rank by trying to remove terms
        # This is a heuristic search; we try removing one term at a time
        # and check if the decomposition still holds
        
        # For efficiency, we only try removing terms that have small norm
        term_norms = []
        for t in range(rank):
            norm = np.sum(U_mat[:, t] ** 2) + np.sum(V_mat[:, t] ** 2) + np.sum(W_mat[:, t] ** 2)
            term_norms.append((norm, t))
        
        # Sort by norm, try removing small terms first
        term_norms.sort()
        
        removed = set()
        current_U, current_V, current_W = U_mat.copy(), V_mat.copy(), W_mat.copy()
        current_rank = rank
        
        for norm, t in term_norms:
            if time.time() > deadline:
                break
            if t in removed:
                continue
            
            # Try removing term t
            new_U = current_U.copy()
            new_V = current_V.copy()
            new_W = current_W.copy()
            
            # Check if removing term t still gives a valid decomposition
            # This is expensive, so we only do it if we have time
            if verify(new_U, new_V, new_W, n, m, p):
                # Term t is redundant
                removed.add(t)
                # We don't actually remove it from the array, we just mark it
                # For now, we just count how many we can remove
                current_rank -= 1
            else:
                # Try removing term t by setting it to zero and checking
                # Actually, we need to check if the tensor equation still holds
                pass
        
        # If we found redundant terms, we could try a more aggressive search
        # But for now, we just return the original rank-55 decomposition
        
    # Convert to lists for the return value
    U_list = U_mat.tolist()
    V_list = V_mat.tolist()
    W_list = W_mat.tolist()
    
    return {'U': U_list, 'V': V_list, 'W': W_list}
# EVOLVE-BLOCK-END
