"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Uses a rank-26 decomposition based on Strassen's algorithm for the 2x2 core
    and standard multiplication for the remaining border terms.
    Attempts to find a lower rank decomposition via systematic linear dependency search.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = DIMENSIONS
    
    def idx_u(i, j):
        return i * m + j
    def idx_v(j, k):
        return j * p + k
    def idx_w(k, i):
        return k * n + i
    
    def verify(U, V, W, R):
        U_reshaped = U.reshape((n, m, R))
        V_reshaped = V.reshape((m, p, R))
        W_reshaped = W.reshape((p, n, R))
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        T_target = np.ones((n, m, p))
        return np.allclose(T_recon, T_target, atol=1e-9)
    
    def build_rank26():
        R = 26
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Strassen 7 terms
        terms = [
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]),
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
        ]
        
        for r, (u_terms, v_terms, w_terms) in enumerate(terms):
            for (i, j, val) in u_terms:
                U[idx_u(i, j), r] = val
            for (j, k, val) in v_terms:
                V[idx_v(j, k), r] = val
            for (k, i, val) in w_terms:
                W[idx_w(k, i), r] = val
        
        # Standard 19 terms
        col = 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        
        return U, V, W, R
    
    U, V, W, R = build_rank26()
    best_U, best_V, best_W, best_R = U.copy(), V.copy(), W.copy(), R
    
    if not verify(U, V, W, R):
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build term-vectors: each term r is a 27-dim vector
    # term_r[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r]
    def get_term_vectors(U, V, W, R):
        vecs = []
        for r in range(R):
            vec = np.zeros(n * m * p)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        vec[i * m * p + j * p + k] = U[idx_u(i,j), r] * V[idx_v(j,k), r] * W[idx_w(k,i), r]
            vecs.append(vec)
        return np.array(vecs)
    
    # Try to find linear dependencies among the 26 terms
    # If sum(c_r * term_r) = 0, we can eliminate one term
    def find_linear_dependency(U, V, W, R, max_iter=5000):
        """
        Find if there's a non-trivial linear combination of terms that sums to zero.
        We search over small integer/rational coefficients.
        """
        term_vecs = get_term_vectors(U, V, W, R)
        # term_vecs shape: (R, 27)
        
        # Try to find a subset of terms that are linearly dependent
        # Use random subset approach
        np.random.seed(123)
        
        for attempt in range(max_iter):
            if time.time() > deadline - 20:
                break
            
            # Pick a random subset of 2-4 terms
            subset_size = np.random.choice([2, 3, 4], p=[0.4, 0.4, 0.2])
            if subset_size > R:
                continue
            indices = np.random.choice(R, size=subset_size, replace=False)
            
            # Try to find coefficients c_1, ..., c_s such that sum(c_i * term_i) ≈ 0
            # This is a system of 27 equations in s unknowns
            # For s < 27, we need a specific structure
            
            # For 2 terms: c1*t1 + c2*t2 = 0 means t1 and t2 are proportional
            if subset_size == 2:
                t1, t2 = term_vecs[indices[0]], term_vecs[indices[1]]
                # Check if proportional
                nonzero = np.abs(t1) > 1e-10
                if np.any(nonzero):
                    ratios = t2[nonzero] / t1[nonzero]
                    if np.allclose(ratios, ratios[0], atol=1e-9):
                        # They are proportional! We can eliminate one
                        return list(indices), [1.0, -ratios[0]]
            elif subset_size == 3:
                # Try small integer coefficients
                for c1 in [-2, -1, 1, 2]:
                    for c2 in [-2, -1, 1, 2]:
                        for c3 in [-2, -1, 1, 2]:
                            combo = c1 * term_vecs[indices[0]] + c2 * term_vecs[indices[1]] + c3 * term_vecs[indices[2]]
                            if np.allclose(combo, 0, atol=1e-9):
                                return list(indices), [c1, c2, c3]
            elif subset_size == 4:
                # Try small integer coefficients
                for c1 in [-2, -1, 1, 2]:
                    for c2 in [-2, -1, 1, 2]:
                        for c3 in [-2, -1, 1, 2]:
                            for c4 in [-2, -1, 1, 2]:
                                combo = c1*term_vecs[indices[0]] + c2*term_vecs[indices[1]] + c3*term_vecs[indices[2]] + c4*term_vecs[indices[3]]
                                if np.allclose(combo, 0, atol=1e-9):
                                    return list(indices), [c1, c2, c3, c4]
        
        return None, None
    
    # Attempt to reduce rank by finding linear dependencies
    if time.time() < deadline:
        dep_indices, dep_coeffs = find_linear_dependency(U, V, W, R)
        
        if dep_indices is not None and len(dep_indices) > 1:
            # We found a linear dependency
            # We can eliminate the last index in dep_indices
            elim_idx = dep_indices[-1]
            keep_indices = [i for i in range(R) if i != elim_idx]
            
            # Scale the remaining terms to compensate
            # sum(c_i * term_i) = 0 means term_elim = -sum_{i != elim} (c_i / c_elim) * term_i
            c_elim = dep_coeffs[-1]
            
            U_new = np.zeros((n * m, R - 1))
            V_new = np.zeros((m * p, R - 1))
            W_new = np.zeros((p * n, R - 1))
            
            for new_r, old_r in enumerate(keep_indices):
                if old_r in dep_indices:
                    # Scale by c_i / c_elim
                    scale = dep_coeffs[dep_indices.index(old_r)] / c_elim
                    U_new[:, new_r] = U[:, old_r] * scale
                    V_new[:, new_r] = V[:, old_r] * scale
                    W_new[:, new_r] = W[:, old_r] * scale
                else:
                    U_new[:, new_r] = U[:, old_r]
                    V_new[:, new_r] = V[:, old_r]
                    W_new[:, new_r] = W[:, old_r]
            
            if verify(U_new, V_new, W_new, R - 1):
                best_U, best_V, best_W, best_R = U_new.copy(), V_new.copy(), W_new.copy(), R - 1
                print(f"Reduced to rank {R-1} via linear dependency")
    
    # Try randomized search for even lower rank
    if time.time() < deadline and best_R > 23:
        np.random.seed(42)
        for target_R in [best_R - 1, 24, 23]:
            if target_R >= best_R or time.time() > deadline - 15:
                continue
            
            for attempt in range(100):
                if time.time() > deadline - 15:
                    break
                
                U_try = np.zeros((n * m, target_R))
                V_try = np.zeros((m * p, target_R))
                W_try = np.zeros((p * n, target_R))
                
                # Initialize with Strassen 7 terms
                terms = [
                    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
                    ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
                    ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
                    ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
                    ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]),
                    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
                    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
                ]
                
                for r, (u_terms, v_terms, w_terms) in enumerate(terms):
                    if r >= target_R:
                        break
                    for (i, j, val) in u_terms:
                        U_try[idx_u(i, j), r] = val
                    for (j, k, val) in v_terms:
                        V_try[idx_v(j, k), r] = val
                    for (k, i, val) in w_terms:
                        W_try[idx_w(k, i), r] = val
                
                # Fill remaining terms
                for r in range(7, target_R):
                    n_u = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                    n_v = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                    n_w = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                    
                    u_positions = np.random.choice(n * m, size=n_u, replace=False)
                    v_positions = np.random.choice(m * p, size=n_v, replace=False)
                    w_positions = np.random.choice(p * n, size=n_w, replace=False)
                    
                    for pos in u_positions:
                        U_try[pos, r] = np.random.choice([-2, -1, 1, 2])
                    for pos in v_positions:
                        V_try[pos, r] = np.random.choice([-2, -1, 1, 2])
                    for pos in w_positions:
                        W_try[pos, r] = np.random.choice([-2, -1, 1, 2])
                
                if verify(U_try, V_try, W_try, target_R):
                    best_U, best_V, best_W, best_R = U_try.copy(), V_try.copy(), W_try.copy(), target_R
                    break
    
    # Final verification
    if not verify(best_U, best_V, best_W, best_R):
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
