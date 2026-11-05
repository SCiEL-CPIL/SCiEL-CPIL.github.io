"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from itertools import product, combinations, combinations_with_replacement

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _get_rank17_solution():
    """Returns the known rank-17 solution using Strassen block."""
    n, m, p = 2, 3, 3
    n_a, n_b, n_c = 6, 9, 6
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    def set_term(r, u_dict, v_dict, w_dict):
        for a, val in u_dict.items():
            U[a, r] = val
        for b, val in v_dict.items():
            V[b, r] = val
        for c, val in w_dict.items():
            W[c, r] = val
            
    # Part 1: Strassen for A1(2x2) x B1_left(2x2) -> C[0:2, 0:2]
    set_term(0, {0:1, 4:1}, {0:1, 4:1}, {0:1, 3:1})
    set_term(1, {3:1, 4:1}, {0:1}, {1:1, 3:-1})
    set_term(2, {0:1}, {1:1, 4:-1}, {2:1, 3:1})
    set_term(3, {4:1}, {3:1, 0:-1}, {0:1, 1:1})
    set_term(4, {0:1, 1:1}, {4:1}, {0:-1, 2:1})
    set_term(5, {3:1, 0:-1}, {0:1, 1:1}, {3:1})
    set_term(6, {1:1, 4:-1}, {3:1, 4:1}, {0:1})
    
    # Part 2: Direct for A1(2x2) x B1_right(2x1) -> C[:, 2]
    set_term(7, {0:1}, {2:1}, {4:1})
    set_term(8, {1:1}, {5:1}, {4:1})
    set_term(9, {3:1}, {2:1}, {5:1})
    set_term(10, {4:1}, {5:1}, {5:1})
    
    # Part 3: Direct for A2(2x1) x B2(1x3) -> C[0:2, 0:3]
    set_term(11, {2:1}, {6:1}, {0:1})
    set_term(12, {2:1}, {7:1}, {2:1})
    set_term(13, {2:1}, {8:1}, {4:1})
    set_term(14, {5:1}, {6:1}, {1:1})
    set_term(15, {5:1}, {7:1}, {3:1})
    set_term(16, {5:1}, {8:1}, {5:1})
    
    return U, V, W

def _combinatorial_rank16_search(T, deadline, max_terms=16):
    """
    Combinatorial support enumeration search for rank-16 decomposition.
    
    Systematically iterates through sparse support patterns for factor matrices
    U, V, W with bounded sparsity per term, solving the linear system to verify
    exact tensor decomposition.
    
    Strategy:
    1. For each term r, choose a small support (subset of indices) for U_r, V_r, W_r
    2. Build the design matrix from these supports
    3. Solve the linear system to find coefficients
    4. Verify exact reconstruction
    
    We use a hierarchical approach:
    - First try supports derived from the rank-17 structure (perturbed)
    - Then try systematically growing support sizes
    - Use time budget to explore more complex patterns
    """
    n_a, n_b, n_c = 6, 9, 6
    T_flat = T.flatten()
    num_entries = T_flat.size
    
    # Get the rank-17 solution for reference supports
    U17, V17, W17 = _get_rank17_solution()
    rank17 = U17.shape[1]
    
    # Extract supports from rank-17
    supports_U_17 = [frozenset(np.nonzero(U17[:, r])[0]) for r in range(rank17)]
    supports_V_17 = [frozenset(np.nonzero(V17[:, r])[0]) for r in range(rank17)]
    supports_W_17 = [frozenset(np.nonzero(W17[:, r])[0]) for r in range(rank17)]
    
    # All possible indices
    all_U = set(range(n_a))
    all_V = set(range(n_b))
    all_W = set(range(n_c))
    
    # Precompute: for each index, which rank-17 terms include it
    U_index_to_terms = {i: set() for i in range(n_a)}
    V_index_to_terms = {i: set() for i in range(n_b)}
    W_index_to_terms = {i: set() for i in range(n_c)}
    for r in range(rank17):
        for idx in supports_U_17[r]:
            U_index_to_terms[idx].add(r)
        for idx in supports_V_17[r]:
            V_index_to_terms[idx].add(r)
        for idx in supports_W_17[r]:
            W_index_to_terms[idx].add(r)
    
    # Cache for design matrix columns
    col_cache = {}
    
    def _build_column(U_supp, V_supp, W_supp):
        """Build a column of the design matrix from supports."""
        key = (U_supp, V_supp, W_supp)
        if key in col_cache:
            return col_cache[key]
        
        u_vec = np.zeros(n_a)
        v_vec = np.zeros(n_b)
        w_vec = np.zeros(n_c)
        for idx in U_supp:
            u_vec[idx] = 1
        for idx in V_supp:
            v_vec[idx] = 1
        for idx in W_supp:
            w_vec[idx] = 1
        
        col = np.einsum('a,b,c->abc', u_vec, v_vec, w_vec).flatten()
        col_cache[key] = col
        return col
    
    def _try_support_pattern(U_supports, V_supports, W_supports, rank):
        """Try a specific support pattern and solve for coefficients."""
        # Build design matrix
        A = np.zeros((num_entries, rank))
        for j in range(rank):
            A[:, j] = _build_column(U_supports[j], V_supports[j], W_supports[j])
        
        # Check rank
        try:
            if np.linalg.matrix_rank(A) < rank:
                return None
        except:
            return None
        
        # Solve
        try:
            alpha, residuals, rank_solve, sv = np.linalg.lstsq(A, T_flat, rcond=1e-10)
            T_recon = A @ alpha
            if np.allclose(T_recon, T_flat, atol=1e-8):
                # Found valid decomposition!
                U16 = np.zeros((n_a, rank))
                V16 = np.zeros((n_b, rank))
                W16 = np.zeros((n_c, rank))
                for j in range(rank):
                    u_vec = np.zeros(n_a)
                    v_vec = np.zeros(n_b)
                    w_vec = np.zeros(n_c)
                    for idx in U_supports[j]:
                        u_vec[idx] = 1
                    for idx in V_supports[j]:
                        v_vec[idx] = 1
                    for idx in W_supports[j]:
                        w_vec[idx] = 1
                    U16[:, j] = alpha[j] * u_vec
                    V16[:, j] = alpha[j] * v_vec
                    W16[:, j] = alpha[j] * w_vec
                return U16, V16, W16
        except:
            return None
        return None
    
    # Phase 1: Try excluding one term from rank-17 and solving
    # This tests if rank-16 is achievable with same supports
    for remove_idx in range(rank17):
        if time.time() > deadline:
            break
        keep = [r for r in range(rank17) if r != remove_idx]
        U_sups = [tuple(sorted(supports_U_17[r])) for r in keep]
        V_sups = [tuple(sorted(supports_V_17[r])) for r in keep]
        W_sups = [tuple(sorted(supports_W_17[r])) for r in keep]
        
        result = _try_support_pattern(U_sups, V_sups, W_sups, 16)
        if result is not None:
            return result
    
    # Phase 2: Systematic support modification
    # For each pair of terms, try merging their supports or swapping entries
    trial_count = 0
    max_trials = 50000
    
    while trial_count < max_trials and time.time() < deadline:
        trial_count += 1
        
        # Choose a base: exclude one term from rank-17
        remove_idx = trial_count % rank17
        keep = [r for r in range(rank17) if r != remove_idx]
        
        U_sups = [frozenset(supports_U_17[r]) for r in keep]
        V_sups = [frozenset(supports_V_17[r]) for r in keep]
        W_sups = [frozenset(supports_W_17[r]) for r in keep]
        
        # Apply 1-3 random modifications
        n_mods = random.randint(1, 3)
        for _ in range(n_mods):
            # Pick a random term to modify
            j = random.randint(0, 15)
            # Pick which matrix
            mat = random.randint(0, 2)
            
            if mat == 0:
                # Modify U support
                if random.random() < 0.5:
                    # Add a random index
                    candidates = list(all_U - U_sups[j])
                    if candidates:
                        U_sups[j] = U_sups[j] | {random.choice(candidates)}
                else:
                    # Remove a random index (if support size > 1)
                    if len(U_sups[j]) > 1:
                        to_remove = random.choice(list(U_sups[j]))
                        U_sups[j] = U_sups[j] - {to_remove}
            elif mat == 1:
                if random.random() < 0.5:
                    candidates = list(all_V - V_sups[j])
                    if candidates:
                        V_sups[j] = V_sups[j] | {random.choice(candidates)}
                else:
                    if len(V_sups[j]) > 1:
                        to_remove = random.choice(list(V_sups[j]))
                        V_sups[j] = V_sups[j] - {to_remove}
            else:
                if random.random() < 0.5:
                    candidates = list(all_W - W_sups[j])
                    if candidates:
                        W_sups[j] = W_sups[j] | {random.choice(candidates)}
                else:
                    if len(W_sups[j]) > 1:
                        to_remove = random.choice(list(W_sups[j]))
                        W_sups[j] = W_sups[j] - {to_remove}
        
        # Try this pattern
        U_tuple = [tuple(sorted(s)) for s in U_sups]
        V_tuple = [tuple(sorted(s)) for s in V_sups]
        W_tuple = [tuple(sorted(s)) for s in W_sups]
        
        result = _try_support_pattern(U_tuple, V_tuple, W_tuple, 16)
        if result is not None:
            return result
    
    return None

def _structured_rank16_search(start_time, deadline):
    """
    Structured search for rank-16 using combinatorial support enumeration.
    """
    n, m, p = 2, 3, 3
    T = _build_tensor(n, m, p)
    
    search_deadline = deadline - 5
    
    result = _combinatorial_rank16_search(T, search_deadline)
    if result is not None:
        U16, V16, W16 = result
        if _verify_decomposition(U16, V16, W16, T):
            return U16, V16, W16
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140 second deadline with safety margin
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    # 1. Attempt combinatorial rank-16 search
    result_16 = _structured_rank16_search(start_time, deadline)
    if result_16 is not None:
        U, V, W = result_16
        T = _build_tensor(n, m, p)
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # 2. Fallback to verified Rank 17
    U, V, W = _get_rank17_solution()
    T = _build_tensor(n, m, p)
    
    if not _verify_decomposition(U, V, W, T):
        # Final fallback: Schoolbook (Rank 18)
        rank_fb = n * m * p
        U_fb = np.zeros((n_a, rank_fb))
        V_fb = np.zeros((n_b, rank_fb))
        W_fb = np.zeros((n_c, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
