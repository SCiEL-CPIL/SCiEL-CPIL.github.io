import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    Uses schoolbook decomposition (rank 100) as guaranteed fallback,
    then attempts rank reduction via vectorized random search and local refinement.
    """
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin before 150s limit

    # Schoolbook decomposition as guaranteed fallback
    base_rank = n * m * p
    U_sb = np.zeros((n * m, base_rank), dtype=np.int64)
    V_sb = np.zeros((m * p, base_rank), dtype=np.int64)
    W_sb = np.zeros((p * n, base_rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_rank = base_rank
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    # Precompute index arrays for vectorized checks
    idx_i = np.array([i for i in range(n) for j in range(m) for k in range(p)])
    idx_j = np.array([j for i in range(n) for j in range(m) for k in range(p)])
    idx_k = np.array([k for i in range(n) for j in range(m) for k in range(p)])
    idx_ui = idx_i * m + idx_j  # U row indices
    idx_vj = idx_j * p + idx_k  # V row indices
    idx_wk = idx_k * n + idx_i  # W row indices
    num_triples = n * m * p

    def check_decomp_vectorized(U, V, W, rank):
        """
        Fast vectorized check: compute all triple products and verify sum == 1.
        Returns True if valid, False otherwise.
        """
        # For each triple (ui, vj, wk), compute sum over r of U[ui,r]*V[vj,r]*W[wk,r]
        # U[ui, :] shape (num_triples, rank)
        # V[vj, :] shape (num_triples, rank)
        # W[wk, :] shape (num_triples, rank)
        U_sel = U[idx_ui, :]  # (num_triples, rank)
        V_sel = V[idx_vj, :]  # (num_triples, rank)
        W_sel = W[idx_wk, :]  # (num_triples, rank)
        
        # Element-wise product then sum over rank
        prod = U_sel * V_sel * W_sel  # (num_triples, rank)
        sums = prod.sum(axis=1)  # (num_triples,)
        
        return np.all(sums == 1)
    
    def count_errors(U, V, W, rank):
        """Count how many triples have sum != 1."""
        U_sel = U[idx_ui, :]
        V_sel = V[idx_vj, :]
        W_sel = W[idx_wk, :]
        prod = U_sel * V_sel * W_sel
        sums = prod.sum(axis=1)
        return np.sum(sums != 1)
    
    def get_error_indices(U, V, W, rank):
        """Return list of triple indices where sum != 1."""
        U_sel = U[idx_ui, :]
        V_sel = V[idx_vj, :]
        W_sel = W[idx_wk, :]
        prod = U_sel * V_sel * W_sel
        sums = prod.sum(axis=1)
        return np.where(sums != 1)[0]
    
    def local_search(U, V, W, rank, max_iters=200):
        """
        Local search: try flipping individual entries to fix errors.
        Uses vectorized error counting for speed.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            err_count = count_errors(U, V, W, rank)
            if err_count == 0:
                return U, V, W
            
            # Get error triple indices
            err_idx = get_error_indices(U, V, W, rank)
            if len(err_idx) == 0:
                return U, V, W
            
            # Pick a random error
            ei = err_idx[np.random.randint(len(err_idx))]
            ui = idx_ui[ei]
            vj = idx_vj[ei]
            wk = idx_wk[ei]
            
            # Try flipping a random column in one of the three matrices
            r = np.random.randint(rank)
            improved = False
            
            # Try U
            old_val = U[ui, r]
            U[ui, r] = 1 - old_val if old_val in [0, 1] else 1
            new_err = count_errors(U, V, W, rank)
            if new_err < err_count:
                improved = True
            else:
                U[ui, r] = old_val
            
            if not improved:
                # Try V
                old_val = V[vj, r]
                V[vj, r] = 1 - old_val if old_val in [0, 1] else 1
                new_err = count_errors(U, V, W, rank)
                if new_err < err_count:
                    improved = True
                else:
                    V[vj, r] = old_val
            
            if not improved:
                # Try W
                old_val = W[wk, r]
                W[wk, r] = 1 - old_val if old_val in [0, 1] else 1
                new_err = count_errors(U, V, W, rank)
                if new_err < err_count:
                    improved = True
                else:
                    W[wk, r] = old_val
            
            if not improved:
                # Try small perturbations: add/subtract 1 to a random entry
                for _ in range(3):
                    mat = np.random.choice(['U', 'V', 'W'])
                    if mat == 'U':
                        row = np.random.randint(n * m)
                        col = np.random.randint(rank)
                        old_val = U[row, col]
                        U[row, col] = old_val + 1
                        new_err = count_errors(U, V, W, rank)
                        if new_err < err_count:
                            improved = True
                        else:
                            U[row, col] = old_val
                    elif mat == 'V':
                        row = np.random.randint(m * p)
                        col = np.random.randint(rank)
                        old_val = V[row, col]
                        V[row, col] = old_val + 1
                        new_err = count_errors(U, V, W, rank)
                        if new_err < err_count:
                            improved = True
                        else:
                            V[row, col] = old_val
                    else:
                        row = np.random.randint(p * n)
                        col = np.random.randint(rank)
                        old_val = W[row, col]
                        W[row, col] = old_val + 1
                        new_err = count_errors(U, V, W, rank)
                        if new_err < err_count:
                            improved = True
                        else:
                            W[row, col] = old_val
                    if improved:
                        break
        
        if count_errors(U, V, W, rank) == 0:
            return U, V, W
        return None
    
    def try_random_search(max_rank, num_attempts=100, max_iters=50):
        """
        Try to find a decomposition of rank max_rank using random initialization
        and local search. Returns (U, V, W) if found, else None.
        """
        nonlocal best_rank, best_U, best_V, best_W
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            # Random binary initialization
            U = np.random.randint(0, 2, (n * m, max_rank), dtype=np.int64)
            V = np.random.randint(0, 2, (m * p, max_rank), dtype=np.int64)
            W = np.random.randint(0, 2, (p * n, max_rank), dtype=np.int64)
            
            if check_decomp_vectorized(U, V, W, max_rank):
                if max_rank < best_rank:
                    best_rank = max_rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                return U, V, W
            
            # Try local search
            result = local_search(U, V, W, max_rank, max_iters=max_iters)
            if result is not None:
                U, V, W = result
                if max_rank < best_rank:
                    best_rank = max_rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                return U, V, W
        
        return None
    
    def try_structured_search(max_rank):
        """
        Try a structured approach: initialize with patterns that respect the tensor structure.
        For each (i,j,k), we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1.
        """
        nonlocal best_rank, best_U, best_V, best_W
        
        # Strategy: assign each triple to a unique column, but try to share columns
        # Start with schoolbook and try to merge columns
        
        U = np.zeros((n * m, max_rank), dtype=np.int64)
        V = np.zeros((m * p, max_rank), dtype=np.int64)
        W = np.zeros((p * n, max_rank), dtype=np.int64)
        
        # Initialize with a random permutation of columns
        col_assign = np.random.permutation(num_triples)[:max_rank]
        
        for r, triple_idx in enumerate(col_assign):
            i = idx_i[triple_idx]
            j = idx_j[triple_idx]
            k = idx_k[triple_idx]
            U[i * m + j, r] = 1
            V[j * p + k, r] = 1
            W[k * n + i, r] = 1
        
        # Now try to fill in remaining triples by adjusting existing columns
        # This is complex, so fall back to local search from this init
        result = local_search(U, V, W, max_rank, max_iters=100)
        if result is not None:
            U, V, W = result
            if max_rank < best_rank:
                best_rank = max_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            return U, V, W
        
        return None
    
    # Try ranks from 99 down to 50
    # First try some promising ranks with more attempts
    ranks_to_try = list(range(99, 49, -1))
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Try structured search first
        result = try_structured_search(rank)
        if result is not None:
            break
        
        # Try random search
        result = try_random_search(rank, num_attempts=50, max_iters=30)
        if result is not None:
            break
    
    # Return the best decomposition found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
