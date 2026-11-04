"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _to_lists(U, V, W):
    def snap(x):
        # Snap to nearest half-integer
        return round(x * 2) / 2.0
    
    U_list = [[snap(x) for x in row] for row in U.tolist()]
    V_list = [[snap(x) for x in row] for row in V.tolist()]
    W_list = [[snap(x) for x in row] for row in W.tolist()]
    
    return {
        "U": U_list,
        "V": V_list,
        "W": W_list
    }

def _verify(U, V, W, n, m, p, tol=1e-8):
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    rank = U.shape[1]
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    # Compute T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # Note: W is indexed by k, i in the tensor definition W[k*n+i, r]
    # So W_t[k, i, r]
    
    # Use einsum for speed
    # U_t[i, j, r]
    # V_t[j, k, r]
    # W_t[k, i, r] -> We need to align indices.
    # Let's transpose W_t to (i, k, r) to match i, j, k order?
    # W_t is (p, n, rank) which is (k, i, r).
    # We want T[i, j, k].
    # Term is U[i, j, r] * V[j, k, r] * W[k, i, r].
    # einsum: 'ijr,jkr,kir->ijk'
    
    try:
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    except Exception:
        return False, 1.0
        
    error = np.max(np.abs(T - 1.0))
    return error < tol, error

def _local_search(U, V, W, n, m, p, max_iters=2000, deadline=None):
    rank = U.shape[1]
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    # Initial error
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    err = T - 1.0
    max_err = np.max(np.abs(err))
    
    if max_err < 1e-8:
        return U, V, W, max_err
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Find worst index
        worst_idx = np.unravel_index(np.argmax(np.abs(err)), err.shape)
        wi, wj, wk = worst_idx
        werr = err[wi, wj, wk]
        
        # Try to adjust one entry in U, V, or W
        # Pick a random column r
        r = np.random.randint(rank)
        
        # Calculate gradient-like direction
        # d/dU[wi, wj, r] T[wi, wj, wk] = V[wj, wk, r] * W[wk, wi, r]
        # We want to reduce |werr|, so we move U in direction -sign(werr) * grad
        
        grad_u = V_t[wj, wk, r] * W_t[wk, wi, r]
        if grad_u != 0:
            # Step size
            step = 0.5 # Try half-integer steps
            # We are on a grid of 0.5. 
            # Current value is U[wi, wj, r].
            # We want to change it to reduce error.
            # If werr > 0, we want to decrease T, so if grad_u > 0, decrease U.
            # If werr < 0, we want to increase T, so if grad_u > 0, increase U.
            
            # Simple heuristic: try adding/subtracting 0.5
            old_u = U_t[wi, wj, r]
            
            # Try a few candidate moves
            candidates = [old_u + 0.5, old_u - 0.5, old_u + 1.0, old_u - 1.0]
            
            best_candidate = old_u
            best_candidate_err = max_err + 1.0 # Must improve global max err or at least local
            
            # Actually, improving the local error might not improve the global max error.
            # But it's a good heuristic.
            
            for c in candidates:
                # Snap to half-integer
                c_snap = round(c * 2) / 2.0
                if c_snap == old_u:
                    continue
                U_t[wi, wj, r] = c_snap
                T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                new_err = np.max(np.abs(T - 1.0))
                if new_err < best_candidate_err:
                    best_candidate_err = new_err
                    best_candidate = c_snap
                U_t[wi, wj, r] = old_u # Restore
            
            if best_candidate != old_u:
                U_t[wi, wj, r] = best_candidate
                T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                err = T - 1.0
                max_err = np.max(np.abs(err))
                if max_err < 1e-8:
                    return U, V, W, max_err
                continue # Restart loop with new error
        
        # If no move in U improved, try V or W similarly
        # To keep it fast, we only try U in this simple version. 
        # A more robust version would try all three.
        
        # If we didn't make a move, we might be stuck. 
        # Add random perturbation
        if np.random.random() < 0.1:
            # Random restart of a few entries
            for _ in range(3):
                idx_u = np.random.randint(n*m)
                r = np.random.randint(rank)
                U[idx_u, r] = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
                T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                err = T - 1.0
                max_err = np.max(np.abs(err))
                if max_err < 1e-8:
                    return U, V, W, max_err

    return U, V, W, max_err

def _local_search_full(U, V, W, n, m, p, max_iters=1000, deadline=None):
    """Improved local search that tries moves in U, V, and W."""
    rank = U.shape[1]
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    # Initial error
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    err = T - 1.0
    max_err = np.max(np.abs(err))
    
    if max_err < 1e-8:
        return U, V, W, max_err
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Find worst index
        worst_idx = np.unravel_index(np.argmax(np.abs(err)), err.shape)
        wi, wj, wk = worst_idx
        werr = err[wi, wj, wk]
        
        # Pick a random column r
        r = np.random.randint(rank)
        
        best_improvement = 0
        best_move = None  # (matrix, index, new_value)
        
        # Try moving U[wi, wj, r]
        grad_u = V_t[wj, wk, r] * W_t[wk, wi, r]
        if grad_u != 0:
            old_u = U_t[wi, wj, r]
            for delta in [-1.0, -0.5, 0.5, 1.0]:
                c = old_u + delta
                c_snap = round(c * 2) / 2.0
                if c_snap == old_u:
                    continue
                U_t[wi, wj, r] = c_snap
                T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                new_err = np.max(np.abs(T - 1.0))
                if new_err < max_err - best_improvement:
                    best_improvement = max_err - new_err
                    best_move = ('U', (wi, wj, r), c_snap)
                U_t[wi, wj, r] = old_u
        
        # Try moving V[wj, wk, r]
        grad_v = U_t[wi, wj, r] * W_t[wk, wi, r]
        if grad_v != 0:
            old_v = V_t[wj, wk, r]
            for delta in [-1.0, -0.5, 0.5, 1.0]:
                c = old_v + delta
                c_snap = round(c * 2) / 2.0
                if c_snap == old_v:
                    continue
                V_t[wj, wk, r] = c_snap
                T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                new_err = np.max(np.abs(T - 1.0))
                if new_err < max_err - best_improvement:
                    best_improvement = max_err - new_err
                    best_move = ('V', (wj, wk, r), c_snap)
                V_t[wj, wk, r] = old_v
        
        # Try moving W[wk, wi, r]
        grad_w = U_t[wi, wj, r] * V_t[wj, wk, r]
        if grad_w != 0:
            old_w = W_t[wk, wi, r]
            for delta in [-1.0, -0.5, 0.5, 1.0]:
                c = old_w + delta
                c_snap = round(c * 2) / 2.0
                if c_snap == old_w:
                    continue
                W_t[wk, wi, r] = c_snap
                T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                new_err = np.max(np.abs(T - 1.0))
                if new_err < max_err - best_improvement:
                    best_improvement = max_err - new_err
                    best_move = ('W', (wk, wi, r), c_snap)
                W_t[wk, wi, r] = old_w
        
        if best_move is not None:
            matrix_name, idx, new_val = best_move
            if matrix_name == 'U':
                U_t[idx[0], idx[1], idx[2]] = new_val
            elif matrix_name == 'V':
                V_t[idx[0], idx[1], idx[2]] = new_val
            elif matrix_name == 'W':
                W_t[idx[0], idx[1], idx[2]] = new_val
            
            T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            err = T - 1.0
            max_err = np.max(np.abs(err))
            if max_err < 1e-8:
                return U, V, W, max_err
        else:
            # No improving move found; random perturbation
            if np.random.random() < 0.15:
                for _ in range(3):
                    which = np.random.choice(['U', 'V', 'W'])
                    if which == 'U':
                        i, j, r = np.random.randint(n), np.random.randint(m), np.random.randint(rank)
                        U_t[i, j, r] = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
                    elif which == 'V':
                        j, k, r = np.random.randint(m), np.random.randint(p), np.random.randint(rank)
                        V_t[j, k, r] = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
                    else:
                        k, i, r = np.random.randint(p), np.random.randint(n), np.random.randint(rank)
                        W_t[k, i, r] = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
                    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                    err = T - 1.0
                    max_err = np.max(np.abs(err))
                    if max_err < 1e-8:
                        return U, V, W, max_err

    return U, V, W, max_err

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with schoolbook
    U0, V0, W0 = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = _to_lists(U0, V0, W0)
    
    # Try to find lower rank decompositions
    # We search for rank R where sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1
    # Rank 36 is schoolbook.
    # We try to go down to 25.
    
    for target_rank in range(best_rank - 1, 24, -1):
        if time.time() > deadline - 10:
            break
        
        # Multiple random restarts
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random half-integers
            # Sparse initialization might be better?
            U = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n * m, target_rank))
            V = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m * p, target_rank))
            W = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(p * n, target_rank))
            
            # Run improved local search that tries U, V, and W
            U, V, W, err = _local_search_full(U, V, W, n, m, p, max_iters=800, deadline=deadline - 2)
            
            if err < 1e-8:
                # Verify and return
                valid, _ = _verify(U, V, W, n, m, p)
                if valid:
                    best_rank = target_rank
                    best_result = _to_lists(U, V, W)
                    return best_result
            
            # If we found a solution with error close to 0 but not exact, snap and check
            if err < 0.1:
                U_snap = np.round(U * 2) / 2.0
                V_snap = np.round(V * 2) / 2.0
                W_snap = np.round(W * 2) / 2.0
                valid, _ = _verify(U_snap, V_snap, W_snap, n, m, p)
                if valid:
                    best_rank = target_rank
                    best_result = _to_lists(U_snap, V_snap, W_snap)
                    return best_result
    
    return best_result
# EVOLVE-BLOCK-END
