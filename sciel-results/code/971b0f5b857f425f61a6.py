import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 150.0 - 15.0  # 15-second safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T as a 3D array
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def compute_error(U, V, W):
        """Compute the max absolute error of the decomposition."""
        r = U.shape[1]
        T_rec = np.zeros((nm, mp, pn))
        for r_idx in range(r):
            T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        return np.max(np.abs(T_rec - T))
    
    def compute_l1_error(U, V, W):
        """Compute the L1 error of the decomposition."""
        r = U.shape[1]
        T_rec = np.zeros((nm, mp, pn))
        for r_idx in range(r):
            T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        return np.sum(np.abs(T_rec - T))
    
    def local_refine(U, V, W, max_iters=200, tol=1e-10):
        """
        Local refinement using coordinate descent on the entries.
        Try to minimize L1 error by adjusting individual entries.
        """
        if time.time() > deadline:
            return U, V, W
        
        current_U = U.copy()
        current_V = V.copy()
        current_W = W.copy()
        current_error = compute_l1_error(current_U, current_V, current_W)
        
        # Possible values to try: 0, ±1, ±0.5, ±2, ±0.25
        possible_values = [0, 0.25, 0.5, 1, 2, -0.25, -0.5, -1, -2]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try modifying a random entry in U, V, or W
            for _ in range(3):
                mat_choice = np.random.randint(0, 3)
                if mat_choice == 0:
                    row, col = np.random.randint(0, nm), np.random.randint(0, current_U.shape[1])
                    old_val = current_U[row, col]
                    for new_val in possible_values:
                        if abs(new_val - old_val) < 1e-12:
                            continue
                        current_U[row, col] = new_val
                        new_error = compute_l1_error(current_U, current_V, current_W)
                        if new_error < current_error - tol:
                            current_error = new_error
                            improved = True
                            break
                    if not improved:
                        current_U[row, col] = old_val
                elif mat_choice == 1:
                    row, col = np.random.randint(0, mp), np.random.randint(0, current_V.shape[1])
                    old_val = current_V[row, col]
                    for new_val in possible_values:
                        if abs(new_val - old_val) < 1e-12:
                            continue
                        current_V[row, col] = new_val
                        new_error = compute_l1_error(current_U, current_V, current_W)
                        if new_error < current_error - tol:
                            current_error = new_error
                            improved = True
                            break
                    if not improved:
                        current_V[row, col] = old_val
                else:
                    row, col = np.random.randint(0, pn), np.random.randint(0, current_W.shape[1])
                    old_val = current_W[row, col]
                    for new_val in possible_values:
                        if abs(new_val - old_val) < 1e-12:
                            continue
                        current_W[row, col] = new_val
                        new_error = compute_l1_error(current_U, current_V, current_W)
                        if new_error < current_error - tol:
                            current_error = new_error
                            improved = True
                            break
                    if not improved:
                        current_W[row, col] = old_val
            
            if not improved:
                break
            
            if current_error < 1e-8:
                break
        
        return current_U, current_V, current_W
    
    def try_structured_search(r, max_attempts=5000):
        """
        Try to find a rank-r decomposition using structured patterns
        that exploit the (i,j,k) index structure.
        """
        if time.time() > deadline:
            return None
        
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            np.random.seed(12345 + attempt * 7919)
            
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            
            strategy = attempt % 6
            
            if strategy == 0:
                # Dense random with small integers, snapped to 0.5
                U = np.random.randint(0, 4, (nm, r)) - 1.5
                V = np.random.randint(0, 4, (mp, r)) - 1.5
                W = np.random.randint(0, 4, (pn, r)) - 1.5
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                
            elif strategy == 1:
                # Sparse: each column has exactly one non-zero per matrix at structured positions
                for r_idx in range(r):
                    i = np.random.randint(0, n)
                    j = np.random.randint(0, m)
                    k = np.random.randint(0, p)
                    U[i * m + j, r_idx] = np.random.choice([-1, 1])
                    V[j * p + k, r_idx] = np.random.choice([-1, 1])
                    W[k * n + i, r_idx] = np.random.choice([-1, 1])
                    
            elif strategy == 2:
                # Grouped: each column covers multiple k values for fixed (i,j)
                for r_idx in range(r):
                    i = np.random.randint(0, n)
                    j = np.random.randint(0, m)
                    U[i * m + j, r_idx] = 1
                    k_count = np.random.randint(1, min(4, p))
                    k_values = np.random.choice(p, size=k_count, replace=False)
                    for k in k_values:
                        V[j * p + k, r_idx] = np.random.choice([-1, 1])
                        W[k * n + i, r_idx] = np.random.choice([-1, 1])
                        
            elif strategy == 3:
                # Hybrid: mix of single and grouped terms
                for r_idx in range(r):
                    if np.random.random() < 0.5:
                        i = np.random.randint(0, n)
                        j = np.random.randint(0, m)
                        k = np.random.randint(0, p)
                        U[i * m + j, r_idx] = np.random.choice([-1, 1])
                        V[j * p + k, r_idx] = np.random.choice([-1, 1])
                        W[k * n + i, r_idx] = np.random.choice([-1, 1])
                    else:
                        i = np.random.randint(0, n)
                        j = np.random.randint(0, m)
                        U[i * m + j, r_idx] = 1
                        k_count = np.random.randint(2, min(5, p))
                        k_values = np.random.choice(p, size=k_count, replace=False)
                        for k in k_values:
                            V[j * p + k, r_idx] = np.random.choice([-1, 1])
                            W[k * n + i, r_idx] = np.random.choice([-1, 1])
                            
            elif strategy == 4:
                # Block-diagonal structure: group by i
                i_blocks = [0, 1, 2]
                for r_idx in range(r):
                    i = i_blocks[r_idx % 3]
                    j = np.random.randint(0, m)
                    k = np.random.randint(0, p)
                    if np.random.random() < 0.3:
                        j = np.random.randint(0, m)
                    U[i * m + j, r_idx] = np.random.choice([-1, 1])
                    V[j * p + k, r_idx] = np.random.choice([-1, 1])
                    W[k * n + i, r_idx] = np.random.choice([-1, 1])
                    
            else:
                # Random with bias toward structured positions
                for r_idx in range(r):
                    if np.random.random() < 0.7:
                        i = np.random.randint(0, n)
                        j = np.random.randint(0, m)
                        k = np.random.randint(0, p)
                        U[i * m + j, r_idx] = np.random.choice([-1, 0, 1])
                        V[j * p + k, r_idx] = np.random.choice([-1, 0, 1])
                        W[k * n + i, r_idx] = np.random.choice([-1, 0, 1])
                    else:
                        u_row = np.random.randint(0, nm)
                        v_row = np.random.randint(0, mp)
                        w_row = np.random.randint(0, pn)
                        U[u_row, r_idx] = np.random.choice([-1, 0, 1])
                        V[v_row, r_idx] = np.random.choice([-1, 0, 1])
                        W[w_row, r_idx] = np.random.choice([-1, 0, 1])
            
            error = compute_l1_error(U, V, W)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if error < 1e-8:
                if compute_error(U, V, W) < 1e-8:
                    return (U, V, W)
            
            # Local refinement for promising candidates
            if error < 5.0 and attempt % 3 == 0:
                U_ref, V_ref, W_ref = local_refine(U, V, W, max_iters=50)
                refined_error = compute_l1_error(U_ref, V_ref, W_ref)
                
                if refined_error < best_error:
                    best_error = refined_error
                    best_U, best_V, best_W = U_ref.copy(), V_ref.copy(), W_ref.copy()
                
                if refined_error < 1e-8:
                    if compute_error(U_ref, V_ref, W_ref) < 1e-8:
                        return (U_ref, V_ref, W_ref)
        
        if best_U is not None:
            if compute_error(best_U, best_V, best_W) < 1e-8:
                return (best_U, best_V, best_W)
        
        return None
    
    def try_from_schoolbook_reduction(target_rank, max_iters=2000):
        """
        Start from schoolbook and try to reduce rank by merging terms.
        """
        if time.time() > deadline:
            return None
        
        if target_rank >= rank_sb:
            return (U_sb.copy(), V_sb.copy(), W_sb.copy())
        
        current_U = U_sb.copy()
        current_V = V_sb.copy()
        current_W = W_sb.copy()
        current_rank = current_U.shape[1]
        
        best_error = compute_l1_error(current_U, current_V, current_W)
        best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            if current_rank <= target_rank:
                break
            
            # Pick a column to try to eliminate
            col_to_eliminate = np.random.randint(0, current_rank)
            
            u_elim = current_U[:, col_to_eliminate]
            v_elim = current_V[:, col_to_eliminate]
            w_elim = current_W[:, col_to_eliminate]
            
            # Check if this column is already zero
            if np.allclose(u_elim, 0) and np.allclose(v_elim, 0) and np.allclose(w_elim, 0):
                keep = [c for c in range(current_rank) if c != col_to_eliminate]
                current_U = current_U[:, keep]
                current_V = current_V[:, keep]
                current_W = current_W[:, keep]
                current_rank = len(keep)
                continue
            
            # Try to distribute this contribution to other columns
            other_cols = [c for c in range(current_rank) if c != col_to_eliminate]
            if len(other_cols) == 0:
                break
            
            target_col = other_cols[np.random.randint(0, len(other_cols))]
            
            found_improvement = False
            for scale in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
                old_U = current_U[:, target_col].copy()
                old_V = current_V[:, target_col].copy()
                old_W = current_W[:, target_col].copy()
                
                current_U[:, target_col] += scale * u_elim
                current_V[:, target_col] += scale * v_elim
                current_W[:, target_col] += scale * w_elim
                
                current_U[:, col_to_eliminate] = 0
                current_V[:, col_to_eliminate] = 0
                current_W[:, col_to_eliminate] = 0
                
                new_error = compute_l1_error(current_U, current_V, current_W)
                
                if new_error < best_error + 1e-10:
                    best_error = new_error
                    best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
                    
                    keep = [c for c in range(current_rank) if c != col_to_eliminate]
                    current_U = current_U[:, keep]
                    current_V = current_V[:, keep]
                    current_W = current_W[:, keep]
                    current_rank = len(keep)
                    found_improvement = True
                    
                    if current_rank <= target_rank and new_error < 1e-8:
                        if compute_error(current_U, current_V, current_W) < 1e-8:
                            return (current_U, current_V, current_W)
                    break
                else:
                    current_U[:, target_col] = old_U
                    current_V[:, target_col] = old_V
                    current_W[:, target_col] = old_W
                    current_U[:, col_to_eliminate] = u_elim
                    current_V[:, col_to_eliminate] = v_elim
                    current_W[:, col_to_eliminate] = w_elim
            
            if not found_improvement:
                if it % 10 == 0:
                    U_ref, V_ref, W_ref = local_refine(current_U, current_V, current_W, max_iters=20)
                    refined_error = compute_l1_error(U_ref, V_ref, W_ref)
                    if refined_error < best_error:
                        best_error = refined_error
                        best_U, best_V, best_W = U_ref.copy(), V_ref.copy(), W_ref.copy()
                        current_U, current_V, current_W = U_ref, V_ref, W_ref
        
        if compute_error(best_U, best_V, best_W) < 1e-8:
            return (best_U, best_V, best_W)
        
        return None
    
    # Try to find the minimum rank
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # Try structured search for various ranks
    ranks_to_try = [50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 21]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        result = try_structured_search(r, max_attempts=3000)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = (U, V, W)
            break
        
        if r < rank_sb:
            result = try_from_schoolbook_reduction(r, max_iters=1000)
            if result is not None:
                U, V, W = result
                best_rank = r
                best_result = (U, V, W)
                break
    
    U, V, W = best_result
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
