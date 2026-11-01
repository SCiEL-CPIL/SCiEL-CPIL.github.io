"""Matrix multiplication 2x3x4 rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
import sys

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 12, 8
    
    # Build the target tensor entries
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.append((u_idx, v_idx, w_idx))
    
    num_entries = len(tensor_entries)  # 24
    
    # Precompute non-entries for verification
    entry_set = set(tensor_entries)
    non_entry_list = []
    for u in range(nm):
        for v in range(mp):
            for w in range(pn):
                if (u, v, w) not in entry_set:
                    non_entry_list.append((u, v, w))
    
    # Schoolbook fallback (rank 24)
    best_rank = n * m * p
    U_sb = [[0] * best_rank for _ in range(nm)]
    V_sb = [[0] * best_rank for _ in range(mp)]
    W_sb = [[0] * best_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    def verify_solution(U, V, W, rank):
        """Check if the decomposition is exact."""
        # Check tensor entries
        for (u, v, w) in tensor_entries:
            val = sum(U[u][r] * V[v][r] * W[w][r] for r in range(rank))
            if abs(val - 1.0) > 1e-10:
                return False
        # Check non-entries
        for (u, v, w) in non_entry_list:
            val = sum(U[u][r] * V[v][r] * W[w][r] for r in range(rank))
            if abs(val) > 1e-10:
                return False
        return True

    def snap_to_half_integers(mat):
        """Snap matrix entries to nearest half-integer."""
        return np.round(mat * 2) / 2.0

    def compute_error(U, V, W, rank):
        """Compute the total squared error of the tensor decomposition."""
        err = 0.0
        for (u, v, w) in tensor_entries:
            val = sum(U[u][r] * V[v][r] * W[w][r] for r in range(rank))
            err += (val - 1.0) ** 2
        for (u, v, w) in non_entry_list:
            val = sum(U[u][r] * V[v][r] * W[w][r] for r in range(rank))
            err += val ** 2
        return err

    def als_step(U, V, W, rank, max_iters=20):
        """Perform Alternating Least Squares to minimize error."""
        U = U.astype(np.float64).copy()
        V = V.astype(np.float64).copy()
        W = W.astype(np.float64).copy()
        
        u_indices = np.array([e[0] for e in tensor_entries])
        v_indices = np.array([e[1] for e in tensor_entries])
        w_indices = np.array([e[2] for e in tensor_entries])
        
        non_u = np.array([e[0] for e in non_entry_list])
        non_v = np.array([e[1] for e in non_entry_list])
        non_w = np.array([e[2] for e in non_entry_list])
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Solve for U
            # For each u, we want to minimize sum_v,w (sum_r U[u,r] V[v,r] W[w,r] - T[u,v,w])^2
            # This is a linear system for U[u, :].
            # Let A_u be a matrix where row (v,w) is vec(V[v,:] * W[w,:])
            # Then A_u * U[u,:] = T_u
            # But T_u depends on v,w.
            # Actually, for fixed V, W, the equation is linear in U[u, :].
            # sum_r U[u,r] * (V[v,r] * W[w,r]) = T[u,v,w]
            # Let B_u be a matrix of size (num_entries_for_u, rank)
            # But entries are specific (u,v,w).
            
            # Efficiently:
            # For each u in 0..nm-1:
            #   Collect all (v,w) such that (u,v,w) is in tensor_entries OR non_entries
            #   Build matrix M_u where M_u[eq, r] = V[v, r] * W[w, r]
            #   Build vector b_u where b_u[eq] = T[u,v,w] (1 if entry, 0 if non-entry)
            #   Solve M_u * U[u, :] = b_u
            
            for u in range(nm):
                # Find all equations involving u
                mask_entries = (u_indices == u)
                mask_non = (non_u == u)
                
                v_e = v_indices[mask_entries]
                w_e = w_indices[mask_entries]
                v_n = non_v[mask_non]
                w_n = non_w[mask_non]
                
                # Combine
                v_all = np.concatenate([v_e, v_n])
                w_all = np.concatenate([w_e, w_n])
                b_all = np.concatenate([np.ones(len(v_e)), np.zeros(len(v_n))])
                
                # Build M_u
                # M_u[i, r] = V[v_all[i], r] * W[w_all[i], r]
                # This can be done via broadcasting
                M_u = V[v_all, :] * W[w_all, :]
                
                # Solve least squares
                try:
                    U[u, :], residuals, rank_u, s = np.linalg.lstsq(M_u, b_all, rcond=None)
                except np.linalg.LinAlgError:
                    pass
            
            # Solve for V
            for v in range(mp):
                mask_entries = (v_indices == v)
                mask_non = (non_v == v)
                
                u_e = u_indices[mask_entries]
                w_e = w_indices[mask_entries]
                u_n = non_u[mask_non]
                w_n = non_w[mask_non]
                
                u_all = np.concatenate([u_e, u_n])
                w_all = np.concatenate([w_e, w_n])
                b_all = np.concatenate([np.ones(len(u_e)), np.zeros(len(u_n))])
                
                M_v = U[u_all, :] * W[w_all, :]
                
                try:
                    V[v, :], residuals, rank_v, s = np.linalg.lstsq(M_v, b_all, rcond=None)
                except np.linalg.LinAlgError:
                    pass
            
            # Solve for W
            for w in range(pn):
                mask_entries = (w_indices == w)
                mask_non = (non_w == w)
                
                u_e = u_indices[mask_entries]
                v_e = v_indices[mask_entries]
                u_n = non_u[mask_non]
                v_n = non_v[mask_non]
                
                u_all = np.concatenate([u_e, u_n])
                v_all = np.concatenate([v_e, v_n])
                b_all = np.concatenate([np.ones(len(u_e)), np.zeros(len(u_n))])
                
                M_w = U[u_all, :] * V[v_all, :]
                
                try:
                    W[w, :], residuals, rank_w, s = np.linalg.lstsq(M_w, b_all, rcond=None)
                except np.linalg.LinAlgError:
                    pass
        
        return U, V, W

    def local_search(U, V, W, rank, max_iters=100):
        """Local search on half-integer lattice to reduce error."""
        U = snap_to_half_integers(U)
        V = snap_to_half_integers(V)
        W = snap_to_half_integers(W)
        
        current_err = compute_error(U, V, W, rank)
        
        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            improved = False
            # Try flipping signs or adding/subtracting 0.5 to random entries
            # To be efficient, only try entries that contribute to high error
            # For now, random perturbations
            
            for _ in range(10): # 10 random moves per iteration
                # Pick a random matrix, row, col
                mat_choice = np.random.choice(3)
                if mat_choice == 0:
                    row = np.random.randint(nm)
                    col = np.random.randint(rank)
                    # Try adding 0.5
                    old_val = U[row, col]
                    U[row, col] += 0.5
                    new_err = compute_error(U, V, W, rank)
                    if new_err < current_err:
                        current_err = new_err
                        improved = True
                    else:
                        U[row, col] = old_val
                        
                elif mat_choice == 1:
                    row = np.random.randint(mp)
                    col = np.random.randint(rank)
                    old_val = V[row, col]
                    V[row, col] += 0.5
                    new_err = compute_error(U, V, W, rank)
                    if new_err < current_err:
                        current_err = new_err
                        improved = True
                    else:
                        V[row, col] = old_val
                        
                else:
                    row = np.random.randint(pn)
                    col = np.random.randint(rank)
                    old_val = W[row, col]
                    W[row, col] += 0.5
                    new_err = compute_error(U, V, W, rank)
                    if new_err < current_err:
                        current_err = new_err
                        improved = True
                    else:
                        W[row, col] = old_val
            
            if not improved:
                # Try larger jumps? Or break?
                # Let's try a few more random restarts of local search
                pass
            
            if current_err < 1e-10:
                break
        
        if verify_solution(U.tolist(), V.tolist(), W.tolist(), rank):
            return U.tolist(), V.tolist(), W.tolist()
        return None

    # Search for minimal rank
    # Theoretical lower bound is 18.
    for target_rank in range(18, best_rank):
        if time.time() > deadline - 10:
            break
        
        found = False
        np.random.seed(42 + target_rank)
        
        # Number of random attempts
        # We have ~130s left. Let's use 500 attempts per rank.
        # Each attempt: ALS (20 iters) + Local Search (100 iters * 10 moves)
        # This might be too slow.
        # Let's optimize:
        # 1. ALS for 10 iters.
        # 2. Snap.
        # 3. Local Search for 50 iters.
        
        num_attempts = 1000
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 2:
                break
            
            # Initialize with small random continuous values
            scale = 0.1
            U_init = np.random.normal(0, scale, size=(nm, target_rank))
            V_init = np.random.normal(0, scale, size=(mp, target_rank))
            W_init = np.random.normal(0, scale, size=(pn, target_rank))
            
            # ALS
            U_curr, V_curr, W_curr = als_step(U_init, V_init, W_init, target_rank, max_iters=10)
            
            # Snap
            U_snapped = snap_to_half_integers(U_curr)
            V_snapped = snap_to_half_integers(V_curr)
            W_snapped = snap_to_half_integers(W_curr)
            
            # Check if already valid
            if verify_solution(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), target_rank):
                best_rank = target_rank
                best_result = {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
                found = True
                break
            
            # Local Search
            result = local_search(U_snapped, V_snapped, W_snapped, target_rank, max_iters=50)
            if result is not None:
                U_res, V_res, W_res = result
                best_rank = target_rank
                best_result = {"U": U_res, "V": V_res, "W": W_res}
                found = True
                break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
