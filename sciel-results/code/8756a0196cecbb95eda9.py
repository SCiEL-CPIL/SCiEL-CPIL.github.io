import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}, rank

    schoolbook_sol, schoolbook_rank = get_schoolbook()
    best_solution = schoolbook_sol
    best_rank = schoolbook_rank

    def compute_reconstruction(U, V, W, rank):
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            Rec += np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
        return Rec

    def is_valid(U, V, W, rank, atol=1e-9):
        Rec = compute_reconstruction(U, V, W, rank)
        return np.allclose(Rec, 1.0, atol=atol)

    def compute_error(U, V, W, rank):
        Rec = compute_reconstruction(U, V, W, rank)
        return np.max(np.abs(Rec - 1.0))

    def polish(U, V, W, rank, max_iters=20000):
        """High-precision polish: local search on half-integers with tighter tolerance."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            Rec += np.einsum('ij,jk,ki->ijk', U[:, r].reshape(n, m), V[:, r].reshape(m, p), W[:, r].reshape(p, n))
        
        current_err = np.max(np.abs(Rec - 1.0))
        if current_err < 1e-9:
            return U, V, W, True
        
        best_err = current_err
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        valid_vals = np.array([x / 2.0 for x in range(-12, 13)])
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try perturbing one entry
            choice = np.random.randint(3)
            
            if choice == 0:
                r = np.random.randint(rank)
                idx = np.random.randint(n * m)
                old_val = U[idx, r]
                new_val = np.random.choice(valid_vals)
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                i_idx = idx // m
                j_idx = idx % m
                v_col = V[:, r].reshape(m, p)
                w_col = W[:, r].reshape(p, n)
                delta_vec = delta * (v_col[j_idx, :] * w_col[:, i_idx])
                Rec[i_idx, j_idx, :] += delta_vec
                U[idx, r] = new_val
                new_err = np.max(np.abs(Rec - 1.0))
                if new_err < current_err:
                    current_err = new_err
                    if new_err < best_err:
                        best_err = new_err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        if best_err < 1e-9:
                            return best_U, best_V, best_W, True
                else:
                    Rec[i_idx, j_idx, :] -= delta_vec
                    U[idx, r] = old_val
                    
            elif choice == 1:
                r = np.random.randint(rank)
                idx = np.random.randint(m * p)
                old_val = V[idx, r]
                new_val = np.random.choice(valid_vals)
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                j_idx = idx // p
                k_idx = idx % p
                u_col = U[:, r].reshape(n, m)
                w_col = W[:, r].reshape(p, n)
                delta_vec = delta * (u_col[:, j_idx] * w_col[k_idx, :])
                Rec[:, j_idx, k_idx] += delta_vec
                V[idx, r] = new_val
                new_err = np.max(np.abs(Rec - 1.0))
                if new_err < current_err:
                    current_err = new_err
                    if new_err < best_err:
                        best_err = new_err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        if best_err < 1e-9:
                            return best_U, best_V, best_W, True
                else:
                    Rec[:, j_idx, k_idx] -= delta_vec
                    V[idx, r] = old_val
                    
            else:
                r = np.random.randint(rank)
                idx = np.random.randint(p * n)
                old_val = W[idx, r]
                new_val = np.random.choice(valid_vals)
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                k_idx = idx // n
                i_idx = idx % n
                u_col = U[:, r].reshape(n, m)
                v_col = V[:, r].reshape(m, p)
                delta_vec = delta * (u_col[i_idx, :] * v_col[:, k_idx])
                Rec[i_idx, :, k_idx] += delta_vec
                W[idx, r] = new_val
                new_err = np.max(np.abs(Rec - 1.0))
                if new_err < current_err:
                    current_err = new_err
                    if new_err < best_err:
                        best_err = new_err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        if best_err < 1e-9:
                            return best_U, best_V, best_W, True
                else:
                    Rec[i_idx, :, k_idx] -= delta_vec
                    W[idx, r] = old_val
        
        return best_U, best_V, best_W, best_err < 1e-9

    def tensor_lifting_init(rank, seed):
        """
        Tensor lifting: exploit structure by initializing from a 3x3x3 base 
        and extending to 3x3x5 using known algebraic patterns.
        """
        np.random.seed(seed)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # For 3x3x5, we can think of it as 3x3x3 + 3x3x2
        # Use a structured approach: group columns in pairs
        
        # Strategy: use rank-18 for the 3x3x3 part and rank-6 for the extra 2 columns
        # Total target: 24, but we search for lower
        
        for r in range(rank):
            # Determine which "block" this rank component belongs to
            # Use a pattern that respects the tensor structure
            
            # For U: 3x3 matrix per rank
            # For V: 3x5 matrix per rank  
            # For W: 5x3 matrix per rank
            
            # Sparse initialization with structure
            # Each rank component should have limited support
            
            # U: choose 2-4 entries
            nz_u = np.random.randint(2, 5)
            idx_u = np.random.choice(n * m, nz_u, replace=False)
            U[idx_u, r] = np.random.choice([-1.0, -0.5, 0.5, 1.0], nz_u)
            
            # V: choose 3-5 entries (more columns)
            nz_v = np.random.randint(3, 6)
            idx_v = np.random.choice(m * p, nz_v, replace=False)
            V[idx_v, r] = np.random.choice([-1.0, -0.5, 0.5, 1.0], nz_v)
            
            # W: choose 2-4 entries
            nz_w = np.random.randint(2, 5)
            idx_w = np.random.choice(p * n, nz_w, replace=False)
            W[idx_w, r] = np.random.choice([-1.0, -0.5, 0.5, 1.0], nz_w)
        
        return U, V, W

    def algebraic_init(rank, seed):
        """
        Use algebraic structure: the 3x3x3 multiplication tensor has rank 18.
        For 3x3x5, we can use a lifting approach.
        """
        np.random.seed(seed)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Known: 3x3x3 has rank 18 (Strassen-like)
        # For 3x3x5, lower bound is around 20-24
        
        # Initialize with a pattern that has some algebraic structure
        # Group ranks into blocks corresponding to tensor factors
        
        for r in range(rank):
            # Use a deterministic pattern with some randomness
            base_r = r % 18
            extra_r = r // 18
            
            # U component: 3x3
            # Use a structured pattern
            if base_r < 9:
                # Diagonal-like
                i = base_r % 3
                j = base_r // 3
                U[i * m + j, r] = 1.0
            elif base_r < 18:
                # Off-diagonal
                i = (base_r - 9) % 3
                j = ((base_r - 9) // 3 + 1) % 3
                U[i * m + j, r] = 0.5
            else:
                # Random
                idx_u = np.random.choice(n * m, 2, replace=False)
                U[idx_u, r] = np.random.choice([-0.5, 0.5], 2)
            
            # V component: 3x5
            if base_r < 9:
                j = base_r // 3
                k = base_r % 3
                V[j * p + k, r] = 1.0
            elif base_r < 18:
                j = (base_r - 9) % 3
                k = ((base_r - 9) // 3 + 1) % 3
                V[j * p + k, r] = 0.5
                # Add extra column contribution
                if extra_r > 0:
                    V[j * p + 3, r] = 0.5
            else:
                idx_v = np.random.choice(m * p, 3, replace=False)
                V[idx_v, r] = np.random.choice([-0.5, 0.5], 3)
            
            # W component: 5x3
            if base_r < 9:
                k = base_r % 3
                i = base_r // 3
                W[k * n + i, r] = 1.0
            elif base_r < 18:
                k = (base_r - 9) % 3
                i = ((base_r - 9) // 3 + 1) % 3
                W[k * n + i, r] = 0.5
                if extra_r > 0:
                    W[3 * n + i, r] = 0.5
            else:
                idx_w = np.random.choice(p * n, 2, replace=False)
                W[idx_w, r] = np.random.choice([-0.5, 0.5], 2)
        
        return U, V, W

    def search_for_solution(max_rank, max_search_time=110.0):
        nonlocal best_solution, best_rank
        search_start = time.time()
        
        # Try ranks from best_rank - 1 down to a lower bound
        # For 3x3x5, the rank is between 20 and 45
        # Known lower bound: 20 (from tensor rank theory)
        for rank in range(min(max_rank, best_rank - 1), 19, -1):
            if time.time() > search_start + max_search_time:
                break
            if rank >= best_rank:
                continue
            
            # Multiple restarts for this rank
            # Use different initialization strategies
            n_attempts_lifting = 30
            n_attempts_algebraic = 20
            
            for attempt in range(n_attempts_lifting):
                if time.time() > search_start + max_search_time:
                    break
                if time.time() > deadline:
                    return
                
                seed = 100 + rank * 1000 + attempt * 7
                U, V, W = tensor_lifting_init(rank, seed)
                
                # Quick check
                if is_valid(U, V, W, rank):
                    best_solution = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    return True
                
                # Run polish
                U_opt, V_opt, W_opt, success = polish(U, V, W, rank, max_iters=8000)
                if success:
                    if is_valid(U_opt, V_opt, W_opt, rank):
                        best_solution = {"U": U_opt, "V": V_opt, "W": W_opt}
                        best_rank = rank
                        return True
            
            for attempt in range(n_attempts_algebraic):
                if time.time() > search_start + max_search_time:
                    break
                if time.time() > deadline:
                    return
                
                seed = 500 + rank * 2000 + attempt * 11
                U, V, W = algebraic_init(rank, seed)
                
                if is_valid(U, V, W, rank):
                    best_solution = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    return True
                
                U_opt, V_opt, W_opt, success = polish(U, V, W, rank, max_iters=8000)
                if success:
                    if is_valid(U_opt, V_opt, W_opt, rank):
                        best_solution = {"U": U_opt, "V": V_opt, "W": W_opt}
                        best_rank = rank
                        return True
        
        return False

    search_for_solution(best_rank - 1)

    def to_lists(mat):
        return mat.tolist()

    return {
        "U": to_lists(best_solution["U"]),
        "V": to_lists(best_solution["V"]),
        "W": to_lists(best_solution["W"])
    }
