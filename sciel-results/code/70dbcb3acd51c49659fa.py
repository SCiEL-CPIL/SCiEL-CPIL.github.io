import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

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

    def is_valid(U, V, W, rank, atol=1e-7):
        Rec = compute_reconstruction(U, V, W, rank)
        return np.allclose(Rec, 1.0, atol=atol)

    def compute_error(U, V, W, rank):
        Rec = compute_reconstruction(U, V, W, rank)
        return np.max(np.abs(Rec - 1.0))

    def gradient_optimize(U, V, W, rank, max_iters=2000, lr_init=0.05, lr_end=0.001):
        """
        Gradient-based continuous optimization to minimize max reconstruction error.
        Uses subgradient of max|Rec - 1|.
        """
        U = U.copy().astype(float)
        V = V.copy().astype(float)
        W = W.copy().astype(float)
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = np.inf
        
        # Precompute for efficiency
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Learning rate schedule
            lr = lr_init * (lr_end / lr_init) ** (it / max_iters)
            
            # Compute reconstruction
            Rec = np.zeros((n, m, p))
            for r in range(rank):
                u = U[:, r].reshape(n, m)
                v = V[:, r].reshape(m, p)
                w = W[:, r].reshape(p, n)
                Rec += np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
            
            err = np.max(np.abs(Rec - 1.0))
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if best_err < 1e-7:
                    break
            
            # Subgradient of max|Rec - 1| w.r.t. each parameter
            # d/dU[i,j,r] max|Rec-1| = sign(Rec[i,j,k*]-1) * V[j,k*,r] * W[k*,i,r] for argmax k*
            # Find the entry with max absolute error
            err_vals = np.abs(Rec - 1.0)
            max_idx = np.unravel_index(np.argmax(err_vals), (n, m, p))
            i_max, j_max, k_max = max_idx
            
            # Sign of the error at the max point
            sign_err = np.sign(Rec[i_max, j_max, k_max] - 1.0)
            
            # Gradients: for each r, compute contribution
            for r in range(rank):
                u = U[:, r].reshape(n, m)
                v = V[:, r].reshape(m, p)
                w = W[:, r].reshape(p, n)
                
                # dU[i,j,r] = sign_err * v[j,k] * w[k,i]
                grad_u = sign_err * v[j_max, k_max] * w[k_max, i_max]
                # dV[j,k,r] = sign_err * u[i,j] * w[k,i]
                grad_v = sign_err * u[i_max, j_max] * w[k_max, i_max]
                # dW[k,i,r] = sign_err * u[i,j] * v[j,k]
                grad_w = sign_err * u[i_max, j_max] * v[j_max, k_max]
                
                U[i_max * m + j_max, r] -= lr * grad_u
                V[j_max * p + k_max, r] -= lr * grad_v
                W[k_max * n + i_max, r] -= lr * grad_w
        
        return best_U, best_V, best_W, best_err

    def snap_to_half_integers(U, V, W, rank):
        """Snap values to nearest half-integer and verify."""
        U_snapped = np.round(U * 2) / 2
        V_snapped = np.round(V * 2) / 2
        W_snapped = np.round(W * 2) / 2
        
        if is_valid(U_snapped, V_snapped, W_snapped, rank):
            return U_snapped, V_snapped, W_snapped, True
        return None, None, None, False

    def structured_init(rank, seed):
        """
        Initialize with structure inspired by known low-rank decompositions.
        For 3x3x5, use a block structure.
        """
        np.random.seed(seed)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Try to initialize with patterns that have some algebraic structure
        # Each rank component should have sparse support
        for r in range(rank):
            # Random sparse with values in {-1, -0.5, 0.5, 1}
            nnz_u = np.random.randint(2, 5)
            nnz_v = np.random.randint(2, 5)
            nnz_w = np.random.randint(2, 5)
            
            idx_u = np.random.choice(n * m, nnz_u, replace=False)
            idx_v = np.random.choice(m * p, nnz_v, replace=False)
            idx_w = np.random.choice(p * n, nnz_w, replace=False)
            
            vals = np.array([-1.0, -0.5, 0.5, 1.0])
            U[idx_u, r] = np.random.choice(vals, nnz_u)
            V[idx_v, r] = np.random.choice(vals, nnz_v)
            W[idx_w, r] = np.random.choice(vals, nnz_w)
        
        return U, V, W

    def search_for_solution(max_rank, max_search_time=110.0):
        nonlocal best_solution, best_rank
        search_start = time.time()
        
        # Known lower bound for 3x3x5 is around 21. Try from 25 down.
        min_rank_to_try = 20
        
        for rank in range(min(max_rank, best_rank - 1), min_rank_to_try - 1, -1):
            if time.time() > search_start + max_search_time:
                break
            if rank >= best_rank:
                continue
            
            # Multiple restarts
            n_attempts = 30
            for attempt in range(n_attempts):
                if time.time() > search_start + max_search_time:
                    break
                if time.time() > deadline:
                    return
                
                seed = 42 + rank * 1000 + attempt * 7
                U, V, W = structured_init(rank, seed)
                
                # Quick validity check
                if is_valid(U, V, W, rank):
                    best_solution = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    return True
                
                # Gradient optimization
                U_opt, V_opt, W_opt, err = gradient_optimize(U, V, W, rank, max_iters=500)
                
                if err < 1e-5:
                    # Try snapping to half-integers
                    U_sn, V_sn, W_sn, snap_ok = snap_to_half_integers(U_opt, V_opt, W_opt, rank)
                    if snap_ok:
                        best_solution = {"U": U_sn, "V": V_sn, "W": W_sn}
                        best_rank = rank
                        return True
                    else:
                        # Use the continuous solution as-is if error is small enough
                        # But we need half-integers... so this won't work directly
                        pass
                
                # If gradient optimization got close, try local discrete search around the solution
                if err < 0.1:
                    # Local discrete search
                    U_local = U_opt.copy()
                    V_local = V_opt.copy()
                    W_local = W_opt.copy()
                    
                    # Snap to half-integers as starting point for local search
                    U_local = np.round(U_local * 2) / 2
                    V_local = np.round(V_local * 2) / 2
                    W_local = np.round(W_local * 2) / 2
                    
                    # Local discrete optimization
                    for local_iter in range(200):
                        if time.time() > deadline:
                            break
                        
                        # Compute current error
                        Rec = compute_reconstruction(U_local, V_local, W_local, rank)
                        curr_err = np.max(np.abs(Rec - 1.0))
                        
                        if curr_err < 1e-7:
                            best_solution = {"U": U_local, "V": V_local, "W": W_local}
                            best_rank = rank
                            return True
                        
                        # Find worst entry
                        err_vals = np.abs(Rec - 1.0)
                        max_idx = np.unravel_index(np.argmax(err_vals), (n, m, p))
                        i_max, j_max, k_max = max_idx
                        
                        # Try adjusting one parameter
                        choice = np.random.randint(3)
                        r = np.random.randint(rank)
                        
                        if choice == 0:
                            idx = i_max * m + j_max
                            old = U_local[idx, r]
                            for delta in [-0.5, 0.5]:
                                new_val = old + delta
                                U_local[idx, r] = new_val
                                new_err = compute_error(U_local, V_local, W_local, rank)
                                if new_err < curr_err:
                                    curr_err = new_err
                                    break
                                else:
                                    U_local[idx, r] = old
                        elif choice == 1:
                            idx = j_max * p + k_max
                            old = V_local[idx, r]
                            for delta in [-0.5, 0.5]:
                                new_val = old + delta
                                V_local[idx, r] = new_val
                                new_err = compute_error(U_local, V_local, W_local, rank)
                                if new_err < curr_err:
                                    curr_err = new_err
                                    break
                                else:
                                    V_local[idx, r] = old
                        else:
                            idx = k_max * n + i_max
                            old = W_local[idx, r]
                            for delta in [-0.5, 0.5]:
                                new_val = old + delta
                                W_local[idx, r] = new_val
                                new_err = compute_error(U_local, V_local, W_local, rank)
                                if new_err < curr_err:
                                    curr_err = new_err
                                    break
                                else:
                                    W_local[idx, r] = old
                    
                    if curr_err < 1e-7:
                        best_solution = {"U": U_local, "V": V_local, "W": W_local}
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
