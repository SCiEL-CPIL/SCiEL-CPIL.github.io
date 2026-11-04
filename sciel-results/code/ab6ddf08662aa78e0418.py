import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

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

    # Valid values: -10 to 10 in steps of 0.5
    valid_vals = np.array([x / 2.0 for x in range(-20, 21)])
    
    # Precompute indices for fast access
    # U idx: i*m + j
    # V idx: j*p + k
    # W idx: k*n + i

    def compute_initial_rec(U, V, W, rank):
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            Rec += np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
        return Rec

    def max_err(Rec):
        return np.max(np.abs(Rec - 1.0))

    def local_search_discrete(U, V, W, rank, max_iters=50000):
        """
        Discrete local search to minimize max error, restricting values to valid_vals.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        Rec = compute_initial_rec(U, V, W, rank)
        current_err = max_err(Rec)
        
        if current_err < 1e-9:
            return U, V, W, True, current_err
            
        best_err = current_err
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_Rec = Rec.copy()
        
        # Neighbors in valid_vals
        # For a value x, neighbors are x-0.5, x+0.5
        
        for iter_ in range(max_iters):
            if time.time() > deadline:
                break
                
            # Pick random term and entry
            r = np.random.randint(0, rank)
            mat_type = np.random.randint(0, 3) # 0:U, 1:V, 2:W
            
            if mat_type == 0:
                idx = np.random.randint(0, n * m)
                i = idx // m
                j = idx % m
                old_val = U[idx, r]
                new_val = old_val + np.random.choice([-0.5, 0.5])
                # Ensure new_val is in valid range, snap to nearest valid if needed
                # For simplicity, just allow +/- 0.5 steps, check bounds
                if new_val < -10 or new_val > 10:
                    continue
                
                # Calculate delta error
                # Rec_ijk += delta * V_jk^r * W_ki^r
                delta = new_val - old_val
                # Update Rec
                # V row j, W col i
                v_row = V[j * p : j * p + p, r]
                w_col = W[0 : n * p : n, r] # W is (p*n, rank), index k*n + i. 
                # Wait, W index is k*n + i. So W[:, r] is vector of length p*n.
                # W[k, r] where k is 0..p-1? No, W shape is (p*n, rank).
                # W[idx, r] where idx = k*n + i.
                # So for fixed i, we access W[k*n + i, r] for k=0..p-1.
                w_vals = W[np.arange(p) * n + i, r]
                
                # Change in Rec for all k:
                # Rec[i, j, k] += delta * v_row[k] * w_vals[k]
                delta_rec = delta * (v_row * w_vals)
                
                # Apply
                Rec[i, j, :] += delta_rec
                new_err = max_err(Rec)
                
                U[idx, r] = new_val
                
                if new_err < best_err - 1e-12:
                    best_err = new_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_Rec = Rec.copy()
                elif new_err > current_err:
                    # Backtrack
                    Rec[i, j, :] -= delta_rec
                    U[idx, r] = old_val
                else:
                    # Accept (simulated annealing style? No, just hill climb with random restarts or simple acceptance if not worse)
                    # Let's stick to strict hill climbing to avoid getting lost
                    Rec[i, j, :] -= delta_rec
                    U[idx, r] = old_val
                current_err = max_err(Rec) # Recompute to be safe? Or track?
                # Tracking max err incrementally is hard if the max element changes.
                # Since dims are small (3x3x5=45 elements), recomputing max_err is cheap.

            elif mat_type == 1:
                idx = np.random.randint(0, m * p)
                j = idx // p
                k = idx % p
                old_val = V[idx, r]
                new_val = old_val + np.random.choice([-0.5, 0.5])
                if new_val < -10 or new_val > 10:
                    continue
                
                delta = new_val - old_val
                # Rec_ijk += delta * U_ij^r * W_ki^r
                u_val = U[i * m + j, r] # Wait, i varies!
                # For fixed j, k.
                # Rec[i, j, k] += delta * U[i, j]^r * W[k, i]^r
                # i ranges 0..n-1.
                
                # U[:, r] is (n*m,). Reshape (n, m).
                # U[i*m + j, r]
                u_vals = U[np.arange(n) * m + j, r]
                # W[k*n + i, r]
                w_vals = W[k * n : k * n + n, r]
                
                delta_rec = delta * (u_vals * w_vals)
                
                Rec[:, j, k] += delta_rec
                new_err = max_err(Rec)
                
                V[idx, r] = new_val
                
                if new_err < best_err - 1e-12:
                    best_err = new_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_Rec = Rec.copy()
                elif new_err > current_err:
                    Rec[:, j, k] -= delta_rec
                    V[idx, r] = old_val
                else:
                    Rec[:, j, k] -= delta_rec
                    V[idx, r] = old_val
                current_err = max_err(Rec)

            else: # mat_type == 2
                idx = np.random.randint(0, p * n)
                k = idx // n
                i = idx % n
                old_val = W[idx, r]
                new_val = old_val + np.random.choice([-0.5, 0.5])
                if new_val < -10 or new_val > 10:
                    continue
                
                delta = new_val - old_val
                # Rec_ijk += delta * U_ij^r * V_jk^r
                # For fixed k, i.
                # Rec[i, j, k] += delta * U[i, j]^r * V[j, k]^r
                # j ranges 0..m-1.
                
                u_vals = U[i * m : i * m + m, r]
                v_vals = V[np.arange(m) * p + k, r]
                
                delta_rec = delta * (u_vals * v_vals)
                
                Rec[i, :, k] += delta_rec
                new_err = max_err(Rec)
                
                W[idx, r] = new_val
                
                if new_err < best_err - 1e-12:
                    best_err = new_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_Rec = Rec.copy()
                elif new_err > current_err:
                    Rec[i, :, k] -= delta_rec
                    W[idx, r] = old_val
                else:
                    Rec[i, :, k] -= delta_rec
                    W[idx, r] = old_val
                current_err = max_err(Rec)
            
            if best_err < 1e-9:
                break

        # Final check
        # The best solution might not have integer/half-integer entries if we drifted?
        # We only moved by 0.5 steps, so if we started at valid values, we stay at valid values.
        # But wait, we start from random initialization which might not be valid?
        # We should ensure initialization is valid.
        
        if best_err < 1e-9:
            # Verify
            Rec_check = compute_initial_rec(best_U, best_V, best_W, rank)
            if np.allclose(Rec_check, 1.0, atol=1e-9):
                return best_U, best_V, best_W, True, best_err
        return best_U, best_V, best_W, False, best_err

    def random_valid_init(rank, seed):
        np.random.seed(seed)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Random valid values
        U[:, :] = np.random.choice(valid_vals, size=(n * m, rank))
        V[:, :] = np.random.choice(valid_vals, size=(m * p, rank))
        W[:, :] = np.random.choice(valid_vals, size=(p * n, rank))
        
        # Scale down to avoid huge errors initially?
        # Or start with small values
        # Let's try small values first
        small_vals = np.array([-1, -0.5, 0, 0.5, 1])
        U[:, :] = np.random.choice(small_vals, size=(n * m, rank))
        V[:, :] = np.random.choice(small_vals, size=(m * p, rank))
        W[:, :] = np.random.choice(small_vals, size=(p * n, rank))
        
        return U, V, W

    def search_for_solution(max_rank, max_search_time=100.0):
        nonlocal best_solution, best_rank
        search_start = time.time()
        
        # Try ranks from best_rank - 1 down to 20
        for rank in range(min(max_rank, best_rank - 1), 15, -1):
            if time.time() > search_start + max_search_time:
                break
            if rank >= best_rank:
                continue
            
            # Try multiple initializations
            n_attempts = 5
            for attempt in range(n_attempts):
                if time.time() > deadline:
                    return
                if time.time() > search_start + max_search_time:
                    break
                    
                seed = 1000 + rank * 100 + attempt
                U, V, W = random_valid_init(rank, seed)
                
                # Check if already valid (unlikely)
                Rec = compute_initial_rec(U, V, W, rank)
                if np.allclose(Rec, 1.0, atol=1e-9):
                    best_solution = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    return True
                
                # Local search
                U_opt, V_opt, W_opt, success, err = local_search_discrete(U, V, W, rank, max_iters=20000)
                if success:
                    best_solution = {"U": U_opt, "V": V_opt, "W": W_opt}
                    best_rank = rank
                    return True

        return False

    # Run search
    search_for_solution(best_rank - 1)

    def to_lists(mat):
        return mat.tolist()

    return {
        "U": to_lists(best_solution["U"]),
        "V": to_lists(best_solution["V"]),
        "W": to_lists(best_solution["W"])
    }
