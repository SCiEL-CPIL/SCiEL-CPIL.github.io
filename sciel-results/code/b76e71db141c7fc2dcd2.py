import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

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

    def is_valid(U, V, W, rank, atol=1e-9):
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            u_reshaped = u.reshape(n, m)
            v_reshaped = v.reshape(m, p)
            w_reshaped = w.reshape(p, n)
            Rec += np.einsum('ij,jk,ki->ijk', u_reshaped, v_reshaped, w_reshaped, optimize=True)
        return np.allclose(Rec, 1.0, atol=atol)

    def compute_error(U, V, W, rank):
        """Compute the error in the tensor reconstruction."""
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            u_reshaped = u.reshape(n, m)
            v_reshaped = v.reshape(m, p)
            w_reshaped = w.reshape(p, n)
            Rec += np.einsum('ij,jk,ki->ijk', u_reshaped, v_reshaped, w_reshaped, optimize=True)
        return np.max(np.abs(Rec - 1.0))

    def local_optimize(U, V, W, rank, max_iters=1000, atol=1e-9):
        """Try to fix small errors by adjusting entries to nearest valid values."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Valid values: integers and half-integers
        valid_vals = [x / 2.0 for x in range(-10, 11)]
        
        best_err = compute_error(U, V, W, rank)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            if best_err < atol:
                break
            
            # Try flipping each entry to nearest valid value
            improved = False
            
            # Try adjusting U entries
            for idx in range(U.size):
                row, col = np.unravel_index(idx, U.shape)
                old_val = U[row, col]
                # Find nearest valid values
                nearest = min(valid_vals, key=lambda v: abs(v - old_val))
                if abs(nearest - old_val) < 1e-12:
                    continue
                U[row, col] = nearest
                err = compute_error(U, V, W, rank)
                if err < best_err - 1e-12:
                    best_err = err
                    improved = True
                else:
                    U[row, col] = old_val
            
            # Try adjusting V entries
            for idx in range(V.size):
                row, col = np.unravel_index(idx, V.shape)
                old_val = V[row, col]
                nearest = min(valid_vals, key=lambda v: abs(v - old_val))
                if abs(nearest - old_val) < 1e-12:
                    continue
                V[row, col] = nearest
                err = compute_error(U, V, W, rank)
                if err < best_err - 1e-12:
                    best_err = err
                    improved = True
                else:
                    V[row, col] = old_val
            
            # Try adjusting W entries
            for idx in range(W.size):
                row, col = np.unravel_index(idx, W.shape)
                old_val = W[row, col]
                nearest = min(valid_vals, key=lambda v: abs(v - old_val))
                if abs(nearest - old_val) < 1e-12:
                    continue
                W[row, col] = nearest
                err = compute_error(U, V, W, rank)
                if err < best_err - 1e-12:
                    best_err = err
                    improved = True
                else:
                    W[row, col] = old_val
            
            if not improved:
                break
        
        if best_err < atol:
            return U, V, W, True
        return U, V, W, False

    def structured_init(rank):
        """Create structured initialization based on partial patterns."""
        np.random.seed(42 + rank)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Use a mix of structured and random initialization
        for r in range(rank):
            # Pattern 1: Single 1 in each matrix
            if r % 3 == 0:
                i = (r // 3) % n
                j = (r // 3) % m
                k = (r // 3) % p
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
            # Pattern 2: Two 1s in U, one in V, one in W
            elif r % 3 == 1:
                i1 = (r // 3) % n
                i2 = (r // 3 + 1) % n
                j = (r // 3) % m
                k1 = (r // 3) % p
                k2 = (r // 3 + 1) % p
                U[i1 * m + j, r] = 1
                U[i2 * m + j, r] = 1
                V[j * p + k1, r] = 1
                W[k1 * n + i1, r] = 1
                W[k2 * n + i2, r] = -1
            # Pattern 3: Random small integers
            else:
                U[:, r] = np.random.randint(-1, 2, n * m)
                V[:, r] = np.random.randint(-1, 2, m * p)
                W[:, r] = np.random.randint(-1, 2, p * n)
        
        return U, V, W

    def search_for_solution(max_rank, max_search_time=60.0):
        nonlocal best_solution, best_rank
        search_start = time.time()
        
        for rank in range(min(max_rank, best_rank - 1), 15, -1):
            if time.time() > search_start + max_search_time:
                break
            if rank >= best_rank:
                continue
            
            # Try multiple structured initializations
            for attempt in range(5):
                if time.time() > search_start + max_search_time:
                    break
                if time.time() > deadline:
                    return
                
                np.random.seed(42 + rank * 100 + attempt)
                U, V, W = structured_init(rank)
                
                # Check if already valid
                if is_valid(U, V, W, rank):
                    best_solution = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    return True
                
                # Try local optimization
                U_opt, V_opt, W_opt, success = local_optimize(U, V, W, rank, max_iters=500)
                if success:
                    best_solution = {"U": U_opt, "V": V_opt, "W": W_opt}
                    best_rank = rank
                    return True
        
        return False

    # Try to find a better solution
    search_for_solution(best_rank - 1)

    def to_lists(mat):
        return mat.tolist()

    return {
        "U": to_lists(best_solution["U"]),
        "V": to_lists(best_solution["V"]),
        "W": to_lists(best_solution["W"])
    }
