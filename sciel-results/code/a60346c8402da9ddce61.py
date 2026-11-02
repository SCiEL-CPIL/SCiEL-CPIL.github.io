import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    total_entries = n * m * p  # 18
    
    start_time = time.time()
    deadline = 140  # safety margin
    
    # Precompute the tensor entries: T[u, v, w] = 1 if (i,j,k) valid
    # u = i*m + j, v = j*p + k, w = k*n + i
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                tensor_entries.append((u, v, w))
                
    # Precompute indices for fast ALS updates
    # For each u, list of (v, w, index_in_entries)
    u_indices = {u: [] for u in range(nm)}
    v_indices = {v: [] for v in range(mp)}
    w_indices = {w: [] for w in range(pn)}
    for idx, (u, v, w) in enumerate(tensor_entries):
        u_indices[u].append((v, w, idx))
        v_indices[v].append((u, w, idx))
        w_indices[w].append((u, v, idx))

    # Schoolbook decomposition (rank 18)
    rank_school = total_entries
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    for col, (u, v, w) in enumerate(tensor_entries):
        U_school[u, col] = 1
        V_school[v, col] = 1
        W_school[w, col] = 1
    
    def verify_solution(U, V, W, tol=1e-9):
        """Check if U, V, W give exact decomposition with half-integer entries."""
        # Check half-integer constraint
        if not np.allclose(U * 2, np.round(U * 2), atol=tol):
            return False
        if not np.allclose(V * 2, np.round(V * 2), atol=tol):
            return False
        if not np.allclose(W * 2, np.round(W * 2), atol=tol):
            return False
        
        # Check exact decomposition
        for (u, v, w) in tensor_entries:
            s = 0.0
            for r in range(U.shape[1]):
                s += U[u, r] * V[v, r] * W[w, r]
            if not np.isclose(s, 1.0, atol=tol):
                return False
        return True

    def calculate_error(U, V, W):
        err = 0.0
        for (u, v, w) in tensor_entries:
            s = 0.0
            for r in range(U.shape[1]):
                s += U[u, r] * V[v, r] * W[w, r]
            err += (s - 1.0) ** 2
        return err

    def als_decompose(U_init, V_init, W_init, R, max_iters=100, tol=1e-10):
        """Alternating least squares for tensor decomposition."""
        U = U_init.copy().astype(np.float64)
        V = V_init.copy().astype(np.float64)
        W = W_init.copy().astype(np.float64)
        
        prev_err = np.inf
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Update U rows
            for u in range(nm):
                if not u_indices[u]:
                    continue
                B = np.zeros((len(u_indices[u]), R))
                for k, (v, w, idx) in enumerate(u_indices[u]):
                    B[k, :] = V[v, :] * W[w, :]
                target = np.ones(len(u_indices[u]))
                U_u, _, _, _ = np.linalg.lstsq(B, target, rcond=None)
                U[u, :] = U_u
            
            # Update V rows
            for v in range(mp):
                if not v_indices[v]:
                    continue
                C = np.zeros((len(v_indices[v]), R))
                for k, (u, w, idx) in enumerate(v_indices[v]):
                    C[k, :] = U[u, :] * W[w, :]
                target = np.ones(len(v_indices[v]))
                V_v, _, _, _ = np.linalg.lstsq(C, target, rcond=None)
                V[v, :] = V_v
            
            # Update W rows
            for w in range(pn):
                if not w_indices[w]:
                    continue
                D = np.zeros((len(w_indices[w]), R))
                for k, (u, v, idx) in enumerate(w_indices[w]):
                    D[k, :] = U[u, :] * V[v, :]
                target = np.ones(len(w_indices[w]))
                W_w, _, _, _ = np.linalg.lstsq(D, target, rcond=None)
                W[w, :] = W_w
            
            # Calculate error
            err = calculate_error(U, V, W)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        U, V, W = best_U, best_V, best_W
        
        # Round to half-integers and verify
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        if verify_solution(U_round, V_round, W_round):
            return U_round, V_round, W_round
            
        # Local search around the rounded solution
        # Try flipping signs or changing 0 to 0.5 etc for a few entries
        # This is a heuristic local search
        best_local_err = calculate_error(U_round, V_round, W_round)
        best_local_sol = (U_round.copy(), V_round.copy(), W_round.copy())
        
        # If the error is small, try to fix it
        if best_local_err < 1.0:
            # Try perturbing individual entries
            # We limit the number of perturbations to avoid timeout
            max_perturbations = 20
            count = 0
            for _ in range(max_perturbations):
                if time.time() > deadline:
                    break
                
                # Pick a random matrix, row, col
                mat_name = np.random.choice(['U', 'V', 'W'])
                if mat_name == 'U':
                    mat = U_round
                    size = nm
                elif mat_name == 'V':
                    mat = V_round
                    size = mp
                else:
                    mat = W_round
                    size = pn
                
                row = np.random.randint(size)
                col = np.random.randint(R)
                
                # Try changing to a nearby half-integer
                old_val = mat[row, col]
                candidates = [old_val + 0.5, old_val - 0.5, -old_val, 0.0, 0.5, -0.5, 1.0, -1.0]
                candidates = [c for c in candidates if abs(c - old_val) > 1e-9]
                
                if not candidates:
                    continue
                
                new_val = np.random.choice(candidates)
                mat[row, col] = new_val
                
                # Check if this improves or solves
                # We don't want to verify full solution every time, just error
                err = calculate_error(U_round, V_round, W_round)
                
                if err < best_local_err:
                    best_local_err = err
                    best_local_sol = (U_round.copy(), V_round.copy(), W_round.copy())
                    if verify_solution(U_round, V_round, W_round):
                        return U_round, V_round, W_round
                else:
                    mat[row, col] = old_val
                    
                count += 1
                
        if verify_solution(best_local_sol[0], best_local_sol[1], best_local_sol[2]):
            return best_local_sol
            
        return None

    def try_rank(R, num_restarts=20, time_limit=10):
        """Try to find a rank-R decomposition with multiple random restarts."""
        start = time.time()
        for restart in range(num_restarts):
            if time.time() - start > time_limit or time.time() > deadline:
                break
            np.random.seed(restart * 1000 + R * 7)
            
            # Try different initialization strategies
            strategy = restart % 3
            
            if strategy == 0:
                # Random initialization
                U_init = np.random.randn(nm, R) * 0.5
                V_init = np.random.randn(mp, R) * 0.5
                W_init = np.random.randn(pn, R) * 0.5
            elif strategy == 1:
                # Sparse half-integer initialization
                U_init = np.zeros((nm, R))
                V_init = np.zeros((mp, R))
                W_init = np.zeros((pn, R))
                for r in range(R):
                    U_init[np.random.randint(nm), r] = np.random.choice([-1, -0.5, 0.5, 1])
                    V_init[np.random.randint(mp), r] = np.random.choice([-1, -0.5, 0.5, 1])
                    W_init[np.random.randint(pn), r] = np.random.choice([-1, -0.5, 0.5, 1])
            else:
                # Structured initialization from tensor entries
                U_init = np.zeros((nm, R))
                V_init = np.zeros((mp, R))
                W_init = np.zeros((pn, R))
                for r in range(min(R, len(tensor_entries))):
                    u, v, w = tensor_entries[r]
                    U_init[u, r] = 1
                    V_init[v, r] = 1
                    W_init[w, r] = 1
                if R > len(tensor_entries):
                    for r in range(len(tensor_entries), R):
                        U_init[np.random.randint(nm), r] = np.random.choice([-0.5, 0.5])
                        V_init[np.random.randint(mp), r] = np.random.choice([-0.5, 0.5])
                        W_init[np.random.randint(pn), r] = np.random.choice([-0.5, 0.5])
            
            res = als_decompose(U_init, V_init, W_init, R, max_iters=100)
            if res is not None:
                return res
        
        return None

    # Strategy: Try ranks from low to high
    # We start from 13, but maybe 14 or 15 is more likely?
    # Actually, if 18 is the answer, we should spend most time trying to beat 18.
    # If we find 17, score is 18/17 = 1.05.
    # If we find 16, score is 18/16 = 1.125.
    
    for target_rank in [13, 14, 15, 16, 17]:
        if time.time() > deadline - 10:
            break
        res = try_rank(target_rank, num_restarts=30, time_limit=10)
        if res is not None:
            U, V, W = res
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
