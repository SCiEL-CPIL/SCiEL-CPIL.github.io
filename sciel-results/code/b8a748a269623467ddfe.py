import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    total_entries = n * m * p
    
    # Precompute indices for vectorized verification
    idx_u = np.array([i * m + j for i in range(n) for j in range(m) for k in range(p)])
    idx_v = np.array([j * p + k for i in range(n) for j in range(m) for k in range(p)])
    idx_w = np.array([k * n + i for i in range(n) for j in range(m) for k in range(p)])
    
    # Precompute the tensor structure for fast verification
    # T[i,j,k] = 1 for all valid (i,j,k)
    # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    def verify_solution(U, V, W, tol=1e-9):
        """Fast verification using precomputed index arrays"""
        R = U.shape[1]
        U_sel = U[idx_u, :]
        V_sel = V[idx_v, :]
        W_sel = W[idx_w, :]
        res = np.sum(U_sel * V_sel * W_sel, axis=1)
        return np.allclose(res, np.ones(total_entries), atol=tol, rtol=0)
    
    def verify_solution_fast(U, V, W):
        """Even faster verification - returns the max error"""
        R = U.shape[1]
        U_sel = U[idx_u, :]
        V_sel = V[idx_v, :]
        W_sel = W[idx_w, :]
        res = np.sum(U_sel * V_sel * W_sel, axis=1)
        return np.max(np.abs(res - 1.0))
    
    def verify_half_integer(U, V, W):
        """Check if all entries are integers or half-integers"""
        for mat in [U, V, W]:
            doubled = mat * 2
            if not np.allclose(doubled, np.round(doubled), atol=1e-9):
                return False
        return True
    
    def get_schoolbook():
        """Return the trivial rank-30 decomposition"""
        R = n * m * p
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, r] = 1.0
                    V[j*p+k, r] = 1.0
                    W[k*n+i, r] = 1.0
                    r += 1
        return U, V, W
    
    def get_structured_init(R, seed=0):
        """
        Initialize using block structure: partition into 2x2x2 Strassen blocks
        and 1x2x2 auxiliary blocks.
        
        For 2x3x5, we can think of this as:
        - Row dimension 2: split into {0} and {1}
        - Col dimension 3: split into {0,1} and {2} 
        - K dimension 5: split into {0,1}, {2,3}, and {4}
        
        This gives us overlapping 2x2x2 sub-tensors that can use Strassen's 7-rank decomposition.
        """
        rng = np.random.RandomState(seed)
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        
        # Strassen's 7 patterns for 2x2x2:
        # Each pattern is (u_pattern, v_pattern, w_pattern) where each is a 2x2 matrix
        # The rank-1 term is outer product of these patterns
        strassen_2x2x2 = [
            # (U_block, V_block, W_block)
            ([[1, 0], [0, 0]], [[1, 0], [0, 0]], [[1, 0], [0, 0]]),  # a00*b00
            ([[1, 0], [0, 0]], [[0, 0], [0, 1]], [[0, 0], [0, 1]]),  # a00*b11
            ([[0, 0], [1, 0]], [[0, 1], [0, 0]], [[0, 1], [0, 0]]),  # a10*b01
            ([[0, 1], [0, 0]], [[0, 0], [1, 0]], [[1, 0], [0, 0]]),  # a01*b10
            ([[0, 0], [0, 1]], [[0, 0], [0, 1]], [[0, 0], [0, 1]]),  # a11*b11
            ([[1, 1], [0, 0]], [[0, 0], [1, 0]], [[1, 0], [0, 0]]),  # (a00+a01)*b10
            ([[0, 0], [1, 1]], [[0, 1], [0, 0]], [[0, 1], [0, 0]]),  # (a10+a11)*b01
        ]
        
        # Define block assignments for 2x3x5
        # We'll create a set of "virtual" 2x2x2 blocks
        blocks = []
        
        # Block type 1: Full 2x2x2 blocks using cols 0,1 and k's 0,1 or 2,3
        for i_off in [0]:  # Row offset (only 0 since n=2)
            for j_off in [0]:  # Col offset for first pair
                for k_off in [0, 2]:  # K offset for pairs
                    blocks.append({
                        'type': 'full_2x2x2',
                        'rows': [i_off, i_off+1],
                        'cols': [j_off, j_off+1],
                        'ks': [k_off, k_off+1]
                    })
        
        # Block type 2: 2x1x2 blocks using col 2
        for k_off in [0, 2]:
            blocks.append({
                'type': '2x1x2',
                'rows': [0, 1],
                'cols': [2],
                'ks': [k_off, k_off+1]
            })
        
        # Block type 3: 2x2x1 blocks using k=4
        for j_off in [0]:
            blocks.append({
                'type': '2x2x1',
                'rows': [0, 1],
                'cols': [j_off, j_off+1],
                'ks': [4]
            })
        
        # Assign ranks to blocks
        # Each full 2x2x2 block needs 7 ranks (Strassen)
        # Each 2x1x2 block needs 4 ranks (trivial)
        # Each 2x2x1 block needs 4 ranks (trivial)
        
        rank_assignments = []
        current_rank = 0
        
        for block in blocks:
            if block['type'] == 'full_2x2x2':
                for s in range(7):
                    rank_assignments.append((block, s))
                    current_rank += 1
            elif block['type'] == '2x1x2':
                for s in range(4):
                    rank_assignments.append((block, s))
                    current_rank += 1
            elif block['type'] == '2x2x1':
                for s in range(4):
                    rank_assignments.append((block, s))
                    current_rank += 1
        
        # If we have more ranks than needed, add random ones
        while len(rank_assignments) < R:
            # Add a random sparse rank
            rank_assignments.append(('random', 0))
        
        # If we have fewer, truncate
        rank_assignments = rank_assignments[:R]
        
        # Now fill in the matrices
        for r, (block, s) in enumerate(rank_assignments):
            if block == 'random':
                # Random sparse initialization
                U[rng.randint(0, nm), r] = rng.choice([-1.0, 1.0])
                V[rng.randint(0, mp), r] = rng.choice([-1.0, 1.0])
                W[rng.randint(0, pn), r] = rng.choice([-1.0, 1.0])
                continue
            
            if block['type'] == 'full_2x2x2':
                rows = block['rows']
                cols = block['cols']
                ks = block['ks']
                
                # Get Strassen pattern s
                U_pat, V_pat, W_pat = strassen_2x2x2[s]
                
                # Map to actual indices
                for ri, row in enumerate(rows):
                    for ci, col in enumerate(cols):
                        for ki, k in enumerate(ks):
                            U_val = U_pat[ri][ci]
                            V_val = V_pat[ci][ki]
                            W_val = W_pat[ki][ri]
                            
                            if U_val != 0:
                                U[row*m + col, r] += U_val
                            if V_val != 0:
                                V[col*p + k, r] += V_val
                            if W_val != 0:
                                W[k*n + row, r] += W_val
            
            elif block['type'] == '2x1x2':
                rows = block['rows']
                col = block['cols'][0]
                ks = block['ks']
                
                # Trivial 4-rank decomposition for 2x1x2
                # We need: for each (i, k), U[i, r] * V[k, r] * W[i, r] = 1
                # Use 4 ranks: (i,k) pairs
                i_idx = s // 2
                k_idx = s % 2
                
                row = rows[i_idx]
                k = ks[k_idx]
                
                U[row*m + col, r] = 1.0
                V[col*p + k, r] = 1.0
                W[k*n + row, r] = 1.0
            
            elif block['type'] == '2x2x1':
                rows = block['rows']
                cols = block['cols']
                k = block['ks'][0]
                
                # Trivial 4-rank decomposition for 2x2x1
                i_idx = s // 2
                j_idx = s % 2
                
                row = rows[i_idx]
                col = cols[j_idx]
                
                U[row*m + col, r] = 1.0
                V[col*p + k, r] = 1.0
                W[k*n + row, r] = 1.0
        
        # Add small random perturbation to break symmetry
        for mat in [U, V, W]:
            mat += rng.randn(*mat.shape) * 0.01
        
        return U, V, W
    
    def try_find_rank(R, time_limit=10.0, seed=0):
        """
        Attempt to find a rank R decomposition using alternating least squares
        with structured initialization.
        """
        if R >= n * m * p:
            return None
        
        rng = np.random.RandomState(seed)
        
        start_time = time.time()
        max_restarts = 15
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            
            # Choose initialization strategy
            if restart % 3 == 0:
                # Structured initialization leveraging block decomposition
                U, V, W = get_structured_init(R, seed=seed + restart)
            elif restart % 3 == 1:
                # Random initialization
                U = rng.uniform(-1, 1, (nm, R))
                V = rng.uniform(-1, 1, (mp, R))
                W = rng.uniform(-1, 1, (pn, R))
            else:
                # Sparse random with ±1 entries
                U = np.zeros((nm, R))
                V = np.zeros((mp, R))
                W = np.zeros((pn, R))
                for r in range(R):
                    U[rng.randint(0, nm), r] = rng.choice([-1.0, 1.0])
                    V[rng.randint(0, mp), r] = rng.choice([-1.0, 1.0])
                    W[rng.randint(0, pn), r] = rng.choice([-1.0, 1.0])
            
            # Phase 1: Alternating Least Squares
            best_err = float('inf')
            for it in range(80):
                if time.time() - start_time > time_limit * 0.6:
                    break
                
                # Update U: for each (i,j), solve A * U_row = ones(p)
                for i in range(n):
                    for j in range(m):
                        V_sub = V[j*p:(j+1)*p, :]  # (p, R)
                        W_rows = W[np.arange(p)*n + i, :]  # (p, R)
                        A = V_sub * W_rows  # (p, R)
                        b = np.ones(p)
                        try:
                            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                            U[i*m+j, :] = sol
                        except:
                            pass
                
                # Update V: for each (j,k), solve A * V_row = ones(n)
                for j in range(m):
                    for k in range(p):
                        U_rows = U[np.arange(n)*m + j, :]  # (n, R)
                        W_rows = W[k*n:(k+1)*n, :]  # (n, R)
                        A = U_rows * W_rows  # (n, R)
                        b = np.ones(n)
                        try:
                            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                            V[j*p+k, :] = sol
                        except:
                            pass
                
                # Update W: for each (k,i), solve A * W_row = ones(m)
                for k in range(p):
                    for i in range(n):
                        U_rows = U[i*m:(i+1)*m, :]  # (m, R)
                        V_rows = V[np.arange(m)*p + k, :]  # (m, R)
                        A = U_rows * V_rows  # (m, R)
                        b = np.ones(m)
                        try:
                            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                            W[k*n+i, :] = sol
                        except:
                            pass
                
                # Check error
                if it % 4 == 0 or it == 79:
                    err = verify_solution_fast(U, V, W)
                    if err < best_err:
                        best_err = err
                        if err < 1e-14:
                            break
            
            # Phase 2: Least Squares Polish
            if best_err > 1e-8:
                continue
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def residuals(x):
                U_flat = x[:nm*R]
                V_flat = x[nm*R:nm*R+mp*R]
                W_flat = x[nm*R+mp*R:]
                
                U_mat = U_flat.reshape(nm, R)
                V_mat = V_flat.reshape(mp, R)
                W_mat = W_flat.reshape(pn, R)
                
                U_sel = U_mat[idx_u, :]
                V_sel = V_mat[idx_v, :]
                W_sel = W_mat[idx_w, :]
                
                res = np.sum(U_sel * V_sel * W_sel, axis=1)
                return res - 1.0
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=5000, 
                                       xtol=1e-14, ftol=1e-14, gtol=1e-14)
                
                if result.cost < 1e-10:
                    x_sol = result.x
                    U_sol = x_sol[:nm*R].reshape(nm, R)
                    V_sol = x_sol[nm*R:nm*R+mp*R].reshape(mp, R)
                    W_sol = x_sol[nm*R+mp*R:].reshape(pn, R)
                    
                    # Try rounding to nearest half-integer
                    U_r = np.round(U_sol * 2) / 2
                    V_r = np.round(V_sol * 2) / 2
                    W_r = np.round(W_sol * 2) / 2
                    
                    err_r = verify_solution_fast(U_r, V_r, W_r)
                    if err_r < 1e-9:
                        return U_r, V_r, W_r
                    
                    # If direct rounding fails, try local search
                    if result.cost < 1e-6 and err_r < 0.5:
                        current_U, current_V, current_W = U_r.copy(), V_r.copy(), W_r.copy()
                        current_err = err_r
                        
                        # Local search: try flipping individual entries
                        max_local_iters = 100
                        for local_iter in range(max_local_iters):
                            if time.time() - start_time > time_limit:
                                break
                            
                            found_improvement = False
                            
                            # Try flipping entries in U
                            for r_idx in range(nm):
                                for c_idx in range(R):
                                    orig = current_U[r_idx, c_idx]
                                    # Try neighboring half-integers
                                    for delta in [-0.5, 0.5]:
                                        new_val = orig + delta
                                        # Check if it's a valid half-integer
                                        if abs(new_val * 2 - round(new_val * 2)) > 1e-9:
                                            continue
                                        
                                        current_U[r_idx, c_idx] = new_val
                                        new_err = verify_solution_fast(current_U, current_V, current_W)
                                        if new_err < current_err - 1e-12:
                                            current_err = new_err
                                            found_improvement = True
                                            if new_err < 1e-9:
                                                return current_U, current_V, current_W
                                        else:
                                            current_U[r_idx, c_idx] = orig
                            
                            # Try flipping entries in V
                            for r_idx in range(mp):
                                for c_idx in range(R):
                                    orig = current_V[r_idx, c_idx]
                                    for delta in [-0.5, 0.5]:
                                        new_val = orig + delta
                                        if abs(new_val * 2 - round(new_val * 2)) > 1e-9:
                                            continue
                                        
                                        current_V[r_idx, c_idx] = new_val
                                        new_err = verify_solution_fast(current_U, current_V, current_W)
                                        if new_err < current_err - 1e-12:
                                            current_err = new_err
                                            found_improvement = True
                                            if new_err < 1e-9:
                                                return current_U, current_V, current_W
                                        else:
                                            current_V[r_idx, c_idx] = orig
                            
                            # Try flipping entries in W
                            for r_idx in range(pn):
                                for c_idx in range(R):
                                    orig = current_W[r_idx, c_idx]
                                    for delta in [-0.5, 0.5]:
                                        new_val = orig + delta
                                        if abs(new_val * 2 - round(new_val * 2)) > 1e-9:
                                            continue
                                        
                                        current_W[r_idx, c_idx] = new_val
                                        new_err = verify_solution_fast(current_U, current_V, current_W)
                                        if new_err < current_err - 1e-12:
                                            current_err = new_err
                                            found_improvement = True
                                            if new_err < 1e-9:
                                                return current_U, current_V, current_W
                                        else:
                                            current_W[r_idx, c_idx] = orig
                            
                            if not found_improvement:
                                break
                        
                        # Final check
                        if verify_solution_fast(current_U, current_V, current_W) < 1e-9:
                            return current_U, current_V, current_W
                    
            except Exception as e:
                pass
                
        return None

    # Main Logic
    U_sb, V_sb, W_sb = get_schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    start_time = time.time()
    timeout = 135.0  # Safety margin before 150s kill
    
    # Try ranks from 29 down to 25
    # Theoretical lower bound is at least max(nm, mp, pn) = max(6, 15, 10) = 15
    # But for matrix multiplication, rank is at least n*m*p / max(n,m,p) = 30/5 = 6
    # Known results suggest rank is around 25-29 for 2x3x5
    
    for R in range(29, 24, -1):
        if time.time() - start_time > timeout:
            break
        
        found = False
        # Use more seeds with shorter time per attempt
        num_seeds = 30
        time_per_seed = min(3.0, (timeout - (time.time() - start_time)) / num_seeds)
        
        for seed in range(num_seeds):
            if time.time() - start_time > timeout:
                break
            res = try_find_rank(R, time_limit=time_per_seed, seed=seed)
            if res is not None:
                U, V, W = res
                best_rank = R
                best_U, best_V, best_W = U, V, W
                found = True
                break
        
        if found:
            continue
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
