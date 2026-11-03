import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    n, m, p = 3, 3, 3
    nm, mp, pn = 9, 9, 9
    
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i
    
    # Precompute target tensor entries
    # T[i,j,k] = 1 for all i,j,k in {0,1,2}
    # The condition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    ones_indices = [(i*m+j, j*p+k, k*n+i) for i in range(n) for j in range(m) for k in range(p)]
    ones_set = set(ones_indices)
    all_indices = [(a, b, c) for a in range(nm) for b in range(mp) for c in range(pn)]
    zero_indices = [idx for idx in all_indices if idx not in ones_set]

    def verify(U, V, W, tol=1e-9):
        """
        Verify if U, V, W form a valid decomposition.
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Check ones
        for a, b, c in ones_indices:
            s = np.sum(W[c, :] * U[a, :] * V[b, :])
            if abs(s - 1.0) > tol:
                return False
        # Check zeros
        for a, b, c in zero_indices:
            s = np.sum(W[c, :] * U[a, :] * V[b, :])
            if abs(s) > tol:
                return False
        return True

    def snap_to_half_integers(U, V, W):
        """Snap to nearest half-integers and verify."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r, tol=1e-9):
            return U_r, V_r, W_r
        return None

    # --- Step 1: Construct Rank-26 Solution (Strassen Core + Border) ---
    R26 = 26
    U26 = np.zeros((nm, R26))
    V26 = np.zeros((mp, R26))
    W26 = np.zeros((pn, R26))
    
    # Strassen terms for 2x2 core (indices 0,1)
    # r=0: (A00+A11)(B00+B11)
    U26[idx_u(0,0), 0] = 1; U26[idx_u(1,1), 0] = 1
    V26[idx_v(0,0), 0] = 1; V26[idx_v(1,1), 0] = 1
    W26[idx_w(0,0), 0] = 1; W26[idx_w(1,1), 0] = 1
    
    # r=1: (A10+A11)B00
    U26[idx_u(1,0), 1] = 1; U26[idx_u(1,1), 1] = 1
    V26[idx_v(0,0), 1] = 1
    W26[idx_w(0,1), 1] = 1; W26[idx_w(1,1), 1] = -1
    
    # r=2: A00(B01-B11)
    U26[idx_u(0,0), 2] = 1
    V26[idx_v(0,1), 2] = 1; V26[idx_v(1,1), 2] = -1
    W26[idx_w(1,0), 2] = 1; W26[idx_w(1,1), 2] = 1
    
    # r=3: A11(B10-B00)
    U26[idx_u(1,1), 3] = 1
    V26[idx_v(1,0), 3] = 1; V26[idx_v(0,0), 3] = -1
    W26[idx_w(0,0), 3] = 1; W26[idx_w(0,1), 3] = 1
    
    # r=4: (A00+A01)B11
    U26[idx_u(0,0), 4] = 1; U26[idx_u(0,1), 4] = 1
    V26[idx_v(1,1), 4] = 1
    W26[idx_w(0,0), 4] = -1; W26[idx_w(1,0), 4] = 1
    
    # r=5: (A10-A00)(B00+B01)
    U26[idx_u(1,0), 5] = 1; U26[idx_u(0,0), 5] = -1
    V26[idx_v(0,0), 5] = 1; V26[idx_v(0,1), 5] = 1
    W26[idx_w(1,1), 5] = 1
    
    # r=6: (A01-A11)(B10+B11)
    U26[idx_u(0,1), 6] = 1; U26[idx_u(1,1), 6] = -1
    V26[idx_v(1,0), 6] = 1; V26[idx_v(1,1), 6] = 1
    W26[idx_w(0,0), 6] = 1
    
    # Border terms (indices involving 2)
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U26[idx_u(i,j), col] = 1
                V26[idx_v(j,k), col] = 1
                W26[idx_w(k,i), col] = 1
                col += 1
                
    assert col == 26, f"Expected 26 columns, got {col}"
    
    # Verify Rank 26
    if not verify(U26, V26, W26):
        # Fallback to Schoolbook Rank 27
        R27 = 27
        U27 = np.zeros((nm, R27))
        V27 = np.zeros((mp, R27))
        W27 = np.zeros((pn, R27))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U27[idx_u(i,j), c] = 1
                    V27[idx_v(j,k), c] = 1
                    W27[idx_w(k,i), c] = 1
                    c += 1
        return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
    
    best_U, best_V, best_W = U26, V26, W26
    best_rank = 26
    
    # --- Step 2: Try to find Rank 25 or 24 using Optimization ---
    # We use the Rank 26 solution as a starting point for lower ranks.
    # The idea is that some columns might be linearly dependent or mergeable.
    # We will use scipy.optimize.least_squares to minimize the error over all 729 entries.
    
    # Precompute all (a,b,c) triples and targets once
    a_idx_all = []
    b_idx_all = []
    c_idx_all = []
    targets_all = []
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                a_idx_all.append(a)
                b_idx_all.append(b)
                c_idx_all.append(c)
                targets_all.append(1.0 if (a, b, c) in ones_set else 0.0)
    a_idx_all = np.array(a_idx_all, dtype=np.int32)
    b_idx_all = np.array(b_idx_all, dtype=np.int32)
    c_idx_all = np.array(c_idx_all, dtype=np.int32)
    targets_all = np.array(targets_all, dtype=float)
    n_entries = len(targets_all)
    
    def try_optimize_rank(R, U_init, V_init, W_init, time_budget):
        """
        Attempt to find a valid decomposition of rank R starting from initial guesses.
        """
        if time.time() > start_time + time_budget:
            return None
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def residual(x):
            U_f = x[:nm*R].reshape(nm, R)
            V_f = x[nm*R:2*nm*R].reshape(mp, R)
            W_f = x[2*nm*R:3*nm*R].reshape(pn, R)
            
            # Compute sum_r U[a,r] V[b,r] W[c,r] for all (a,b,c) efficiently
            # For each r, compute U[:,r] @ V[:,r] @ W[:,r] contribution
            # Use broadcasting: for each r, U_f[:, r][:, None] * V_f[:, r][None, :] gives outer product
            # But we need the triple product summed over r
            # Efficient approach: for each (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r]
            # Vectorize: create matrix M_r where M_r[a,b] = U[a,r]*V[b,r], then for each c,
            # res[:, :, c] = sum_r M_r * W[c,r]
            # Let's do it with einsum-like approach using loops over r
            res = np.zeros(n_entries)
            for r in range(R):
                # For this r, contribution to entry (a,b,c) is U[a,r]*V[b,r]*W[c,r]
                # We can compute this as: for all (a,b,c), U[a,r] * V[b,r] * W[c,r]
                # Vectorized: outer products
                UV = np.outer(U_f[:, r], V_f[:, r])  # shape (nm, mp)
                # For each c, add UV * W[c, r] to res
                # res is indexed by flat index of (a,b,c)
                # a_idx_all, b_idx_all, c_idx_all give the indices
                contrib = UV[a_idx_all, b_idx_all] * W_f[c_idx_all, r]
                res += contrib
            res -= targets_all
            return res

        from scipy.optimize import least_squares
        best_cost = float('inf')
        best_x = x0
        
        # We will do a few local optimizations
        max_iter = 5
        for i in range(max_iter):
            if time.time() > start_time + time_budget:
                break
            
            try:
                sol = least_squares(residual, x0, method='trf', max_nfev=3000, ftol=1e-14, xtol=1e-14, gtol=1e-14)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_x = sol.x
                    if sol.cost < 1e-12:
                        break
            except Exception:
                pass
            
            # Perturb for next restart
            if i < max_iter - 1:
                x0 = best_x + np.random.randn(len(best_x)) * 0.01

        if best_cost < 1e-10:
            U_sol = best_x[:nm*R].reshape(nm, R)
            V_sol = best_x[nm*R:2*nm*R].reshape(mp, R)
            W_sol = best_x[2*nm*R:3*nm*R].reshape(pn, R)
            
            # Try to snap to half-integers
            snapped = snap_to_half_integers(U_sol, V_sol, W_sol)
            if snapped:
                return snapped
                
        return None

    # Try Rank 25
    remaining_time = deadline - time.time()
    if remaining_time > 15:
        # Use structured initialization based on the rank-26 solution
        np.random.seed(42)
        for R in [25, 24]:
            if time.time() > deadline - 10:
                break
            
            found = False
            # Number of restarts depends on time
            num_restarts = max(2, int((deadline - time.time()) / 6.0))
            
            for restart in range(num_restarts):
                if time.time() > deadline - 5:
                    break
                
                # Initialize by truncating the rank-26 solution and adding small random perturbation
                # This gives a structured starting point closer to a valid solution
                U_init = U26[:, :R].copy()
                V_init = V26[:, :R].copy()
                W_init = W26[:, :R].copy()
                
                # Add small random perturbation to escape exact degeneracy
                U_init += np.random.randn(nm, R) * 0.05
                V_init += np.random.randn(mp, R) * 0.05
                W_init += np.random.randn(pn, R) * 0.05
                
                res = try_optimize_rank(R, U_init, V_init, W_init, time_budget=5.0)
                if res is not None:
                    best_U, best_V, best_W = res
                    best_rank = R
                    found = True
                    break # Found a better rank, break inner loop
                    
            if found:
                break # Found a better rank, break outer loop
                
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
