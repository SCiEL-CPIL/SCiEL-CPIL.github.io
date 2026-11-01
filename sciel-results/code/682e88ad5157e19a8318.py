import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute the target tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    start_time = time.time()
    deadline = start_time + 140

    def compute_R(U, V, W):
        return np.einsum('as,bs,cs->abc', U, V, W, optimize=True)

    def check_exact(U, V, W, T, tol=1e-9):
        R = compute_R(U, V, W)
        if np.allclose(R, T, atol=tol):
            return True
        return False

    def snap_to_half_int(mat):
        # Round to nearest 0.5
        return np.round(mat * 2) / 2.0

    def local_repair(U, V, W, T, rank, max_steps=1000):
        """
        Try to fix a nearly exact solution by adjusting single entries.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for step in range(max_steps):
            R = compute_R(U, V, W)
            diff = R - T
            sq = np.sum(diff**2)
            if sq < 1e-12:
                return U, V, W, True
            
            # Find the worst error entry
            flat_diff = diff.flatten()
            idx = np.argmax(np.abs(flat_diff))
            a, b, c = np.unravel_index(idx, diff.shape)
            
            # Try to adjust one entry in U, V, or W at column s
            # We want to reduce diff[a,b,c]
            target_change = -diff[a, b, c]
            
            found_improvement = False
            best_sq = sq
            
            # Try adjusting U[a, s]
            for s in range(rank):
                v_val = V[b, s]
                w_val = W[c, s]
                denom = v_val * w_val
                if abs(denom) > 1e-9:
                    # Current contribution: U[a,s] * denom
                    # We want new contribution to change by target_change
                    # (U_new - U_old) * denom = target_change
                    # U_new = U_old + target_change / denom
                    u_new = U[a, s] + target_change / denom
                    u_snapped = snap_to_half_int(np.array([u_new]))[0]
                    
                    if u_snapped != U[a, s]:
                        old_u = U[a, s]
                        U[a, s] = u_snapped
                        R_new = compute_R(U, V, W)
                        sq_new = np.sum((R_new - T)**2)
                        if sq_new < best_sq:
                            best_sq = sq_new
                            found_improvement = True
                            if sq_new < 1e-12:
                                return U, V, W, True
                        else:
                            U[a, s] = old_u
            
            # Try adjusting V[b, s]
            for s in range(rank):
                u_val = U[a, s]
                w_val = W[c, s]
                denom = u_val * w_val
                if abs(denom) > 1e-9:
                    v_new = V[b, s] + target_change / denom
                    v_snapped = snap_to_half_int(np.array([v_new]))[0]
                    
                    if v_snapped != V[b, s]:
                        old_v = V[b, s]
                        V[b, s] = v_snapped
                        R_new = compute_R(U, V, W)
                        sq_new = np.sum((R_new - T)**2)
                        if sq_new < best_sq:
                            best_sq = sq_new
                            found_improvement = True
                            if sq_new < 1e-12:
                                return U, V, W, True
                        else:
                            V[b, s] = old_v

            # Try adjusting W[c, s]
            for s in range(rank):
                u_val = U[a, s]
                v_val = V[b, s]
                denom = u_val * v_val
                if abs(denom) > 1e-9:
                    w_new = W[c, s] + target_change / denom
                    w_snapped = snap_to_half_int(np.array([w_new]))[0]
                    
                    if w_snapped != W[c, s]:
                        old_w = W[c, s]
                        W[c, s] = w_snapped
                        R_new = compute_R(U, V, W)
                        sq_new = np.sum((R_new - T)**2)
                        if sq_new < best_sq:
                            best_sq = sq_new
                            found_improvement = True
                            if sq_new < 1e-12:
                                return U, V, W, True
                        else:
                            W[c, s] = old_w

            if not found_improvement:
                # Try random perturbation to escape local minimum
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    r, c = np.random.randint(nm), np.random.randint(rank)
                    U[r, c] += np.random.choice([-0.5, 0.5])
                elif mat_idx == 1:
                    r, c = np.random.randint(mp), np.random.randint(rank)
                    V[r, c] += np.random.choice([-0.5, 0.5])
                else:
                    r, c = np.random.randint(pn), np.random.randint(rank)
                    W[r, c] += np.random.choice([-0.5, 0.5])

        return U, V, W, False

    def gradient_descent(U, V, W, T, rank, lr=0.01, steps=500):
        """
        Real-valued gradient descent to get close to a solution.
        """
        for step in range(steps):
            R = compute_R(U, V, W)
            diff = R - T
            
            # Gradient for U: dE/dU[a,s] = 2 * sum_{b,c} diff[a,b,c] * V[b,s] * W[c,s]
            grad_U = 2 * np.einsum('abc,bs,cs->as', diff, V, W, optimize=True)
            grad_V = 2 * np.einsum('abc,as,cs->bs', diff, U, W, optimize=True)
            grad_W = 2 * np.einsum('abc,as,bs->cs', diff, U, V, optimize=True)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Optional: Project to half-integers every few steps? 
            # No, keep real for smooth optimization, snap at the end.
            
            if time.time() > deadline - 10:
                break

        return U, V, W

    # Try ranks from 12 to 18
    for rank in [12, 13, 14, 15, 16, 17, 18]:
        if time.time() > deadline - 20:
            break
            
        # For rank 18, we have a known solution
        if rank == 18:
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1.0
                        V[j * p + k, col] = 1.0
                        W[k * n + i, col] = 1.0
                        col += 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        # For lower ranks, try multiple restarts
        max_restarts = 50
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize with random values in [-1, 1]
            np.random.seed(r) # Reproducibility for debugging, remove in prod?
            U = np.random.uniform(-1, 1, (nm, rank))
            V = np.random.uniform(-1, 1, (mp, rank))
            W = np.random.uniform(-1, 1, (pn, rank))
            
            # Run Gradient Descent
            U, V, W = gradient_descent(U, V, W, T, rank, lr=0.05, steps=200)
            
            # Snap to half-integers
            U_snapped = snap_to_half_int(U)
            V_snapped = snap_to_half_int(V)
            W_snapped = snap_to_half_int(W)
            
            # Check if exact
            if check_exact(U_snapped, V_snapped, W_snapped, T):
                return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
                
            # If not exact, try local repair
            U_rep, V_rep, W_rep, success = local_repair(U_snapped, V_snapped, W_snapped, T, rank, max_steps=50)
            if success:
                return {"U": U_rep.tolist(), "V": V_rep.tolist(), "W": W_rep.tolist()}
            
            # Also try repair on the non-snapped solution? 
            # The non-snapped solution might be closer to a real solution, but we need integers.
            # Maybe snap again after a few GD steps?
            
    # Fallback
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
