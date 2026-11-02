"""Matrix multiplication 2x3x4: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    dim_u = n * m # 6
    dim_v = m * p # 12
    dim_w = p * n # 8
    
    # Build target tensor T
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    def compute_error(U, V, W):
        P = np.einsum('at,bt,ct->abc', U, V, W)
        return np.sum((P - T) ** 2)
        
    def verify(U, V, W):
        P = np.einsum('at,bt,ct->abc', U, V, W)
        return np.allclose(P, T, atol=1e-9)
        
    def schoolbook():
        r = n*m*p
        U = np.zeros((dim_u, r))
        V = np.zeros((dim_v, r))
        W = np.zeros((dim_w, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W
        
    def local_search(U, V, W, max_iters=2000):
        """
        Gradient-guided local search with random restarts on steps.
        Returns (U, V, W, error).
        """
        candidates = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
        cur_err = compute_error(U, V, W)
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        rank = U.shape[1]
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            if cur_err < 1e-9:
                break
                
            # Compute gradients to guide the search
            # G_U[a, t] = sum_{b,c} (P_abc - T_abc) V_bt W_ct
            P = np.einsum('at,bt,ct->abc', U, V, W)
            R = P - T
            G_U = np.einsum('abc,bt,ct->at', R, V, W)
            G_V = np.einsum('abc,at,ct->bt', R, U, W)
            G_W = np.einsum('abc,at,bt->ct', R, U, V)
            
            # Find the variable with the largest gradient magnitude
            # We can sample a few random variables or pick the max
            # Picking max is better for convergence
            max_grad_u = np.max(np.abs(G_U))
            max_grad_v = np.max(np.abs(G_V))
            max_grad_w = np.max(np.abs(G_W))
            
            max_grad = max(max_grad_u, max_grad_v, max_grad_w)
            
            if max_grad < 1e-6:
                # Stuck in a flat region, try random perturbation
                mat_id = np.random.choice(3)
                if mat_id == 0:
                    idx = np.random.randint(dim_u)
                    t = np.random.randint(rank)
                elif mat_id == 1:
                    idx = np.random.randint(dim_v)
                    t = np.random.randint(rank)
                else:
                    idx = np.random.randint(dim_w)
                    t = np.random.randint(rank)
                    
                old_val = U[idx, t] if mat_id == 0 else (V[idx, t] if mat_id == 1 else W[idx, t])
                new_val = np.random.choice(candidates)
                if new_val == old_val: continue
                
                if mat_id == 0: U[idx, t] = new_val
                elif mat_id == 1: V[idx, t] = new_val
                else: W[idx, t] = new_val
                
                new_err = compute_error(U, V, W)
                if new_err < cur_err:
                    cur_err = new_err
                else:
                    if mat_id == 0: U[idx, t] = old_val
                    elif mat_id == 1: V[idx, t] = old_val
                    else: W[idx, t] = old_val
                continue

            # Select the matrix with the largest gradient
            if max_grad == max_grad_u:
                mat_id = 0
                idx, t = np.unravel_index(np.argmax(np.abs(G_U)), G_U.shape)
            elif max_grad == max_grad_v:
                mat_id = 1
                idx, t = np.unravel_index(np.argmax(np.abs(G_V)), G_V.shape)
            else:
                mat_id = 2
                idx, t = np.unravel_index(np.argmax(np.abs(G_W)), G_W.shape)
                
            # Try moving this variable to a candidate value that reduces error
            # We try the current value's neighbors in the candidate set
            old_val = U[idx, t] if mat_id == 0 else (V[idx, t] if mat_id == 1 else W[idx, t])
            
            # Sort candidates by distance to old_val
            # Or just try all candidates? 9 candidates is small.
            # But we want to be fast. Try the 2 closest candidates first.
            # Since old_val is likely already a candidate, try neighbors.
            
            best_new_err = cur_err
            best_new_val = old_val
            
            # Try all candidates, but stop early if we find a significant improvement?
            # No, just find the best among candidates.
            for c in candidates:
                if c == old_val: continue
                if mat_id == 0: U[idx, t] = c
                elif mat_id == 1: V[idx, t] = c
                else: W[idx, t] = c
                
                new_err = compute_error(U, V, W)
                if new_err < best_new_err:
                    best_new_err = new_err
                    best_new_val = c
                # Restore for next trial
                if mat_id == 0: U[idx, t] = old_val
                elif mat_id == 1: V[idx, t] = old_val
                else: W[idx, t] = old_val
                
            # Apply the best move
            if best_new_val != old_val:
                if mat_id == 0: U[idx, t] = best_new_val
                elif mat_id == 1: V[idx, t] = best_new_val
                else: W[idx, t] = best_new_val
                cur_err = best_new_err
            else:
                # No improvement from this variable, try a random one
                mat_id = np.random.choice(3)
                if mat_id == 0:
                    idx = np.random.randint(dim_u)
                    t = np.random.randint(rank)
                    old_val = U[idx, t]
                elif mat_id == 1:
                    idx = np.random.randint(dim_v)
                    t = np.random.randint(rank)
                    old_val = V[idx, t]
                else:
                    idx = np.random.randint(dim_w)
                    t = np.random.randint(rank)
                    old_val = W[idx, t]
                    
                new_val = np.random.choice(candidates)
                if new_val == old_val: continue
                
                if mat_id == 0: U[idx, t] = new_val
                elif mat_id == 1: V[idx, t] = new_val
                else: W[idx, t] = new_val
                
                new_err = compute_error(U, V, W)
                if new_err < cur_err:
                    cur_err = new_err
                else:
                    if mat_id == 0: U[idx, t] = old_val
                    elif mat_id == 1: V[idx, t] = old_val
                    else: W[idx, t] = old_val
                    
        return U, V, W, cur_err

    best_result = None
    best_rank = n*m*p
    best_U, best_V, best_W = schoolbook()
    best_err = 0.0
    
    # Try to reduce rank from 23 down to 14
    # We use the best solution of rank r+1 to initialize rank r
    current_U, current_V, current_W = best_U, best_V, best_W
    
    for target_rank in range(n*m*p - 1, 13, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        max_restarts = 50
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize from previous best solution
            if restart == 0 and current_U.shape[1] > target_rank:
                # Take first target_rank columns
                U_init = current_U[:, :target_rank].copy()
                V_init = current_V[:, :target_rank].copy()
                W_init = current_W[:, :target_rank].copy()
            else:
                # Random initialization with sparse structure
                np.random.seed(restart * 123 + target_rank)
                U_init = np.zeros((dim_u, target_rank))
                V_init = np.zeros((dim_v, target_rank))
                W_init = np.zeros((dim_w, target_rank))
                # Fill with random candidates
                for t in range(target_rank):
                    # Pick random non-zero entries
                    for _ in range(3):
                        U_init[np.random.randint(dim_u), t] = np.random.choice([-1, 0.5, 1])
                        V_init[np.random.randint(dim_v), t] = np.random.choice([-1, 0.5, 1])
                        W_init[np.random.randint(dim_w), t] = np.random.choice([-1, 0.5, 1])
            
            U_sol, V_sol, W_sol, err = local_search(U_init, V_init, W_sol if False else W_init, max_iters=1000)
            
            if err < 1e-9:
                found = True
                best_rank = target_rank
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                best_err = err
                current_U, current_V, current_W = best_U, best_V, best_W
                break
                
        if found:
            # Continue to next lower rank
            pass
        else:
            # If we didn't find a solution for this rank, we can still try lower ranks?
            # Usually if rank r fails, rank r-1 is harder. But random restarts might find it.
            # However, if the "directed" search from r+1 failed, random might work.
            # We continue the loop.
            pass

    if best_result is None or best_err > 1e-9:
        # Fallback to schoolbook
        U, V, W = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
