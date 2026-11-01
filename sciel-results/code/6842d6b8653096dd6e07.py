"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[i,j,k] where
    i indexes U (n*m), j indexes V (m*p), k indexes W (p*n).
    T[i,j,k] = 1 if the triple contributes to the product."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    T = _build_tensor(n, m, p)
    # Reconstruct: for each (u,v), sum_c U[u,c]*V[v,c] should give T[u,v,w] for all w
    # Actually: sum_c U[u,c]*V[v,c]*W[w,c] = T[u,v,w] for all u,v,w
    # This is: (U @ V^T) doesn't work directly. The reconstruction is:
    # T_recon[u,v,w] = sum_c U[u,c] * V[v,c] * W[w,c]
    # We can check: for each w, sum_c U[u,c]*V[v,c]*W[w,c] = T[u,v,w]
    # Or: sum_c (U[:,c] * V[:,c]) * W[:,c] = T flattened
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    # Build tensor: T_recon[u,v,w] = sum_c U[u,c]*V[v,c]*W[w,c]
    # This is an outer product sum
    # T_recon = np.einsum('uc,vc,wc->uvw', U_np, V_np, W_np)
    # For memory efficiency with small dimensions:
    T_recon = np.einsum('uc,vc,wc->uvw', U_np, V_np, W_np)
    if not np.allclose(T_recon, T, atol=tol):
        return False
    return True

def _try_remove_column(U, V, W, col_idx, n, m, p, tol=1e-10):
    """Try removing column col_idx and see if the remaining columns still decompose T."""
    rank = U.shape[1]
    if rank <= 1:
        return None
    U_new = [row[:col_idx] + row[col_idx+1:] for row in U]
    V_new = [row[:col_idx] + row[col_idx+1:] for row in V]
    W_new = [row[:col_idx] + row[col_idx+1:] for row in W]
    if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol):
        return (U_new, V_new, W_new)
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin: 120s
    
    # Start with schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * m * 5 // 5)]  # m*p = 4*5 = 20
    # Fix: V has m*p = 20 rows
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try to reduce rank by removing columns
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = rank
    
    # Greedy column removal
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        current_rank = len(best_U[0])
        for c in range(current_rank):
            if time.time() > deadline - 5.0:
                break
            result = _try_remove_column(best_U, best_V, best_W, c, n, m, p)
            if result is not None:
                best_U, best_V, best_W = result
                best_rank = current_rank - 1
                changed = True
                break  # Restart scanning from 0 after a removal
    
    # Try local search: perturb entries slightly to find valid lower-rank decompositions
    # Use random restarts with local optimization
    rng = random.Random(42)
    
    # Try different ranks from best_rank-1 down to a reasonable minimum
    for target_rank in range(best_rank - 1, 24, -1):
        if time.time() > deadline - 10.0:
            break
        
        # Try several random initializations
        for attempt in range(5):
            if time.time() > deadline - 10.0:
                break
            
            # Initialize with small random perturbations of a subset of schoolbook columns
            # Pick target_rank columns from the schoolbook and perturb
            if target_rank >= best_rank:
                continue
            
            # Start from current best and try to fix after removing a column
            # Or: use a random valid decomposition attempt
            
            # Simple approach: start from schoolbook, remove random columns,
            # then try to adjust entries to make it work
            U_try = [row[:] for row in U]
            V_try = [row[:] for row in V]
            W_try = [row[:] for row in W]
            
            # Remove random columns to get to target_rank
            cols_to_keep = list(range(rank))
            cols_to_remove = sorted(rng.sample(range(rank), rank - target_rank))
            
            # Keep only the columns we want
            keep_cols = [c for c in range(rank) if c not in set(cols_to_remove)]
            U_try = [row[c] for row in U for c in keep_cols]
            # Reshape back
            U_try = [[0]*target_rank for _ in range(n*m)]
            V_try = [[0]*target_rank for _ in range(m*p)]
            W_try = [[0]*target_rank for _ in range(p*n)]
            for new_idx, old_idx in enumerate(keep_cols):
                for r in range(n*m):
                    U_try[r][new_idx] = U[r][old_idx]
                for r in range(m*p):
                    V_try[r][new_idx] = V[r][old_idx]
                for r in range(p*n):
                    W_try[r][new_idx] = W[r][old_idx]
            
            # Now try to adjust entries to make it a valid decomposition
            # Use gradient-free local search: try small perturbations
            best_error = float('inf')
            for step in range(200):
                if time.time() > deadline - 5.0:
                    break
                
                # Check current error
                U_np = np.array(U_try, dtype=np.float64)
                V_np = np.array(V_try, dtype=np.float64)
                W_np = np.array(W_try, dtype=np.float64)
                T_recon = np.einsum('uc,vc,wc->uvw', U_np, V_np, W_np)
                T_target = _build_tensor(n, m, p)
                error = np.linalg.norm(T_recon - T_target)
                
                if error < 1e-10:
                    # Found a valid decomposition!
                    if target_rank < best_rank:
                        best_U = U_try
                        best_V = V_try
                        best_W = W_try
                        best_rank = target_rank
                    break
                
                # Try a random perturbation
                # Pick a random entry in U, V, or W and perturb it
                matrix_choice = rng.choice(['U', 'V', 'W'])
                if matrix_choice == 'U':
                    r = rng.randint(0, n*m-1)
                    c = rng.randint(0, target_rank-1)
                    old_val = U_try[r][c]
                    U_try[r][c] = old_val + rng.choice([-1, 1]) * rng.choice([0.5, 1.0, 2.0])
                elif matrix_choice == 'V':
                    r = rng.randint(0, m*p-1)
                    c = rng.randint(0, target_rank-1)
                    old_val = V_try[r][c]
                    V_try[r][c] = old_val + rng.choice([-1, 1]) * rng.choice([0.5, 1.0, 2.0])
                else:
                    r = rng.randint(0, p*n-1)
                    c = rng.randint(0, target_rank-1)
                    old_val = W_try[r][c]
                    W_try[r][c] = old_val + rng.choice([-1, 1]) * rng.choice([0.5, 1.0, 2.0])
                
                # Check if it improved
                U_np = np.array(U_try, dtype=np.float64)
                V_np = np.array(V_try, dtype=np.float64)
                W_np = np.array(W_try, dtype=np.float64)
                T_recon = np.einsum('uc,vc,wc->uvw', U_np, V_np, W_np)
                new_error = np.linalg.norm(T_recon - T_target)
                
                if new_error >= best_error:
                    # Revert
                    if matrix_choice == 'U':
                        U_try[r][c] = old_val
                    elif matrix_choice == 'V':
                        V_try[r][c] = old_val
                    else:
                        W_try[r][c] = old_val
                else:
                    best_error = new_error
    
    # Verify the best solution
    if _verify_decomposition(best_U, best_V, best_W, n, m, p):
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        # Fallback to schoolbook
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

