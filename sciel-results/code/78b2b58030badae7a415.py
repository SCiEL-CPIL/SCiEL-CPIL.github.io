"""Rank-minimal exact tensor decomposition for 2x3 x 3x8 matmul using block decomposition with improved search."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 8]

def _verify(U, V, W, n, m, p):
    """Verify if U, V, W form a valid decomposition."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _compute_error(U, V, W, n, m, p):
    """Compute L2 error of the decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    errors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                errors.append(val - 1.0)
    return np.sqrt(np.sum(np.array(errors) ** 2))

def _search_decomp(n, m, p, target_rank, time_limit, seed=42, max_val=5):
    """
    Search for a decomposition of rank target_rank for n x m by m x p.
    Uses hill climbing with L2 error minimization.
    Returns (U, V, W) as lists, or None.
    """
    start_time = time.time()
    random.seed(seed)
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    best_exact = None
    
    # Try multiple random initializations
    max_initializations = 50
    for init in range(max_initializations):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization
        U = np.random.randint(-max_val, max_val + 1, size=(dim_u, target_rank), dtype=float)
        V = np.random.randint(-max_val, max_val + 1, size=(dim_v, target_rank), dtype=float)
        W = np.random.randint(-max_val, max_val + 1, size=(dim_w, target_rank), dtype=float)
        
        current_error = _compute_error(U, V, W, n, m, p)
        
        if current_error < 1e-9:
            best_exact = (U.copy(), V.copy(), W.copy())
            break
        
        # Hill climbing
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() - start_time < time_limit and no_improve_count < max_no_improve:
            # Pick a random matrix, row, and column to perturb
            mat_choice = random.randint(0, 2)
            if mat_choice == 0:
                mat = U
                dim = dim_u
            elif mat_choice == 1:
                mat = V
                dim = dim_v
            else:
                mat = W
                dim = dim_w
            
            row = random.randint(0, dim - 1)
            col = random.randint(0, target_rank - 1)
            
            old_val = mat[row, col]
            # Try a small perturbation
            delta = random.choice([-2, -1, 1, 2])
            mat[row, col] = old_val + delta
            
            new_error = _compute_error(U, V, W, n, m, p)
            
            if new_error < current_error - 1e-10:
                current_error = new_error
                no_improve_count = 0
                if current_error < 1e-9:
                    best_exact = (U.copy(), V.copy(), W.copy())
                    break
            else:
                mat[row, col] = old_val
                no_improve_count += 1
        
        if best_exact is not None:
            break
        
        # Track best
        if current_error < best_error:
            best_error = current_error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if best_error < 1e-6:
            # Try to snap to integers
            U_snap = np.round(U).astype(int)
            V_snap = np.round(V).astype(int)
            W_snap = np.round(W).astype(int)
            if _verify(U_snap, V_snap, W_snap, n, m, p):
                best_exact = (U_snap, V_snap, W_snap)
                break
    
    if best_exact is not None:
        U, V, W = best_exact
        return U.tolist(), V.tolist(), W.tolist()
    
    # If we got very close, try snapping
    if best_U is not None and best_error < 1e-4:
        U_snap = np.round(best_U).astype(int)
        V_snap = np.round(best_V).astype(int)
        W_snap = np.round(best_W).astype(int)
        if _verify(U_snap, V_snap, W_snap, n, m, p):
            return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Leave 15s margin
    
    # Strategy: Decompose B (3x8) into B1 (3x3), B2 (3x3), B3 (3x2)
    # Find decomp for (2, 3, 3) and (2, 3, 2)
    
    d1 = None
    d2 = None
    
    # Search for (2, 3, 3) - try ranks 12, 13
    for r1 in [12, 13]:
        if time.time() > deadline - 50:
            break
        d1 = _search_decomp(2, 3, 3, r1, time_limit=15, seed=r1*100, max_val=4)
        if d1:
            print(f"Found (2,3,3) decomp with rank {r1}")
            break
    
    # Search for (2, 3, 2) - try ranks 8, 9, 10
    for r2 in [8, 9, 10]:
        if time.time() > deadline - 10:
            break
        d2 = _search_decomp(2, 3, 2, r2, time_limit=15, seed=r2*100, max_val=4)
        if d2:
            print(f"Found (2,3,2) decomp with rank {r2}")
            break

    # Construct final result
    if d1 and d2:
        U1, V1, W1 = d1
        U2, V2, W2 = d2
        
        r1 = len(U1[0])
        r2 = len(U2[0])
        total_rank = 2 * r1 + r2
        
        # Initialize full matrices
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # Fill Block 1 (cols 0-2 of B)
        for i in range(n * m):
            for c in range(r1):
                U[i][c] = U1[i][c]
        for j in range(m):
            for k in range(3):
                v_idx = j * p + k
                v1_idx = j * 3 + k
                for c in range(r1):
                    V[v_idx][c] = V1[v1_idx][c]
        for k in range(3):
            for i in range(n):
                w_idx = k * n + i
                w1_idx = k * n + i
                for c in range(r1):
                    W[w_idx][c] = W1[w1_idx][c]
                    
        # Fill Block 2 (cols 3-5 of B)
        offset1 = r1
        for i in range(n * m):
            for c in range(r1):
                U[i][offset1 + c] = U1[i][c]
        for j in range(m):
            for k in range(3):
                v_idx = j * p + (3 + k)
                v1_idx = j * 3 + k
                for c in range(r1):
                    V[v_idx][offset1 + c] = V1[v1_idx][c]
        for k in range(3):
            for i in range(n):
                w_idx = (3 + k) * n + i
                w1_idx = k * n + i
                for c in range(r1):
                    W[w_idx][offset1 + c] = W1[w1_idx][c]
                    
        # Fill Block 3 (cols 6-7 of B)
        offset2 = 2 * r1
        for i in range(n * m):
            for c in range(r2):
                U[i][offset2 + c] = U2[i][c]
        for j in range(m):
            for k in range(2):
                v_idx = j * p + (6 + k)
                v2_idx = j * 3 + k
                for c in range(r2):
                    V[v_idx][offset2 + c] = V2[v2_idx][c]
        for k in range(2):
            for i in range(n):
                w_idx = (6 + k) * n + i
                w2_idx = k * n + i
                for c in range(r2):
                    W[w_idx][offset2 + c] = W2[w2_idx][c]
                    
        # Final verification
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        else:
            print("Verification failed for combined decomp")

    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
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
    return {"U": U, "V": V, "W": W}
