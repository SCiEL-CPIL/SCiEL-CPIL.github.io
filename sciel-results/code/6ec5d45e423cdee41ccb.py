"""Matrix multiplication 2x3x4 rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 12, 8
    
    # Build the target tensor entries
    # T[u,v,w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.append((u_idx, v_idx, w_idx))
    
    num_entries = len(tensor_entries)  # 24
    
    # Schoolbook fallback (rank 24)
    best_rank = n * m * p
    U_sb = [[0] * best_rank for _ in range(nm)]
    V_sb = [[0] * best_rank for _ in range(mp)]
    W_sb = [[0] * best_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    def verify_solution(U, V, W, rank):
        """Check if the decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Check all 24 entries
        for (u, v, w) in tensor_entries:
            val = np.dot(U[u], V[v] * W[w])
            if abs(val - 1.0) > 1e-9:
                return False
        
        # Check non-entries
        # To speed this up, we can compute the full tensor and check zeros
        # But 6*12*8 = 576 is small enough to check directly if needed
        # However, checking all 576 - 24 = 552 zeros is fast.
        for u in range(nm):
            for v in range(mp):
                for w in range(pn):
                    if (u, v, w) in tensor_entries:
                        continue
                    val = np.dot(U[u], V[v] * W[w])
                    if abs(val) > 1e-9:
                        return False
        return True

    def compute_error(U, V, W):
        """Compute the sum of squared errors for the 24 entries."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        err = 0.0
        for (u, v, w) in tensor_entries:
            val = np.dot(U[u], V[v] * W[w])
            err += (val - 1.0)**2
        return err

    def local_search(U, V, W, rank, max_iterations=20):
        """Perform local search on the half-integer grid to reduce error."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        current_err = compute_error(U, V, W)
        
        for _ in range(max_iterations):
            improved = False
            # Try to improve W
            for w_idx in range(pn):
                for r in range(rank):
                    old_val = W[w_idx, r]
                    # Try neighbors: old_val +/- 0.5
                    for new_val in [old_val - 0.5, old_val + 0.5]:
                        W[w_idx, r] = new_val
                        new_err = compute_error(U, V, W)
                        if new_err < current_err - 1e-12:
                            current_err = new_err
                            improved = True
                        else:
                            W[w_idx, r] = old_val
            # Try to improve U
            for u_idx in range(nm):
                for r in range(rank):
                    old_val = U[u_idx, r]
                    for new_val in [old_val - 0.5, old_val + 0.5]:
                        U[u_idx, r] = new_val
                        new_err = compute_error(U, V, W)
                        if new_err < current_err - 1e-12:
                            current_err = new_err
                            improved = True
                        else:
                            U[u_idx, r] = old_val
            # Try to improve V
            for v_idx in range(mp):
                for r in range(rank):
                    old_val = V[v_idx, r]
                    for new_val in [old_val - 0.5, old_val + 0.5]:
                        V[v_idx, r] = new_val
                        new_err = compute_error(U, V, W)
                        if new_err < current_err - 1e-12:
                            current_err = new_err
                            improved = True
                        else:
                            V[v_idx, r] = old_val
                            
            if not improved:
                break
                
        # Final check if error is 0
        if current_err < 1e-9:
            # Verify full solution
            if verify_solution(U.tolist(), V.tolist(), W.tolist(), rank):
                return U, V, W
        return None

    # Search for lower rank
    np.random.seed(42)
    
    # We know rank 24 works. Let's try 23, 22, ...
    # Based on literature, 2x3x4 might have rank around 18-20.
    # Let's try from 18 upwards to be safe, or 20 downwards?
    # Lower is better, so we want the minimum.
    # Let's try 18, 19, 20, 21, 22, 23.
    
    found = False
    for target_rank in range(18, 24):
        if time.time() > deadline - 20:
            break
            
        if found:
            break
            
        # Multiple restarts
        num_restarts = 50
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize
            # Use a mix of random and structured
            if restart < 10:
                # Structured initialization: start with schoolbook-like and perturb
                U_init = np.zeros((nm, target_rank))
                V_init = np.zeros((mp, target_rank))
                W_init = np.zeros((pn, target_rank))
                
                # Fill first min(target_rank, 24) columns with schoolbook patterns
                # But we have fewer columns than schoolbook rank if target_rank < 24
                # So we can't just copy schoolbook.
                # Instead, use random half-integers.
                pass
            
            U_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(nm, target_rank))
            V_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(mp, target_rank))
            W_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(pn, target_rank))
            
            # Alternating Least Squares with Rounding and Local Search
            U_curr = U_init.copy()
            V_curr = V_init.copy()
            W_curr = W_init.copy()
            
            for iteration in range(50):
                if time.time() > deadline - 2:
                    break
                    
                # Solve for W
                # A_W shape: (24, pn * target_rank)
                A_W = np.zeros((num_entries, pn * target_rank))
                for e_idx, (u_idx, v_idx, w_idx) in enumerate(tensor_entries):
                    for r in range(target_rank):
                        col = w_idx * target_rank + r
                        A_W[e_idx, col] = U_curr[u_idx, r] * V_curr[v_idx, r]
                
                W_flat, _, _, _ = np.linalg.lstsq(A_W, np.ones(num_entries), rcond=None)
                W_curr = W_flat.reshape(pn, target_rank)
                W_curr = np.round(W_curr * 2) / 2
                
                # Solve for U
                A_U = np.zeros((num_entries, nm * target_rank))
                for e_idx, (u_idx, v_idx, w_idx) in enumerate(tensor_entries):
                    for r in range(target_rank):
                        col = u_idx * target_rank + r
                        A_U[e_idx, col] = V_curr[v_idx, r] * W_curr[w_idx, r]
                
                U_flat, _, _, _ = np.linalg.lstsq(A_U, np.ones(num_entries), rcond=None)
                U_curr = U_flat.reshape(nm, target_rank)
                U_curr = np.round(U_curr * 2) / 2
                
                # Solve for V
                A_V = np.zeros((num_entries, mp * target_rank))
                for e_idx, (u_idx, v_idx, w_idx) in enumerate(tensor_entries):
                    for r in range(target_rank):
                        col = v_idx * target_rank + r
                        A_V[e_idx, col] = U_curr[u_idx, r] * W_curr[w_idx, r]
                
                V_flat, _, _, _ = np.linalg.lstsq(A_V, np.ones(num_entries), rcond=None)
                V_curr = V_flat.reshape(mp, target_rank)
                V_curr = np.round(V_curr * 2) / 2
                
                # Local Search
                res = local_search(U_curr, V_curr, W_curr, target_rank)
                if res is not None:
                    U_curr, V_curr, W_curr = res
                    best_rank = target_rank
                    best_result = {
                        "U": U_curr.tolist(),
                        "V": V_curr.tolist(),
                        "W": W_curr.tolist()
                    }
                    found = True
                    break
            
            if found:
                break

    if found:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
