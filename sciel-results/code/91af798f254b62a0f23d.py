"""Matrix multiplication 2x3x4 rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 12, 8
    
    # Build the target tensor entries
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
        # Check tensor entries
        for (u, v, w) in tensor_entries:
            val = sum(U[u][r] * V[v][r] * W[w][r] for r in range(rank))
            if abs(val - 1.0) > 1e-10:
                return False
        # Check non-entries
        entry_set = set(tensor_entries)
        for u in range(nm):
            for v in range(mp):
                for w in range(pn):
                    if (u, v, w) not in entry_set:
                        val = sum(U[u][r] * V[v][r] * W[w][r] for r in range(rank))
                        if abs(val) > 1e-10:
                            return False
        return True

    def snap_to_half_integers(mat, tol=1e-6):
        """Snap matrix entries to nearest half-integer if within tolerance."""
        snapped = np.round(mat * 2) / 2.0
        if np.max(np.abs(mat - snapped)) < tol:
            return snapped, True
        return None, False

    def try_gradient_refinement(U_init, V_init, W_init, target_rank, max_iters=500):
        """
        Apply gradient-based refinement to U, V, W to minimize the tensor error,
        then snap to half-integers.
        """
        U = U_init.astype(np.float64).copy()
        V = V_init.astype(np.float64).copy()
        W = W_init.astype(np.float64).copy()
        
        # Precompute index arrays for efficiency
        u_indices = np.array([e[0] for e in tensor_entries])
        v_indices = np.array([e[1] for e in tensor_entries])
        w_indices = np.array([e[2] for e in tensor_entries])
        
        # Non-entry indices
        entry_set = set(tensor_entries)
        non_entry_list = []
        for u in range(nm):
            for v in range(mp):
                for w in range(pn):
                    if (u, v, w) not in entry_set:
                        non_entry_list.append((u, v, w))
        non_u = np.array([e[0] for e in non_entry_list])
        non_v = np.array([e[1] for e in non_entry_list])
        non_w = np.array([e[2] for e in non_entry_list])
        
        lr = 0.01
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute gradients
            # Error = sum_entries (sum_r U[u,r]*V[v,r]*W[w,r] - 1)^2 + sum_non_entries (sum_r U[u,r]*V[v,r]*W[w,r])^2
            
            # d/dU[u,r] = 2 * sum_entries (err) * V[v,r]*W[w,r] + 2 * sum_non_entries (err) * V[v,r]*W[w,r]
            # where err = sum_r' U[u,r']*V[v,r']*W[w,r] - T[u,v,w]
            
            # For entries: T=1, for non-entries: T=0
            
            # Compute current values for entries
            entry_vals = np.zeros(num_entries)
            for r in range(target_rank):
                entry_vals += U[u_indices, r] * V[v_indices, r] * W[w_indices, r]
            entry_err = entry_vals - 1.0
            
            # Compute current values for non-entries
            non_vals = np.zeros(len(non_entry_list))
            for r in range(target_rank):
                non_vals += U[non_u, r] * V[non_v, r] * W[non_w, r]
            non_err = non_vals
            
            # Gradient for U
            dU = np.zeros_like(U)
            for r in range(target_rank):
                # From entries
                dU[u_indices, r] += 2 * entry_err * (V[v_indices, r] * W[w_indices, r])
                # From non-entries
                dU[non_u, r] += 2 * non_err * (V[non_v, r] * W[non_w, r])
            
            # Gradient for V
            dV = np.zeros_like(V)
            for r in range(target_rank):
                dV[v_indices, r] += 2 * entry_err * (U[u_indices, r] * W[w_indices, r])
                dV[non_v, r] += 2 * non_err * (U[non_u, r] * W[non_w, r])
            
            # Gradient for W
            dW = np.zeros_like(W)
            for r in range(target_rank):
                dW[w_indices, r] += 2 * entry_err * (U[u_indices, r] * V[v_indices, r])
                dW[non_w, r] += 2 * non_err * (U[non_u, r] * V[non_v, r])
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check convergence
            total_err = np.sum(entry_err**2) + np.sum(non_err**2)
            if total_err < 1e-12:
                break
            
            # Decay learning rate
            if it % 100 == 0 and it > 0:
                lr *= 0.9
        
        # Snap to half-integers
        U_snapped, okU = snap_to_half_integers(U, tol=1e-4)
        V_snapped, okV = snap_to_half_integers(V, tol=1e-4)
        W_snapped, okW = snap_to_half_integers(W, tol=1e-4)
        
        if okU and okV and okW:
            if verify_solution(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), target_rank):
                return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
        
        return None
    
    # Search for minimal rank using randomized linear solving
    # Theoretical lower bound is 18 for 2x3x4. We start from 18 to save time.
    # If 18 fails, we go up to 24.
    
    # We use a strategy where we fix U and V randomly (sparse, small values)
    # and solve for W. Then we check if W is "simple" (half-integer).
    
    # To make this work, we need the system A_W * W_flat = b_W to be consistent.
    # A_W is built from U and V.
    
    # We will try ranks from 18 to 23.
    for target_rank in range(18, best_rank):
        if time.time() > deadline - 5:
            break
        
        found = False
        np.random.seed(42 + target_rank)
        
        # Number of random attempts
        num_attempts = 2000
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 2:
                break
            
            # Generate random U and V with entries in {-1, 0, 1}
            # Using sparse matrices increases the chance of solvability and simple W
            U_init = np.random.choice([-1, 0, 1], size=(nm, target_rank))
            V_init = np.random.choice([-1, 0, 1], size=(mp, target_rank))
            
            # Ensure U and V are not zero vectors
            if np.all(U_init == 0) or np.all(V_init == 0):
                continue
            
            # Build A_W and b_W
            # Equation: sum_r U[u,r] * V[v,r] * W[w,r] = T[u,v,w]
            # For each (u,v,w) in tensor_entries:
            #   sum_r (U[u,r] * V[v,r]) * W[w,r] = 1
            # This is a linear system in W[w,r].
            
            # We have num_entries equations.
            # Unknowns are W[w,r] for w in 0..pn-1, r in 0..target_rank-1.
            # Total unknowns: pn * target_rank.
            
            A_W = np.zeros((num_entries, pn * target_rank))
            b_W = np.ones(num_entries)
            
            for e_idx, (u_idx, v_idx, w_idx) in enumerate(tensor_entries):
                for r in range(target_rank):
                    col = w_idx * target_rank + r
                    A_W[e_idx, col] = U_init[u_idx, r] * V_init[v_idx, r]
            
            # Solve least squares
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A_W, b_W, rcond=None)
            except np.linalg.LinAlgError:
                continue
                
            W_curr = W_flat.reshape(pn, target_rank)
            
            # Check if W_curr is close to half-integers
            # We want W_curr * 2 to be close to integers
            W_check = np.round(W_curr * 2) / 2.0
            
            # Calculate error if we snap to half-integers
            # We need to verify the full tensor equation with the snapped W
            # and the original U, V
            
            # Fast check: only check tensor entries first
            error = 0.0
            for (u, v, w) in tensor_entries:
                val = sum(U_init[u, r] * V_init[v, r] * W_check[w, r] for r in range(target_rank))
                error += abs(val - 1.0)
                
            if error > 1e-6:
                # Try gradient refinement if we're close
                if error < 1.0:
                    result = try_gradient_refinement(U_init, V_init, W_check, target_rank, max_iters=200)
                    if result is not None:
                        U_res, V_res, W_res = result
                        best_rank = target_rank
                        best_result = {"U": U_res, "V": V_res, "W": W_res}
                        found = True
                        break
                continue
                
            # Full verification
            if verify_solution(U_init.tolist(), V_init.tolist(), W_check.tolist(), target_rank):
                best_rank = target_rank
                best_result = {
                    "U": U_init.tolist(),
                    "V": V_init.tolist(),
                    "W": W_check.tolist()
                }
                found = True
                break
        
        if found:
            break
    
    # If we found a solution, return it
    if best_rank < n * m * p:
        return best_result
    
    # Return schoolbook if nothing better found
    return best_result
# EVOLVE-BLOCK-END
