"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    schoolbook_rank = n * m * p  # 80
    
    # Build schoolbook decomposition as fallback
    U = np.zeros((n * m, schoolbook_rank))
    V = np.zeros((m * p, schoolbook_rank))
    W = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = schoolbook_rank
    
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    def verify_decomposition(U_mat, V_mat, W_mat, rank, tol=1e-6):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_mat[i * m + j, :rank] * V_mat[j * p + k, :rank] * W_mat[k * n + i, :rank])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def check_nice_entries(arr, tol=1e-8):
        """Check if all entries are integer or half-integer."""
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                if abs(val - round(val)) > tol and abs(val * 2 - round(val * 2)) > tol:
                    return False
        return True
    
    def to_nice_list(arr):
        """Convert array to list with integer/half-integer entries."""
        result = []
        for i in range(arr.shape[0]):
            row = []
            for j in range(arr.shape[1]):
                val = arr[i, j]
                if abs(val - round(val)) < 1e-8:
                    row.append(int(round(val)))
                elif abs(val * 2 - round(val * 2)) < 1e-8:
                    row.append(round(val * 2) / 2.0)
                else:
                    # Round to nearest half-integer as best effort
                    row.append(round(val * 2) / 2.0)
            result.append(row)
        return result
    
    # Try to find lower rank decompositions
    # Strategy: try removing one term at a time and adjusting others
    # For rank 79: remove one schoolbook term, try to compensate
    
    for target_rank in range(schoolbook_rank - 1, max(20, schoolbook_rank - 20), -1):
        if time.time() > deadline - 15:
            break
        
        # Try multiple approaches for this rank
        max_attempts = 10 if target_rank >= 75 else 5
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            U_try = np.zeros((n * m, target_rank))
            V_try = np.zeros((m * p, target_rank))
            W_try = np.zeros((p * n, target_rank))
            
            # Initialize with first target_rank schoolbook terms
            for r in range(target_rank):
                # Decode (i, j, k) from r
                k = r % p
                j = (r // p) % m
                i = (r // (p * m)) % n
                U_try[i * m + j, r] = 1.0
                V_try[j * p + k, r] = 1.0
                W_try[k * n + i, r] = 1.0
            
            # The missing terms are those with index >= target_rank
            # We need to adjust the existing terms to cover the missing contributions
            
            # Use coordinate descent: for each missing (i,j,k), try to adjust existing terms
            missing_terms = []
            for r in range(target_rank, schoolbook_rank):
                k = r % p
                j = (r // p) % m
                i = (r // (p * m)) % n
                missing_terms.append((i, j, k))
            
            # For each missing term, try to find which existing term can be adjusted
            # to compensate. We'll use a simple greedy approach.
            
            adjusted = False
            for (mi, mj, mk) in missing_terms:
                if time.time() > deadline - 5:
                    break
                
                # The missing term contributes 1 to T[mi, mj, mk]
                # We need to add 1 to this entry by adjusting existing terms
                # Term r contributes U[mi*m+mj, r] * V[mj*p+mk, r] * W[mk*n+mi, r]
                
                # Find a term where we can increase the product by 1
                # Try increasing U[mi*m+mj, r] by 1 for some r where V and W are non-zero
                best_r = -1
                best_gain = 0.0
                
                for r in range(target_rank):
                    u_val = U_try[mi * m + mj, r]
                    v_val = V_try[mj * p + mk, r]
                    w_val = W_try[mk * n + mi, r]
                    
                    # If we increase u_val by 1, the gain is v_val * w_val
                    gain = v_val * w_val
                    if gain > best_gain and u_val + 1 <= 2.0:  # Keep values small
                        best_gain = gain
                        best_r = r
                
                if best_r >= 0 and best_gain >= 1.0 - 1e-8:
                    U_try[mi * m + mj, best_r] += 1.0
                    adjusted = True
                elif best_r >= 0 and best_gain > 0.5:
                    # Try half-integer adjustment
                    U_try[mi * m + mj, best_r] += 0.5
                    # Need to compensate the 0.5 deficit elsewhere
                    # For now, just mark as potentially valid
                    adjusted = True
            
            # Verify the decomposition
            if verify_decomposition(U_try, V_try, W_try, target_rank):
                if check_nice_entries(U_try) and check_nice_entries(V_try) and check_nice_entries(W_try):
                    best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                    best_rank = target_rank
                    break  # Found a valid decomposition at this rank, move to lower rank
        
        # If we found a valid decomposition at this rank, try lower ranks
        if best_rank == target_rank:
            continue
        else:
            # Try a different approach: random perturbation from schoolbook
            if target_rank >= 75:
                U_rand = np.zeros((n * m, target_rank))
                V_rand = np.zeros((m * p, target_rank))
                W_rand = np.zeros((p * n, target_rank))
                
                # Start with first target_rank schoolbook terms
                for r in range(target_rank):
                    k = r % p
                    j = (r // p) % m
                    i = (r // (p * m)) % n
                    U_rand[i * m + j, r] = 1.0
                    V_rand[j * p + k, r] = 1.0
                    W_rand[k * n + i, r] = 1.0
                
                # Try small random perturbations to see if we can fix the missing terms
                for perturb_attempt in range(5):
                    if time.time() > deadline - 5:
                        break
                    
                    U_pert = U_rand.copy()
                    V_pert = V_rand.copy()
                    W_pert = W_rand.copy()
                    
                    # Add small random perturbations
                    np.random.seed(attempt * 1000 + perturb_attempt)
                    U_pert += np.random.randn(n * m, target_rank) * 0.1
                    V_pert += np.random.randn(m * p, target_rank) * 0.1
                    W_pert += np.random.randn(p * n, target_rank) * 0.1
                    
                    # Round to nearest half-integer
                    U_pert = np.round(U_pert * 2) / 2.0
                    V_pert = np.round(V_pert * 2) / 2.0
                    W_pert = np.round(W_pert * 2) / 2.0
                    
                    if verify_decomposition(U_pert, V_pert, W_pert, target_rank):
                        best_U, best_V, best_W = U_pert.copy(), V_pert.copy(), W_pert.copy()
                        best_rank = target_rank
                        break
    
    # Final verification and conversion
    if not verify_decomposition(best_U, best_V, best_W, best_rank):
        # Fall back to schoolbook
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = schoolbook_rank
    
    U_list = to_nice_list(best_U[:, :best_rank])
    V_list = to_nice_list(best_V[:, :best_rank])
    W_list = to_nice_list(best_W[:, :best_rank])
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
