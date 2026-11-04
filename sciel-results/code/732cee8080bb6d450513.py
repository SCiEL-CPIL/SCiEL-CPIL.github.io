"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or as low as possible) tensor decomposition
    for 3x4 * 4x4 matrix multiplication.
    
    Strategy:
    1. Start with the schoolbook decomposition (rank = 48).
    2. Use vectorized verification for speed.
    3. Use local search with perturbation to find rank reductions.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    def _schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        
        return U, V, W, rank
    
    def _verify_fast(U, V, W, tol=1e-10):
        """
        Vectorized verification.
        T[i,j,k,l] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+l, r]
        should equal 1 if i==l and j==k, else 0
        """
        rank = U.shape[1]
        # Reshape for efficient computation
        # For each r, we have U_r (n*m,), V_r (m*p,), W_r (p*n,)
        # The contribution to the tensor is outer product
        
        # Build the full tensor T of shape (n, m, p, p)
        # T[i,j,k,l] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+l,r]
        
        # Efficient approach: for each r, compute the rank-1 tensor and sum
        # But that's still O(rank * n*m*p*p)
        
        # Alternative: use einsum
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_r = U.reshape(n, m, rank)  # U_r[i,j,r]
        V_r = V.reshape(m, p, rank)  # V_r[j,k,r]
        W_r = W.reshape(p, n, rank)  # W_r[k,l,r]
        
        # T[i,j,k,l] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,l,r]
        # Use einsum: 'ijr,jkr,klr->ijkl'
        T = np.einsum('ijr,jkr,klr->ijkl', U_r, V_r, W_r, optimize=True)
        
        # Expected: T[i,j,k,l] = 1 if i==l and j==k, else 0
        # Build expected tensor
        expected = np.zeros((n, m, p, p), dtype=float)
        for i in range(n):
            for j in range(m):
                expected[i, j, j, i] = 1.0
        
        max_err = np.max(np.abs(T - expected))
        return max_err < tol, max_err
    
    def _try_reduce_rank(U, V, W, current_rank, max_time=30.0):
        """
        Try to reduce the rank by finding linear dependencies.
        Uses a more systematic approach: try to eliminate each column
        by solving for coefficients in all three matrices simultaneously.
        """
        start = time.time()
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = current_rank
        
        # Try to find columns that can be eliminated
        # A column r can be eliminated if U[:,r] is in span of other U columns,
        # V[:,r] is in span of other V columns, AND W[:,r] is in span of other W columns
        
        for attempt in range(200):
            if time.time() - start > max_time:
                break
            if time.time() > deadline:
                break
            
            # Pick a random column to try to eliminate
            r_elim = random.randint(0, best_rank - 1)
            
            other_cols = [c for c in range(best_rank) if c != r_elim]
            if len(other_cols) < 1:
                break
            
            U_other = best_U[:, other_cols]
            V_other = best_V[:, other_cols]
            W_other = best_W[:, other_cols]
            
            U_target = best_U[:, r_elim]
            V_target = best_V[:, r_elim]
            W_target = best_W[:, r_elim]
            
            # Solve for coefficients using least squares
            try:
                a_coeff, _, _, _ = np.linalg.lstsq(U_other, U_target, rcond=None)
                b_coeff, _, _, _ = np.linalg.lstsq(V_other, V_target, rcond=None)
                c_coeff, _, _, _ = np.linalg.lstsq(W_other, W_target, rcond=None)
            except:
                continue
            
            U_recon = U_other @ a_coeff
            V_recon = V_other @ b_coeff
            W_recon = W_other @ c_coeff
            
            U_err = np.linalg.norm(U_recon - U_target) / (np.linalg.norm(U_target) + 1e-10)
            V_err = np.linalg.norm(V_recon - V_target) / (np.linalg.norm(V_target) + 1e-10)
            W_err = np.linalg.norm(W_recon - W_target) / (np.linalg.norm(W_target) + 1e-10)
            
            if U_err < 1e-8 and V_err < 1e-8 and W_err < 1e-8:
                # Great! We can eliminate this column
                new_U = np.delete(best_U, r_elim, axis=1)
                new_V = np.delete(best_V, r_elim, axis=1)
                new_W = np.delete(best_W, r_elim, axis=1)
                
                # Verify using fast method
                valid, _ = _verify_fast(new_U, new_V, new_W, tol=1e-8)
                if valid:
                    best_U, best_V, best_W = new_U, new_V, new_W
                    best_rank -= 1
                    print(f"Reduced rank to {best_rank}")
        
        return best_U, best_V, best_W, best_rank
    
    def _local_search(U, V, W, current_rank, max_time=40.0):
        """
        Local search: perturb entries and try to create dependencies.
        This is more powerful than just trying to eliminate columns.
        """
        start = time.time()
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = current_rank
        best_err = float('inf')
        
        # First, verify current solution
        valid, best_err = _verify_fast(best_U, best_V, best_W, tol=1e-8)
        if not valid:
            # Try to repair
            pass
        
        # Local search: try small perturbations
        for iteration in range(500):
            if time.time() - start > max_time:
                break
            if time.time() > deadline:
                break
            
            # Pick random entries to perturb
            n_perturb = random.randint(1, 3)
            
            U_pert = best_U.copy()
            V_pert = best_V.copy()
            W_pert = best_W.copy()
            
            for _ in range(n_perturb):
                # Perturb a random entry in one of the matrices
                mat_choice = random.choice(['U', 'V', 'W'])
                if mat_choice == 'U':
                    row = random.randint(0, U_pert.shape[0] - 1)
                    col = random.randint(0, U_pert.shape[1] - 1)
                    U_pert[row, col] += random.uniform(-0.1, 0.1)
                elif mat_choice == 'V':
                    row = random.randint(0, V_pert.shape[0] - 1)
                    col = random.randint(0, V_pert.shape[1] - 1)
                    V_pert[row, col] += random.uniform(-0.1, 0.1)
                else:
                    row = random.randint(0, W_pert.shape[0] - 1)
                    col = random.randint(0, W_pert.shape[1] - 1)
                    W_pert[row, col] += random.uniform(-0.1, 0.1)
            
            # Check if perturbed solution is still valid
            valid, err = _verify_fast(U_pert, V_pert, W_pert, tol=1e-6)
            
            if valid and err < best_err:
                best_U, best_V, best_W = U_pert, V_pert, W_pert
                best_err = err
            
            # After some perturbations, try to reduce rank
            if iteration % 10 == 0:
                U_r, V_r, W_r, rank_r = _try_reduce_rank(
                    best_U, best_V, best_W, best_rank, max_time=5.0
                )
                if rank_r < best_rank:
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = rank_r
                    print(f"Local search reduced rank to {best_rank}")
        
        return best_U, best_V, best_W, best_rank
    
    # Start with schoolbook
    U, V, W, rank = _schoolbook()
    print(f"Initial rank: {rank}")
    
    # Try to reduce rank
    U, V, W, rank = _try_reduce_rank(U, V, W, rank, max_time=30.0)
    print(f"After rank reduction: {rank}")
    
    # Try local search for further reduction
    U, V, W, rank = _local_search(U, V, W, rank, max_time=50.0)
    print(f"After local search: {rank}")
    
    # Convert to list format with integer/half-integer entries if possible
    def _to_clean_list(arr):
        """Convert array to list, rounding to nearest half-integer if close."""
        result = []
        for row in arr:
            clean_row = []
            for val in row:
                # Check if close to integer or half-integer
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2.0
                if abs(val - nearest_int) < 1e-10:
                    clean_row.append(int(nearest_int))
                elif abs(val - nearest_half) < 1e-10:
                    clean_row.append(nearest_half)
                else:
                    clean_row.append(float(val))
            result.append(clean_row)
        return result
    
    return {"U": _to_clean_list(U), "V": _to_clean_list(V), "W": _to_clean_list(W)}
# EVOLVE-BLOCK-END
