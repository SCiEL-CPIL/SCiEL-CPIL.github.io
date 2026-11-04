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
    2. Use systematic dependency detection to reduce rank.
    3. Apply local search with tighter tolerances for final polish.
    4. Try structured reductions using tensor rank properties.
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
    
    def _verify(U, V, W, tol=1e-10):
        """Verify that the decomposition is correct."""
        n, m, p = DIMENSIONS
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Check: for all i in [0,n), j in [0,m), k in [0,p), l in [0,p):
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+l, r] = 1 if i==l and j==k else 0
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for l in range(p):
                        expected = 1.0 if (i == l and j == k) else 0.0
                        actual = 0.0
                        for r in range(U.shape[1]):
                            actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+l, r]
                        if abs(actual - expected) > tol:
                            return False
        return True
    
    def _find_and_eliminate_dependencies(U, V, W, current_rank, max_time=5.0, tol=1e-10):
        """
        Systematically find and eliminate linear dependencies.
        Uses structured approach: for each column, check if it's in span of others
        across all three matrices simultaneously.
        """
        start = time.time()
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = current_rank
        eliminated = set()
        
        # Multiple passes to handle cascading dependencies
        for pass_num in range(10):
            if time.time() - start > max_time:
                break
            if time.time() > deadline:
                break
            
            found_elimination = False
            
            for r_elim in range(best_rank):
                if time.time() - start > max_time or time.time() > deadline:
                    break
                
                if r_elim in eliminated:
                    continue
                
                other_cols = [c for c in range(best_rank) if c != r_elim and c not in eliminated]
                if len(other_cols) < 1:
                    continue
                
                # Check if column r_elim is in the span of others in all three matrices
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
                
                # Check reconstruction error
                U_recon = U_other @ a_coeff
                V_recon = V_other @ b_coeff
                W_recon = W_other @ c_coeff
                
                U_norm = np.linalg.norm(U_target)
                V_norm = np.linalg.norm(V_target)
                W_norm = np.linalg.norm(W_target)
                
                U_err = np.linalg.norm(U_recon - U_target) / (U_norm + 1e-15)
                V_err = np.linalg.norm(V_recon - V_target) / (V_norm + 1e-15)
                W_err = np.linalg.norm(W_recon - W_target) / (W_norm + 1e-15)
                
                if U_err < tol and V_err < tol and W_err < tol:
                    # This column can be eliminated
                    new_U = np.delete(best_U, r_elim, axis=1)
                    new_V = np.delete(best_V, r_elim, axis=1)
                    new_W = np.delete(best_W, r_elim, axis=1)
                    
                    # Verify before accepting
                    if _verify(new_U, new_V, new_W, tol=1e-9):
                        best_U, best_V, best_W = new_U, new_V, new_W
                        best_rank -= 1
                        eliminated.add(r_elim)
                        found_elimination = True
        
        return best_U, best_V, best_W, best_rank
    
    def _local_search_polish(U, V, W, current_rank, max_time=10.0, tol=1e-10):
        """
        Local search with tighter tolerances for final polish.
        Tries random column eliminations with strict verification.
        """
        start = time.time()
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = current_rank
        
        for attempt in range(200):
            if time.time() - start > max_time:
                break
            if time.time() > deadline:
                break
            
            if best_rank <= 1:
                break
            
            r_elim = random.randint(0, best_rank - 1)
            
            other_cols = [c for c in range(best_rank) if c != r_elim]
            
            U_other = best_U[:, other_cols]
            V_other = best_V[:, other_cols]
            W_other = best_W[:, other_cols]
            
            U_target = best_U[:, r_elim]
            V_target = best_V[:, r_elim]
            W_target = best_W[:, r_elim]
            
            try:
                a_coeff, _, _, _ = np.linalg.lstsq(U_other, U_target, rcond=None)
                b_coeff, _, _, _ = np.linalg.lstsq(V_other, V_target, rcond=None)
                c_coeff, _, _, _ = np.linalg.lstsq(W_other, W_target, rcond=None)
            except:
                continue
            
            U_recon = U_other @ a_coeff
            V_recon = V_other @ b_coeff
            W_recon = W_other @ c_coeff
            
            U_err = np.linalg.norm(U_recon - U_target)
            V_err = np.linalg.norm(V_recon - V_target)
            W_err = np.linalg.norm(W_recon - W_target)
            
            total_err = U_err + V_err + W_err
            
            if total_err < tol:
                new_U = np.delete(best_U, r_elim, axis=1)
                new_V = np.delete(best_V, r_elim, axis=1)
                new_W = np.delete(best_W, r_elim, axis=1)
                
                if _verify(new_U, new_V, new_W, tol=1e-9):
                    best_U, best_V, best_W = new_U, new_V, new_W
                    best_rank -= 1
        
        return best_U, best_V, best_W, best_rank
    
    def _polish_with_tighter_tol(U, V, W, current_rank, max_time=5.0):
        """
        Final polish pass with very tight tolerance to catch near-misses.
        """
        start = time.time()
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = current_rank
        
        # Use extremely tight tolerance
        tight_tol = 1e-12
        
        for attempt in range(50):
            if time.time() - start > max_time:
                break
            if time.time() > deadline:
                break
            
            if best_rank <= 1:
                break
            
            r_elim = random.randint(0, best_rank - 1)
            
            other_cols = [c for c in range(best_rank) if c != r_elim]
            
            U_other = best_U[:, other_cols]
            V_other = best_V[:, other_cols]
            W_other = best_W[:, other_cols]
            
            U_target = best_U[:, r_elim]
            V_target = best_V[:, r_elim]
            W_target = best_W[:, r_elim]
            
            try:
                a_coeff, _, _, _ = np.linalg.lstsq(U_other, U_target, rcond=None)
                b_coeff, _, _, _ = np.linalg.lstsq(V_other, V_target, rcond=None)
                c_coeff, _, _, _ = np.linalg.lstsq(W_other, W_target, rcond=None)
            except:
                continue
            
            U_recon = U_other @ a_coeff
            V_recon = V_other @ b_coeff
            W_recon = W_other @ c_coeff
            
            # Use tighter error metric
            U_err = np.max(np.abs(U_recon - U_target))
            V_err = np.max(np.abs(V_recon - V_target))
            W_err = np.max(np.abs(W_recon - W_target))
            
            max_err = max(U_err, V_err, W_err)
            
            if max_err < tight_tol:
                new_U = np.delete(best_U, r_elim, axis=1)
                new_V = np.delete(best_V, r_elim, axis=1)
                new_W = np.delete(best_W, r_elim, axis=1)
                
                if _verify(new_U, new_V, new_W, tol=1e-11):
                    best_U, best_V, best_W = new_U, new_V, new_W
                    best_rank -= 1
        
        return best_U, best_V, best_W, best_rank
    
    # Start with schoolbook
    U, V, W, rank = _schoolbook()
    
    # Systematic dependency elimination
    U, V, W, rank = _find_and_eliminate_dependencies(U, V, W, rank, max_time=25.0, tol=1e-10)
    
    # Local search polish
    U, V, W, rank = _local_search_polish(U, V, W, rank, max_time=30.0, tol=1e-10)
    
    # Final tight polish
    U, V, W, rank = _polish_with_tighter_tol(U, V, W, rank, max_time=15.0)
    
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
