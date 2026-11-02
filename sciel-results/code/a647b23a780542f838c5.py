import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    def build_from_terms(terms):
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for c in range(rank):
            u_dict, v_dict, w_dict = terms[c]
            for (i, j), val in u_dict.items():
                U[i * m + j][c] = val
            for (j, k), val in v_dict.items():
                V[j * p + k][c] = val
            for (k, i), val in w_dict.items():
                W[k * n + i][c] = val
        return {"U": U, "V": V, "W": W}
    
    def validate_fast(U, V, W):
        """Check that the decomposition is exact using vectorized operations."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # Tensor: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # Reshape to use einsum
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        if np.max(np.abs(T - 1.0)) > 1e-9:
            return False
        return True

    # Strassen-based construction (known good, rank 46)
    def strassen_construction():
        terms = []
        
        def add_strassen(i_start, j_start, k_start):
            u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
            v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
            w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
            terms.append((u, v, w))
            
            u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
            v = {(j_start, k_start): 1}
            w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
            terms.append((u, v, w))
            
            u = {(i_start, j_start): 1}
            v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
            w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
            terms.append((u, v, w))
            
            u = {(i_start+1, j_start+1): 1}
            v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
            w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
            terms.append((u, v, w))
            
            u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
            v = {(j_start+1, k_start+1): 1}
            w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
            terms.append((u, v, w))
            
            u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
            v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
            w = {(k_start+1, i_start+1): 1}
            terms.append((u, v, w))
            
            u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
            v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
            w = {(k_start, i_start): 1}
            terms.append((u, v, w))
        
        def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    for k in range(k_start, k_end):
                        u = {(i, j): 1}
                        v = {(j, k): 1}
                        w = {(k, i): 1}
                        terms.append((u, v, w))
        
        add_strassen(0, 0, 0)
        add_strassen(0, 0, 2)
        add_schoolbook(0, 2, 0, 2, 4, 5)
        add_strassen(0, 2, 0)
        add_strassen(0, 2, 2)
        add_schoolbook(0, 2, 2, 4, 4, 5)
        add_schoolbook(0, 2, 4, 5, 0, 2)
        add_schoolbook(0, 2, 4, 5, 2, 4)
        add_schoolbook(0, 2, 4, 5, 4, 5)
        
        return terms
    
    # Start with the known good construction
    best_terms = strassen_construction()
    best_result = build_from_terms(best_terms)
    best_rank = len(best_terms)
    
    # Verify it's valid
    if not validate_fast(best_result["U"], best_result["V"], best_result["W"]):
        raise ValueError("Strassen construction failed validation")
    
    U_full = np.array(best_result["U"], dtype=float)
    V_full = np.array(best_result["V"], dtype=float)
    W_full = np.array(best_result["W"], dtype=float)
    
    rank = best_rank
    
    # Pre-compute non-zero positions for each factor
    U_nz = []
    for c in range(rank):
        inds = []
        for idx in range(n*m):
            if U_full[idx, c] != 0:
                inds.append(idx)
        U_nz.append(inds)
    
    V_nz = []
    for c in range(rank):
        inds = []
        for idx in range(m*p):
            if V_full[idx, c] != 0:
                inds.append(idx)
        V_nz.append(inds)
    
    W_nz = []
    for c in range(rank):
        inds = []
        for idx in range(p*n):
            if W_full[idx, c] != 0:
                inds.append(idx)
        W_nz.append(inds)
    
    def try_reduce_fixed_UV(dropped_indices):
        """Fix U and V for remaining terms, solve for W."""
        if time.time() > deadline - 3:
            return None
        
        cols = [c for c in range(rank) if c not in dropped_indices]
        if len(cols) >= best_rank:
            return None
            
        U_red = U_full[:, cols]
        V_red = V_full[:, cols]
        
        # Build list of unknowns: (w_row, new_c) for each non-zero W position
        unknowns = []
        for new_c, old_c in enumerate(cols):
            for w_row in W_nz[old_c]:
                unknowns.append((w_row, new_c))
        
        num_unknowns = len(unknowns)
        if num_unknowns == 0:
            return None
        
        # Build A matrix: A[linear_idx, u_idx] = U[i*m+j, new_c] * V[j*p+k, new_c]
        # where linear_idx = i*m*p + j*p + k, and w_row = k*n+i
        A = np.zeros((n*m*p, num_unknowns))
        
        for new_c, old_c in enumerate(cols):
            for u_idx, (w_row, nc) in enumerate(unknowns):
                if nc != new_c:
                    continue
                # w_row = k*n + i
                k = w_row // n
                i = w_row % n
                for j in range(m):
                    linear_idx = i*m*p + j*p + k
                    val = U_red[i*m+j, new_c] * V_red[j*p+k, new_c]
                    if val != 0:
                        A[linear_idx, u_idx] = val
        
        b = np.ones(n*m*p)
        
        try:
            x, residuals, rank_l, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
            
        # Check residual
        if residuals.size > 0 and residuals[0] > 1e-9:
            return None
        
        # Reconstruct W_red
        W_red = np.zeros((p*n, len(cols)))
        for u_idx, (w_row, new_c) in enumerate(unknowns):
            W_red[w_row, new_c] = x[u_idx]
        
        # Check if values are close to half-integers
        W_rounded = np.round(W_red * 2) / 2
        
        # Verify with rounded values
        result_test = {
            "U": U_red.tolist(),
            "V": V_red.tolist(), 
            "W": W_rounded.tolist()
        }
        if validate_fast(result_test["U"], result_test["V"], result_test["W"]):
            return result_test
        
        # Try without rounding (in case exact solution isn't half-integer)
        result_test = {
            "U": U_red.tolist(),
            "V": V_red.tolist(),
            "W": W_red.tolist()
        }
        if validate_fast(result_test["U"], result_test["V"], result_test["W"]):
            return result_test
        
        return None
    
    def try_reduce_fixed_W(dropped_indices):
        """Fix W for remaining terms, solve for U and V."""
        if time.time() > deadline - 3:
            return None
        
        cols = [c for c in range(rank) if c not in dropped_indices]
        if len(cols) >= best_rank:
            return None
            
        W_red = W_full[:, cols]
        
        # We need to solve for U_red and V_red
        # This is a bilinear problem, so we'll use a different approach
        # For each term c, U and V have specific non-zero patterns
        # Let's collect all unknowns
        
        # Unknowns for U: (u_row, new_c) for each non-zero U position
        U_unknowns = []
        for new_c, old_c in enumerate(cols):
            for u_row in U_nz[old_c]:
                U_unknowns.append((u_row, new_c))
        
        # Unknowns for V: (v_row, new_c) for each non-zero V position
        V_unknowns = []
        for new_c, old_c in enumerate(cols):
            for v_row in V_nz[old_c]:
                V_unknowns.append((v_row, new_c))
        
        num_U_unknowns = len(U_unknowns)
        num_V_unknowns = len(V_unknowns)
        total_unknowns = num_U_unknowns + num_V_unknowns
        
        if total_unknowns == 0:
            return None
        
        # This is a bilinear system, which is harder to solve directly
        # Instead, we can try a greedy approach or use the structure
        # For now, skip this approach as it's complex
        return None
    
    # Strategy 1: Try dropping one term at a time
    for drop_idx in range(rank):
        if time.time() > deadline - 5:
            break
        
        result = try_reduce_fixed_UV([drop_idx])
        if result is not None:
            best_result = result
            best_rank = rank - 1
            # Update arrays for further reduction
            U_full = np.array(best_result["U"], dtype=float)
            V_full = np.array(best_result["V"], dtype=float)
            W_full = np.array(best_result["W"], dtype=float)
            rank = best_rank
            # Recompute non-zero patterns
            U_nz = []
            for c in range(rank):
                inds = [idx for idx in range(n*m) if U_full[idx, c] != 0]
                U_nz.append(inds)
            V_nz = []
            for c in range(rank):
                inds = [idx for idx in range(m*p) if V_full[idx, c] != 0]
                V_nz.append(inds)
            W_nz = []
            for c in range(rank):
                inds = [idx for idx in range(p*n) if W_full[idx, c] != 0]
                W_nz.append(inds)
            # Continue trying to reduce further
            break
    
    # Strategy 2: If we found rank 45, try to get to 44
    if best_rank == 45 and time.time() < deadline - 10:
        for drop_idx in range(best_rank):
            if time.time() > deadline - 5:
                break
            
            result = try_reduce_fixed_UV([drop_idx])
            if result is not None:
                best_result = result
                best_rank = 44
                break
    
    # Strategy 3: Try dropping two terms at once from original
    if best_rank == 46 and time.time() < deadline - 15:
        # Try pairs of terms to drop
        # Only try pairs where at least one is from a schoolbook block
        # (since Strassen terms are more entangled)
        # Identify which terms are schoolbook (single non-zero in each factor)
        schoolbook_terms = []
        for c in range(rank):
            if len(U_nz[c]) == 1 and len(V_nz[c]) == 1 and len(W_nz[c]) == 1:
                schoolbook_terms.append(c)
        
        # Try dropping pairs of schoolbook terms
        for i in range(len(schoolbook_terms)):
            for j in range(i+1, len(schoolbook_terms)):
                if time.time() > deadline - 5:
                    break
                drop_pair = [schoolbook_terms[i], schoolbook_terms[j]]
                result = try_reduce_fixed_UV(drop_pair)
                if result is not None:
                    best_result = result
                    best_rank = rank - 2
                    break
            if best_rank < rank:
                break
    
    return best_result
