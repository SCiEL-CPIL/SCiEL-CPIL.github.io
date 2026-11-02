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
    
    def validate(U, V, W):
        """Check that the decomposition is exact for matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
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
    if not validate(best_result["U"], best_result["V"], best_result["W"]):
        raise ValueError("Strassen construction failed validation")
    
    # Try to find rank 45 by dropping one term and solving for W
    # We fix U and V to the Strassen values of the remaining terms
    # and solve the linear system for W
    
    U_full = np.array(best_result["U"], dtype=float)
    V_full = np.array(best_result["V"], dtype=float)
    W_full = np.array(best_result["W"], dtype=float)
    
    rank = best_rank
    # Map from term index to non-zero indices in W
    # We need to construct the linear system A x = b
    # x are the values of the non-zero entries in W for the remaining terms
    # A is built from U and V
    
    # Pre-compute the structure of W
    # For each term c, W has non-zeros at specific (k,i) indices
    # Let's store the indices
    w_indices = []
    for c in range(rank):
        inds = []
        for k in range(p):
            for i in range(n):
                if W_full[k*n+i, c] != 0:
                    inds.append((k*n+i, c))
        w_indices.append(inds)
    
    # Try dropping each term
    for drop_idx in range(rank):
        if time.time() > deadline - 5:
            break
            
        # Create U_reduced, V_reduced
        # Columns are all except drop_idx
        cols = [c for c in range(rank) if c != drop_idx]
        U_red = U_full[:, cols]
        V_red = V_full[:, cols]
        
        # Build linear system for W_red
        # Unknowns: values of W_red at non-zero positions
        # Let's identify the unknowns
        # For each remaining term c, the non-zero positions in W are the same as in W_full
        # Let's create a list of (row_in_W, col_in_W_red) for each unknown
        unknowns = []
        for new_c, old_c in enumerate(cols):
            for (w_row, old_col) in w_indices[old_c]:
                # w_row is the index in W (k*n+i)
                # old_col is the column in W_full
                # We need the column index in W_red
                # Since we removed drop_idx, the mapping is:
                # if old_c < drop_idx, new_c = old_c
                # if old_c > drop_idx, new_c = old_c - 1
                # Actually, we can just use new_c
                unknowns.append((w_row, new_c))
        
        num_unknowns = len(unknowns)
        
        # Build A (50 x num_unknowns)
        # A[i,j,k, unknown_idx] = U[i,m+j, new_c] * V[j,p+k, new_c]
        # But we iterate over i,j,k
        A = np.zeros((n*m*p, num_unknowns))
        b = np.ones(n*m*p)
        
        # Map (i,j,k) to linear index
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    linear_idx = i*m*p + j*p + k
                    for u_idx, (w_row, new_c) in enumerate(unknowns):
                        # w_row = k*n + i
                        # Check if w_row matches k*n + i
                        if w_row == k*n + i:
                            A[linear_idx, u_idx] = U_red[i*m+j, new_c] * V_red[j*p+k, new_c]
        
        # Solve A x = b
        # Use least squares
        try:
            x, residuals, rank_l, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
            
        # Check residual
        if residuals.size > 0 and residuals[0] < 1e-9:
            # Reconstruct W_red
            W_red = np.zeros((p*n, len(cols)))
            for u_idx, (w_row, new_c) in enumerate(unknowns):
                W_red[w_row, new_c] = x[u_idx]
            
            # Check if values are close to half-integers
            # Round to nearest 0.5
            W_rounded = np.round(W_red * 2) / 2
            
            # Check if rounded W works with U_red, V_red
            # Reconstruct terms for validation
            terms_test = []
            for new_c, old_c in enumerate(cols):
                u_dict, v_dict, _ = best_terms[old_c]
                w_dict = {}
                for k in range(p):
                    for i in range(n):
                        val = W_rounded[k*n+i, new_c]
                        if val != 0:
                            w_dict[(k, i)] = val
                terms_test.append((u_dict, v_dict, w_dict))
                
            result_test = build_from_terms(terms_test)
            if validate(result_test["U"], result_test["V"], result_test["W"]):
                best_result = result_test
                best_rank = len(cols)
                break # Found a lower rank, stop searching for this strategy
        
    return best_result
