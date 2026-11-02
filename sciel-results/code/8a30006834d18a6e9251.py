"""
Matrix multiplication 2x5 x 5x5: rank-minimal exact tensor decomposition.

Strategy:
1. Build initial decomposition using Strassen on 2x2x2 sub-blocks + schoolbook
2. Run local optimization to try to reduce rank by finding linear dependencies
"""

import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Build the full tensor T[a, b, c]
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def build_decomposition_from_terms(terms):
        """Build U, V, W matrices from a list of (u_dict, v_dict, w_dict) terms."""
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for r, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), val in u_dict.items():
                U[i * m + j, r] = val
            for (j, k), val in v_dict.items():
                V[j * p + k, r] = val
            for (k, i), val in w_dict.items():
                W[k * n + i, r] = val
        
        return U, V, W
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the tensor T."""
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T - Rec))
        return err < tol, err
    
    def add_strassen(i_start, j_start, k_start, terms):
        """Adds 7 terms for 2x2x2 block at (i_start, j_start, k_start)"""
        # M1 = (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
        u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        # M2 = (A[i1,j0]+A[i1,j1])*B[j0,k0]
        u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1}
        w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
        terms.append((u, v, w))
        
        # M3 = A[i0,j0]*(B[j0,k1]-B[j1,k1])
        u = {(i_start, j_start): 1}
        v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
        w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        # M4 = A[i1,j1]*(B[j1,k0]-B[j0,k0])
        u = {(i_start+1, j_start+1): 1}
        v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
        w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
        terms.append((u, v, w))
        
        # M5 = (A[i0,j0]+A[i0,j1])*B[j1,k1]
        u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
        v = {(j_start+1, k_start+1): 1}
        w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
        terms.append((u, v, w))
        
        # M6 = (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1])
        u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
        v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
        w = {(k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        # M7 = (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1])
        u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
        v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1}
        terms.append((u, v, w))
    
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end, terms):
        """Adds schoolbook terms for the specified range"""
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    terms.append((u, v, w))
    
    # Build initial decomposition
    terms = []
    
    # For each combination of i-block, j-block, k-block:
    # i can be {0,1} (size 2)
    # j can be split as {0,1}, {2,3}, {4}
    # k can be split as {0,1}, {2,3}, {4}
    
    # j in {0,1}:
    #   k in {0,1}: Strassen
    add_strassen(0, 0, 0, terms)
    #   k in {2,3}: Strassen
    add_strassen(0, 0, 2, terms)
    #   k in {4}: Schoolbook
    add_schoolbook(0, 2, 0, 2, 4, 5, terms)
    
    # j in {2,3}:
    #   k in {0,1}: Strassen
    add_strassen(0, 2, 0, terms)
    #   k in {2,3}: Strassen
    add_strassen(0, 2, 2, terms)
    #   k in {4}: Schoolbook
    add_schoolbook(0, 2, 2, 4, 4, 5, terms)
    
    # j in {4}:
    #   k in {0,1}: Schoolbook
    add_schoolbook(0, 2, 4, 5, 0, 2, terms)
    #   k in {2,3}: Schoolbook
    add_schoolbook(0, 2, 4, 5, 2, 4, terms)
    #   k in {4}: Schoolbook
    add_schoolbook(0, 2, 4, 5, 4, 5, terms)
    
    U, V, W = build_decomposition_from_terms(terms)
    initial_rank = U.shape[1]
    
    # Verify initial decomposition
    valid, err = verify_decomposition(U, V, W)
    if not valid:
        # Fallback to pure schoolbook
        terms = []
        add_schoolbook(0, 2, 0, 5, 0, 5, terms)
        U, V, W = build_decomposition_from_terms(terms)
        initial_rank = U.shape[1]
    
    # Local optimization to reduce rank
    def try_reduce_rank(U, V, W, max_reduction=8, steps_per_trial=200):
        """Try to reduce rank by removing dependent terms and re-optimizing."""
        R = U.shape[1]
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = R
        
        # Compute norms to identify potentially removable terms
        norms = np.linalg.norm(U, axis=0) * np.linalg.norm(V, axis=0) * np.linalg.norm(W, axis=0)
        sorted_indices = np.argsort(norms)
        
        for reduction in range(1, max_reduction + 1):
            if time.time() > deadline - 20:
                break
                
            new_R = R - reduction
            if new_R < 1:
                break
            
            # Try removing the smallest-norm terms
            remove_indices = sorted_indices[:reduction]
            keep_indices = np.setdiff1d(np.arange(R), remove_indices)
            
            U_new = U[:, keep_indices].copy()
            V_new = V[:, keep_indices].copy()
            W_new = W[:, keep_indices].copy()
            
            # Local optimization with the reduced rank
            lr = 0.01
            momentum = 0.9
            v_u, v_v, v_w = np.zeros_like(U_new), np.zeros_like(V_new), np.zeros_like(W_new)
            
            for step in range(steps_per_trial):
                if time.time() > deadline - 15:
                    break
                
                # Compute reconstruction
                Rec = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
                grad_U = np.einsum('abc,br,cr->ar', (T - Rec), V_new, W_new)
                grad_V = np.einsum('abc,ar,cr->br', (T - Rec), U_new, W_new)
                grad_W = np.einsum('abc,ar,br->cr', (T - Rec), U_new, V_new)
                
                # Update with momentum
                v_u = momentum * v_u - lr * grad_U
                v_v = momentum * v_v - lr * grad_V
                v_w = momentum * v_w - lr * grad_W
                
                U_new += v_u
                V_new += v_v
                W_new += v_w
                
                # Check convergence
                if step % 20 == 0:
                    Rec_check = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
                    err_check = np.max(np.abs(T - Rec_check))
                    if err_check < 1e-9:
                        break
            
            # Final check
            Rec_final = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
            err_final = np.max(np.abs(T - Rec_final))
            
            if err_final < 1e-9:
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = new_R
        
        return best_U, best_V, best_W, best_rank
    
    # Try to reduce rank
    if time.time() < deadline - 30:
        U_opt, V_opt, W_opt, final_rank = try_reduce_rank(U, V, W)
        if final_rank < initial_rank:
            U, V, W = U_opt, V_opt, W_opt
    
    # Convert to list of lists with integer values where possible
    def to_list(mat):
        result = []
        for i in range(mat.shape[0]):
            row = []
            for c in range(mat.shape[1]):
                val = mat[i, c]
                if abs(val - round(val)) < 1e-9:
                    row.append(int(round(val)))
                else:
                    row.append(float(val))
            result.append(row)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
