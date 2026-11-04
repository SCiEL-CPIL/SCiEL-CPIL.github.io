import numpy as np
import time

DIMENSIONS = [4, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0  # 5s margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    deadline = time.time() + TIME_LIMIT - 5  # safety margin
    
    U_rows = n * m  # 16
    V_rows = m * p  # 24
    W_rows = p * n  # 24
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(np.array(u_vec, dtype=np.float64))
        V_cols.append(np.array(v_vec, dtype=np.float64))
        W_cols.append(np.array(w_vec, dtype=np.float64))
    
    def strassen_2x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add 7 rank-1 terms for Strassen 2x2*2x2."""
        def ai(i,j): return (a_r+i)*m + (a_c+j)
        def bi(i,j): return (b_r+i)*p + (b_c+j)
        def ci(i,j): return (c_c+j)*n + (c_r+i)
        
        def term(a_entries, b_entries, c_entries):
            u = [0.0]*U_rows; v = [0.0]*V_rows; w = [0.0]*W_rows
            for idx, val in a_entries:
                u[idx] = val
            for idx, val in b_entries:
                v[idx] = val
            for idx, val in c_entries:
                w[idx] = val
            add_term(u, v, w)
        
        # M1 = (A11+A22)*(B11+B22)
        term([(ai(0,0),1.0),(ai(1,1),1.0)], [(bi(0,0),1.0),(bi(1,1),1.0)], [(ci(0,0),1.0),(ci(1,1),1.0)])
        # M2 = (A21+A22)*B11
        term([(ai(1,0),1.0),(ai(1,1),1.0)], [(bi(0,0),1.0)], [(ci(1,0),1.0),(ci(1,1),-1.0)])
        # M3 = A11*(B12-B22)
        term([(ai(0,0),1.0)], [(bi(0,1),1.0),(bi(1,1),-1.0)], [(ci(0,1),1.0),(ci(1,1),1.0)])
        # M4 = A22*(B21-B11)
        term([(ai(1,1),1.0)], [(bi(1,0),1.0),(bi(0,0),-1.0)], [(ci(0,0),1.0),(ci(1,0),1.0)])
        # M5 = (A11+A12)*B22
        term([(ai(0,0),1.0),(ai(0,1),1.0)], [(bi(1,1),1.0)], [(ci(0,0),-1.0),(ci(0,1),1.0)])
        # M6 = (A21-A11)*(B11+B12)
        term([(ai(1,0),1.0),(ai(0,0),-1.0)], [(bi(0,0),1.0),(bi(0,1),1.0)], [(ci(1,1),1.0)])
        # M7 = (A12-A22)*(B21+B22)
        term([(ai(0,1),1.0),(ai(1,1),-1.0)], [(bi(1,0),1.0),(bi(1,1),1.0)], [(ci(0,0),1.0)])
    
    # Build the rank-84 decomposition
    for block in range(3):
        b_col_offset = block * 2
        c_col_offset = block * 2
        
        strassen_2x2_2x2(0, 0, 0, b_col_offset, 0, c_col_offset)
        strassen_2x2_2x2(0, 2, 2, b_col_offset, 0, c_col_offset)
        strassen_2x2_2x2(2, 0, 0, b_col_offset, 2, c_col_offset)
        strassen_2x2_2x2(2, 2, 2, b_col_offset, 2, c_col_offset)
    
    rank = len(U_cols)
    
    if time.time() > deadline:
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        def to_list(mat):
            return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # Build target tensor
    T = np.zeros((U_rows, V_rows, W_rows), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def check(U, V, W, atol=1e-9):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=atol)
    
    def err_of(U, V, W):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T - recon) ** 2)
    
    best_U = np.column_stack(U_cols)
    best_V = np.column_stack(V_cols)
    best_W = np.column_stack(W_cols)
    best_rank = rank
    best_err = err_of(best_U, best_V, best_W)
    
    # Strategy: Use SVD-based global linear compression
    # The idea is to find linear dependencies among the rank-1 terms
    # and eliminate them while preserving the tensor decomposition.
    
    def svd_compression(U, V, W, T, max_iter=200):
        """Try to compress the decomposition using SVD-based methods."""
        R = U.shape[1]
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try eliminating each column
            for r in range(R):
                if time.time() > deadline:
                    break
                
                mask = [i for i in range(R) if i != r]
                U_trial = U[:, mask]
                V_trial = V[:, mask]
                W_trial = W[:, mask]
                
                recon = np.einsum('ar,br,cr->abc', U_trial, V_trial, W_trial)
                err = np.sum((T - recon) ** 2)
                
                if err < 1e-10:
                    # Found a removable column
                    U = U_trial
                    V = V_trial
                    W = W_trial
                    R -= 1
                    improved = True
                    break
            
            if not improved:
                # Try a more sophisticated approach:
                # Check if any two columns can be merged or if there's
                # a linear relationship that allows elimination
                
                # For each pair of columns, check if their difference
                # or some linear combination can be absorbed
                found = False
                for r1 in range(R):
                    if time.time() > deadline:
                        break
                    for r2 in range(r1 + 1, R):
                        if time.time() > deadline:
                            break
                        
                        # Try to find if column r2 can be expressed as
                        # a linear combination involving column r1 and others
                        # This is complex, so try a simpler heuristic:
                        # Check if removing both and adjusting works
                        
                        mask = [i for i in range(R) if i not in [r1, r2]]
                        if len(mask) == 0:
                            continue
                        
                        U_trial = U[:, mask]
                        V_trial = V[:, mask]
                        W_trial = W[:, mask]
                        
                        recon = np.einsum('ar,br,cr->abc', U_trial, V_trial, W_trial)
                        err = np.sum((T - recon) ** 2)
                        
                        if err < 1e-10:
                            U = U_trial
                            V = V_trial
                            W = W_trial
                            R -= 2
                            improved = True
                            found = True
                            break
                    
                    if found:
                        break
                
                if not improved:
                    break
        
        return U, V, W, R
    
    # Apply SVD compression
    best_U, best_V, best_W, best_rank = svd_compression(best_U, best_V, best_W, T)
    
    # Try additional random transformations to find more dependencies
    rng = np.random.RandomState(42)
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Apply a random orthogonal transformation to the columns
        R = best_rank
        Q, _ = np.linalg.qr(rng.randn(R, R))
        
        U_new = best_U @ Q
        V_new = best_V @ Q
        W_new = best_W @ Q
        
        # Check if tensor is preserved
        recon = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
        err = np.sum((T - recon) ** 2)
        
        if err < 1e-10:
            # Try compression on the transformed version
            U_trial, V_trial, W_trial, R_trial = svd_compression(U_new, V_new, W_new, T)
            
            if R_trial < best_rank:
                best_U = U_trial
                best_V = V_trial
                best_W = W_trial
                best_rank = R_trial
    
    # Final verification
    if check(best_U, best_V, best_W):
        def to_list(mat):
            return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
    
    # Fallback to original
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
    return {"U": to_list(np.column_stack(U_cols)), "V": to_list(np.column_stack(V_cols)), "W": to_list(np.column_stack(W_cols))}
