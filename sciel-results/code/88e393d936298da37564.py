import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Reshape to matrix M of size (nm, mp*pn)
    M = T.reshape(nm, mp * pn)
    
    best_rank = None
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from 20 to 24
    for R in range(20, 25):
        # Compute SVD
        try:
            U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        except:
            continue
        
        if R > len(s):
            continue
            
        # Take top R components
        U_top = U_svd[:, :R]
        s_top = s[:R]
        Vt_top = Vt[:R, :]
        
        # Reconstruct M_R
        M_R = (U_top * s_top) @ Vt_top
        
        # Check if M_R approximates M
        if not np.allclose(M_R, M, atol=1e-6):
            continue
        
        # Vt_top has shape (R, mp*pn), reshape to (R, mp, pn)
        B_reshaped = Vt_top.reshape(R, mp, pn)
        
        # Check if each B_r is rank 1
        valid = True
        for r in range(R):
            Br = B_reshaped[r]
            if np.linalg.matrix_rank(Br) > 1:
                valid = False
                break
        
        if not valid:
            continue
        
        # Factor each rank-1 matrix
        U_mat = np.zeros((nm, R))
        V_mat = np.zeros((mp, R))
        W_mat = np.zeros((pn, R))
        
        success = True
        for r in range(R):
            Br = B_reshaped[r]
            if np.allclose(Br, 0, atol=1e-10):
                continue
                
            # Find a non-zero row
            row_sums = np.abs(Br).sum(axis=1)
            if np.max(row_sums) < 1e-10:
                continue
            row_idx = np.argmax(row_sums)
            v_r = Br[row_idx].copy()
            
            # Find b_idx where v_r[b_idx] != 0
            non_zero = np.abs(v_r) > 1e-10
            if not np.any(non_zero):
                success = False
                break
            b_idx = np.where(non_zero)[0][0]
            w_r = Br[b_idx] / v_r[b_idx]
            
            # Verify
            Br_check = np.outer(v_r, w_r)
            if not np.allclose(Br, Br_check, atol=1e-6):
                success = False
                break
            
            A_r = U_top[:, r] * s_top[r]
            U_mat[:, r] = A_r
            V_mat[:, r] = v_r
            W_mat[:, r] = w_r
        
        if not success:
            continue
        
        # Check if entries are close to half-integers
        all_half_int = True
        for mat in [U_mat, V_mat, W_mat]:
            diff = np.abs(2 * mat - np.round(2 * mat))
            if np.max(diff) > 1e-4:
                all_half_int = False
                break
        
        if not all_half_int:
            continue
        
        # Round to half-integers
        U_mat = np.round(2 * U_mat) / 2
        V_mat = np.round(2 * V_mat) / 2
        W_mat = np.round(2 * W_mat) / 2
        
        # Verify the decomposition
        T_check = np.zeros((nm, mp, pn))
        for r in range(R):
            T_check += np.einsum('ar,br,cr->abc', U_mat[:, r], V_mat[:, r], W_mat[:, r])
        
        if np.allclose(T_check, T, atol=1e-6):
            best_rank = R
            best_U = U_mat
            best_V = V_mat
            best_W = W_mat
            break
    
    if best_rank is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Fallback to schoolbook
    rank = nm * mp * pn
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
