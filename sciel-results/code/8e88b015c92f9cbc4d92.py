import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    nm, mp, pn = n * m, m * p, p * n
    
    # The problem decomposes into 3 independent 2x3x3 matrix multiplications.
    # We find a decomposition for 2x3x3 and replicate it.
    
    def solve_2x3x3(max_rank=12, max_trials=2000):
        # Tensor T2[i*m+j, j*p3+k, k*n+i] for n=2, m=3, p=3
        n2, m2, p2 = 2, 3, 3
        nm2, mp2, pn2 = n2*m2, m2*p2, p2*n2
        T2 = np.zeros((nm2, mp2, pn2))
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    T2[i*m2+j, j*p2+k, k*n2+i] = 1.0
        
        def verify(U, V, W, T, tol=1e-9):
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            return np.allclose(T_approx, T, atol=tol)
        
        def is_half_integer(mat, tol=1e-6):
            scaled = 2.0 * mat
            rounded = np.round(scaled)
            return np.all(np.abs(scaled - rounded) < tol)

        def try_solve_for_U(V, W, T, r):
            M = (V[:, None, :] * W[None, :, :]).reshape(mp2 * pn2, r)
            # Check rank of M
            if np.linalg.matrix_rank(M) < r:
                return None
            
            U = np.zeros((nm2, r))
            for a in range(nm2):
                target = T[a, :, :].flatten()
                try:
                    sol, residuals, rank_m, sv = np.linalg.lstsq(M, target, rcond=None)
                    if residuals[0] > 1e-8:
                        return None
                    U[a, :] = sol
                except:
                    return None
            return U

        for r in range(10, max_rank + 1):
            for trial in range(max_trials):
                np.random.seed(10000 + r * 100 + trial)
                
                # Random V and W
                # Using sparse random might help
                V = np.random.randint(-1, 2, size=(mp2, r))
                W = np.random.randint(-1, 2, size=(pn2, r))
                
                U = try_solve_for_U(V, W, T2, r)
                if U is not None:
                    # Check if U is half-integer
                    if is_half_integer(U):
                        U_snapped = np.round(2.0 * U) / 2.0
                        if verify(U_snapped, V, W, T2):
                            return U_snapped, V, W, r
                    else:
                        # If not half-integer, check if it's close to integer/half-integer
                        # and if snapping works
                        U_snapped = np.round(2.0 * U) / 2.0
                        if verify(U_snapped, V, W, T2):
                            return U_snapped, V, W, r
        
        return None

    deadline = time.time() + 110
    
    # Try to find 2x3x3 decomposition
    sol_233 = solve_2x3x3(max_rank=12, max_trials=1000)
    
    if sol_233 is not None:
        U2, V2, W2, r2 = sol_233
        # Assemble full solution
        # The full tensor is T = T2 (for k=0,1,2) + T2 (for k=3,4,5) + T2 (for k=6,7,8)
        # U is same for all blocks (A is shared)
        # V and W are block diagonal in the rank dimension
        
        U_full = np.zeros((nm, r2 * 3))
        V_full = np.zeros((mp, r2 * 3))
        W_full = np.zeros((pn, r2 * 3))
        
        for block in range(3):
            # U columns are the same for all blocks?
            # Wait, U depends on A. A is shared.
            # In the tensor decomposition T = sum U_r \otimes V_r \otimes W_r
            # For block b, the tensor is T_b.
            # T_b[i j, j k_b, k_b i] = 1
            # This is isomorphic to T2[i j, j k, k i].
            # The decomposition T2 = sum_r u_r \otimes v_r \otimes w_r
            # For block b, we use the same u_r, but v_r is placed in rows j*p + (b*3+k), and w_r in rows (b*3+k)*n + i.
            
            # U part:
            U_full[:, block*r2:(block+1)*r2] = U2
            
            # V part:
            # V2 has rows j*p2 + k.
            # We need to map these to V_full rows j*p + (b*3 + k).
            for j in range(m2):
                for k in range(p2):
                    row_233 = j * p2 + k
                    row_full = j * p + (block * 3 + k)
                    V_full[row_full, block*r2:(block+1)*r2] = V2[row_233, :]
            
            # W part:
            # W2 has rows k*n2 + i.
            # We need to map these to W_full rows (b*3 + k)*n + i.
            for k in range(p2):
                for i in range(n2):
                    row_233 = k * n2 + i
                    row_full = (block * 3 + k) * n + i
                    W_full[row_full, block*r2:(block+1)*r2] = W2[row_233, :]
        
        # Verify full solution
        T_full = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_full[i*m+j, j*p+k, k*n+i] = 1.0
        
        T_approx = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
        if np.allclose(T_approx, T_full, atol=1e-9):
            return {
                "U": [[float(x) for x in row] for row in U_full],
                "V": [[float(x) for x in row] for row in V_full],
                "W": [[float(x) for x in row] for row in W_full]
            }
        
    # Fallback: Schoolbook
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    return {
        "U": [[float(x) for x in row] for row in U_sb],
        "V": [[float(x) for x in row] for row in V_sb],
        "W": [[float(x) for x in row] for row in W_sb]
    }
