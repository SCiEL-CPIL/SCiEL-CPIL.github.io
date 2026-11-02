import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 36
    
    # Build the target tensor T flattened
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Trivial solution as fallback
    U_triv = np.zeros((n * m, trivial_rank))
    V_triv = np.zeros((m * p, trivial_rank))
    W_triv = np.zeros((p * n, trivial_rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j, idx] = 1.0
                V_triv[j * p + k, idx] = 1.0
                W_triv[k * n + i, idx] = 1.0
                idx += 1

    deadline = time.time() + 110  # Safety margin
    
    # Pre-compute flattened T views
    T_flat_U = T.reshape(n * m, m * p * p * n)
    T_flat_V = T.reshape(m * p, n * m * p * n)
    T_flat_W = T.reshape(p * n, n * m * m * p)
    
    def polish_solution(U, V, W, r, max_iters=50):
        """High-precision polish pass with tighter tolerance"""
        for it in range(max_iters):
            if time.time() > deadline:
                return None, None, None, None
            
            # Update U
            D_U = np.einsum('br,cr->rbc', V, W).reshape(r, m * p * p * n)
            try:
                U_new = np.linalg.lstsq(D_U, T_flat_U.T, rcond=1e-12)[0].T
            except np.linalg.LinAlgError:
                return None, None, None, None
                
            # Update V
            D_V = np.einsum('ar,cr->rac', U_new, W).reshape(r, n * m * p * n)
            try:
                V_new = np.linalg.lstsq(D_V, T_flat_V.T, rcond=1e-12)[0].T
            except np.linalg.LinAlgError:
                return None, None, None, None

            # Update W
            D_W = np.einsum('ar,br->rab', U_new, V_new).reshape(r, n * m * m * p)
            try:
                W_new = np.linalg.lstsq(D_W, T_flat_W.T, rcond=1e-12)[0].T
            except np.linalg.LinAlgError:
                return None, None, None, None

            U, V, W = U_new, V_new, W_new
            
            # Check convergence with tighter tolerance
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(T - T_approx)
            
            if residual < 1e-10:
                return U, V, W, residual
        
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        residual = np.linalg.norm(T - T_approx)
        return U, V, W, residual
    
    def try_round_and_verify(U, V, W, r):
        """Try rounding to half-integers and verify"""
        for scale in [2, 4, 8, 16]:
            U_rounded = np.round(U * scale) / scale
            V_rounded = np.round(V * scale) / scale
            W_rounded = np.round(W * scale) / scale
            
            T_approx_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
            residual_rounded = np.linalg.norm(T - T_approx_rounded)
            
            if residual_rounded < 1e-8:
                return U_rounded, V_rounded, W_rounded, scale
        
        return None, None, None, None

    def strassen_init_2x2_block(i_start, j_start, k_start, U, V, W, base_idx):
        """
        Initialize a 2x2x2 Strassen block within the larger tensor.
        i_start, j_start, k_start: starting indices for the 2x2x2 block
        base_idx: starting column index
        """
        # Strassen's 7 multiplications for 2x2 matrices
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # For each of the 7 Strassen terms, we add a rank-1 component
        # The indices in U are (i, j), V are (j, k), W are (k, i)
        # But we need to map the 2x2 sub-block indices to the global indices
        
        # Let's define the local indices for the 2x2x2 block
        # i in {i_start, i_start+1}, j in {j_start, j_start+1}, k in {k_start, k_start+1}
        
        # Strassen coefficients for U (A matrix factors)
        # M1: A11=1, A22=1 -> U[i_start, j_start] += 1, U[i_start+1, j_start+1] += 1
        # M2: A21=1, A22=1 -> U[i_start+1, j_start] += 1, U[i_start+1, j_start+1] += 1
        # M3: A11=1 -> U[i_start, j_start] += 1
        # M4: A22=1 -> U[i_start+1, j_start+1] += 1
        # M5: A11=1, A12=1 -> U[i_start, j_start] += 1, U[i_start, j_start+1] += 1
        # M6: A21=1, A11=-1 -> U[i_start+1, j_start] += 1, U[i_start, j_start] -= 1
        # M7: A12=1, A22=-1 -> U[i_start, j_start+1] += 1, U[i_start+1, j_start+1] -= 1
        
        # Strassen coefficients for V (B matrix factors)
        # M1: B11=1, B22=1 -> V[j_start, k_start] += 1, V[j_start+1, k_start+1] += 1
        # M2: B11=1 -> V[j_start, k_start] += 1
        # M3: B12=1, B22=-1 -> V[j_start, k_start+1] += 1, V[j_start+1, k_start+1] -= 1
        # M4: B21=1, B11=-1 -> V[j_start+1, k_start] += 1, V[j_start, k_start] -= 1
        # M5: B22=1 -> V[j_start+1, k_start+1] += 1
        # M6: B11=1, B12=1 -> V[j_start, k_start] += 1, V[j_start, k_start+1] += 1
        # M7: B21=1, B22=1 -> V[j_start+1, k_start] += 1, V[j_start+1, k_start+1] += 1
        
        # Strassen coefficients for W (C matrix reconstruction)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # W[k, i] where k and i are local indices
        # M1: W[k_start, i_start] += 1, W[k_start+1, i_start+1] += 1
        # M2: W[k_start, i_start+1] -= 1
        # M3: W[k_start+1, i_start] += 1, W[k_start+1, i_start+1] += 1
        # M4: W[k_start, i_start] += 1, W[k_start+1, i_start+1] += 1
        # M5: W[k_start+1, i_start] -= 1
        # M6: W[k_start+1, i_start+1] += 1
        # M7: W[k_start, i_start] += 1
        
        # Let's implement this more carefully
        # For each Strassen term s (0 to 6), we add to U, V, W at column base_idx + s
        
        # Define the Strassen patterns
        # Each pattern is a list of (index_offset, coefficient) for U, V, W
        
        strassen_U = [
            [(0, 1), (1, 1)],  # M1: (i1,j1)=1, (i2,j2)=1
            [(1, 1), (1, 1)],  # M2: (i2,j1)=1, (i2,j2)=1
            [(0, 1)],          # M3: (i1,j1)=1
            [(1, 1)],          # M4: (i2,j2)=1
            [(0, 1), (0, 1)],  # M5: (i1,j1)=1, (i1,j2)=1
            [(1, 1), (0, -1)], # M6: (i2,j1)=1, (i1,j1)=-1
            [(0, 1), (1, -1)], # M7: (i1,j2)=1, (i2,j2)=-1
        ]
        
        strassen_V = [
            [(0, 1), (1, 1)],  # M1: (j1,k1)=1, (j2,k2)=1
            [(0, 1)],          # M2: (j1,k1)=1
            [(0, 1), (1, -1)], # M3: (j1,k2)=1, (j2,k2)=-1
            [(1, 1), (0, -1)], # M4: (j2,k1)=1, (j1,k1)=-1
            [(1, 1)],          # M5: (j2,k2)=1
            [(0, 1), (0, 1)],  # M6: (j1,k1)=1, (j1,k2)=1
            [(1, 1), (1, 1)],  # M7: (j2,k1)=1, (j2,k2)=1
        ]
        
        strassen_W = [
            [(0, 1), (1, 1)],  # M1: (k1,i1)=1, (k2,i2)=1
            [(0, -1)],         # M2: (k1,i2)=-1
            [(1, 1), (1, 1)],  # M3: (k2,i1)=1, (k2,i2)=1
            [(0, 1), (1, 1)],  # M4: (k1,i1)=1, (k2,i2)=1
            [(1, -1)],         # M5: (k2,i1)=-1
            [(1, 1)],          # M6: (k2,i2)=1
            [(0, 1)],          # M7: (k1,i1)=1
        ]
        
        for s in range(7):
            col = base_idx + s
            
            # Add U contributions
            for (offset, coeff) in strassen_U[s]:
                i_local = offset // 2
                j_local = offset % 2
                U[i_start + i_local, col] += coeff * (1 if j_start == 0 else 1)
                # Actually, let me be more careful with the indexing
                # offset encodes (i_local, j_local)
                # For 2x2, offset 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            
            # Let me redo this with explicit indices
            # Clear and redo
            U[i_start, col] = 0
            U[i_start+1, col] = 0
            U[i_start, col] = 0  # Reset
            U[i_start+1, col] = 0
            
            # This is getting complicated. Let me use a different approach.
            pass
        
        return base_idx + 7

    # Try to find a lower rank decomposition
    # We will try ranks from 35 down to 28
    best_result = None
    best_rank = trivial_rank
    
    for r in range(35, 27, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        num_trials = 50 if r >= 33 else 30
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
                
            # Different initialization strategies
            if trial % 3 == 0:
                # Strategy 1: Random subset of trivial columns
                perm = np.random.permutation(trivial_rank)
                idxs = perm[:r]
                U = U_triv[:, idxs].copy()
                V = V_triv[:, idxs].copy()
                W = W_triv[:, idxs].copy()
                U += np.random.randn(n * m, r) * 0.05
                V += np.random.randn(m * p, r) * 0.05
                W += np.random.randn(p * n, r) * 0.05
            elif trial % 3 == 1:
                # Strategy 2: Random Gaussian initialization
                U = np.random.randn(n * m, r) * 0.5
                V = np.random.randn(m * p, r) * 0.5
                W = np.random.randn(p * n, r) * 0.5
            else:
                # Strategy 3: Mixed - some trivial, some random
                perm = np.random.permutation(trivial_rank)
                idxs = perm[:r//2]
                U = U_triv[:, idxs].copy()
                V = V_triv[:, idxs].copy()
                W = W_triv[:, idxs].copy()
                
                U_rand = np.random.randn(n * m, r - r//2) * 0.5
                V_rand = np.random.randn(m * p, r - r//2) * 0.5
                W_rand = np.random.randn(p * n, r - r//2) * 0.5
                
                U = np.hstack([U, U_rand])
                V = np.hstack([V, V_rand])
                W = np.hstack([W, W_rand])
            
            # ALS Iterations
            for it in range(80):
                if time.time() > deadline:
                    break
                    
                # Update U
                D_U = np.einsum('br,cr->rbc', V, W).reshape(r, m * p * p * n)
                try:
                    U_new = np.linalg.lstsq(D_U, T_flat_U.T, rcond=None)[0].T
                except np.linalg.LinAlgError:
                    continue
                    
                # Update V
                D_V = np.einsum('ar,cr->rac', U_new, W).reshape(r, n * m * p * n)
                try:
                    V_new = np.linalg.lstsq(D_V, T_flat_V.T, rcond=None)[0].T
                except np.linalg.LinAlgError:
                    continue

                # Update W
                D_W = np.einsum('ar,br->rab', U_new, V_new).reshape(r, n * m * m * p)
                try:
                    W_new = np.linalg.lstsq(D_W, T_flat_W.T, rcond=None)[0].T
                except np.linalg.LinAlgError:
                    continue

                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                residual = np.linalg.norm(T - T_approx)
                
                if residual < 1e-8:
                    # Found a solution! Try to polish and round
                    U_polished, V_polished, W_polished, res_polished = polish_solution(U, V, W, r, max_iters=30)
                    
                    if U_polished is not None:
                        U_rounded, V_rounded, W_rounded, scale = try_round_and_verify(U_polished, V_polished, W_polished, r)
                        
                        if U_rounded is not None:
                            return {
                                "U": U_rounded.tolist(),
                                "V": V_rounded.tolist(),
                                "W": W_rounded.tolist()
                            }
                    
                    # If polishing didn't help, try direct rounding
                    U_rounded, V_rounded, W_rounded, scale = try_round_and_verify(U, V, W, r)
                    if U_rounded is not None:
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }

    # Fallback to trivial
    return {
        "U": U_triv.tolist(),
        "V": V_triv.tolist(),
        "W": W_triv.tolist()
    }
