import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = 145.0

    n, m, p = DIMENSIONS  # 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Build the multiplication tensor T of shape (nm, mp, pn)
    # T[a,b,c] = 1 iff a=(i*m+j), b=(j*p+k), c=(k*n+i) for some i,j,k
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify_fast(U_mat, V_mat, W_mat, tol=1e-8):
        """Vectorized verification of tensor decomposition."""
        r = U_mat.shape[1]
        # T_decomposed[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Check all non-zero and zero entries
        # Build full tensor via einsum
        T_dec = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        return np.allclose(T_dec, T, atol=tol)

    def strassen_basis_entries():
        """
        For a 2x2x2 Strassen block, the 7 multiplications M1..M7
        produce contributions to the 8 output entries.
        Returns a list of 7 entries, each a dict mapping (a_local, b_local, c_local) -> coeff.
        Local indices: a in {0,1} (i), b in {0,1} (j->k part), c in {0,1} (k->i part)
        Actually for 2x2x2: A is 2x2 (i,j), B is 2x2 (j,k), C is 2x2 (k,i)
        Strassen:
          M1 = (A00+A11)(B00+B11) -> contributes to C00(+1), C11(+1)
          M2 = (A10+A11)B00       -> contributes to C10(+1), C11(-1)
          M3 = A00(B01-B11)       -> contributes to C01(+1), C11(+1)
          M4 = A11(B10-B00)       -> contributes to C00(+1), C10(+1)
          M5 = (A00+A01)B11       -> contributes to C00(-1), C01(+1)
          M6 = (A10-A00)(B00+B01) -> contributes to C11(+1)
          M7 = (A01-A11)(B10+B11) -> contributes to C00(+1)
        
        In terms of tensor entries T[a,b,c] where a=idx(A entry), b=idx(B entry), c=idx(C entry):
        A00=(0,0)->a=0, A01=(0,1)->a=1, A10=(1,0)->a=2, A11=(1,1)->a=3
        B00=(0,0)->b=0, B01=(0,1)->b=1, B10=(1,0)->b=2, B11=(1,1)->b=3
        C00=(0,0)->c=0, C01=(0,1)->c=1, C10=(1,0)->c=2, C11=(1,1)->c=3
        
        M1: U has A00,A11; V has B00,B11; W has C00,C11
        M2: U has A10,A11; V has B00; W has C10,C11
        M3: U has A00; V has B01,B11; W has C01,C11
        M4: U has A11; V has B10,B00; W has C00,C10
        M5: U has A00,A01; V has B11; W has C00,C01
        M6: U has A10,A00; V has B00,B01; W has C11
        M7: U has A01,A11; V has B10,B11; W has C00
        """
        basis = []
        # M1
        basis.append({(0,0,0):1, (0,3,3):1, (3,0,0):1, (3,3,3):1})
        # M2
        basis.append({(2,0,2):1, (3,0,2):1, (2,0,3):-1, (3,0,3):-1})
        # M3
        basis.append({(0,1,1):1, (0,3,1):1, (0,1,3):-1, (0,3,3):-1})
        # M4
        basis.append({(3,2,0):1, (3,0,0):-1, (3,2,2):1, (3,0,2):-1})
        # M5
        basis.append({(0,3,0):-1, (1,3,0):-1, (0,3,1):1, (1,3,1):1})
        # M6
        basis.append({(2,0,3):1, (0,0,3):-1, (2,1,3):1, (0,1,3):-1})
        # M7
        basis.append({(1,2,0):1, (3,2,0):-1, (1,3,0):1, (3,3,0):-1})
        return basis

    # For 2x4x6, we have tilings at (i_start, j_start, k_start) for i_start in {0}, j_start in {0,2}, k_start in {0,2,4}
    # That's 1*2*3 = 6 blocks, each using 7 Strassen terms = 42 total
    
    # But we can also use overlapping windows. Let's enumerate all possible 2x2x2 windows:
    # i_start in {0}, j_start in {0,1,2,3} but must be even for 2-wide? No, j can be 0,2 for 2-wide
    # Actually for 2x4x6: i in {0}, j_start in {0, 2} (2 positions), k_start in {0, 2, 4} (3 positions)
    # These are the non-overlapping tilings. For overlapping:
    # i_start in {0}, j_start in {0, 1, 2, 3} -> but 2-wide means j_start+1 <= 4, so j_start in {0,1,2,3}
    # k_start in {0, 1, 2, 3, 4} -> k_start+1 <= 6, so k_start in {0,1,2,3,4}
    # Total overlapping windows: 1 * 4 * 5 = 20 windows, each with 7 terms = 140 basis vectors
    
    # Build the full basis: for each window (is, js, ks) and each Strassen term t in 0..6
    # The global indices are:
    #   A entry: (is + a_local, js + b_local) -> global a = (is + a_local)*m + (js + b_local)
    #   B entry: (js + b_local, ks + c_local) -> global b = (js + b_local)*p + (ks + c_local)
    #   C entry: (ks + c_local, is + a_local) -> global c = (ks + c_local)*n + (is + a_local)
    
    windows = []
    for is_ in range(0, n, 1):  # i_start: 0 only since n=2
        if is_ + 1 > n:
            continue
        for js_ in range(0, m - 1):  # j_start: 0,1,2,3
            for ks_ in range(0, p - 1):  # k_start: 0,1,2,3,4
                windows.append((is_, js_, ks_))
    
    # Build basis vectors as sparse representations
    # Each basis vector is a set of (a, b, c, coeff) tuples
    basis_entries = []
    for (is_, js_, ks_) in windows:
        for t in range(7):
            entries = []
            for (al, bl, cl), coeff in strassen_basis_entries()[t].items():
                a = (is_ + al) * m + (js_ + bl)
                b = (js_ + bl) * p + (ks_ + cl)
                c = (ks_ + cl) * n + (is_ + al)
                if 0 <= a < nm and 0 <= b < mp and 0 <= c < pn:
                    entries.append((a, b, c, coeff))
            basis_entries.append(entries)
    
    num_basis = len(basis_entries)
    
    # Now: we want to find coefficients x_0, ..., x_{num_basis-1} (each can be any real)
    # such that sum_t x_t * basis_t reproduces T
    # This is a linear system. But we want MINIMUM RANK decomposition.
    # Rank of decomposition = number of nonzero "rank-1 slices" in U,V,W.
    # 
    # Alternative approach: directly solve for U, V, W with low rank r.
    # T[a,b,c] = sum_{r=1}^R U[a,r] * V[b,r] * W[c,r]
    # This is a system of nm*mp*pn = 8*24*12 = 2304 equations in R*(nm+mp+pn) = R*44 unknowns.
    # For R=28: 28*44 = 1232 unknowns < 2304 equations. Underdetermined is not the issue; it's that
    # the equations are nonlinear (products). 
    #
    # Better: fix R, and try to solve iteratively.
    # Use the fact that if we fix two of U,V,W, the third is determined linearly.
    # 
    # Strategy: Start with R=28, random init, and do alternating least squares (ALS):
    #   Fix V,W: solve for U (linear system)
    #   Fix U,W: solve for V (linear system)
    #   Fix U,V: solve for W (linear system)
    # Repeat until convergence or deadline.
    
    def als_decompose(R, max_iters=200, seed=42):
        """Alternating least squares for rank-R tensor decomposition."""
        rng = np.random.RandomState(seed)
        U = rng.randn(nm, R) * 0.1
        V = rng.randn(mp, R) * 0.1
        W = rng.randn(pn, R) * 0.1
        
        # Precompute T as flat
        T_flat = T.flatten()
        T_shape = T.shape
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each a, solve sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all b,c
            # This is: U[a,:] @ (V * W^T) = T[a,:,:]
            # where (V*W^T)[b,c] = sum_r V[b,r]*W[c,r]
            VW = V.T @ W  # shape (R, R) -- no wait
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed a: T[a,:,:] = U[a,:] outer product with... no
            # T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
            # Let M_r = V[:,r] outer W[:,r], shape (mp, pn)
            # T[a,:,:] = sum_r U[a,r] * M_r
            # So T[a,:,:] = U[a,:] @ (stack of M_r)
            # M_r has shape (mp, pn), so stack them: M_all shape (R, mp, pn)
            # T[a,b,c] = sum_r U[a,r] * M_all[r,b,c]
            # For each a: T[a,:,:] (shape mp*pn) = U[a,:] (shape R) dot M_all[:, :, :] reshaped
            # Linear system: A_U * U[a,:]^T = T[a,:,:]^T where A_U[r, b*pn+c] = M_all[r,b,c]
            
            M_all = np.einsum('br,cr->rbc', V, W)  # shape (R, mp, pn)
            A_U = M_all.reshape(R, mp * pn)  # shape (R, mp*pn)
            # T[a,:,:] flattened has shape (mp*pn,)
            # U[a,:] shape (R,)
            # A_U @ U[a,:]^T should equal T[a,:,:].T? No:
            # (A_U @ U[a,:]^T)[b*pn+c] = sum_r A_U[r, b*pn+c] * U[a,r] = sum_r M_all[r,b,c] * U[a,r] = T[a,b,c] ✓
            # So: A_U @ U[a,:]^T = T[a,:,:].flatten()
            # This is A_U (R x mp*pn) @ U[a,:]^T (mp*pn x 1) = ... wait, dimensions don't match.
            # A_U is (R, mp*pn), U[a,:]^T is (R, 1)^T = (1, R)... 
            # We need: for each (b,c): sum_r U[a,r] * A_U[r, b*pn+c] = T[a,b,c]
            # This is: (U[a,:] @ A_U^T)[b*pn+c] = T[a,b,c]
            # So: U[a,:] @ A_U^T = T[a,:,:].flatten()
            # A_U^T shape: (mp*pn, R)
            # U[a,:] shape: (R,)
            # U[a,:] @ A_U^T shape: (mp*pn,) ✓
            # Solve: A_U @ U[a,:]^T = T[a,:,:].flatten() -> this is overdetermined (mp*pn > R)
            # Use least squares: U[a,:] = lstsq(A_U^T, T[a,:,:].flatten())
            
            AU_T = A_U.T  # shape (mp*pn, R)
            for a in range(nm):
                rhs = T[a, :, :].flatten()
                # Solve AU_T @ U[a,:]^T = rhs
                sol, _, _, _ = np.linalg.lstsq(AU_T, rhs, rcond=None)
                U[a, :] = sol
            
            # Update V similarly
            MW = np.einsum('ar,cr->rac', U, W)  # shape (nm, R, pn) -- no
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed b: T[:,b,:] (shape nm*pn) = V[b,:] @ (U*W)^T... 
            # T[a,b,c] = sum_r V[b,r] * (U[a,r] * W[c,r])
            # M2[r, a*pn+c] = U[a,r] * W[c,r]
            M2 = np.einsum('ar,cr->rac', U, W).reshape(R, nm * pn)
            AV_T = M2.T  # shape (nm*pn, R)
            for b in range(mp):
                rhs = T[:, b, :].flatten()
                sol, _, _, _ = np.linalg.lstsq(AV_T, rhs, rcond=None)
                V[b, :] = sol
            
            # Update W
            M3 = np.einsum('ar,br->rab', U, V).reshape(R, nm * mp)
            AW_T = M3.T  # shape (nm*mp, R)
            for c in range(pn):
                rhs = T[:, :, c].flatten()
                sol, _, _, _ = np.linalg.lstsq(AW_T, rhs, rcond=None)
                W[c, :] = sol
        
        return U, V, W

    def check_decomposition(U, V, W, tol=1e-6):
        T_dec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_dec, T, atol=tol)

    # Try ranks from low to high
    best_result = None
    for R in [24, 28, 32, 36, 40, 42]:
        if time.time() > deadline - 10:
            break
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            U, V, W = als_decompose(R, max_iters=100, seed=42 + attempt * 1000)
            if check_decomposition(U, V, W, tol=1e-6):
                # Polish with tighter tolerance
                for polish in range(20):
                    if time.time() > deadline - 3:
                        break
                    # One more ALS iteration with tighter tol
                    M_all = np.einsum('br,cr->rbc', V, W)
                    A_U = M_all.reshape(R, mp * pn)
                    AU_T = A_U.T
                    for a in range(nm):
                        rhs = T[a, :, :].flatten()
                        sol, _, _, _ = np.linalg.lstsq(AU_T, rhs, rcond=None)
                        U[a, :] = sol
                    M2 = np.einsum('ar,cr->rac', U, W).reshape(R, nm * pn)
                    AV_T = M2.T
                    for b in range(mp):
                        rhs = T[:, b, :].flatten()
                        sol, _, _, _ = np.linalg.lstsq(AV_T, rhs, rcond=None)
                        V[b, :] = sol
                    M3 = np.einsum('ar,br->rab', U, V).reshape(R, nm * mp)
                    AW_T = M3.T
                    for c in range(pn):
                        rhs = T[:, :, c].flatten()
                        sol, _, _, _ = np.linalg.lstsq(AW_T, rhs, rcond=None)
                        W[c, :] = sol
                    if check_decomposition(U, V, W, tol=1e-8):
                        break
                
                if check_decomposition(U, V, W, tol=1e-8):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                elif check_decomposition(U, V, W, tol=1e-6):
                    if best_result is None or R < best_result[0]:
                        best_result = (R, U.copy(), V.copy(), W.copy())
                    break
            if time.time() > deadline - 5:
                break
        if best_result is not None:
            break

    # Fallback: rank-42 Strassen
    if time.time() < deadline - 2:
        rank42 = 42
        U42 = np.zeros((nm, rank42), dtype=float)
        V42 = np.zeros((mp, rank42), dtype=float)
        W42 = np.zeros((pn, rank42), dtype=float)
        
        def add_strassen_block(U, V, W, i_start, j_start, k_start, offset):
            a00 = i_start * m + j_start
            a01 = i_start * m + j_start + 1
            a10 = (i_start + 1) * m + j_start
            a11 = (i_start + 1) * m + j_start + 1
            b00 = j_start * p + k_start
            b01 = j_start * p + k_start + 1
            b10 = (j_start + 1) * p + k_start
            b11 = (j_start + 1) * p + k_start + 1
            w00 = k_start * n + i_start
            w01 = (k_start + 1) * n + i_start
            w10 = k_start * n + i_start + 1
            w11 = (k_start + 1) * n + i_start + 1
            
            U[a00, offset] += 1.0; U[a11, offset] += 1.0
            V[b00, offset] += 1.0; V[b11, offset] += 1.0
            W[w00, offset] += 1.0; W[w11, offset] += 1.0
            
            U[a10, offset+1] += 1.0; U[a11, offset+1] += 1.0
            V[b00, offset+1] += 1.0
            W[w10, offset+1] += 1.0; W[w11, offset+1] -= 1.0
            
            U[a00, offset+2] += 1.0
            V[b01, offset+2] += 1.0; V[b11, offset+2] -= 1.0
            W[w01, offset+2] += 1.0; W[w11, offset+2] += 1.0
            
            U[a11, offset+3] += 1.0
            V[b10, offset+3] += 1.0; V[b00, offset+3] -= 1.0
            W[w00, offset+3] += 1.0; W[w10, offset+3] += 1.0
            
            U[a00, offset+4] += 1.0; U[a01, offset+4] += 1.0
            V[b11, offset+4] += 1.0
            W[w00, offset+4] -= 1.0; W[w01, offset+4] += 1.0
            
            U[a10, offset+5] += 1.0; U[a00, offset+5] -= 1.0
            V[b00, offset+5] += 1.0; V[b01, offset+5] += 1.0
            W[w11, offset+5] += 1.0
            
            U[a01, offset+6] += 1.0; U[a11, offset+6] -= 1.0
            V[b10, offset+6] += 1.0; V[b11, offset+6] += 1.0
            W[w00, offset+6] += 1.0
        
        offset = 0
        for i_start in range(0, n, 2):
            for j_start in range(0, m, 2):
                for k_start in range(0, p, 2):
                    add_strassen_block(U42, V42, W42, i_start, j_start, k_start, offset)
                    offset += 7
        
        if verify_fast(U42, V42, W42, tol=1e-8):
            return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}

    # Final fallback: schoolbook
    rank0 = n * m * p
    U0 = np.zeros((nm, rank0), dtype=float)
    V0 = np.zeros((mp, rank0), dtype=float)
    W0 = np.zeros((pn, rank0), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * (m * p) + j * p + k
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
