"""
Rank-minimal exact tensor decomposition for 2x2 * 2x15 matrix multiplication.
Strategy: Randomized Gaussian elimination heuristic to find sparse decompositions
with rank < 53, exploiting cross-column linear dependencies.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook fallback: rank = n*m*p = 60
    rank_schoolbook = n * m * p
    U_fb = np.zeros((n * m, rank_schoolbook))
    V_fb = np.zeros((m * p, rank_schoolbook))
    W_fb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    best_result = {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    best_rank = rank_schoolbook

    # Strassen-based decomposition for 2x2 * 2x15 (rank 53)
    # This is our baseline to beat
    R_strassen = 53
    U_s = np.zeros((n * m, R_strassen))
    V_s = np.zeros((m * p, R_strassen))
    W_s = np.zeros((p * n, R_strassen))
    col_offset = 0

    # Process 7 blocks of 2 columns each (columns 0-13)
    for b in range(7):
        k0 = 2 * b
        k1 = 2 * b + 1

        # M1: (A00+A11) * (B0,k0 + B1,k1)
        c = col_offset
        U_s[0, c] = 1
        U_s[3, c] = 1
        V_s[0 * p + k0, c] = 1
        V_s[1 * p + k1, c] = 1
        W_s[k0 * n + 0, c] = 1
        W_s[k1 * n + 1, c] = 1
        col_offset += 1

        # M2: (A10+A11) * B0,k0
        c = col_offset
        U_s[2, c] = 1
        U_s[3, c] = 1
        V_s[0 * p + k0, c] = 1
        W_s[k0 * n + 1, c] = 1
        col_offset += 1

        # M3: A00 * (B1,k0 - B1,k1)
        c = col_offset
        U_s[0, c] = 1
        V_s[1 * p + k0, c] = 1
        V_s[1 * p + k1, c] = -1
        W_s[k1 * n + 0, c] = 1
        col_offset += 1

        # M4: A11 * (B0,k1 - B0,k0)
        c = col_offset
        U_s[3, c] = 1
        V_s[0 * p + k1, c] = 1
        V_s[0 * p + k0, c] = -1
        W_s[k0 * n + 0, c] = 1
        W_s[k0 * n + 1, c] = 1
        col_offset += 1

        # M5: (A00+A01) * B1,k1
        c = col_offset
        U_s[0, c] = 1
        U_s[1, c] = 1
        V_s[1 * p + k1, c] = 1
        W_s[k0 * n + 0, c] = -1
        W_s[k1 * n + 0, c] = 1
        col_offset += 1

        # M6: (A10-A00) * (B0,k0 + B1,k0)
        c = col_offset
        U_s[2, c] = 1
        U_s[0, c] = -1
        V_s[0 * p + k0, c] = 1
        V_s[1 * p + k0, c] = 1
        W_s[k1 * n + 1, c] = 1
        col_offset += 1

        # M7: (A01-A11) * (B0,k1 + B1,k1)
        c = col_offset
        U_s[1, c] = 1
        U_s[3, c] = -1
        V_s[0 * p + k1, c] = 1
        V_s[1 * p + k1, c] = 1
        W_s[k0 * n + 0, c] = 1
        col_offset += 1

    # Last column (k=14) with schoolbook (rank 4)
    k_last = 14
    for i in range(n):
        for j in range(m):
            c = col_offset
            U_s[i * m + j, c] = 1
            V_s[j * p + k_last, c] = 1
            W_s[k_last * n + i, c] = 1
            col_offset += 1

    assert col_offset == R_strassen

    # Verify Strassen decomposition
    errors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(R_strassen):
                    total += U_s[i * m + j, c] * V_s[j * p + k, c] * W_s[k * n + i, c]
                if abs(total - 1.0) > 1e-10:
                    errors.append((i, j, k, total))

    if len(errors) == 0:
        best_result = {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        best_rank = R_strassen

    # Now attempt to find a lower-rank decomposition using randomized search
    # Target: rank < 53
    
    # The tensor equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # for all i in {0,1}, j in {0,1}, k in {0..14}
    #
    # This is a system of 4*15 = 60 bilinear equations.
    #
    # Strategy: Try rank R = 52, 51, 50, ... down to some minimum.
    # For each R, use randomized initialization + iterative refinement.
    
    def try_decomposition(R, max_iter=5000, tol=1e-8):
        """Try to find a decomposition of rank R."""
        if time.time() > deadline - 10:
            return None
        
        # Random initialization with sparse structure
        np.random.seed(42 + R)
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Initialize with random ±1 entries, sparsified
        for c in range(R):
            # Each column of U, V, W should be sparse
            # U has 4 entries, V has 30, W has 30
            nz_u = np.random.choice(4, size=max(1, 4//3), replace=False)
            nz_v = np.random.choice(30, size=max(1, 30//5), replace=False)
            nz_w = np.random.choice(30, size=max(1, 30//5), replace=False)
            
            for idx in nz_u:
                U[idx, c] = np.random.choice([-1, 1])
            for idx in nz_v:
                V[idx, c] = np.random.choice([-1, 1])
            for idx in nz_w:
                W[idx, c] = np.random.choice([-1, 1])
        
        # Iterative refinement: pick a random equation, adjust one variable
        # to reduce the residual
        # Equations: for each (i,j,k), residual = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                return None
            
            # Check convergence
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0.0
                        for c in range(R):
                            total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        err = abs(total - 1.0)
                        if err > max_err:
                            max_err = err
            if max_err < tol:
                return U, V, W
            
            # Pick a random equation with large error
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            
            total = 0.0
            for c in range(R):
                total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
            residual = total - 1.0
            
            if abs(residual) < 1e-6:
                continue
            
            # Try to adjust one variable to reduce residual
            # We can adjust U[i*m+j, c], V[j*p+k, c], or W[k*n+i, c]
            # for some c
            c = np.random.randint(R)
            
            # Current contribution of this term
            u_val = U[i * m + j, c]
            v_val = V[j * p + k, c]
            w_val = W[k * n + i, c]
            contrib = u_val * v_val * w_val
            
            # We want to change one of these to reduce |residual|
            # If we change u_val to u_new, new residual = residual + (u_new - u_val)*v_val*w_val
            # We want new residual to be closer to 0
            
            # Try flipping signs
            best_change = None
            best_new_err = abs(residual)
            
            # Option 1: flip U
            if v_val * w_val != 0:
                delta_u = -2 * u_val  # flip sign
                new_resid = residual + delta_u * v_val * w_val
                new_err = abs(new_resid)
                if new_err < best_new_err - 1e-10:
                    best_change = ('U', i * m + j, c, -u_val)
                    best_new_err = new_err
            
            # Option 2: flip V
            if u_val * w_val != 0:
                delta_v = -2 * v_val
                new_resid = residual + u_val * delta_v * w_val
                new_err = abs(new_resid)
                if new_err < best_new_err - 1e-10:
                    best_change = ('V', j * p + k, c, -v_val)
                    best_new_err = new_err
            
            # Option 3: flip W
            if u_val * v_val != 0:
                delta_w = -2 * w_val
                new_resid = residual + u_val * v_val * delta_w
                new_err = abs(new_resid)
                if new_err < best_new_err - 1e-10:
                    best_change = ('W', k * n + i, c, -w_val)
                    best_new_err = new_err
            
            if best_change is not None:
                mat, idx, col_idx, new_val = best_change
                if mat == 'U':
                    U[idx, col_idx] = new_val
                elif mat == 'V':
                    V[idx, col_idx] = new_val
                elif mat == 'W':
                    W[idx, col_idx] = new_val
        
        return None
    
    # Try ranks from 52 down to 45
    for R in range(52, 44, -1):
        if time.time() > deadline - 15:
            break
        
        result = try_decomposition(R)
        if result is not None:
            U, V, W = result
            # Final verification
            errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0.0
                        for c in range(R):
                            total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        if abs(total - 1.0) > 1e-8:
                            errors.append((i, j, k, total))
            
            if len(errors) == 0:
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = R
                break  # Found a valid lower-rank decomposition
    
    if time.time() > deadline - 5:
        return best_result

    return best_result
# EVOLVE-BLOCK-END
