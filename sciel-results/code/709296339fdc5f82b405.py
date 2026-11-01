import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin for 150s limit

    n, m, p = DIMENSIONS

    # Build the target tensor T[i,j,k] for 2x4x4
    # i = row_a * 4 + col_a (A is 2x4)
    # j = row_b * 4 + col_b (B is 4x4)
    # k = col_c * 2 + row_c (C is 2x4, stored as W[k*n+i,r] convention: k = col*n + row)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for row_a in range(n):
        for col_a in range(m):
            i = row_a * m + col_a
            for row_b in range(m):
                for col_b in range(p):
                    j = row_b * p + col_b
                    for row_c in range(n):
                        for col_c in range(p):
                            k = col_c * n + row_c
                            if col_a == row_b and row_a == row_c and col_b == col_c:
                                T[i, j, k] = 1.0

    # Strassen 2x2x2 factors
    U_s = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, -1, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, -1],
        [1, 1, 0, 1, 0, -1, 0]
    ], dtype=float)

    V_s = np.array([
        [1, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 1]
    ], dtype=float)

    # Derive W_s by solving the linear system for 2x2x2
    T_2 = np.zeros((4, 4, 4), dtype=float)
    for ra in range(2):
        for ca in range(2):
            ii = ra * 2 + ca
            for rb in range(2):
                for cb in range(2):
                    jj = rb * 2 + cb
                    for rc in range(2):
                        for cc in range(2):
                            kk = cc * 2 + rc
                            if ca == rb and ra == rc and cb == cc:
                                T_2[ii, jj, kk] = 1.0

    M_sys = np.zeros((64, 28), dtype=float)
    T_vec = np.zeros(64, dtype=float)
    for ii in range(4):
        for jj in range(4):
            for kk in range(4):
                eq = ii * 16 + jj * 4 + kk
                T_vec[eq] = T_2[ii, jj, kk]
                for s in range(7):
                    M_sys[eq, kk * 7 + s] = U_s[ii, s] * V_s[jj, s]

    W_vec, _, _, _ = np.linalg.lstsq(M_sys, T_vec, rcond=None)
    W_s = W_vec.reshape(4, 7)

    # Verify Strassen decomposition
    err_s = 0.0
    for ii in range(4):
        for jj in range(4):
            for kk in range(4):
                val = np.sum(U_s[ii, :] * V_s[jj, :] * W_s[kk, :])
                err_s = max(err_s, abs(val - T_2[ii, jj, kk]))

    # Fallback: schoolbook rank 32
    def get_fallback():
        U_fb = np.zeros((8, 32))
        V_fb = np.zeros((16, 32))
        W_fb = np.zeros((8, 32))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    if err_s > 1e-9:
        return get_fallback()

    # Build the rank-28 Strassen decomposition for 2x4x4
    def build_rank28():
        U_full = np.zeros((8, 28), dtype=float)
        V_full = np.zeros((16, 28), dtype=float)
        W_full = np.zeros((8, 28), dtype=float)

        offset = 0
        for bi in range(2):
            for bj in range(2):
                # Map local indices to global
                for ra in range(2):
                    for ca in range(2):
                        local_i = ra * 2 + ca
                        global_i = ra * 4 + (2 * bi + ca)
                        U_full[global_i, offset:offset + 7] = U_s[local_i, :]

                for rb in range(2):
                    for cb in range(2):
                        local_j = rb * 2 + cb
                        global_j = (2 * bi + rb) * 4 + (2 * bj + cb)
                        V_full[global_j, offset:offset + 7] = V_s[local_j, :]

                for rc in range(2):
                    for cc in range(2):
                        local_k = cc * 2 + rc
                        global_k = (2 * bj + cc) * 2 + rc
                        W_full[global_k, offset:offset + 7] = W_s[local_k, :]

                offset += 7

        return U_full, V_full, W_full

    U28, V28, W28 = build_rank28()

    # Verify rank-28 decomposition
    def verify_decomposition(U_full, V_full, W_full, T_target):
        max_err = 0.0
        for i in range(U_full.shape[0]):
            for j in range(V_full.shape[0]):
                for k in range(W_full.shape[0]):
                    val = np.dot(U_full[i, :] * V_full[j, :], W_full[k, :])
                    err = abs(val - T_target[i, j, k])
                    if err > max_err:
                        max_err = err
        return max_err

    err28 = verify_decomposition(U28, V28, W28, T)
    
    if err28 > 1e-9:
        return get_fallback()

    # NEW STRATEGY: Structured search for Rank 27
    # We hypothesize that the tensor M_{2,4,4} might have rank 27.
    # We will attempt to find a decomposition of rank 27 by fixing a structured basis
    # derived from the rank-28 solution but with one degree of freedom removed or modified.
    
    # Specifically, we will try to solve for a rank-27 decomposition by fixing U and V 
    # to be "almost" the rank-28 U and V (dropping one column or merging two) and solving for W.
    # However, simply dropping a column from U28/V28 won't work because the span changes.
    
    # Better approach: Use the fact that the tensor is a sum of 4 2x2x2 tensors.
    # Let T = T1 + T2 + T3 + T4.
    # Rank(T) <= Rank(T1)+Rank(T2)+Rank(T3)+Rank(T4) = 28.
    # Can we do better? 
    # Some literature suggests R(M_{2,4,4}) = 28, but let's try to find a counter-example 
    # or a specific low-rank form with small integers.
    
    # We will perform a randomized search for rank 27, but guided by the structure.
    # We will fix U and V to be sparse matrices with entries in {-1, 0, 1} 
    # and solve for W. If W has entries in {-1, -0.5, 0, 0.5, 1}, we accept.
    
    # To make this efficient, we will only try a limited number of trials and ensure
    # we return the rank-28 solution if we don't find a rank-27 one quickly.
    
    def try_structured_rank27(num_trials=100):
        # Use a random seed for reproducibility in this attempt, but vary it
        # Actually, let's use a deterministic set of "structured" perturbations first.
        
        # We will try to express T as sum of 27 terms.
        # We fix U and V to be random sparse matrices.
        # We solve for W.
        # We check if W is "nice" (integer or half-integer).
        
        for trial in range(num_trials):
            if time.time() - start_time > deadline - 10:
                return None
            
            # Generate sparse U and V
            # Density ~ 0.3
            U_rand = np.zeros((8, 27))
            V_rand = np.zeros((16, 27))
            
            # Ensure some structure: maybe reuse Strassen patterns?
            # Let's just use random sparse integers.
            U_rand = np.random.randint(-1, 2, (8, 27)).astype(float)
            V_rand = np.random.randint(-1, 2, (16, 27)).astype(float)
            
            # Solve for W
            # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
            # For each k, T[:,:,k] = sum_r W[k,r] (U[:,r] V[:,r]^T)
            W_sol = np.zeros((8, 27))
            valid = True
            
            for k in range(8):
                T_k_vec = T[:, :, k].flatten()
                A_k = np.zeros((128, 27))
                for r in range(27):
                    A_k[:, r] = np.outer(U_rand[:, r], V_rand[:, r]).flatten()
                
                W_k, residuals, rank, sv = np.linalg.lstsq(A_k, T_k_vec, rcond=None)
                if len(residuals) > 0 and residuals[0] > 1e-5:
                    valid = False
                    break
                W_sol[k, :] = W_k
            
            if not valid:
                continue
            
            # Verify
            err = verify_decomposition(U_rand, V_rand, W_sol, T)
            if err < 1e-5:
                # Check for nice entries in W
                # Round to nearest 0.5
                W_rounded = np.round(W_sol * 2) / 2
                err_rounded = verify_decomposition(U_rand, V_rand, W_rounded, T)
                if err_rounded < 1e-5:
                    return U_rand, V_rand, W_rounded
        
        return None

    result_27 = try_structured_rank27(num_trials=50)
    
    if result_27 is not None:
        U_out, V_out, W_out = result_27
        return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
        
    # If rank 27 not found, return rank 28
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
