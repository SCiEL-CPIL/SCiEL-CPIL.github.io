import numpy as np
import time
import sys

# The task is to find a rank-minimal exact tensor decomposition for 2x4x4 matrix multiplication.
# Dimensions: n=2, m=4, p=4.
# U shape: (n*m, R) = (8, R)
# V shape: (m*p, R) = (16, R)
# W shape: (p*n, R) = (8, R)
# Identity: sum_r U[i, r] V[j, r] W[k, r] = delta_{i, i'} delta_{j, j'} delta_{k, k'} 
# Wait, the indices in the prompt are:
# U[i'*m+j', r], V[j''*p+k', r], W[k*n+i, r]
# Sum over r equals delta_{i', i} delta_{k', k} delta_{j', j''}
# This is the standard matrix multiplication tensor T_{n,m,p}.
# n=2, m=4, p=4.

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 16
    pn = p * n  # 8
    
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit (or 150s if kill is 150)
    # Prompt says "Each evaluation must finish within 120 s." and "KILLS the program after 150s".
    # I will use 110s to be safe.

    def get_errors(U, V, W):
        """
        Compute errors for the tensor identity.
        Returns a flat array of errors.
        """
        U_arr = np.asarray(U, dtype=np.float64)
        V_arr = np.asarray(V, dtype=np.float64)
        W_arr = np.asarray(W, dtype=np.float64)
        
        # Identity:
        # C[i, k] = sum_j A[i, j] B[j, k]
        # Tensor form:
        # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]  (This is for standard MM where output is C[i,k])
        # But the prompt specifies:
        # U index: i'*m + j'  (A indices: row i', col j')
        # V index: j''*p + k' (B indices: row j'', col k')
        # W index: k*n + i    (C indices: row k, col i ?? No, usually C is n x p)
        # Prompt: "output factor follows the official trace convention W[k*n+i,r]"
        # And identity: delta_{i', i} delta_{k', k} delta_{j', j''}
        # This implies:
        # A has indices (i, j) where i in 0..n-1, j in 0..m-1.
        # B has indices (j, k) where j in 0..m-1, k in 0..p-1.
        # C has indices (i, k) where i in 0..n-1, k in 0..p-1.
        # Wait, the prompt says W[k*n+i]. This is column-major for C if C is n x p?
        # If C is n x p, index is i*p + k (row major) or k*n + i (col major).
        # The identity is delta_{i', i} delta_{k', k} delta_{j', j''}.
        # This matches C[i, k] = sum_j A[i, j] B[j, k].
        
        # We need to check:
        # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = delta_{i, i'} delta_{k, k'} delta_{j, j'} ??
        # No, the sum is over r. The result should be the MM tensor.
        # The MM tensor T is defined such that T[A, B] = C.
        # T_{i, j, k} = 1 if valid indices, else 0?
        # Actually, the decomposition is:
        # MM(A, B)_{i, k} = sum_j A_{i, j} B_{j, k}
        # This can be written as sum_r (sum_j A_{i, j} U_{j, r}) (sum_k B_{j, k} V_{k, r}) ... no.
        
        # Standard decomposition:
        # C_{i, k} = sum_r M_r
        # M_r = (sum_{i,j} A_{i,j} U_{i,j,r}) * (sum_{j,k} B_{j,k} V_{j,k,r})
        # Then C_{i,k} = sum_r W_{i,k,r} M_r.
        
        # The prompt's indices:
        # U[i*m+j, r] -> A_{i,j} coefficient.
        # V[j*p+k, r] -> B_{j,k} coefficient.
        # W[k*n+i, r] -> C_{i,k} coefficient.
        
        # So the identity to satisfy is:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] 
        # This sum should equal... what?
        # The tensor T for matrix multiplication is:
        # T_{i, j, k} = 1 if we are computing C_{i,k} from A_{i,j} and B_{j,k}?
        # No, the tensor is a 3-way array T_{i, j, k} such that
        # C_{i, k} = sum_j A_{i, j} B_{j, k} T_{i, j, k} ??
        # No, T is the multiplication tensor.
        # T_{i, j, k} = 1 if the indices are consistent with the multiplication.
        # Specifically, for fixed i, j, k, the term A_{i, j} B_{j, k} contributes to C_{i, k}.
        # So the tensor element is 1 if (i, j, k) is a valid path?
        # Actually, the decomposition is:
        # T = sum_r U_r \otimes V_r \otimes W_r
        # where (U_r \otimes V_r \otimes W_r)_{i, j, k} = U_{i, r} V_{j, r} W_{k, r}.
        # And T_{i, j, k} = 1 if j is the shared index?
        # Let's check the identity in the prompt:
        # "exactly decompose the matrix multiplication tensor"
        # "Identity: sum_r U[i'*m+j', r] V[j''*p+k', r] W[k*n+i, r] = delta_{i', i} delta_{k', k} delta_{j', j''}"
        
        # Let's analyze the RHS:
        # It is 1 if i'=i, k'=k, j'=j''.
        # This implies that for a fixed term r, the product U[i', j'] V[j'', k'] W[k, i]
        # should sum to 1 only when i'=i, k'=k, j'=j''.
        # This is exactly the condition for:
        # sum_r U_{i', j', r} V_{j'', k', r} W_{k, i, r} = \delta_{i', i} \delta_{k', k} \delta_{j', j''}
        
        # This is equivalent to:
        # Let A have entries A_{i', j'}.
        # Let B have entries B_{j'', k'}.
        # Let C have entries C_{i, k}.
        # The standard MM is C_{i, k} = sum_j A_{i, j} B_{j, k}.
        # This can be rewritten as:
        # sum_{i', j', j'', k'} A_{i', j'} B_{j'', k'} (sum_r U_{i', j', r} V_{j'', k', r} W_{k, i, r}) = C_{i, k}
        # For this to hold for all A, B, the kernel K_{i', j', j'', k'} = sum_r ... must be
        # \delta_{i', i} \delta_{k', k} \delta_{j', j''} ??
        # No, if we set i'=i, k'=k, j'=j'', the kernel is 1.
        # If we set j' != j'', the kernel is 0.
        # This means the sum_r term acts as a delta function picking out the matching indices.
        # So yes, the condition is:
        # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if j=j (trivial) and i,i and k,k match?
        # Wait, the indices in the sum are i', j', j'', k'.
        # The result is indexed by i, k?
        # The RHS is \delta_{i', i} \delta_{k', k} \delta_{j', j''}.
        # This means if we fix i, k, the sum over r of U[i', j'] V[j'', k'] W[k, i]
        # is 1 if i'=i, k'=k, j'=j''.
        # This is exactly the requirement.
        
        errors = np.zeros((n, m, m, p)) # i, j', j'', k
        # Or just flatten
        err_flat = np.zeros(n * m * m * p)
        idx = 0
        for i in range(n):
            for k in range(p):
                w_idx = k * n + i
                for j_prime in range(m):
                    for j_double_prime in range(m):
                        for i_prime in range(n):
                            # i_prime is 0..n-1
                            # j_prime is 0..m-1
                            # j_double_prime is 0..m-1
                            # k is 0..p-1
                            # We need to sum over r: U[i_prime, j_prime, r] * V[j_double_prime, k, r] * W[k, i, r]
                            # Wait, V index is j''*p + k'. Here k' is the column of B.
                            # In the identity, k' is the column index of B.
                            # In the MM, C_{i, k} = sum_j A_{i, j} B_{j, k}.
                            # So the B index is (j, k).
                            # So V index should be j_double_prime * p + k.
                            # And W index is k * n + i.
                            # U index is i_prime * m + j_prime.
                            
                            u_idx = i_prime * m + j_prime
                            v_idx = j_double_prime * p + k
                            
                            # The target is 1 if i_prime == i and j_prime == j_double_prime and k == k (always true for k in V/W match?)
                            # The RHS is delta_{i', i} delta_{k', k} delta_{j', j''}
                            # Here k' is the column of B, which is k.
                            # So target is 1 if i_prime == i and j_prime == j_double_prime.
                            # Note: k is fixed in the loop.
                            
                            target = 1.0 if (i_prime == i and j_prime == j_double_prime) else 0.0
                            
                            # Compute sum_r
                            # U[u_idx, :] * V[v_idx, :] * W[w_idx, :]
                            # This is element-wise product, then sum.
                            val = np.dot(U_arr[u_idx] * V_arr[v_idx], W_arr[w_idx])
                            errors[idx] = val - target
                            idx += 1
                            
        return errors

    def check_identity(U, V, W):
        errors = get_errors(U, V, W)
        return np.max(np.abs(errors))

    def strassen_rank_28():
        """
        Constructs the known rank-28 solution.
        """
        R = 28
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        
        # Split A (2x4) into A_L (2x2) and A_R (2x2)
        # Split B (4x4) into B_LL, B_LR, B_RL, B_RR (all 2x2)
        # C = A B = [A_L B_LL + A_R B_RL, A_L B_LR + A_R B_RR]
        
        # We use Strassen for each 2x2x2 multiplication.
        # Strassen for M1 = A_L B_LL:
        # M1_00, M1_01, M1_10, M1_11
        # Uses 7 products.
        
        # We will define the 7 Strassen products for a generic 2x2 mat mult X * Y.
        # P1 = (X00 + X11)(Y00 + Y11)
        # P2 = (X10 + X11)Y00
        # P3 = X00(Y01 - Y11)
        # P4 = X11(Y10 - Y00)
        # P5 = (X00 + X01)Y11
        # P6 = (X10 - X00)(Y00 + Y01)
        # P7 = (X01 - X11)(Y10 + Y11)
        
        # Result:
        # Z00 = P1 + P4 - P5 + P7
        # Z01 = P3 + P5
        # Z10 = P2 + P4
        # Z11 = P1 - P2 + P3 + P6
        
        # We need to map these to the global U, V, W.
        # For each of the 4 block multiplications (LL, LR, RL, RR), we add 7 terms.
        
        blocks = [
            ('L', 'LL', 0, 0), # A_L, B_LL -> C_L
            ('L', 'LR', 0, 2), # A_L, B_LR -> C_R
            ('R', 'RL', 2, 0), # A_R, B_RL -> C_L
            ('R', 'RR', 2, 2), # A_R, B_RR -> C_R
        ]
        
        col = 0
        for a_side, b_block, c_col_offset in [
            (0, 0, 0), # A_L (cols 0-1), B_LL (rows 0-1, cols 0-1), C_L (cols 0-1)
            (0, 2, 2), # A_L (cols 0-1), B_LR (rows 0-1, cols 2-3), C_R (cols 2-3)
            (2, 0, 0), # A_R (cols 2-3), B_RL (rows 2-3, cols 0-1), C_L (cols 0-1)
            (2, 2, 2)  # A_R (cols 2-3), B_RR (rows 2-3, cols 2-3), C_R (cols 2-3)
        ]:
            # a_side: 0 for A_L (rows 0-1, cols 0-1 of A), 2 for A_R (rows 0-1, cols 2-3 of A)
            # Wait, A is 2x4.
            # A_L is A[:, 0:2]. A_R is A[:, 2:4].
            # So for A_L, i is 0,1. j is 0,1.
            # For A_R, i is 0,1. j is 2,3.
            
            # b_block: 0 for B_LL (rows 0-1, cols 0-1), 2 for B_LR (rows 0-1, cols 2-3)
            #          4 for B_RL (rows 2-3, cols 0-1), 6 for B_RR (rows 2-3, cols 2-3)
            # Let's just use offsets.
            
            a_row_off = 0 # A always starts at row 0
            a_col_off = a_side # 0 or 2
            
            b_row_off = 0 if b_block in [0, 2] else 2 # LL, LR are top; RL, RR are bottom
            # Wait, B_LL is rows 0-1. B_LR is rows 0-1.
            # B_RL is rows 2-3. B_RR is rows 2-3.
            # So b_row_off = 0 for LL, LR. b_row_off = 2 for RL, RR.
            
            # My loop variables:
            # 0,0,0 -> A_L (col 0), B_LL (row 0), C_L (col 0)
            # 0,2,2 -> A_L (col 0), B_LR (row 0), C_R (col 2) -- Wait, B_LR is rows 0-1, cols 2-3.
            # So b_row_off should be 0 for LR.
            # 2,0,0 -> A_R (col 2), B_RL (row 2), C_L (col 0) -- B_RL is rows 2-3, cols 0-1.
            # 2,2,2 -> A_R (col 2), B_RR (row 2), C_R (col 2) -- B_RR is rows 2-3, cols 2-3.
            
            # Correct offsets:
            a_col_off = [0, 0, 2, 2][col//7] # This is getting messy. Let's hardcode the 4 cases.
            
        # Let's just hardcode the 4 Strassen decompositions.
        
        def add_strassen_term(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0, s):
            """
            Adds 7 terms for Strassen multiplication of 2x2 block A[a_r0:a_r0+2, a_c0:a_c0+2]
            by B[b_r0:b_r0+2, b_c0:b_c0+2] into C[c_r0:c_r0+2, c_c0:c_c0+2].
            s is the column offset in U, V, W.
            """
            # P1 = (A00 + A11)(B00 + B11)
            # U: A00, A11
            # V: B00, B11
            # W: C00, C01, C10, C11 (with coefficients)
            
            # We will define the 7 products explicitly.
            # Product 1: (A00 + A11) * (B00 + B11)
            U[a_r0 + 0, a_c0 + 0, s] += 1
            U[a_r0 + 1, a_c0 + 1, s] += 1
            V[b_r0 + 0, b_c0 + 0, s] += 1
            V[b_r0 + 1, b_c0 + 1, s] += 1
            # C00 = P1 + P4 - P5 + P7
            # C01 = P3 + P5
            # C10 = P2 + P4
            # C11 = P1 - P2 + P3 + P6
            # So P1 contributes to C00 (+1) and C11 (+1)
            W[(c_c0 + 0) * n + (c_r0 + 0), s] += 1 # C00
            W[(c_c0 + 1) * n + (c_r0 + 1), s] += 1 # C11
            
            # Product 2: (A10 + A11) * B00
            U[a_r0 + 1, a_c0 + 0, s] += 1
            U[a_r0 + 1, a_c0 + 1, s] += 1
            V[b_r0 + 0, b_c0 + 0, s] += 1
            # P2 contributes to C10 (+1) and C11 (-1)
            W[(c_c0 + 0) * n + (c_r0 + 1), s] += 1 # C10
            W[(c_c0 + 1) * n + (c_r0 + 1), s] -= 1 # C11
            
            # Product 3: A00 * (B01 - B11)
            U[a_r0 + 0, a_c0 + 0, s] += 1
            V[b_r0 + 0, b_c0 + 1, s] += 1
            V[b_r0 + 1, b_c0 + 1, s] -= 1
            # P3 contributes to C01 (+1) and C11 (+1)
            W[(c_c0 + 1) * n + (c_r0 + 0), s] += 1 # C01
            W[(c_c0 + 1) * n + (c_r0 + 1), s] += 1 # C11
            
            # Product 4: A11 * (B10 - B00)
            U[a_r0 + 1, a_c0 + 1, s] += 1
            V[b_r0 + 1, b_c0 + 0, s] += 1
            V[b_r0 + 0, b_c0 + 0, s] -= 1
            # P4 contributes to C00 (+1) and C10 (+1)
            W[(c_c0 + 0) * n + (c_r0 + 0), s] += 1 # C00
            W[(c_c0 + 0) * n + (c_r0 + 1), s] += 1 # C10
            
            # Product 5: (A00 + A01) * B11
            U[a_r0 + 0, a_c0 + 0, s] += 1
            U[a_r0 + 0, a_c0 + 1, s] += 1
            V[b_r0 + 1, b_c0 + 1, s] += 1
            # P5 contributes to C00 (-1) and C01 (+1)
            W[(c_c0 + 0) * n + (c_r0 + 0), s] -= 1 # C00
            W[(c_c0 + 1) * n + (c_r0 + 0), s] += 1 # C01
            
            # Product 6: (A10 - A00) * (B00 + B01)
            U[a_r0 + 1, a_c0 + 0, s] += 1
            U[a_r0 + 0, a_c0 + 0, s] -= 1
            V[b_r0 + 0, b_c0 + 0, s] += 1
            V[b_r0 + 0, b_c0 + 1, s] += 1
            # P6 contributes to C11 (+1)
            W[(c_c0 + 1) * n + (c_r0 + 1), s] += 1 # C11
            
            # Product 7: (A01 - A11) * (B10 + B11)
            U[a_r0 + 0, a_c0 + 1, s] += 1
            U[a_r0 + 1, a_c0 + 1, s] -= 1
            V[b_r0 + 1, b_c0 + 0, s] += 1
            V[b_r0 + 1, b_c0 + 1, s] += 1
            # P7 contributes to C00 (+1)
            W[(c_c0 + 0) * n + (c_r0 + 0), s] += 1 # C00
            
            return s + 7

        s = 0
        # 1. A_L * B_LL -> C_L
        # A_L: rows 0-1, cols 0-1. B_LL: rows 0-1, cols 0-1. C_L: rows 0-1, cols 0-1.
        s = add_strassen_term(0, 0, 0, 0, 0, 0, s)
        
        # 2. A_L * B_LR -> C_R
        # A_L: rows 0-1, cols 0-1. B_LR: rows 0-1, cols 2-3. C_R: rows 0-1, cols 2-3.
        s = add_strassen_term(0, 0, 0, 2, 0, 2, s)
        
        # 3. A_R * B_RL -> C_L
        # A_R: rows 0-1, cols 2-3. B_RL: rows 2-3, cols 0-1. C_L: rows 0-1, cols 0-1.
        s = add_strassen_term(0, 2, 2, 0, 0, 0, s)
        
        # 4. A_R * B_RR -> C_R
        # A_R: rows 0-1, cols 2-3. B_RR: rows 2-3, cols 2-3. C_R: rows 0-1, cols 2-3.
        s = add_strassen_term(0, 2, 2, 2, 0, 2, s)
        
        return U, V, W

    # 1. Construct Rank 28
    U, V, W = strassen_rank_28()
    
    # Verify
    err = check_identity(U, V, W)
    if err > 1e-9:
        # Fallback to schoolbook if Strassen construction fails (shouldn't happen)
        R = nm * mp # 32
        U_s = np.zeros((nm, R))
        V_s = np.zeros((mp, R))
        W_s = np.zeros((pn, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m + j, col] = 1
                    V_s[j*p + k, col] = 1
                    W_s[k*n + i, col] = 1
                    col += 1
        return {
            "U": U_s.tolist(),
            "V": V_s.tolist(),
            "W": W_s.tolist()
        }

    # 2. Try Rank 27
    # Use the remaining time to search for a rank 27 solution.
    if time.time() < deadline - 20:
        try:
            from scipy.optimize import minimize
            
            R = 27
            U_init = U[:, :R]
            V_init = V[:, :R]
            W_init = W[:, :R]
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(params):
                U_flat = params[:R*nm].reshape(nm, R)
                V_flat = params[R*nm:R*nm+R*mp].reshape(mp, R)
                W_flat = params[R*nm+R*mp:R*nm+R*mp+R*pn].reshape(pn, R)
                errors = get_errors(U_flat, V_flat, W_flat)
                return np.sum(errors**2)
            
            # Try a few random restarts or just one from the truncated rank 28
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'maxfun': 500})
            
            if res.fun < 1e-6:
                U_opt = res.x[:R*nm].reshape(nm, R)
                V_opt = res.x[R*nm:R*nm+R*mp].reshape(mp, R)
                W_opt = res.x[R*nm+R*mp:R*nm+R*mp+R*pn].reshape(pn, R)
                
                # Round to half-integers
                U_r = np.round(U_opt * 2) / 2.0
                V_r = np.round(V_opt * 2) / 2.0
                W_r = np.round(W_opt * 2) / 2.0
                
                err = check_identity(U_r, V_r, W_r)
                if err < 1e-9:
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
        except Exception:
            pass

    # Return Rank 28
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    print("Success")
