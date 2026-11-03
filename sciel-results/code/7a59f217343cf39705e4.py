import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 120  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Build the tensor T[a,b,c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((n*m, m*p, p*n), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school), dtype=object)
    V_school = np.zeros((m*p, rank_school), dtype=object)
    W_school = np.zeros((p*n, rank_school), dtype=object)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Peeling algorithm
    T_work = T.copy()
    U_list = []
    V_list = []
    W_list = []
    
    def is_zero_tensor(T_arr, tol=1e-10):
        return np.all(np.abs(T_arr) < tol)
    
    def verify_decomposition(U_arr, V_arr, W_arr, n, m, p, tol=1e-10):
        """Verify that sum_r U[:,r]*V[:,r]*W[:,r] equals the tensor"""
        rank = U_arr.shape[1]
        T_recon = np.zeros((n*m, m*p, p*n), dtype=float)
        for r in range(rank):
            T_recon += np.outer(np.outer(U_arr[:,r], V_arr[:,r]), W_arr[:,r])
        # Check against expected
        T_expected = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b = j*p+k
                    c = k*n+i
                    T_expected[a,b,c] = 1
        return np.allclose(T_recon, T_expected, atol=tol)
    
    # Try peeling with random W vectors
    np.random.seed(42)
    max_peels = 84  # don't exceed schoolbook
    
    for peel_count in range(max_peels):
        if time.time() > deadline - 30:
            break
        if is_zero_tensor(T_work):
            break
            
        # Try to find a rank-1 component
        found = False
        for attempt in range(20):
            if time.time() > deadline - 30:
                break
            
            # Random W vector
            w = np.random.randn(p*n)
            # Form matrix M[a,b] = sum_c T_work[a,b,c] * w[c]
            M = np.tensordot(T_work, w, axes=([2],[0]))
            
            # Check if M has rank 1 (or close)
            # Use SVD
            try:
                U_svd, s, Vt_svd = np.linalg.svd(M, full_matrices=False)
            except:
                continue
            
            if len(s) == 0:
                continue
            
            # If the second singular value is negligible, M is approximately rank 1
            if len(s) > 1 and s[1] > 1e-8:
                continue
            
            if s[0] < 1e-10:
                continue
            
            # M = s[0] * U_svd[:,0] * Vt_svd[0,:]
            u = s[0] * U_svd[:, 0]
            v = Vt_svd[0, :]
            
            # Now we have T_work ≈ u * v^T * w^T + residual
            # But we need exact. Let's check if this is exact by computing residual
            T_component = np.outer(np.outer(u, v), w)
            T_new = T_work - T_component
            
            # Check if T_new is still a valid tensor (non-negative or structured)
            # Actually, for exact decomposition, we just need the sum to work out
            # Let's check if the residual is "clean"
            if np.allclose(T_new, np.round(T_new), atol=1e-8):
                # Good, we found a clean component
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)
                T_work = T_new
                found = True
                break
        
        if not found:
            # Try with structured W vectors
            for k in range(p):
                for i in range(n):
                    if time.time() > deadline - 30:
                        break
                    w_struct = np.zeros(p*n)
                    w_struct[k*n+i] = 1
                    M = np.tensordot(T_work, w_struct, axes=([2],[0]))
                    try:
                        U_svd, s, Vt_svd = np.linalg.svd(M, full_matrices=False)
                    except:
                        continue
                    if len(s) == 0 or s[0] < 1e-10:
                        continue
                    if len(s) > 1 and s[1] > 1e-8:
                        continue
                    u = s[0] * U_svd[:, 0]
                    v = Vt_svd[0, :]
                    T_component = np.outer(np.outer(u, v), w_struct)
                    T_new = T_work - T_component
                    if np.allclose(T_new, np.round(T_new), atol=1e-8):
                        U_list.append(u)
                        V_list.append(v)
                        W_list.append(w_struct)
                        T_work = T_new
                        found = True
                        break
                if found:
                    break
            if not found:
                break
    
    if len(U_list) > 0 and not is_zero_tensor(T_work):
        # Didn't fully peel, use schoolbook
        pass
    elif len(U_list) > 0:
        # Successfully peeled
        U_new = np.column_stack(U_list)
        V_new = np.column_stack(V_list)
        W_new = np.column_stack(W_list)
        new_rank = len(U_list)
        if new_rank < best_rank:
            if verify_decomposition(U_new, V_new, W_new, n, m, p):
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = new_rank
    
    # Convert to lists of lists with Fraction for exactness
    def to_fraction_list(arr):
        rows = []
        for row in arr:
            r = []
            for val in row:
                if abs(val) < 1e-10:
                    r.append(0)
                elif abs(val - round(val)) < 1e-10:
                    r.append(int(round(val)))
                else:
                    # Try fraction
                    frac = Fraction(val).limit_denominator(1000000)
                    r.append(frac)
            rows.append(r)
        return rows
    
    U_out = to_fraction_list(np.array(best_U, dtype=float))
    V_out = to_fraction_list(np.array(best_V, dtype=float))
    W_out = to_fraction_list(np.array(best_W, dtype=float))
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
