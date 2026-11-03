import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 150s limit
    
    # Target Tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify_decomposition(U, V, W):
        """Fast verification using einsum."""
        D = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(D, T, atol=1e-9)

    def is_half_integer(arr, atol=1e-6):
        """Check if all entries are close to half-integers."""
        return np.allclose(arr * 2, np.round(arr * 2), atol=atol)

    def is_rank_1_tensor(tens, tol=1e-8):
        """
        Check if a 3-way tensor of shape (I, J, K) is rank 1.
        Returns (True, u, v, w) if rank 1, else (False, None, None, None).
        """
        I, J, K = tens.shape
        
        # Check mode-1 matricization: (I, J*K)
        mat1 = tens.reshape(I, J * K)
        U_svd, s, Vt = np.linalg.svd(mat1, full_matrices=False)
        
        if s[1] > tol:
            return False, None, None, None
            
        # Extract u (first left singular vector)
        u = U_svd[:, 0]
        scale = s[0]
        
        # The right part reshaped to (J, K)
        M = (scale * Vt[0, :]).reshape(J, K)
        
        # Check if M is rank 1
        U_m, s_m, Vt_m = np.linalg.svd(M, full_matrices=False)
        if s_m[1] > tol:
            return False, None, None, None
            
        v = U_m[:, 0]
        w = Vt_m[0, :]
        
        # Reconstruct to verify
        recon = np.einsum('i,j,k->ijk', u, v, s_m[0] * w)
        if not np.allclose(recon, tens, atol=tol):
            return False, None, None, None
            
        # Additional check: mode-2 matricization should also be rank 1
        mat2 = tens.transpose(1, 0, 2).reshape(J, I * K)
        s2 = np.linalg.svd(mat2, full_matrices=False, compute_uv=False)
        if s2[1] > tol:
            return False, None, None, None
            
        # Mode-3 matricization
        mat3 = tens.transpose(2, 0, 1).reshape(K, I * J)
        s3 = np.linalg.svd(mat3, full_matrices=False, compute_uv=False)
        if s3[1] > tol:
            return False, None, None, None
            
        return True, u, v, s_m[0] * w

    def try_scale_to_half_int(u, v, w):
        """
        Try to find scalings such that all entries become half-integers.
        """
        # Try simple scaling factors
        for alpha in [1.0, 0.5, 2.0, 0.25, 4.0, 1.0/3.0, 3.0, 0.5/3.0, 1.5]:
            u_s = u * alpha
            v_s = v / alpha
            if is_half_integer(u_s) and is_half_integer(v_s) and is_half_integer(w):
                return u_s, v_s, w
        
        # Try scaling w instead
        for beta in [1.0, 0.5, 2.0, 0.25, 4.0, 1.0/3.0, 3.0]:
            w_s = w * beta
            u_s = u / beta
            if is_half_integer(u_s) and is_half_integer(v) and is_half_integer(w_s):
                return u_s, v, w_s
        
        # Try scaling v
        for gamma in [1.0, 0.5, 2.0, 0.25, 4.0, 1.0/3.0, 3.0]:
            v_s = v * gamma
            w_s = w / gamma
            if is_half_integer(u) and is_half_integer(v_s) and is_half_integer(w_s):
                return u, v_s, w_s
        
        return None, None, None

    def get_strassen_block_terms(i_base, j_base, k_base):
        """
        Returns U, V, W vectors for a 2x2x2 Strassen block.
        """
        U_terms = []
        V_terms = []
        W_terms = []
        
        def u_idx(i, j): return i * m + j
        def v_idx(j, k): return j * p + k
        def w_idx(k, i): return k * n + i
        
        # Term 1: (A00+A11)(B00+B11)
        U1 = {u_idx(i_base, j_base): 1, u_idx(i_base+1, j_base+1): 1}
        V1 = {v_idx(j_base, k_base): 1, v_idx(j_base+1, k_base+1): 1}
        W1 = {w_idx(k_base, i_base): 1, w_idx(k_base+1, i_base+1): 1}
        U_terms.append(U1); V_terms.append(V1); W_terms.append(W1)
        
        # Term 2: (A10+A11)B00
        U2 = {u_idx(i_base+1, j_base): 1, u_idx(i_base+1, j_base+1): 1}
        V2 = {v_idx(j_base, k_base): 1}
        W2 = {w_idx(k_base, i_base+1): 1}
        U_terms.append(U2); V_terms.append(V2); W_terms.append(W2)
        
        # Term 3: A00(B01-B11)
        U3 = {u_idx(i_base, j_base): 1}
        V3 = {v_idx(j_base, k_base+1): 1, v_idx(j_base+1, k_base+1): -1}
        W3 = {w_idx(k_base+1, i_base): 1}
        U_terms.append(U3); V_terms.append(V3); W_terms.append(W3)
        
        # Term 4: A11(B10-B00)
        U4 = {u_idx(i_base+1, j_base+1): 1}
        V4 = {v_idx(j_base+1, k_base): 1, v_idx(j_base, k_base): -1}
        W4 = {w_idx(k_base, i_base+1): 1}
        U_terms.append(U4); V_terms.append(V4); W_terms.append(W4)
        
        # Term 5: (A00+A01)B11
        U5 = {u_idx(i_base, j_base): 1, u_idx(i_base, j_base+1): 1}
        V5 = {v_idx(j_base+1, k_base+1): 1}
        W5 = {w_idx(k_base+1, i_base): 1}
        U_terms.append(U5); V_terms.append(V5); W_terms.append(W5)
        
        # Term 6: (A10-A00)(B00+B01)
        U6 = {u_idx(i_base+1, j_base): 1, u_idx(i_base, j_base): -1}
        V6 = {v_idx(j_base, k_base): 1, v_idx(j_base, k_base+1): 1}
        W6 = {w_idx(k_base, i_base+1): 1, w_idx(k_base+1, i_base+1): 1}
        U_terms.append(U6); V_terms.append(V6); W_terms.append(W6)
        
        # Term 7: (A01-A11)(B10+B11)
        U7 = {u_idx(i_base, j_base+1): 1, u_idx(i_base+1, j_base+1): -1}
        V7 = {v_idx(j_base+1, k_base): 1, v_idx(j_base+1, k_base+1): 1}
        W7 = {w_idx(k_base, i_base): 1, w_idx(k_base+1, i_base): 1}
        U_terms.append(U7); V_terms.append(V7); W_terms.append(W7)
        
        return U_terms, V_terms, W_terms

    # Initialize with Strassen blocks for (0,0,0) and (0,0,2)
    strassen_covered = set()
    blocks = [
        (0, 0, 0), 
        (0, 0, 2)  
    ]
    
    for ib, jb, kb in blocks:
        for i in range(ib, ib+2):
            for j in range(jb, jb+2):
                for k in range(kb, kb+2):
                    strassen_covered.add((i, j, k))
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Add Strassen columns
    for ib, jb, kb in blocks:
        U_terms, V_terms, W_terms = get_strassen_block_terms(ib, jb, kb)
        for u_t, v_t, w_t in zip(U_terms, V_terms, W_terms):
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            
            for idx, val in u_t.items():
                u_vec[idx] = val
            for idx, val in v_t.items():
                v_vec[idx] = val
            for idx, val in w_t.items():
                w_vec[idx] = val
            
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)
    
    # Add Schoolbook columns for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in strassen_covered:
                    continue
                u_vec = np.zeros(n * m)
                v_vec = np.zeros(m * p)
                w_vec = np.zeros(p * n)
                
                u_vec[i * m + j] = 1
                v_vec[j * p + k] = 1
                w_vec[k * n + i] = 1
                
                U_cols.append(u_vec)
                V_cols.append(v_vec)
                W_cols.append(w_vec)
    
    U_curr = np.column_stack(U_cols)
    V_curr = np.column_stack(V_cols)
    W_curr = np.column_stack(W_cols)
    
    rank_curr = U_curr.shape[1]
    best_rank = rank_curr
    best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
    
    if not verify_decomposition(U_curr, V_curr, W_curr):
        # Fallback to schoolbook if Strassen construction failed
        U_curr = np.zeros((n * m, n * m * p))
        V_curr = np.zeros((m * p, n * m * p))
        W_curr = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_curr[i * m + j, col] = 1.0
                    V_curr[j * p + k, col] = 1.0
                    W_curr[k * n + i, col] = 1.0
                    col += 1
        rank_curr = U_curr.shape[1]
        best_rank = rank_curr
        best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()

    D_current = T.copy()
    
    attempts = 0
    max_attempts = 3000  # Reduced attempts to fit in time, since start rank is lower
    
    while rank_curr > 20 and time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        idxs = list(range(rank_curr))
        random.shuffle(idxs)
        
        found = False
        for pair_idx in range(min(3, rank_curr - 1)):
            if time.time() >= deadline:
                break
                
            i_rem = idxs[pair_idx]
            j_rem = idxs[(pair_idx + 1) % rank_curr]
            if i_rem == j_rem:
                j_rem = idxs[(pair_idx + 2) % rank_curr]
                if i_rem == j_rem:
                    continue
            
            u_i = U_curr[:, i_rem]
            v_i = V_curr[:, i_rem]
            w_i = W_curr[:, i_rem]
            term_i = np.einsum('a,b,c->abc', u_i, v_i, w_i)
            
            u_j = U_curr[:, j_rem]
            v_j = V_curr[:, j_rem]
            w_j = W_curr[:, j_rem]
            term_j = np.einsum('a,b,c->abc', u_j, v_j, w_j)
            
            D_others = D_current - term_i - term_j
            Delta = T - D_others
            
            is_r1, u_new, v_new, w_new = is_rank_1_tensor(Delta)
            
            if is_r1:
                u_h, v_h, w_h = try_scale_to_half_int(u_new, v_new, w_new)
                
                if u_h is None:
                    if is_half_integer(u_new) and is_half_integer(v_new) and is_half_integer(w_new):
                        u_h, v_h, w_h = u_new, v_new, w_new
                    else:
                        continue
                
                mask = [k for k in range(rank_curr) if k != i_rem and k != j_rem]
                U_new = np.column_stack((U_curr[:, mask], u_h))
                V_new = np.column_stack((V_curr[:, mask], v_h))
                W_new = np.column_stack((W_curr[:, mask], w_h))
                
                new_rank = rank_curr - 1
                
                if verify_decomposition(U_new, V_new, W_new):
                    D_current = D_current - term_i - term_j + np.einsum('a,b,c->abc', u_h, v_h, w_h)
                    
                    U_curr = U_new
                    V_curr = V_new
                    W_curr = W_new
                    rank_curr = new_rank
                    if rank_curr < best_rank:
                        best_rank = rank_curr
                        best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
                    found = True
                    break
        
        if not found and rank_curr > 20:
            continue

    def to_list(arr):
        return arr.tolist()
        
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
