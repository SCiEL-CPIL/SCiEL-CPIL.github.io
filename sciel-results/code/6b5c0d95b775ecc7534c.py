import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 8
    dim_v = m * p  # 28
    dim_w = p * n  # 14
    
    # Build the multiplication tensor T as a flat vector
    # T[i*m+j, j*p+k, k*n+i] = 1
    T_flat = np.zeros(dim_u * dim_v * dim_w)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T_flat[idx_u * dim_v * dim_w + idx_v * dim_w + idx_w] = 1.0
    
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = np.zeros((dim_u, rank_school))
    V_school = np.zeros((dim_v, rank_school))
    W_school = np.zeros((dim_w, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    start_time = time.time()
    deadline = 115  # 115s limit for safety
    
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Helper: check if entries are integer or half-integer
    def is_half_integer(x, atol=1e-10):
        x2 = x * 2
        return np.allclose(x2, np.round(x2), atol=atol)
    
    # Helper: convert to clean half-integers
    def clean_half_int(x):
        x2 = x * 2
        x_rounded = np.round(x2) / 2
        return x_rounded
    
    # Helper: verify decomposition
    def verify_decomposition(U, V, W, T_flat, atol=1e-10):
        T_recon = np.einsum('it,jt,kt->ijk', U, V, W).flatten()
        return np.allclose(T_recon, T_flat, atol=atol)
    
    # Strategy 1: Exploit Kronecker structure m=4 = 2x2
    # Index j = j1*2 + j2 where j1, j2 in {0,1}
    # The tensor T[i, j, k] = T[i, j1*2+j2, k]
    # We can write this as a sum over j1, j2 of terms involving (i, j1) and (j2, k)
    
    # Build the base 2x2x7 tensor: multiplying (2x2) by (2x7)
    # This has dimensions: U_base: 2*2=4, V_base: 2*7=14, W_base: 7*2=14
    n_base, m_base, p_base = 2, 2, 7
    dim_u_base = n_base * m_base  # 4
    dim_v_base = m_base * p_base  # 14
    dim_w_base = p_base * n_base  # 14
    
    T_base_flat = np.zeros(dim_u_base * dim_v_base * dim_w_base)
    for i in range(n_base):
        for j in range(m_base):
            for k in range(p_base):
                idx_u = i * m_base + j
                idx_v = j * p_base + k
                idx_w = k * n_base + i
                T_base_flat[idx_u * dim_v_base * dim_w_base + idx_v * dim_w_base + idx_w] = 1.0
    
    # Try to find a low-rank decomposition of the base tensor
    # Then use Kronecker products to build the full decomposition
    
    def find_base_decomposition(T_base_flat, dim_u_b, dim_v_b, dim_w_b, target_rank, time_limit):
        """Find a rank-r decomposition of the base 2x2x7 tensor"""
        # Start with schoolbook
        U_b = np.zeros((dim_u_b, dim_u_b * dim_v_b // dim_w_b))  # placeholder
        V_b = np.zeros((dim_v_b, dim_u_b * dim_v_b // dim_w_b))
        W_b = np.zeros((dim_w_b, dim_u_b * dim_v_b // dim_w_b))
        
        # Schoolbook for base: rank = 2*2*7 = 28
        rank_sb = n_base * m_base * p_base
        U_b = np.zeros((dim_u_b, rank_sb))
        V_b = np.zeros((dim_v_b, rank_sb))
        W_b = np.zeros((dim_w_b, rank_sb))
        col = 0
        for i in range(n_base):
            for j in range(m_base):
                for k in range(p_base):
                    U_b[i * m_base + j, col] = 1
                    V_b[j * p_base + k, col] = 1
                    W_b[k * n_base + i, col] = 1
                    col += 1
        
        best_r = rank_sb
        best_Ub, best_Vb, best_Wb = U_b.copy(), V_b.copy(), W_b.copy()
        
        # Try greedy decomposition for the base tensor
        residual = T_base_flat.copy()
        U_list, V_list, W_list = [], [], []
        
        for r in range(min(target_rank, 30)):
            if time.time() - start_time > time_limit:
                break
            
            best_score = -np.inf
            best_u, best_v, best_w = None, None, None
            
            for trial in range(30):
                if time.time() - start_time > time_limit:
                    break
                
                nnz_u = random.randint(1, 3)
                nnz_v = random.randint(1, 4)
                nnz_w = random.randint(1, 3)
                
                u = np.zeros(dim_u_b)
                v = np.zeros(dim_v_b)
                w = np.zeros(dim_w_b)
                
                idx_u = random.sample(range(dim_u_b), nnz_u)
                idx_v = random.sample(range(dim_v_b), nnz_v)
                idx_w = random.sample(range(dim_w_b), nnz_w)
                
                u[idx_u] = np.random.choice([-1, 1], nnz_u)
                v[idx_v] = np.random.choice([-1, 1], nnz_v)
                w[idx_w] = np.random.choice([-1, 1], nnz_w)
                
                tensor = np.einsum('i,j,k->ijk', u, v, w).flatten()
                score = np.dot(tensor, residual)
                
                if score > best_score:
                    best_score = score
                    best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
            
            if best_score <= 1e-10:
                break
            
            tensor = np.einsum('i,j,k->ijk', best_u, best_v, best_w).flatten()
            residual -= tensor
            
            U_list.append(best_u)
            V_list.append(best_v)
            W_list.append(best_w)
            
            if np.allclose(residual, 0, atol=1e-10):
                break
        
        if len(U_list) > 0:
            U_g = np.column_stack(U_list)
            V_g = np.column_stack(V_list)
            W_g = np.column_stack(W_list)
            rank_g = len(U_list)
            
            # Try to refine with least squares
            M = np.einsum('it,jt,kt->ijt', U_g, V_g, W_g).reshape(-1, rank_g)
            try:
                c, _, _, _ = np.linalg.lstsq(M, T_base_flat, rcond=None)
                T_recon = (M @ c).reshape(dim_u_b, dim_v_b, dim_w_b)
                if np.allclose(T_recon, T_base_flat, atol=1e-10):
                    if is_half_integer(c, atol=1e-8):
                        U_final = U_g * c[None, :]
                        V_final = V_g
                        W_final = W_g
                        U_final = clean_half_int(U_final)
                        V_final = clean_half_int(V_final)
                        W_final = clean_half_int(W_final)
                        
                        if verify_decomposition(U_final, V_final, W_final, T_base_flat):
                            if rank_g < best_r:
                                best_r = rank_g
                                best_Ub, best_Vb, best_Wb = U_final, V_final, W_final
            except:
                pass
        
        return best_Ub, best_Vb, best_Wb, best_r
    
    # Find base decomposition
    Ub, Vb, Wb, rank_b = find_base_decomposition(
        T_base_flat, dim_u_base, dim_v_base, dim_w_base, 25, deadline - 20
    )
    
    # Now build the full 2x4x7 decomposition using Kronecker structure
    # j = j1*2 + j2
    # T[i, j1*2+j2, k] needs to be expressed
    # We can use: T_full = sum_{j1,j2} T_base[i, j1, k] * delta_{j2, ?}
    # Actually, we need to be more careful
    
    # The full tensor has indices (i, j, k) where j = j1*2 + j2
    # We can decompose j into (j1, j2) and use the base tensor structure
    
    # For each (j1, j2) pair, we have a contribution
    # The key insight: we can use Kronecker products of base vectors
    
    # Build full decomposition from base
    # U_full[i, j] where j = j1*2+j2
    # We need: sum_t U_full[i, j1*2+j2, t] * V_full[j1*2+j2, k, t] * W_full[k, i, t] = T[i, j1*2+j2, k]
    
    # Using Kronecker: if we have base vectors u_b[i,j1], v_b[j1,k], w_b[k,i]
    # Then for the full problem, we can use:
    # U_full[i, j1*2+j2, (t1, t2)] = u_b[i, j1, t1] * delta_{j2, t2}
    # V_full[j1*2+j2, k, (t1, t2)] = v_b[j1, k, t1] * delta_{j2, t2}  
    # W_full[k, i, (t1, t2)] = w_b[k, i, t1] * 1
    
    # This gives rank = rank_b * 2 (for the two j2 values)
    
    if rank_b < rank_school // 2:
        # Build Kronecker decomposition
        rank_full = rank_b * 2
        
        U_full = np.zeros((dim_u, rank_full))
        V_full = np.zeros((dim_v, rank_full))
        W_full = np.zeros((dim_w, rank_full))
        
        for t1 in range(rank_b):
            for j2 in range(2):
                t = t1 * 2 + j2
                
                # U: index (i, j1*2+j2)
                for i in range(n_base):
                    for j1 in range(m_base):
                        j = j1 * 2 + j2
                        U_full[i * m + j, t] = Ub[i * m_base + j1, t1]
                
                # V: index (j1*2+j2, k)
                for j1 in range(m_base):
                    for k in range(p_base):
                        j = j1 * 2 + j2
                        V_full[j * p + k, t] = Vb[j * p_base + k, t1]
                
                # W: index (k, i)
                for k in range(p_base):
                    for i in range(n_base):
                        W_full[k * n + i, t] = Wb[k * n_base + i, t1]
        
        # Verify
        if verify_decomposition(U_full, V_full, W_full, T_flat):
            if rank_full < best_rank:
                best_rank = rank_full
                best_U = U_full
                best_V = V_full
                best_W = W_full
    
    # Strategy 2: Direct greedy search with structured initialization
    if time.time() - start_time < deadline - 10:
        # Try a more targeted greedy search
        residual = T_flat.copy()
        U_list, V_list, W_list = [], [], []
        
        for r in range(min(40, deadline - 15 - int(time.time() - start_time))):
            if time.time() - start_time > deadline - 5:
                break
            
            best_score = -np.inf
            best_u, best_v, best_w = None, None, None
            
            for trial in range(40):
                if time.time() - start_time > deadline - 5:
                    break
                
                # Structured random: exploit the j = j1*2+j2 structure
                nnz_u = random.randint(2, 6)
                nnz_v = random.randint(3, 8)
                nnz_w = random.randint(2, 5)
                
                u = np.zeros(dim_u)
                v = np.zeros(dim_v)
                w = np.zeros(dim_w)
                
                # Sample indices with structure
                idx_u = random.sample(range(dim_u), nnz_u)
                idx_v = random.sample(range(dim_v), nnz_v)
                idx_w = random.sample(range(dim_w), nnz_w)
                
                u[idx_u] = np.random.choice([-1, 1], nnz_u)
                v[idx_v] = np.random.choice([-1, 1], nnz_v)
                w[idx_w] = np.random.choice([-1, 1], nnz_w)
                
                tensor = np.einsum('i,j,k->ijk', u, v, w).flatten()
                score = np.dot(tensor, residual)
                
                if score > best_score:
                    best_score = score
                    best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
            
            if best_score <= 1e-10:
                break
            
            tensor = np.einsum('i,j,k->ijk', best_u, best_v, best_w).flatten()
            residual -= tensor
            
            U_list.append(best_u)
            V_list.append(best_v)
            W_list.append(best_w)
            
            if np.allclose(residual, 0, atol=1e-10):
                break
        
        if len(U_list) > 0:
            U_g = np.column_stack(U_list)
            V_g = np.column_stack(V_list)
            W_g = np.column_stack(W_list)
            rank_g = len(U_list)
            
            M = np.einsum('it,jt,kt->ijt', U_g, V_g, W_g).reshape(-1, rank_g)
            try:
                c, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
                T_recon = (M @ c).reshape(dim_u, dim_v, dim_w)
                if np.allclose(T_recon, T_flat, atol=1e-10):
                    if is_half_integer(c, atol=1e-8):
                        U_final = U_g * c[None, :]
                        V_final = V_g
                        W_final = W_g
                        U_final = clean_half_int(U_final)
                        V_final = clean_half_int(V_final)
                        W_final = clean_half_int(W_final)
                        
                        if verify_decomposition(U_final, V_final, W_final, T_flat):
                            if rank_g < best_rank:
                                best_rank = rank_g
                                best_U = U_final
                                best_V = V_final
                                best_W = W_final
            except:
                pass
    
    if best_rank == rank_school:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
