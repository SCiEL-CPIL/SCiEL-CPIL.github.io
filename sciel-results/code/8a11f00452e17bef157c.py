import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m      # 8
    dim_v = m * p      # 28
    dim_w = p * n      # 14
    rank_school = n * m * p  # 56

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

    # Schoolbook solution
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
    deadline = 110  # Safety margin before 120s limit

    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    def is_half_integer(x, atol=1e-10):
        x2 = x * 2
        return np.allclose(x2, np.round(x2), atol=atol)

    def clean_half_int(x):
        x2 = x * 2
        return np.round(x2) / 2

    def verify_decomposition(U, V, W, T_flat, atol=1e-10):
        T_recon = np.einsum('it,jt,kt->ijk', U, V, W).flatten()
        return np.allclose(T_recon, T_flat, atol=atol)

    # Strategy: Use structured decomposition based on splitting m=4 into 2+2 blocks
    # and p=7 into 3+4 blocks, then use Strassen-like methods on sub-blocks
    
    # Key insight: Split j into two groups of 2: j = j1*2 + j2 where j1 in {0,1}, j2 in {0,1}
    # Split k into two groups: k = k1*3 + k2 where k1 in {0,1}, k2 in {0,1,2}
    # This gives us sub-problems of size 2x2x3 and 2x2x4
    
    # For 2x2x3: can use rank 7 (Strassen for 2x2x2 + schoolbook for 2x2x1)
    # For 2x2x4: can use rank 7 (Strassen) + rank 4 (schoolbook for 2x2x2) = 11? Or better?
    
    # Actually, let's use a more systematic approach:
    # Decompose the tensor using the fact that it's a "product" tensor with specific structure
    
    # The tensor T[i,j,k] = 1 for all valid (i,j,k)
    # We can write T as sum of rank-1 tensors
    
    # Let's try a direct structured search with better initialization
    
    # Approach 1: Use the known lower bound and try to find decompositions near it
    # Lower bound for 2x4x7 is at least max(8, 28, 14) = 28, but likely higher
    
    # Let's use a greedy algorithm with structured patterns
    
    residual = T_flat.copy()
    U_list, V_list, W_list = [], [], []
    
    # Define structured patterns based on the block decomposition
    # j = j1*2 + j2, k = k1*3 + k2
    
    # For each (j1, k1) pair, we have a 2x2x3 or 2x2x4 sub-problem
    # Let's handle them separately
    
    # Sub-problem 1: j1=0, k1=0 -> j in {0,1}, k in {0,1,2} -> 2x2x3
    # Sub-problem 2: j1=0, k1=1 -> j in {0,1}, k in {3,4,5,6} -> 2x2x4
    # Sub-problem 3: j1=1, k1=0 -> j in {2,3}, k in {0,1,2} -> 2x2x3
    # Sub-problem 4: j1=1, k1=1 -> j in {2,3}, k in {3,4,5,6} -> 2x2x4
    
    # For 2x2x3: rank 7 (Strassen 7 + schoolbook 1 for the extra k)
    # Wait, Strassen is for 2x2x2. For 2x2x3, we need rank at least 7?
    # Actually, 2x2x3 can be done in rank 7: use Strassen for k=0,1 and schoolbook for k=2
    # But that's 7 + 4 = 11. Or we can be smarter.
    
    # Let's use a different approach: direct optimization
    
    # Use alternating least squares with random restarts
    
    def als_decomposition(T_flat, dim_u, dim_v, dim_w, rank, max_iter=100, n_restarts=5, seed=42):
        """Alternating least squares for tensor decomposition"""
        rng = np.random.RandomState(seed)
        best_rank_found = rank_school
        best_U, best_V, best_W = None, None, None
        
        for restart in range(n_restarts):
            if time.time() - start_time > deadline:
                break
            
            # Initialize with random sparse vectors
            U = np.zeros((dim_u, rank))
            V = np.zeros((dim_v, rank))
            W = np.zeros((dim_w, rank))
            
            for t in range(rank):
                # Sparse random initialization
                nnz_u = max(1, dim_u // 3)
                nnz_v = max(1, dim_v // 3)
                nnz_w = max(1, dim_w // 3)
                
                idx_u = rng.choice(dim_u, nnz_u, replace=False)
                idx_v = rng.choice(dim_v, nnz_v, replace=False)
                idx_w = rng.choice(dim_w, nnz_w, replace=False)
                
                U[idx_u, t] = rng.choice([-1, 1], nnz_u)
                V[idx_v, t] = rng.choice([-1, 1], nnz_v)
                W[idx_w, t] = rng.choice([-1, 1], nnz_w)
            
            # ALS iterations
            for it in range(max_iter):
                if time.time() - start_time > deadline:
                    break
                
                # Update U
                T_uv = np.einsum('jkt,jt->kt', T_flat.reshape(dim_u, dim_v, dim_w), V)
                for t in range(rank):
                    # U[:, t] = argmin ||T - sum_t U_t outer V_t outer W_t||
                    # This is a linear system
                    pass  # Skip for now, too complex
            
            # Verify
            if verify_decomposition(U, V, W, T_flat):
                if rank < best_rank_found:
                    best_rank_found = rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        return best_rank_found, best_U, best_V, best_W
    
    # Strategy 2: Use the block structure more explicitly
    # The tensor has the property that T[i,j,k] = 1 for all (i,j,k)
    # This means we can factor it as:
    # T = sum_{a,b,c} u_a v_b w_c where u_a[i] * v_b[j] * w_c[k] = delta_{i,j,k}
    
    # Let's try a more clever approach: use the fact that the tensor is "all ones"
    # in the valid region
    
    # For 2x4x7, the tensor is 8x28x14 with 1s at specific positions
    
    # Let's use a greedy approach with better pattern matching
    
    max_steps = 50
    for step in range(max_steps):
        if time.time() - start_time > deadline:
            break
        
        best_score = -np.inf
        best_u, best_v, best_w = None, None, None
        
        # Try multiple structured patterns
        n_trials = 50
        for trial in range(n_trials):
            if time.time() - start_time > deadline:
                break
            
            # Pattern based on block structure
            if trial % 4 == 0:
                # Focus on j1=0 blocks
                nnz_u = random.randint(3, 6)
                nnz_v = random.randint(8, 16)
                nnz_w = random.randint(4, 8)
            elif trial % 4 == 1:
                # Focus on j1=1 blocks
                nnz_u = random.randint(3, 6)
                nnz_v = random.randint(8, 16)
                nnz_w = random.randint(4, 8)
            elif trial % 4 == 2:
                # Focus on k1=0 blocks
                nnz_u = random.randint(4, 8)
                nnz_v = random.randint(6, 12)
                nnz_w = random.randint(3, 6)
            else:
                # Focus on k1=1 blocks
                nnz_u = random.randint(4, 8)
                nnz_v = random.randint(6, 12)
                nnz_w = random.randint(3, 6)
            
            u = np.zeros(dim_u)
            v = np.zeros(dim_v)
            w = np.zeros(dim_w)
            
            idx_u = random.sample(range(dim_u), nnz_u)
            idx_v = random.sample(range(dim_v), nnz_v)
            idx_w = random.sample(range(dim_w), nnz_w)
            
            # Use structured values: ±1 with some zeros
            u[idx_u] = np.random.choice([-1, 0, 1], nnz_u, p=[0.3, 0.2, 0.5])
            v[idx_v] = np.random.choice([-1, 0, 1], nnz_v, p=[0.3, 0.2, 0.5])
            w[idx_w] = np.random.choice([-1, 0, 1], nnz_w, p=[0.3, 0.2, 0.5])
            
            tensor = np.einsum('i,j,k->ijk', u, v, w).flatten()
            score = np.dot(tensor, residual)
            
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
        
        if best_score <= 1e-10 or best_u is None:
            break
        
        tensor = np.einsum('i,j,k->ijk', best_u, best_v, best_w).flatten()
        residual -= tensor
        
        U_list.append(best_u)
        V_list.append(best_v)
        W_list.append(best_w)
        
        if np.allclose(residual, 0, atol=1e-10):
            break
    
    # Try to refine with least squares
    if len(U_list) > 0 and len(U_list) < best_rank:
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
    
    # If we still have the schoolbook rank, return it
    if best_rank == rank_school:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
