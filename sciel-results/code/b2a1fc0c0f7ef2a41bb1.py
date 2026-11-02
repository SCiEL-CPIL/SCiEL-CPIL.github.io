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
    
    # Try to find a lower rank decomposition using structured search
    # Strategy: Use Kronecker product structure and greedy refinement
    
    start_time = time.time()
    deadline = 120  # 120s limit, use 115s for safety
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Helper: check if a decomposition is valid
    def check_decomposition(U, V, W, T_flat, atol=1e-10):
        T_recon = np.einsum('it,jt,kt->ijk', U, V, W)
        T_recon_flat = T_recon.flatten()
        return np.allclose(T_recon_flat, T_flat, atol=atol)
    
    # Helper: check if entries are integer or half-integer
    def is_half_integer(x, atol=1e-10):
        x2 = x * 2
        return np.allclose(x2, np.round(x2), atol=atol)
    
    # Helper: convert to clean half-integers
    def clean_half_int(x):
        x2 = x * 2
        x_rounded = np.round(x2) / 2
        return x_rounded
    
    # Try greedy rank-1 decomposition with exact refinement
    # For each target rank R, try to find U, V, W with small integer entries
    
    # First, try a structured approach using the block structure
    # Split m=4 into 2+2, so we have two 2x2x7 problems
    # Actually, let's try to use the fact that we can group terms
    
    # Greedy approach: iteratively find rank-1 terms that best match the residual
    def greedy_decompose(T_flat, dim_u, dim_v, dim_w, max_rank, time_limit):
        residual = T_flat.copy()
        U_list = []
        V_list = []
        W_list = []
        
        for r in range(max_rank):
            if time.time() - start_time > time_limit:
                break
            
            # Find rank-1 tensor that best matches residual
            # Use random initialization + local search
            best_score = -np.inf
            best_u, best_v, best_w = None, None, None
            
            # Try several random initializations
            for trial in range(50):
                if time.time() - start_time > time_limit:
                    break
                
                # Random sparse vectors
                nnz_u = random.randint(1, 4)
                nnz_v = random.randint(1, 4)
                nnz_w = random.randint(1, 3)
                
                u = np.zeros(dim_u)
                v = np.zeros(dim_v)
                w = np.zeros(dim_w)
                
                idx_u = random.sample(range(dim_u), nnz_u)
                idx_v = random.sample(range(dim_v), nnz_v)
                idx_w = random.sample(range(dim_w), nnz_w)
                
                u[idx_u] = np.random.choice([-1, 1], nnz_u)
                v[idx_v] = np.random.choice([-1, 1], nnz_v)
                w[idx_w] = np.random.choice([-1, 1], nnz_w)
                
                # Compute inner product with residual
                tensor = np.einsum('i,j,k->ijk', u, v, w).flatten()
                score = np.dot(tensor, residual)
                
                if score > best_score:
                    best_score = score
                    best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
            
            if best_score <= 1e-10:
                break
            
            # Subtract the best rank-1 term
            tensor = np.einsum('i,j,k->ijk', best_u, best_v, best_w).flatten()
            residual -= tensor
            
            U_list.append(best_u)
            V_list.append(best_v)
            W_list.append(best_w)
            
            # Check if residual is zero
            if np.allclose(residual, 0, atol=1e-10):
                break
        
        if len(U_list) == 0:
            return None
        
        U = np.column_stack(U_list)
        V = np.column_stack(V_list)
        W = np.column_stack(W_list)
        
        return U, V, W, len(U_list)
    
    # Try greedy with different max ranks
    for target_rank in [25, 28, 32, 35, 40, 45, 50]:
        if time.time() - start_time > deadline - 10:
            break
        
        result = greedy_decompose(T_flat, dim_u, dim_v, dim_w, target_rank, deadline - 15)
        if result is not None:
            U, V, W, rank = result
            
            # Check if the decomposition is exact (up to scaling)
            # The greedy method gives approximate coefficients, so we need to refine
            # Solve for exact coefficients using least squares
            M = np.einsum('it,jt,kt->ijt', U, V, W).reshape(-1, rank)
            
            try:
                c, residuals, rank_m, sv = np.linalg.lstsq(M, T_flat, rcond=None)
            except:
                continue
            
            T_recon = (M @ c).reshape(dim_u, dim_v, dim_w)
            if np.allclose(T_recon, T_flat, atol=1e-10):
                # Check if coefficients are half-integers
                c2 = c * 2
                if np.allclose(c2, np.round(c2), atol=1e-10):
                    # Absorb coefficients into U
                    U_final = U * c[None, :]
                    V_final = V
                    W_final = W
                    
                    # Clean to half-integers
                    U_final = clean_half_int(U_final)
                    V_final = clean_half_int(V_final)
                    W_final = clean_half_int(W_final)
                    
                    # Final verification
                    T_check = np.einsum('it,jt,kt->ijk', U_final, V_final, W_final).flatten()
                    if np.allclose(T_check, T_flat, atol=1e-10):
                        if rank < best_rank:
                            best_rank = rank
                            best_U = U_final
                            best_V = V_final
                            best_W = W_final
    
    # If greedy didn't find anything better, try a different approach
    # Use the known rank 25 construction if we can find it
    # For now, return the best found
    
    if best_rank == rank_school:
        # Return schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
