# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    
    # 1. Initial Heuristic: Rank Matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    fac_order = np.argsort(-a_row_sums, kind="stable")
    loc_order = np.argsort(b_row_sums, kind="stable")
    
    p_init = np.empty(n, dtype=np.int64)
    p_init[fac_order] = loc_order
    
    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    current_cost = compute_cost(p_init)
    best_cost = current_cost
    best_p = p_init.copy()
    
    start_time = time.time()
    deadline = start_time + 145.0
    
    # 2. ADMM on Birkhoff Polytope
    # Minimize tr(A X B X^T) subject to X in Birkhoff polytope
    # ADMM: X^{k+1} = argmin tr(A X B X^T) + (rho/2)||X - Z^k + U^k||_F^2
    #        Z^{k+1} = proj_Birkhoff(X^{k+1} - U^k)
    #        U^{k+1} = U^k + X^{k+1} - Z^{k+1}
    
    A_sym = (a + a.T) / 2.0
    B_sym = (b + b.T) / 2.0
    
    # Start from rank-matching solution
    Z = np.zeros((n, n), dtype=np.float64)
    Z[np.arange(n), p_init] = 1.0
    
    rho = 1.0  # ADMM penalty parameter
    U = np.zeros((n, n), dtype=np.float64)  # Dual variable
    
    # We'll track the best solution from ADMM iterations
    admm_best_p = p_init.copy()
    admm_best_cost = best_cost
    
    # Precompute spectral norms for step size
    norm_A = np.linalg.norm(A_sym, ord=2)
    norm_B = np.linalg.norm(B_sym, ord=2)
    
    n_admm_iters = 300
    n_qp_steps = 8  # Inner QP iterations
    
    for k in range(n_admm_iters):
        if time.time() > deadline - 60.0:
            break
        
        # X-update: minimize tr(A X B X^T) + (rho/2)||X - (Z+U)||_F^2
        # Target for X is Z + U
        target = Z + U
        
        # Gradient of tr(A X B X^T) is A X B + B X A (for symmetric A,B)
        # Gradient of (rho/2)||X - target||_F^2 is rho*(X - target)
        # Full gradient: A X B + B X A + rho*(X - target)
        
        step_size = 0.5 / (norm_A * norm_B + rho)
        
        X = Z.copy()  # Initialize X at current Z
        for inner in range(n_qp_steps):
            AX = A_sym @ X
            grad_qp = AX @ B_sym + B_sym @ AX + rho * (X - target)
            X -= step_size * grad_qp
        
        # Z-update: project (X - U) to Birkhoff
        proj_arg = X - U
        row_ind, col_ind = linear_sum_assignment(-proj_arg)
        Z_new = np.zeros((n, n), dtype=np.float64)
        Z_new[row_ind, col_ind] = 1.0
        
        # Extract permutation from Z_new
        p_admm = np.empty(n, dtype=np.int64)
        p_admm[row_ind] = col_ind
        cost_admm = compute_cost(p_admm)
        
        if cost_admm < admm_best_cost:
            admm_best_cost = cost_admm
            admm_best_p = p_admm.copy()
            if cost_admm < best_cost:
                best_cost = cost_admm
                best_p = p_admm.copy()
        
        # U-update
        U = U + X - Z_new
        
        Z = Z_new
    
    # Update best from ADMM
    if admm_best_cost < best_cost:
        best_cost = admm_best_cost
        best_p = admm_best_p.copy()
    
    # 3. Local Search: Vectorized 2-Swap
    p = best_p.copy()
    current_cost = best_cost
    
    # Precompute for faster delta computation
    # For swap (i,j) with u=p[i], v=p[j]:
    # delta = sum_k a[i,k]*(b[v,p[k]]-b[u,p[k]]) + sum_k a[k,i]*(b[p[k],v]-b[p[k],u])
    #         + sum_k a[j,k]*(b[u,p[k]]-b[v,p[k]]) + sum_k a[k,j]*(b[p[k],u]-b[p[k],v])
    #         + a[i,j]*(b[v,u]-b[u,v]) + a[j,i]*(b[u,v]-b[v,u])
    #         - a[i,i]*(b[u,u]-b[u,u]) - a[j,j]*(b[v,v]-b[v,v])  [these are 0]
    #         - cross terms for k=i,j in the sums
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        best_delta = 0
        best_i = -1
        best_j = -1
        
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
                
            u = p[i]
            
            # Precompute contributions involving i
            a_i_row = a[i, :]
            a_i_col = a[:, i]
            b_u_row = b[u, :]
            b_u_col = b[:, u]
            
            for j in range(i + 1, n):
                if time.time() > deadline - 5.0:
                    break
                
                v = p[j]
                
                a_j_row = a[j, :]
                a_j_col = a[:, j]
                b_v_row = b[v, :]
                b_v_col = b[:, v]
                
                # Compute delta efficiently
                # Terms involving i and u:
                # sum_k a[i,k] * (b[v,p[k]] - b[u,p[k]]) for k != i,j
                # sum_k a[k,i] * (b[p[k],v] - b[p[k],u]) for k != i,j
                
                # Vectorized: (a_i_row - a_j_row) @ (b_v_row[p] - b_u_row[p])
                # but need to exclude i and j
                
                diff_a_row = a_i_row - a_j_row
                diff_a_col = a_i_col - a_j_col
                
                diff_b_row = b_v_row[p] - b_u_row[p]  # b[v,p[k]] - b[u,p[k]] for all k
                diff_b_col = b_v_col[p] - b_u_col[p]  # b[p[k],v] - b[p[k],u] for all k
                
                # Full sum
                sum_row = diff_a_row @ diff_b_row
                sum_col = diff_a_col @ diff_b_col
                
                # Subtract i and j contributions
                # For k=i: a[i,i]*(b[v,u]-b[u,u]) + a[i,i]*(b[u,v]-b[u,v]) = a[i,i]*(b[v,u]-b[u,v])
                # For k=j: a[j,j]*(b[v,v]-b[u,v]) + a[j,j]*(b[u,u]-b[v,u]) = a[j,j]*(b[u,u]-b[u,v] + b[v,v]-b[u,v])... 
                # Let's be more careful:
                # sum_row includes k=i: a[i,i]*(b[v,u]-b[u,u]) and k=j: a[j,j]*(b[v,v]-b[u,v])
                # sum_col includes k=i: a[i,i]*(b[u,v]-b[u,v]) = 0 and k=j: a[j,j]*(b[v,v]-b[u,v])... 
                
                # Actually, let's just compute directly for clarity
                # The delta formula:
                # delta = sum_{k!=i,j} [a[i,k]*(b[v,p[k]]-b[u,p[k]]) + a[k,i]*(b[p[k],v]-b[p[k],u])]
                #       + sum_{k!=i,j} [a[j,k]*(b[u,p[k]]-b[v,p[k]]) + a[k,j]*(b[p[k],u]-b[p[k],v])]
                #       + a[i,j]*(b[v,u]-b[u,v]) + a[j,i]*(b[u,v]-b[v,u])
                #       + a[i,i]*(b[v,u]-b[u,u]) + a[i,i]*(b[u,v]-b[u,v])  [k=i in first sum]
                #       + a[j,j]*(b[u,v]-b[v,v]) + a[j,j]*(b[v,u]-b[u,u])  [k=j in second sum... wait]
                
                # Let me use the original correct formula:
                D1 = a_i_row - a_j_row
                D2 = a_i_col - a_j_col
                
                diff_b_vu = b_v_row[p] - b_u_row[p]
                diff_b_pv_u = b_v_col[p] - b_u_col[p]
                
                Sum1_all = D1 @ diff_b_vu
                Sum2_all = D2 @ diff_b_pv_u
                
                # Subtract i and j terms
                # k=i: D1[i]*(b[v,u]-b[u,u]) + D2[i]*(b[u,v]-b[u,v])
                # k=j: D1[j]*(b[v,v]-b[u,v]) + D2[j]*(b[v,v]-b[v,u])
                
                Sub_i_1 = D1[i] * (b[v, u] - b[u, u])
                Sub_i_2 = D2[i] * (b[u, v] - b[u, u])
                
                Sub_j_1 = D1[j] * (b[v, v] - b[u, v])
                Sub_j_2 = D2[j] * (b[v, v] - b[v, u])
                
                Sum1 = Sum1_all - Sub_i_1 - Sub_j_1
                Sum2 = Sum2_all - Sub_i_2 - Sub_j_2
                
                Cross = a[i,j] * (b[v,u] - b[u,v]) + a[j,i] * (b[u,v] - b[v,u])
                
                delta = Cross + Sum1 + Sum2
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_delta < 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            improved = True
        else:
            break
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
