"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def _compute_cost(a, b, p):
    """Compute total QAP cost for permutation p."""
    p = np.asarray(p, dtype=np.int64)
    return int(np.sum(a * b[np.ix_(p, p)]))

def _delta_swap(a, D, i, j):
    """
    Exact delta cost for swapping positions i and j in permutation p,
    where D = b[p, p].
    """
    # Cost contribution of elements at positions i and j before swap
    # After swap, position i holds what was at j, and vice versa.
    
    # The change in cost comes from:
    # 1. Terms where one index is i or j (in the a matrix row/col)
    # 2. Terms where one index is the facility at i or j (in the b matrix)
    
    # Let's compute delta = new_cost - old_cost
    # new_cost - old_cost = sum over all k of a[k,i]*D_new[k,i] + a[k,j]*D_new[k,j] 
    #                       + a[i,k]*D_new[i,k] + a[j,k]*D_new[j,k]
    #                       - (same with D_old)
    # But D_new differs from D_old only in rows/cols i and j.
    
    # Actually, let me think more carefully.
    # Cost = sum_{m,n} a[m,n] * D[m,n]
    # D[m,n] = b[p[m], p[n]]
    # After swapping positions i and j: p'[i] = p[j], p'[j] = p[i]
    # D'[m,n] = b[p'[m], p'[n]]
    
    # D' differs from D only where m in {i,j} or n in {i,j}
    
    # Delta = sum_{m,n} a[m,n] * (D'[m,n] - D[m,n])
    # Non-zero only when m in {i,j} or n in {i,j}
    
    # Case 1: m=i, n not in {i,j}: D'[i,n] = b[p[j], p[n]], D[i,n] = b[p[i], p[n]]
    #   Contribution: a[i,n] * (b[p[j],p[n]] - b[p[i],p[n]]) for n not in {i,j}
    # Case 2: m=j, n not in {i,j}: a[j,n] * (b[p[i],p[n]] - b[p[j],p[n]])
    # Case 3: n=i, m not in {i,j}: a[m,i] * (b[p[m],p[j]] - b[p[m],p[i]])
    # Case 4: n=j, m not in {i,j}: a[m,j] * (b[p[m],p[i]] - b[p[m],p[j]])
    # Case 5: m=i, n=i: a[i,i]*(b[p[j],p[j]] - b[p[i],p[i]])
    # Case 6: m=j, n=j: a[j,j]*(b[p[i],p[i]] - b[p[j],p[j]])
    # Case 7: m=i, n=j: a[i,j]*(b[p[j],p[i]] - b[p[i],p[j]])
    # Case 8: m=j, n=i: a[j,i]*(b[p[i],p[j]] - b[p[j],p[i]])
    
    # But we don't have p directly, we have D. However D[m,n] = b[p[m],p[n]]
    # So b[p[j],p[n]] = D[j,n] and b[p[i],p[n]] = D[i,n]
    # Similarly b[p[m],p[j]] = D[m,j] and b[p[m],p[i]] = D[m,i]
    
    # Case 1: sum_{n != i,j} a[i,n] * (D[j,n] - D[i,n])
    # Case 2: sum_{n != i,j} a[j,n] * (D[i,n] - D[j,n])
    # Case 3: sum_{m != i,j} a[m,i] * (D[m,j] - D[m,i])
    # Case 4: sum_{m != i,j} a[m,j] * (D[m,i] - D[m,j])
    # Case 5: a[i,i] * (D[j,j] - D[i,i])
    # Case 6: a[j,j] * (D[i,i] - D[j,j])
    # Case 7: a[i,j] * (D[j,i] - D[i,j])
    # Case 8: a[j,i] * (D[i,j] - D[j,i])
    
    # Combine:
    # Case 1+2: sum_{n != i,j} (a[i,n] - a[j,n]) * (D[j,n] - D[i,n])
    #   = sum_{n != i,j} (a[i,n] - a[j,n]) * D[j,n] - sum_{n != i,j} (a[i,n] - a[j,n]) * D[i,n]
    #   = (a[i,:] - a[j,:]) dot (D[j,:] - D[i,:]) - (a[i,i]-a[j,i])*(D[j,i]-D[i,i]) - (a[i,j]-a[j,j])*(D[j,j]-D[i,j])
    
    # This is getting complex. Let me just compute it directly with vectorized ops.
    
    # Vectorized computation:
    # delta = sum over all (m,n) where m in {i,j} or n in {i,j} of a[m,n]*(D'[m,n]-D[m,n])
    
    # Let's build the difference matrix for D
    # D'[m,n] - D[m,n] is non-zero only when m in {i,j} or n in {i,j}
    
    # For efficiency, compute:
    # row_i_diff: D'[i,:] - D[i,:] -> only non-zero at positions i and j
    #   D'[i,i] = D[j,j], D'[i,j] = D[j,i], D'[i,n] = D[i,n] for n != i,j
    #   So D'[i,:] - D[i,:] has: at i: D[j,j]-D[i,i], at j: D[j,i]-D[i,j], else 0
    
    # Similarly for row j, col i, col j.
    
    # Let me just do it the straightforward way with the four cases + four corner cases.
    
    # Cases 1-4 (excluding corners i,j):
    # We need sums over indices excluding i and j.
    
    ai = a[i].copy()
    aj = a[j].copy()
    ai[i] = 0; ai[j] = 0
    aj[i] = 0; aj[j] = 0
    
    Di = D[i].copy()
    Dj = D[j].copy()
    Di[i] = 0; Di[j] = 0
    Dj[i] = 0; Dj[j] = 0
    
    # Case 1+2 combined
    part1 = np.dot(ai, Dj - Di) + np.dot(aj, Di - Dj)
    # = np.dot(ai - aj, Dj - Di)
    
    # Cases 3+4 combined
    ai_col = a[:, i].copy()
    aj_col = a[:, j].copy()
    ai_col[i] = 0; ai_col[j] = 0
    aj_col[i] = 0; aj_col[j] = 0
    
    Di_col = D[:, i].copy()
    Dj_col = D[:, j].copy()
    Di_col[i] = 0; Di_col[j] = 0
    Dj_col[i] = 0; Dj_col[j] = 0
    
    part2 = np.dot(ai_col, Dj_col - Di_col) + np.dot(aj_col, Di_col - Dj_col)
    
    # Corner cases 5-8
    part5 = a[i, i] * (D[j, j] - D[i, i])
    part6 = a[j, j] * (D[i, i] - D[j, j])
    part7 = a[i, j] * (D[j, i] - D[i, j])
    part8 = a[j, i] * (D[i, j] - D[j, i])
    
    return part1 + part2 + part5 + part6 + part7 + part8

def _local_search(a, b, p, time_limit, start_time):
    """
    2-opt local search for QAP.
    """
    n = len(p)
    p = np.asarray(p, dtype=np.int64).copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                return p, current_cost
            for j in range(i + 1, n):
                delta = _delta_swap(a, D, i, j)
                if delta < 0:
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    
                    # Update D
                    temp = D[i, :].copy()
                    D[i, :] = D[j, :]
                    D[j, :] = temp
                    
                    temp = D[:, i].copy()
                    D[:, i] = D[:, j]
                    D[:, j] = temp
                    
                    current_cost += int(delta)
                    improved = True
                    # Restart scan from beginning to find more improvements
                    break
            if improved:
                break
    
    return p, current_cost

def _hungarian_init(a, b):
    """
    Initialize using Hungarian algorithm on a cost matrix derived from flow and distance.
    """
    n = len(a)
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    # Use the principal eigenvectors to create a cost matrix
    try:
        _, eigvecs_a = np.linalg.eigh(a_sym)
        _, eigvecs_b = np.linalg.eigh(b_sym)
        
        v_a = eigvecs_a[:, -1]
        v_b = eigvecs_b[:, -1]
        
        # Create cost matrix: cost[i,j] = -v_a[i] * v_b[j] (we want to maximize correlation)
        cost_matrix = -np.outer(v_a, v_b)
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.zeros(n, dtype=np.int64)
        p[row_ind] = col_ind
        
        return p
    except Exception:
        return None

def _sa_search(a, b, p, start_temp, end_temp, time_limit, start_time, max_iters=100):
    """
    Simulated Annealing for QAP.
    """
    n = len(p)
    p = np.asarray(p, dtype=np.int64).copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    best_p = p.copy()
    best_cost = current_cost
    
    temp = start_temp
    total_iters = max_iters
    
    for it in range(total_iters):
        if (time.time() - start_time) > time_limit:
            break
        
        # Update temperature
        temp = start_temp * (end_temp / start_temp) ** (it / total_iters)
        
        # Random swap
        i, j = random.sample(range(n), 2)
        delta = _delta_swap(a, D, i, j)
        
        if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
            # Accept
            p[i], p[j] = p[j], p[i]
            
            # Update D
            temp_row = D[i, :].copy()
            D[i, :] = D[j, :]
            D[j, :] = temp_row
            
            temp_col = D[:, i].copy()
            D[:, i] = D[:, j]
            D[:, j] = temp_col
            
            current_cost += int(delta)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    return best_p, best_cost

def _perturb(p, k):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 145  # leave 5s buffer before 150s limit
    
    best_p = None
    best_cost = np.inf
    
    # Hungarian-based initialization
    p_init = _hungarian_init(a, b)
    if p_init is None:
        rng = random.Random(42)
        p_init = np.arange(n, dtype=np.int64)
        rng.shuffle(p_init.tolist())
        p_init = np.array(p_init, dtype=np.int64)
    
    # First local search
    elapsed = time.time() - start_time
    ls_time = min(30, time_budget - 20)
    p_ls, cost_ls = _local_search(a, b, p_init, ls_time, start_time)
    best_p = p_ls
    best_cost = cost_ls
    
    # Simulated Annealing from the local optimum
    elapsed = time.time() - start_time
    if time_budget - elapsed > 10:
        sa_time = min(20, time_budget - elapsed - 5)
        p_sa, cost_sa = _sa_search(a, b, best_p, start_temp=1000.0, end_temp=1.0,
                                   time_limit=sa_time, start_time=start_time, max_iters=200)
        if cost_sa < best_cost:
            best_cost = cost_sa
            best_p = p_sa
    
    # Iterated Local Search with adaptive perturbation
    k_perturb = 5
    while (time.time() - start_time) < (time_budget - 5):
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5:
            break
        
        # Perturb
        p_pert = _perturb(best_p, k_perturb)
        
        # Local search
        ls_time = min(remaining - 2, 15)
        p_ls, cost_ls = _local_search(a, b, p_pert, ls_time, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
            k_perturb = max(3, k_perturb - 1)
        else:
            k_perturb = min(15, k_perturb + 1)
    
    # Final polish
    elapsed = time.time() - start_time
    if time_budget - elapsed > 3:
        p_final, cost_final = _local_search(a, b, best_p, time_budget - elapsed - 1, start_time)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
