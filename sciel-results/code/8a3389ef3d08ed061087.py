import time
import numpy as np
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0
    
    # Precompute for speed
    # Ensure a and b are float for linear algebra stability, though they are ints
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)

    def compute_cost(p):
        B_p = b[np.ix_(p, p)]
        return int(np.sum(a * B_p))

    def rank_init():
        # Rank matching: high flow to close locations
        # Sum of flows from/to each facility
        flow_sums = a_f.sum(axis=1) + a_f.sum(axis=0)
        # Sum of distances to/from each location
        dist_sums = b_f.sum(axis=1) + b_f.sum(axis=0)
        
        # Sort facilities by flow descending
        fac_order = np.argsort(-flow_sums)
        # Sort locations by distance ascending (close locations get high flow)
        loc_order = np.argsort(dist_sums)
        
        p = np.empty(n, dtype=np.int64)
        p[fac_order] = loc_order
        return p

    def spectral_init():
        try:
            a_sym = (a_f + a_f.T) / 2.0
            b_sym = (b_f + b_f.T) / 2.0
            
            # Use top k eigenvectors
            k = max(10, n // 10)
            
            evals_a, evecs_a = np.linalg.eigh(a_sym)
            evals_b, evecs_b = np.linalg.eigh(b_sym)
            
            # Sort by absolute value
            idx_a = np.argsort(-np.abs(evals_a))[:k]
            idx_b = np.argsort(-np.abs(evals_b))[:k]
            
            V_a = evecs_a[:, idx_a]
            V_b = evecs_b[:, idx_b]
            
            # Match
            M = V_a.T @ V_b
            row_ind, col_ind = linear_sum_assignment(-M)
            
            p = np.empty(n, dtype=np.int64)
            p[row_ind] = col_ind
            # Fill remaining if k < n? linear_sum_assignment returns n pairs if M is n x n.
            # Here M is k x k. We need to map back to n.
            # This spectral init is tricky to map back to full permutation directly.
            # Better to use full eigenvectors if n is small (150 is small).
            
            # Let's use full eigenvectors
            idx_a_full = np.argsort(-np.abs(evals_a))
            idx_b_full = np.argsort(-np.abs(evals_b))
            V_a_full = evecs_a[:, idx_a_full]
            V_b_full = evecs_b[:, idx_b_full]
            
            M_full = V_a_full.T @ V_b_full
            row_ind, col_ind = linear_sum_assignment(-M_full)
            p[row_ind] = col_ind
            return p
        except:
            return None

    def hungarian_refine(p_cur):
        # Linearization: C = a @ b[:, p].T + a.T @ b[p, :]
        b_col_p = b_f[:, p_cur]
        b_row_p = b_f[p_cur, :]
        C = a_f @ b_col_p.T + a_f.T @ b_row_p
        row_ind, col_ind = linear_sum_assignment(C)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        return p_new

    def simulated_annealing(p, time_limit):
        p = p.copy()
        current_cost = compute_cost(p)
        best_p = p.copy()
        best_cost = current_cost
        
        # Precompute row_cost for O(1) delta
        # row_cost[i, l] = sum_k a[i,k] b[l, p[k]]
        row_cost = a_f @ b_f[:, p]
        
        # SA Parameters
        T = 1000.0 # Initial temp
        cooling = 0.99
        min_T = 1e-6
        
        while time.time() < time_limit and T > min_T:
            # Pick random i, j
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
                
            u = p[i]
            v = p[j]
            
            # Calculate Delta
            # Delta = row_cost[i, v] - row_cost[i, u] + row_cost[j, u] - row_cost[j, v] 
            #       + (a[i,i]-a[i,j])*(b[v,v]-b[v,u]) + (a[j,i]-a[j,j])*(b[u,v]-b[u,u])
            #       + sum_{k!=i,j} (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])
            
            # Vectorized sum part:
            # S = (a[:, i] - a[:, j]) @ (b[p, v] - b[p, u])
            # Subtract k=i, k=j terms
            diff_a = a_f[:, i] - a_f[:, j]
            diff_b = b_f[p, v] - b_f[p, u]
            S = np.dot(diff_a, diff_b)
            S -= (a_f[i, i] - a_f[i, j]) * (b_f[u, v] - b_f[u, u])
            S -= (a_f[j, i] - a_f[j, j]) * (b_f[v, u] - b_f[v, v])
            
            delta = (row_cost[i, v] - row_cost[i, u]) + (row_cost[j, u] - row_cost[j, v])
            delta += (a_f[i, i] - a_f[i, j]) * (b_f[v, v] - b_f[v, u])
            delta += (a_f[j, i] - a_f[j, j]) * (b_f[u, v] - b_f[u, u])
            delta += S
            
            # Acceptance
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                # Accept
                p[i], p[j] = v, u
                current_cost += delta
                
                # Update row_cost
                # row_cost += outer(a[:, i] - a[:, j], b[:, v] - b[:, u])
                da = a_f[:, i] - a_f[:, j]
                db = b_f[:, v] - b_f[:, u]
                row_cost += np.outer(da, db)
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            
            T *= cooling
        
        return best_p, best_cost

    # Main Logic
    best_p_overall = None
    best_cost_overall = float('inf')
    
    inits = []
    
    # 1. Rank Init
    p_rank = rank_init()
    inits.append(p_rank)
    
    # 2. Spectral Init
    p_spec = spectral_init()
    if p_spec is not None:
        inits.append(p_spec)
        
    # 3. Random Inits
    for _ in range(5):
        inits.append(np.random.permutation(n))
        
    # Time budget per restart
    # We have ~115s. 8 restarts -> ~14s each.
    time_per_restart = 110.0 / len(inits)
    
    for idx, p_init in enumerate(inits):
        if time.time() > deadline - 5:
            break
            
        # Hungarian refine first
        p_hung = hungarian_refine(p_init)
        c_hung = compute_cost(p_hung)
        
        # Pick better of init and hung
        c_init = compute_cost(p_init)
        if c_hung < c_init:
            p_cur = p_hung
            c_cur = c_hung
        else:
            p_cur = p_init
            c_cur = c_init
            
        # Simulated Annealing
        p_sa, c_sa = simulated_annealing(p_cur, deadline - 5 - (len(inits) - 1 - idx) * time_per_restart * 0.5) # Conservative time
        # Actually, just give each restart a fixed slice of time
        # Let's simplify: run SA for (deadline - time.time() - margin) / remaining_restarts
        
        if c_sa < best_cost_overall:
            best_cost_overall = c_sa
            best_p_overall = p_sa
            
    # Final Polish: Greedy Local Search on best solution
    if best_p_overall is not None and time.time() < deadline - 2:
        # Implement fast greedy local search
        p = best_p_overall.copy()
        cost = compute_cost(p)
        row_cost = a_f @ b_f[:, p]
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(n):
                if time.time() > deadline - 1:
                    break
                # Find best j > i
                best_delta = 0
                best_j = -1
                u = p[i]
                # Vectorized delta for all j
                # This is complex to fully vectorize for all j at once due to S term
                # So we loop j
                for j in range(i + 1, n):
                    v = p[j]
                    diff_a = a_f[:, i] - a_f[:, j]
                    diff_b = b_f[p, v] - b_f[p, u]
                    S = np.dot(diff_a, diff_b)
                    S -= (a_f[i, i] - a_f[i, j]) * (b_f[u, v] - b_f[u, u])
                    S -= (a_f[j, i] - a_f[j, j]) * (b_f[v, u] - b_f[v, v])
                    
                    delta = (row_cost[i, v] - row_cost[i, u]) + (row_cost[j, u] - row_cost[j, v])
                    delta += (a_f[i, i] - a_f[i, j]) * (b_f[v, v] - b_f[v, u])
                    delta += (a_f[j, i] - a_f[j, j]) * (b_f[u, v] - b_f[u, u])
                    delta += S
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_j = j
                        
                if best_j != -1:
                    # Apply swap
                    j = best_j
                    u = p[i]
                    v = p[j]
                    p[i], p[j] = v, u
                    cost += best_delta
                    da = a_f[:, i] - a_f[:, j]
                    db = b_f[:, v] - b_f[:, u]
                    row_cost += np.outer(da, db)
                    improved = True
                    
        if cost < best_cost_overall:
            best_cost_overall = cost
            best_p_overall = p
            
    return best_p_overall.tolist()
