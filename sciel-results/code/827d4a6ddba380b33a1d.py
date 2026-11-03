# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = 118.0  # safety margin before 120s/150s kill

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    # --- Initialization strategies ---
    
    # 1. Spectral initialization
    A_sym = (a + a.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    idx_a = np.argmax(eigvals)
    v_a = eigvecs[:, idx_a]

    B_sym = (b + b.T) / 2.0
    eigvals_b, eigvecs_b = np.linalg.eigh(B_sym)
    idx_b = np.argmax(eigvals_b)
    v_b = eigvecs_b[:, idx_b]

    fac_order = np.argsort(-v_a, kind="stable")
    loc_order = np.argsort(v_b, kind="stable")
    p_spectral = np.empty(n, dtype=np.int64)
    p_spectral[fac_order] = loc_order
    cost_spectral = compute_cost(p_spectral)

    # 2. Rank-based initialization
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    cost_rank = compute_cost(p_rank)

    # 3. Hungarian algorithm initialization (Linear Surrogate)
    # We want to minimize sum_i a_row_sum[i] * b[p[i], :] ... no, that's not a standard assignment.
    # Standard QAP linearization: min sum_i sum_j a[i,j] b[p[i], p[j]]
    # A common surrogate is to ignore the p[j] dependency in the second index and just minimize
    # sum_i (sum_j a[i,j]) * (sum_j b[i, j])? No, that doesn't use p.
    # Let's use the "centroid" method or similar.
    # Actually, a better surrogate for assignment is:
    # Cost matrix C[i, j] = sum_k a[i, k] * b[j, k] ? No, that's summing over k which is location.
    # Let's try: C[i, j] = (a[i, :] @ b[j, :]) ? This is the dot product of facility i's flow vector and location j's distance vector.
    # Minimizing sum_i a[i,:] @ b[p[i], :] is a valid linear assignment problem.
    C_hung = np.zeros((n, n))
    for i in range(n):
        C_hung[i, :] = np.dot(a[i, :], b)  # This computes sum_k a[i,k] b[j,k] for each j? No.
        # b is (n,n). b[j, :] is row j.
        # np.dot(a[i, :], b) computes sum_k a[i,k] b[k, :] -> resulting in a vector of length n where element j is sum_k a[i,k] b[k,j].
        # This is (A B)[i, j].
        # Let's check dimensions. a[i,:] is (1,n). b is (n,n). Result is (1,n).
        # So C_hung[i, j] = sum_k a[i, k] b[k, j].
    
    # Solve linear assignment
    row_ind, col_ind = linear_sum_assignment(C_hung)
    p_hung = np.empty(n, dtype=np.int64)
    p_hung[row_ind] = col_ind
    cost_hung = compute_cost(p_hung)

    # Select best initial
    candidates = [
        (p_spectral, cost_spectral),
        (p_rank, cost_rank),
        (p_hung, cost_hung)
    ]
    p, best_cost = min(candidates, key=lambda x: x[1])

    # --- Vectorized Local Search (First Improvement) ---
    # Optimized delta calculation
    def get_delta(p, i, j):
        u = p[i]
        v = p[j]
        # delta = 2 * ( (a[i]-a[j]) @ (b[v,p]-b[u,p]) - correction )
        # Let's use the exact formula derived in the previous thought process but optimized.
        # delta = (a[i,:]-a[j,:]) @ (b[v,p]-b[u,p]) + (a[:,i]-a[:,j]) @ (b[p,v]-b[p,u]) + correction
        # correction handles the i,i, j,j, i,j, j,i terms which are double counted or wrong in the vectorized part.
        
        # Vector part:
        diff_a_row = a[i] - a[j]
        diff_a_col = a[:, i] - a[:, j]
        diff_b_row = b[v, p] - b[u, p]
        diff_b_col = b[p, v] - b[p, u]
        
        val = np.dot(diff_a_row, diff_b_row) + np.dot(diff_a_col, diff_b_col)
        
        # Correction terms:
        # The vectorized sum includes k=i and k=j terms which are incorrect because p[i] and p[j] change.
        # We need to subtract the incorrect contributions and add the correct ones.
        # Incorrect in row part (k=i): (a[i,i]-a[j,i]) * (b[v,u]-b[u,u])
        # Incorrect in row part (k=j): (a[i,j]-a[j,j]) * (b[v,v]-b[u,v])
        # Incorrect in col part (k=i): (a[i,i]-a[i,j]) * (b[u,v]-b[u,u])
        # Incorrect in col part (k=j): (a[j,i]-a[j,j]) * (b[v,v]-b[u,v])
        
        # Let's just compute the full delta directly for safety and speed (n=100 is small)
        # Direct computation is O(n) per swap.
        # old_i = a[i,:] @ b[u,p] + a[:,i] @ b[p,u]
        # old_j = a[j,:] @ b[v,p] + a[:,j] @ b[p,v]
        # Note: b[u,p] includes b[u,u] at index i and b[u,v] at index j.
        # This is getting complicated. Let's stick to the simple O(n) swap check which is fast enough for n=100.
        
        # Simple O(n) check:
        # Construct p_new? No, just compute delta.
        # delta = (new_i + new_j) - (old_i + old_j)
        
        # old contribution of i and j to total cost:
        # C_old = sum_k a[i,k] b[u, p[k]] + sum_k a[k,i] b[p[k], u] + sum_k a[j,k] b[v, p[k]] + sum_k a[k,j] b[p[k], v]
        # But we must be careful not to double count a[i,j] b[u,v] and a[j,i] b[v,u] in the sums if we just sum rows/cols.
        # Actually, C = sum_i sum_j a[i,j] b[p[i], p[j]].
        # Terms involving i or j:
        # 1. a[i,k] b[u, p[k]] for all k.
        # 2. a[k,i] b[p[k], u] for all k.
        # 3. a[j,k] b[v, p[k]] for all k.
        # 4. a[k,j] b[p[k], v] for all k.
        # The term a[i,j] b[u,v] appears in 1 (k=j) and 4 (k=i).
        # The term a[j,i] b[v,u] appears in 3 (k=i) and 2 (k=j).
        # So if we sum 1+2+3+4, we count a[i,j]b[u,v] twice and a[j,i]b[v,u] twice.
        # But in the total cost C, a[i,j]b[u,v] appears only once.
        # So C_old_contribution = (sum 1 + sum 2 + sum 3 + sum 4) - a[i,j]b[u,v] - a[j,i]b[v,u]
        
        # New contribution:
        # p_new[i]=v, p_new[j]=u.
        # 1'. a[i,k] b[v, p_new[k]]
        # 2'. a[k,i] b[p_new[k], v]
        # 3'. a[j,k] b[u, p_new[k]]
        # 4'. a[k,j] b[p_new[k], u]
        # a[i,j] b[v,u] appears in 1' (k=j, p_new[j]=u) and 4' (k=i, p_new[i]=v).
        # a[j,i] b[u,v] appears in 3' (k=i, p_new[i]=v) and 2' (k=j, p_new[j]=u).
        # C_new_contribution = (sum 1' + sum 2' + sum 3' + sum 4') - a[i,j]b[v,u] - a[j,i]b[u,v]
        
        # We can compute these sums efficiently.
        # sum 1 = a[i,:] @ b[u, p]
        # sum 2 = a[:,i] @ b[p, u]
        # sum 3 = a[j,:] @ b[v, p]
        # sum 4 = a[:,j] @ b[p, v]
        
        # For new:
        # p_new is p with i,j swapped.
        # sum 1' = a[i,:] @ b[v, p_new]
        # b[v, p_new] is b[v, p] with b[v, p[i]] changed to b[v, v] and b[v, p[j]] changed to b[v, u].
        # p[i]=u, p[j]=v.
        # b[v, p] has b[v, u] at index i and b[v, v] at index j.
        # b[v, p_new] has b[v, v] at index i and b[v, u] at index j.
        # So sum 1' = a[i,:] @ b[v, p] - a[i,i]*b[v,u] - a[i,j]*b[v,v] + a[i,i]*b[v,v] + a[i,j]*b[v,u]
        #           = a[i,:] @ b[v, p] + a[i,i]*(b[v,v]-b[v,u]) + a[i,j]*(b[v,u]-b[v,v])
        
        # Similarly for others.
        
        old_1 = np.dot(a[i], b[u, p])
        old_2 = np.dot(a[:, i], b[p, u])
        old_3 = np.dot(a[j], b[v, p])
        old_4 = np.dot(a[:, j], b[p, v])
        old_val = old_1 + old_2 + old_3 + old_4 - a[i,j]*b[u,v] - a[j,i]*b[v,u]
        
        new_1 = old_3 # a[j,:] @ b[v,p] is same as a[i,:] @ b[v,p] if we swap i/j? No.
        # Let's just compute them.
        # b[v, p_new] vector:
        # It's b[v, p] but index i has b[v,v] and index j has b[v,u].
        # b[v, p] index i is b[v,u], index j is b[v,v].
        # diff_1 = b[v,v] - b[v,u] at i, b[v,u] - b[v,v] at j.
        # new_1 = old_3 + a[i,i]*(b[v,v]-b[v,u]) + a[i,j]*(b[v,u]-b[v,v])
        # Wait, old_3 is a[j,:] @ b[v,p]. We need a[i,:] @ b[v,p_new].
        # Let's just create the vector b_v_p_new.
        # b_v_p_new = b[v, p].copy()
        # b_v_p_new[i] = b[v, v]
        # b_v_p_new[j] = b[v, u]
        # new_1 = np.dot(a[i], b_v_p_new)
        
        # This is O(n) memory allocation per swap. Might be slow.
        # Let's stick to the algebraic derivation.
        
        # new_1 = np.dot(a[i], b[v, p]) + a[i,i]*(b[v,v]-b[v,u]) + a[i,j]*(b[v,u]-b[v,v])
        # new_2 = np.dot(a[:,i], b[p, v]) + a[i,i]*(b[v,v]-b[u,v]) + a[j,i]*(b[u,v]-b[v,v])
        #        Note: b[p, v] index i is b[u,v], index j is b[v,v].
        #        b[p_new, v] index i is b[v,v], index j is b[u,v].
        #        So diff is a[i,i]*(b[v,v]-b[u,v]) + a[j,i]*(b[u,v]-b[v,v])
        
        # new_3 = np.dot(a[j], b[u, p]) + a[j,i]*(b[u,u]-b[u,v]) + a[j,j]*(b[u,v]-b[u,u])
        #        b[u, p] index i is b[u,u], index j is b[u,v].
        #        b[u, p_new] index i is b[u,v], index j is b[u,u].
        #        So diff is a[j,i]*(b[u,v]-b[u,u]) + a[j,j]*(b[u,u]-b[u,v])
        
        # new_4 = np.dot(a[:,j], b[p, u]) + a[i,j]*(b[v,u]-b[u,u]) + a[j,j]*(b[u,u]-b[v,u])
        #        b[p, u] index i is b[u,u], index j is b[v,u].
        #        b[p_new, u] index i is b[v,u], index j is b[u,u].
        #        So diff is a[i,j]*(b[v,u]-b[u,u]) + a[j,j]*(b[u,u]-b[v,u])
        
        new_1 = np.dot(a[i], b[v, p]) + a[i,i]*(b[v,v]-b[v,u]) + a[i,j]*(b[v,u]-b[v,v])
        new_2 = np.dot(a[:,i], b[p, v]) + a[i,i]*(b[v,v]-b[u,v]) + a[j,i]*(b[u,v]-b[v,v])
        new_3 = np.dot(a[j], b[u, p]) + a[j,i]*(b[u,v]-b[u,u]) + a[j,j]*(b[u,u]-b[u,v])
        new_4 = np.dot(a[:,j], b[p, u]) + a[i,j]*(b[v,u]-b[u,u]) + a[j,j]*(b[u,u]-b[v,u])
        
        new_val = new_1 + new_2 + new_3 + new_4 - a[i,j]*b[v,u] - a[j,i]*b[u,v]
        
        return new_val - old_val

    def local_search(p, current_cost):
        p = p.copy()
        improved = True
        while improved:
            if time.time() - t_start > deadline:
                break
            improved = False
            # First improvement
            for i in range(n):
                if time.time() - t_start > deadline:
                    break
                for j in range(i+1, n):
                    delta = get_delta(p, i, j)
                    if delta < -1e-9:
                        p[i], p[j] = p[j], p[i]
                        current_cost += int(delta)
                        improved = True
                        break # Restart inner loop? Or just continue? First improvement usually restarts.
                if improved:
                    break
        return p, current_cost

    # Run local search on best initial
    p, best_cost = local_search(p, best_cost)

    # --- Iterated Local Search ---
    rng = np.random.RandomState(42)
    no_improve_count = 0
    max_no_improve = 20

    while time.time() - t_start < deadline - 2.0:
        # Perturbation
        # Use a mix of random swaps and maybe 3-opt?
        # Let's try random 2-4 swaps.
        n_swaps = rng.randint(2, 5)
        p_pert = p.copy()
        for _ in range(n_swaps):
            i, j = rng.choice(n, 2, replace=False)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
        
        p_new, cost_new = local_search(p_pert, compute_cost(p_pert))
        
        if cost_new < best_cost:
            p = p_new
            best_cost = cost_new
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Stronger perturbation
                no_improve_count = 0
                n_swaps = rng.randint(5, 10)
                p_pert = p.copy()
                for _ in range(n_swaps):
                    i, j = rng.choice(n, 2, replace=False)
                    p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
                p_new, cost_new = local_search(p_pert, compute_cost(p_pert))
                if cost_new < best_cost:
                    p = p_new
                    best_cost = cost_new

    return p.tolist()
# EVOLVE-BLOCK-END
