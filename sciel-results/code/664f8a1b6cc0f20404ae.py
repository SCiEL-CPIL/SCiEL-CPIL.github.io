# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta_vec(a, b, p, i, j):
    """
    Compute the exact change in cost when swapping facilities at positions i and j in permutation p.
    Uses vectorized operations for O(n) complexity.
    """
    n = len(p)
    pi = p[i]
    pj = p[j]
    
    # We derive the delta by considering the contribution of the pairs involving i or j.
    # Let u = pi, v = pj.
    # Old cost terms involving i or j:
    # Sum_{k} a[i,k] b[u, p[k]] + Sum_{k} a[k,i] b[p[k], u]
    # Sum_{k} a[j,k] b[v, p[k]] + Sum_{k} a[k,j] b[p[k], v]
    # Note: The term a[i,j]b[u,v] and a[j,i]b[v,u] are counted in both sums above if we are not careful,
    # but actually the standard QAP cost is sum_{i,j} a[i,j] b[p[i], p[j]].
    # When we swap i and j, p[i] becomes v and p[j] becomes u.
    # The positions of all other k remain the same.
    
    # The change comes from:
    # 1. Terms where one index is i or j and the other is k (k != i, j).
    # 2. Terms where both indices are i and j.
    
    # Let's compute the difference directly.
    # Old contribution of i (as row): sum_k a[i,k] b[u, p[k]]
    # New contribution of i (as row, now at v): sum_k a[i,k] b[v, p[k]]
    # Diff_row_i = sum_k a[i,k] (b[v, p[k]] - b[u, p[k]])
    
    # Old contribution of j (as row, now at u): sum_k a[j,k] b[u, p[k]]
    # New contribution of j (as row, now at v... wait, j moves to u): sum_k a[j,k] b[u, p[k]] -> No, j moves to u.
    # Wait, p[i] was u, p[j] was v.
    # After swap: p[i] is v, p[j] is u.
    # So facility i is now at v. Facility j is now at u.
    
    # Row i terms:
    # Old: sum_k a[i,k] b[u, p[k]]
    # New: sum_k a[i,k] b[v, p[k]]
    # Note: for k=i, p[k] changes from u to v. For k=j, p[k] changes from v to u. For k!=i,j, p[k] is same.
    # It is safer to compute the sum over all k and adjust for the specific changes in p[k] for k=i,j.
    
    # Let's use the formula:
    # Delta = sum_{k != i, j} [ a[i,k](b[v, p[k]] - b[u, p[k]]) + a[j,k](b[u, p[k]] - b[v, p[k]]) 
    #                           + a[k,i](b[p[k], v] - b[p[k], u]) + a[k,j](b[p[k], u] - b[p[k], v]) ]
    #       + (Terms for k=i and k=j explicitly)
    
    # Vectorized approach:
    # Let diff_b_row[k] = b[v, p[k]] - b[u, p[k]]
    # Let diff_b_col[k] = b[p[k], v] - b[p[k], u]
    
    # The sum over ALL k of a[i,k] * diff_b_row[k] includes:
    # k=i: a[i,i] * (b[v, u] - b[u, u])  (since p[i] becomes v, but in the vector b[v, p] we use OLD p? 
    # No, b[v, p] uses the CURRENT p array. p[i] is currently u. So b[v, p[i]] is b[v, u].
    # But in the NEW state, p[i] is v. So the term a[i,i] b[p[i], p[i]] becomes a[i,i] b[v, v].
    # The vector b[v, p] has at index i: b[v, u]. At index j: b[v, v].
    # The vector b[u, p] has at index i: b[u, u]. At index j: b[u, v].
    
    # This "vectorized over all k" approach requires careful subtraction of the i and j terms
    # because the "p[k]" inside the b lookup changes for k=i and k=j.
    
    # Let's stick to the robust derivation:
    # Delta = (New Cost) - (Old Cost)
    
    # Terms involving i or j in the sum sum_{r,s} a[r,s] b[p[r], p[s]]:
    # Pairs (i,k), (k,i), (j,k), (k,j) for all k.
    # Plus (i,i), (j,j), (i,j), (j,i).
    
    # Let's compute the contribution of the set of indices {i, j} to the total cost.
    # Let S = {i, j}.
    # Old Cost_S = sum_{r in S, s in S} a[r,s] b[p[r], p[s]] + sum_{r in S, s not in S} a[r,s] b[p[r], p[s]] + sum_{r not in S, s in S} a[r,s] b[p[r], p[s]]
    
    # This is getting complicated. Let's use the standard efficient delta formula.
    # delta = sum_{k != i, j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]]) 
    #       + sum_{k != i, j} (a[k,i] - a[k,j]) * (b[p[k], v] - b[p[k], u])
    #       + (a[i,i] - a[j,j]) * (b[v, v] - b[u, u])  # Diagonal terms change
    #       + a[i,j] * (b[v, u] - b[u, v]) 
    #       + a[j,i] * (b[u, v] - b[v, u])
    #       + a[i,i] * (b[v, u] - b[u, u]) + a[j,j] * (b[u, v] - b[v, v]) # Wait, these are covered?
    
    # Let's re-verify the "cross terms" formula from literature.
    # For a swap of i and j (u=p[i], v=p[j]):
    # delta = 2 * sum_{k != i, j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])  <-- This is for symmetric A and B?
    # If A and B are not symmetric, we need to be careful.
    
    # Let's use the explicit 4-sum method which is robust for non-symmetric matrices.
    # S1 = sum_k a[i,k] * (b[v, p[k]] - b[u, p[k]])
    # S2 = sum_k a[j,k] * (b[u, p[k]] - b[v, p[k]])
    # S3 = sum_k a[k,i] * (b[p[k], v] - b[p[k], u])
    # S4 = sum_k a[k,j] * (b[p[k], u] - b[p[k], v])
    
    # This sums over ALL k.
    # Let's check the k=i and k=j terms in these sums.
    # In S1 (k=i): a[i,i] * (b[v, u] - b[u, u]). 
    #   In the true delta, the term a[i,i] changes from b[u,u] to b[v,v].
    #   So the change is a[i,i]*(b[v,v] - b[u,u]).
    #   The vectorized S1 has a[i,i]*(b[v,u] - b[u,u]).
    #   Difference: a[i,i]*(b[v,v] - b[v,u]).
    
    # This indicates that simply summing over all k with the current p array does NOT account for the fact that p[i] and p[j] change values in the b lookup.
    
    # Correct Approach:
    # Compute the sum over k != i, j using the vectorized dot product.
    # Then handle k=i and k=j explicitly.
    
    # Let mask be all True, then set i and j to False.
    # But creating a mask every time is slow.
    
    # Alternative:
    # Sum_all = np.dot(a[i, :], b[v, p] - b[u, p])
    # This includes k=i: a[i,i]*(b[v,u]-b[u,u]) and k=j: a[i,j]*(b[v,v]-b[u,v]).
    
    # We want:
    # Sum_{k!=i,j} a[i,k]*(b[v,p[k]]-b[u,p[k]]) + (explicit i,j terms)
    
    # Let's calculate the explicit terms for the pairs involving i and j.
    # Old Cost terms involving i or j:
    # C_old = a[i,i]b[u,u] + a[i,j]b[u,v] + a[j,i]b[v,u] + a[j,j]b[v,v]
    #        + sum_{k!=i,j} (a[i,k]b[u,p[k]] + a[k,i]b[p[k],u] + a[j,k]b[v,p[k]] + a[k,j]b[p[k],v])
    
    # New Cost terms involving i or j (p[i]=v, p[j]=u):
    # C_new = a[i,i]b[v,v] + a[i,j]b[v,u] + a[j,i]b[u,v] + a[j,j]b[u,u]
    #        + sum_{k!=i,j} (a[i,k]b[v,p[k]] + a[k,i]b[p[k],v] + a[j,k]b[u,p[k]] + a[k,j]b[p[k],u])
    
    # Delta = C_new - C_old
    # Delta = [a[i,i](b[v,v]-b[u,u]) + a[i,j](b[v,u]-b[u,v]) + a[j,i](b[u,v]-b[v,u]) + a[j,j](b[u,u]-b[v,v])]
    #        + sum_{k!=i,j} [ a[i,k](b[v,p[k]]-b[u,p[k]]) + a[k,i](b[p[k],v]-b[p[k],u]) 
    #                         + a[j,k](b[u,p[k]]-b[v,p[k]]) + a[k,j](b[p[k],u]-b[p[k],v]) ]
    
    # Let's compute the sum part using vectorization.
    # Let V1 = b[v, p] - b[u, p]  (length n vector)
    # Let V2 = b[p, v] - b[p, u]  (length n vector)
    
    # Sum1 = sum_k a[i,k] V1[k]
    # Sum2 = sum_k a[k,i] V2[k]
    # Sum3 = sum_k a[j,k] (-V1[k])
    # Sum4 = sum_k a[k,j] (-V2[k])
    
    # Total_Sum = Sum1 + Sum2 + Sum3 + Sum4
    # This includes k=i and k=j terms.
    
    # k=i term in Total_Sum:
    # Sum1[k=i]: a[i,i] * (b[v,u] - b[u,u])
    # Sum2[k=i]: a[i,i] * (b[u,v] - b[u,u])
    # Sum3[k=i]: a[j,i] * (-(b[v,u] - b[u,u])) = a[j,i]*(b[u,u]-b[v,u])
    # Sum4[k=i]: a[i,j] * (-(b[u,v] - b[u,u])) = a[i,j]*(b[u,u]-b[u,v])
    
    # k=j term in Total_Sum:
    # Sum1[k=j]: a[i,j] * (b[v,v] - b[u,v])
    # Sum2[k=j]: a[j,i] * (b[v,v] - b[u,v])  # Wait, V2[j] = b[p[j], v] - b[p[j], u] = b[v,v] - b[v,u]? No.
    # p[j] is v. So V2[j] = b[v, v] - b[v, u].
    # Sum2[k=j]: a[j,i] * (b[v,v] - b[v,u])
    # Sum3[k=j]: a[j,j] * (-(b[v,v] - b[u,v])) = a[j,j]*(b[u,v]-b[v,v])
    # Sum4[k=j]: a[j,j] * (-(b[v,v] - b[v,u])) = a[j,j]*(b[v,u]-b[v,v])
    
    # This is getting messy. Let's just compute the delta by summing over k != i, j using a loop?
    # No, n=100. A Python loop of 100 iterations is fast enough if the inner op is vectorized?
    # Actually, np.dot is very fast. The overhead of the logic is the issue.
    
    # Let's try a different vectorized formulation that avoids the mask.
    # We can compute the full dot product and subtract the i and j components.
    
    # V1 = b[v, p] - b[u, p]
    # V2 = b[p, v] - b[p, u]
    
    # Full_Sum1 = np.dot(a[i, :], V1)
    # Full_Sum2 = np.dot(a[:, i], V2)
    # Full_Sum3 = np.dot(a[j, :], -V1)
    # Full_Sum4 = np.dot(a[:, j], -V2)
    
    # Full_Total = Full_Sum1 + Full_Sum2 + Full_Sum3 + Full_Sum4
    
    # Now subtract the k=i and k=j parts from Full_Total, and add the "True" k=i and k=j parts from the explicit formula above.
    
    # True_k_i = a[i,i](b[v,v]-b[u,u]) + a[i,j](b[v,u]-b[u,v]) + a[j,i](b[u,v]-b[v,u]) + a[j,j](b[u,u]-b[v,v]) is the TOTAL delta for the 2x2 block.
    # Wait, the explicit formula for the 2x2 block is:
    # Block_Delta = a[i,i](b[v,v]-b[u,u]) + a[i,j](b[v,u]-b[u,v]) + a[j,i](b[u,v]-b[v,u]) + a[j,j](b[u,u]-b[v,v])
    
    # The Full_Total includes the "incorrect" k=i and k=j contributions derived from the vector V1/V2 which use OLD p[i] and p[j].
    # We need to remove those and add the Block_Delta.
    
    # Incorrect_k_i_in_Full_Total = 
    #   a[i,i]*(b[v,u]-b[u,u]) + a[i,i]*(b[u,v]-b[u,u]) + a[j,i]*(b[u,u]-b[v,u]) + a[i,j]*(b[u,u]-b[u,v])
    # Incorrect_k_j_in_Full_Total =
    #   a[i,j]*(b[v,v]-b[u,v]) + a[j,i]*(b[v,v]-b[v,u]) + a[j,j]*(b[u,v]-b[v,v]) + a[j,j]*(b[v,u]-b[v,v])
    
    # This is too error-prone to derive manually in code comments. 
    # Let's use the "Exclude i,j" method with a precomputed mask? No, mask creation is O(n).
    # For n=100, O(n) is fine.
    
    # Actually, the simplest and most robust way for n=100 is:
    # 1. Compute the delta for the 2x2 block explicitly (O(1)).
    # 2. Compute the delta for the rest (k != i, j) using vectorization.
    #    To do this, we can use the full dot product and subtract the i and j terms.
    
    # Let's implement that.
    
    V1 = b[v, p] - b[u, p]
    V2 = b[p, v] - b[p, u]
    
    # Full sums
    S1_full = np.dot(a[i, :], V1)
    S2_full = np.dot(a[:, i], V2)
    S3_full = np.dot(a[j, :], -V1)
    S4_full = np.dot(a[:, j], -V2)
    
    Full_Sum = S1_full + S2_full + S3_full + S4_full
    
    # Subtract k=i terms
    # V1[i] = b[v, u] - b[u, u]
    # V2[i] = b[u, v] - b[u, u]
    term_i = a[i, i]*V1[i] + a[i, i]*V2[i] + a[j, i]*(-V1[i]) + a[i, j]*(-V2[i])
    
    # Subtract k=j terms
    # V1[j] = b[v, v] - b[u, v]
    # V2[j] = b[v, v] - b[v, u]  (since p[j]=v)
    term_j = a[i, j]*V1[j] + a[j, i]*V2[j] + a[j, j]*(-V1[j]) + a[j, j]*(-V2[j])
    
    # Sum over k != i, j
    Sum_rest = Full_Sum - term_i - term_j
    
    # Add explicit 2x2 block delta
    Block_Delta = (a[i,i]*(b[v,v]-b[u,u]) + 
                   a[i,j]*(b[v,u]-b[u,v]) + 
                   a[j,i]*(b[u,v]-b[v,u]) + 
                   a[j,j]*(b[u,u]-b[v,v]))
    
    return int(Sum_rest + Block_Delta)


def _hungarian_initial(a, b):
    """
    Create an initial permutation using Hungarian algorithm on a linear surrogate.
    Surrogate cost: C[i, j] = -a_row_sums[i] * b_row_sums[j]
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # We want to assign high flow to high distance centrality?
    # Usually, high flow facilities should be close to each other.
    # So we want to minimize sum a[i,j] b[p[i], p[j]].
    # If a[i] is large, we want b[p[i], p[j]] to be small.
    # This is not a simple rank matching.
    # The surrogate C[i, j] = -a_row_sums[i] * b_row_sums[j] assumes that high flow should go to low distance sum?
    # If b_row_sums[j] is small, location j is "central" (close to everything).
    # So we want high flow facilities at central locations.
    # Thus, we want to minimize sum a_row_sums[i] * b_row_sums[p[i]].
    # This is equivalent to maximizing sum a_row_sums[i] * (-b_row_sums[p[i]]).
    # So the cost matrix for Hungarian (which minimizes) should be:
    # Cost[i, j] = a_row_sums[i] * b_row_sums[j]
    # Wait, if we minimize sum a_i * b_j, we pair large a with small b.
    # That is correct.
    
    C = np.outer(a_row_sums, b_row_sums)
    
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p


def _local_search_vnd(a, b, p, deadline, best_cost, best_p):
    """Variable Neighborhood Descent (VND) with 2-opt."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            pi = p[i]
            for j in range(i + 1, n):
                d = _swap_delta_vec(a, b, p, i, j)
                if d < 0:
                    p[i], p[j] = p[j], p[i]
                    cur += d
                    improved = True
                    if cur < best_cost:
                        best_cost = cur
                        best_p = p.copy()
                    break  # First improvement
    
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
        
    return best_cost, best_p


def _anneal(a, b, p, deadline, best_cost, best_p, rng, t_start=50.0, t_end=0.1, alpha=0.995):
    """Simulated annealing polish."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    
    t = t_start
    while time.time() < deadline and t > t_end:
        i = rng.integers(n)
        j = rng.integers(n)
        if i == j:
            continue
        d = _swap_delta_vec(a, b, p, i, j)
        if d < 0 or rng.random() < np.exp(-d / max(t, 1e-10)):
            p[i], p[j] = p[j], p[i]
            cur += d
            if cur < best_cost:
                best_cost = cur
                best_p = p.copy()
        t *= alpha
    
    # Final local search
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    best_cost, best_p = _local_search_vnd(a, b, p, deadline, best_cost, best_p)
    return best_cost, best_p


def solve(a, b):
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    n = len(a)
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin before 120s kill

    rng = np.random.default_rng(12345)
    
    # --- Constructive starts ---
    best_cost = np.iinfo(np.int64).max
    best_p = None
    
    # Strategy 1: Rank matching (simple)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central
    c1 = _qap_cost(a, b, p1)
    if c1 < best_cost:
        best_cost = c1
        best_p = p1.copy()
        
    # Strategy 2: Hungarian
    p_hung = _hungarian_initial(a, b)
    c_hung = _qap_cost(a, b, p_hung)
    if c_hung < best_cost:
        best_cost = c_hung
        best_p = p_hung.copy()

    # Strategy 3: Greedy
    p2 = np.zeros(n, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    order = np.argsort(-a.sum(axis=1), kind="stable")
    for idx in order:
        if not np.any(used):
            p2[idx] = int(np.argmin(b.sum(axis=1)))
        else:
            placed = np.where(~used)[0]
            scores = np.zeros(n, dtype=np.float64)
            for j in placed:
                scores += a[idx, j] * b[:, p2[j]] + a[j, idx] * b[p2[j], :]
            candidates = np.where(~used)[0]
            p2[idx] = int(candidates[np.argmin(scores[candidates])])
        used[p2[idx]] = True
    c2 = _qap_cost(a, b, p2)
    if c2 < best_cost:
        best_cost = c2
        best_p = p2.copy()

    # --- Refinement ---
    p = best_p.copy()
    best_cost, best_p = _local_search_vnd(a, b, p, deadline, best_cost, best_p)
    
    # --- Diversification with Annealing ---
    while time.time() < deadline - 5.0:
        # Perturb current best
        p = best_p.copy()
        # Random 2-swap perturbation
        for _ in range(3):
            i = rng.integers(n)
            j = rng.integers(n)
            if i != j:
                p[i], p[j] = p[j], p[i]
                
        # Anneal from perturbed state
        best_cost, best_p = _anneal(a, b, p, deadline, best_cost, best_p, rng)
        
        # If we didn't improve, try a fresh random start occasionally
        if rng.random() < 0.1:
            p = rng.permutation(n).astype(np.int64)
            best_cost, best_p = _local_search_vnd(a, b, p, deadline, best_cost, best_p)

    return best_p.tolist()
# EVOLVE-BLOCK-END
