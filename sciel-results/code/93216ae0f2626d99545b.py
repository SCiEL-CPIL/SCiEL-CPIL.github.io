"""
Iterated Local Search (ILS) solver for QAPLIB wil100.
Strategy:
1. Precompute matrix properties.
2. Use a greedy initial solution.
3. Perform Iterated Local Search:
   - Phase 1 (Repair/Descent): Evaluate the cost delta for ALL possible swaps (i, j) simultaneously using vectorized NumPy operations.
   - Select the best improving move. Repeat until no improving move exists (local optimum).
   - This "error_repair" mechanism ensures we always end up at a local minimum after perturbation,
     which structurally differs from the stochastic acceptance of SA.
   - Phase 2 (Perturbation): Apply a random sequence of k swaps to escape the local minimum.
   - Repeat until time limit.
4. Track the best solution found.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _greedy_initial(a, b):
    """
    Create an initial permutation using a greedy heuristic.
    Matches facilities with high flow to locations with low distance.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy = np.argsort(-a_row_sums, kind="stable")
    # Sort locations by ascending distance sum
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _vectorized_best_improvement(a, b, p):
    """
    Calculate the cost delta for all swaps (i, j) with i < j.
    Returns the best delta and the indices (i, j) of the best move.
    Returns None if no improving move is found.
    """
    n = len(p)
    # We want to compute delta for all pairs (i, j) where i < j.
    # Delta(i, j) = T1 + T2 + T3
    # T1: sum_{k!=i,j} (a[k,i]-a[k,j]) * (b[p[k],p[j]]-b[p[k],p[i]])
    # T2: sum_{k!=i,j} (a[i,k]-a[j,k]) * (b[p[j],p[k]]-b[p[i],p[k]])
    # T3: (a[i,j]-a[j,i]) * (b[p[j],p[i]]-b[p[i],p[j]])
    
    # To vectorize this for all pairs, we can exploit the structure.
    # However, full vectorization for n=100 (4950 pairs) might be memory heavy if done naively 
    # (100x100x100 is 1M elements, which is fine, but the operations are complex).
    # 
    # Alternative: Loop over i (100 times), and vectorize over j > i (up to 99 times).
    # This is O(n^2) numpy ops, which is very fast.
    
    best_delta = 0
    best_i = -1
    best_j = -1
    
    # Precompute views
    # For a fixed i, we want to compute deltas for all j > i.
    # Let J be the array of indices [i+1, ..., n-1].
    
    for i in range(n - 1):
        J = np.arange(i + 1, n)
        if len(J) == 0:
            continue
        
        # Extract relevant parts of A and B
        # a[:, i] is shape (n,)
        # a[:, J] is shape (n, |J|)
        # p[i] is scalar
        # p[J] is shape (|J|,)
        
        # Term 1: sum_{k!=i,j} (a[k,i]-a[k,j]) * (b[p[k],p[j]]-b[p[k],p[i]])
        # We compute sum over all k, then subtract k=i and k=j.
        
        # diff_a_col[k, j_idx] = a[k, i] - a[k, J[j_idx]]
        diff_a_col = a[:, i][:, None] - a[:, J] # shape (n, |J|)
        
        # diff_b_row[k, j_idx] = b[p[k], p[J[j_idx]]] - b[p[k], p[i]]
        # b[p, p[J]] is a matrix where row k is b[p[k], p[J[0]]], ..., b[p[k], p[J[-1]]]
        # b[p, p[i]] is a vector where row k is b[p[k], p[i]]
        b_p_pJ = b[p, p[J]] # shape (n, |J|)
        b_p_pi = b[p, p[i]] # shape (n,)
        diff_b_row = b_p_pJ - b_p_pi[:, None] # shape (n, |J|)
        
        sum_all_T1 = np.sum(diff_a_col * diff_b_row, axis=0) # shape (|J|)
        
        # Subtract k=i term:
        # (a[i,i]-a[i,J]) * (b[p[i],p[J]]-b[p[i],p[i]])
        term_1_i = (a[i, i] - a[i, J]) * (b[p[i], p[J]] - b[p[i], p[i]]) # shape (|J|)
        
        # Subtract k=j term (for each j in J):
        # (a[J, i]-a[J, J]) * (b[p[J],p[J]]-b[p[J],p[i]])
        # Note: a[J, J] is the diagonal of the submatrix? No, a[j, j].
        # a[J, i] is vector. a[J, J] is matrix. We need diagonal of a[J, J]?
        # The term is for specific pair (i, j). So a[j, j] is scalar for that j.
        # In vectorized form:
        # a_J_i = a[J, i] # shape (|J|)
        # a_J_J_diag = np.diag(a[J, J]) # shape (|J|)
        # b_pJ_pJ_diag = np.diag(b[p[J], p[J]]) # shape (|J|)
        # b_pJ_pi = b[p[J], p[i]] # shape (|J|)
        
        a_J_i = a[J, i]
        a_J_J_diag = np.diag(a[J, J])
        b_pJ_pJ_diag = np.diag(b[p[J], p[J]])
        b_pJ_pi = b[p[J], p[i]]
        
        term_1_j = (a_J_i - a_J_J_diag) * (b_pJ_pJ_diag - b_pJ_pi) # shape (|J|)
        
        T1 = sum_all_T1 - term_1_i - term_1_j # shape (|J|)
        
        # Term 2: sum_{k!=i,j} (a[i,k]-a[j,k]) * (b[p[j],p[k]]-b[p[i],p[k]])
        
        # diff_a_row[j_idx, k] = a[i, k] - a[J[j_idx], k]
        diff_a_row = a[i, :][None, :] - a[J, :] # shape (|J|, n)
        
        # diff_b_col[j_idx, k] = b[p[J[j_idx]], p[k]] - b[p[i], p[k]]
        # b[p[J], p] is matrix shape (|J|, n)
        # b[p[i], p] is vector shape (n,)
        b_pJ_p = b[p[J], p] # shape (|J|, n)
        b_pi_p = b[p[i], p] # shape (n,)
        diff_b_col = b_pJ_p - b_pi_p[None, :] # shape (|J|, n)
        
        sum_all_T2 = np.sum(diff_a_row * diff_b_col, axis=1) # shape (|J|)
        
        # Subtract k=i term:
        # (a[i,i]-a[J,i]) * (b[p[J],p[i]]-b[p[i],p[i]])
        term_2_i = (a[i, i] - a[J, i]) * (b[p[J], p[i]] - b[p[i], p[i]]) # shape (|J|)
        
        # Subtract k=j term:
        # (a[i,J]-a[J,J]) * (b[p[J],p[J]]-b[p[i],p[J]])
        a_i_J = a[i, J]
        a_J_J_diag = np.diag(a[J, J]) # Reuse
        b_pJ_pJ_diag = np.diag(b[p[J], p[J]]) # Reuse
        b_pi_pJ = b[p[i], p[J]]
        
        term_2_j = (a_i_J - a_J_J_diag) * (b_pJ_pJ_diag - b_pi_pJ) # shape (|J|)
        
        T2 = sum_all_T2 - term_2_i - term_2_j # shape (|J|)
        
        # Term 3: (a[i,J]-a[J,i]) * (b[p[J],p[i]]-b[p[i],p[J]])
        T3 = (a[i, J] - a[J, i]) * (b[p[J], p[i]] - b[p[i], p[J]]) # shape (|J|)
        
        deltas = T1 + T2 + T3 # shape (|J|)
        
        # Find best delta in this batch
        min_idx = np.argmin(deltas)
        min_val = deltas[min_idx]
        
        if min_val < best_delta:
            best_delta = min_val
            best_i = i
            best_j = J[min_idx]
            
    if best_delta < 0:
        return best_delta, best_i, best_j
    return None

def solve(a, b):
    n = len(a)
    total_deadline = 115.0 # Leave 5s buffer for the 120s limit
    start_time = time.time()
    
    # 1. Initial Solution
    p = _greedy_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Initial Local Search (Repair)
    while time.time() - start_time < total_deadline - 2:
        res = _vectorized_best_improvement(a, b, p)
        if res is None:
            break
        delta, i, j = res
        p[i], p[j] = p[j], p[i]
        current_cost += delta
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
    # 3. Iterated Local Search
    # Perturbation strength: number of random swaps
    # Start with small perturbations, increase if stuck?
    # Or fixed moderate perturbation.
    # For QAP, 5-10 swaps is often good.
    
    stagnant_rounds = 0
    current_perturb_size = 5
    
    while time.time() - start_time < total_deadline - 2:
        # Perturb
        p = best_p.copy()
        current_cost = best_cost
        
        # Apply random swaps
        for _ in range(current_perturb_size):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                p[i], p[j] = p[j], p[i]
        
        # Repair (Local Search)
        while time.time() - start_time < total_deadline - 2:
            res = _vectorized_best_improvement(a, b, p)
            if res is None:
                break
            delta, i, j = res
            p[i], p[j] = p[j], p[i]
            current_cost += delta
        
        # Check improvement
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            stagnant_rounds = 0
            # Reduce perturbation if we improved? Or keep it?
            # Let's try to reduce perturbation to fine-tune
            current_perturb_size = max(3, current_perturb_size - 1)
        else:
            stagnant_rounds += 1
            # Increase perturbation to escape
            current_perturb_size = min(20, current_perturb_size + 2)
            
    return best_p.tolist()

# EVOLVE-BLOCK-END
