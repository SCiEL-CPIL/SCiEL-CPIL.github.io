"""
Vectorized Iterated Local Search (ILS) for QAPLIB wil100.
Strategy:
1. Compute the exact delta for ALL possible swaps (i, j) simultaneously in O(n^2) 
   using vectorized NumPy matrix operations. This is orders of magnitude faster 
   than evaluating random moves one-by-one.
2. Greedily select the best improving swap and update the solution.
3. Repeat until no improving move exists (Local Optimum).
4. Apply a perturbation (shuffling a random block of facilities) to escape the 
   local minimum.
5. Repeat until the time limit is reached.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _greedy_initial(a, b):
    """
    Create a strong initial permutation using a greedy heuristic.
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

def _compute_all_deltas(a, b, p):
    """
    Compute the cost delta for swapping every pair of positions (i, j) in the 
    current permutation p.
    
    Returns a matrix D where D[i, j] is the cost change if we swap p[i] and p[j].
    """
    n = len(p)
    
    # Precompute views of B permuted by p
    # Bp[i, j] = b[p[i], p[j]]
    Bp = b[np.ix_(p, p)]
    
    # Term 1: sum_{k != i, j} (a[k, i] - a[k, j]) * (Bp[k, j] - Bp[k, i])
    # Let's compute sum_{k} (a[:, i] - a[:, j]) * (Bp[:, j] - Bp[:, i])
    # We can vectorize this over i and j.
    
    # diff_a[k, i, j] = a[k, i] - a[k, j]
    # This is an n x n x n tensor. For n=100, this is 1,000,000 elements. 
    # This is small enough to handle in memory and compute quickly.
    
    # A_diff = a[:, :, None] - a[:, None, :]  # Shape (n, n, n): A_diff[k, i, j]
    A_diff = a[:, :, None] - a[:, None, :]
    
    # Bp_diff[k, i, j] = Bp[k, j] - Bp[k, i]
    Bp_diff = Bp[:, :, None] - Bp[:, None, :]
    
    # T1_raw[i, j] = sum_k A_diff[k, i, j] * Bp_diff[k, i, j]
    T1_raw = np.einsum('kij,kij->ij', A_diff, Bp_diff)
    
    # We included k=i and k=j in the sum. We need to subtract them.
    # For k=i:
    # Term = (a[i, i] - a[i, j]) * (Bp[i, j] - Bp[i, i])
    # Let's compute the correction for all i, j.
    # a_ii is a vector.
    a_ii = np.diag(a)
    b_ii = np.diag(b) # b[p[i], p[i]] is 0 usually, but let's be general.
    # Actually Bp[i, i] = b[p[i], p[i]].
    Bp_ii = np.diag(Bp)
    
    # Correction 1 (k=i):
    # (a[i, i] - a[i, j]) * (Bp[i, j] - Bp[i, i])
    # a[i, i] is a[i]. a[i, j] is a[i].
    # Let's construct the matrix.
    # C1[i, j] = (a[i, i] - a[i, j]) * (Bp[i, j] - Bp[i, i])
    # a[i, i] - a[i, j] = a[i, i] - a[i]
    # Let A_row_i = a[i]
    # Let A_ii = a[i, i]
    # Diff_A_row = A_ii[:, None] - a
    # Diff_Bp_row = Bp - Bp_ii[:, None]
    # C1 = Diff_A_row * Diff_Bp_row
    
    # Correction 2 (k=j):
    # (a[j, i] - a[j, j]) * (Bp[j, j] - Bp[j, i])
    # a[j, i] - a[j, j] = a[:, j] - a[j, j]
    # Diff_A_col = a - a_ii[None, :]
    # Bp[j, j] - Bp[j, i] = Bp_ii[j] - Bp[j]
    # Diff_Bp_col = Bp_ii[None, :] - Bp
    # C2 = Diff_A_col * Diff_Bp_col
    
    # Wait, let's verify indices.
    # T1 is sum over k.
    # k=i term: (a[i, i] - a[i, j]) * (Bp[i, j] - Bp[i, i])
    #   a[i, i] is scalar for fixed i. a[i, j] is element of row i.
    #   So for fixed i, vector over j: (a[i, i] - a[i, :]) * (Bp[i, :] - Bp[i, i])
    #   This is C1[i, :].
    # k=j term: (a[j, i] - a[j, j]) * (Bp[j, j] - Bp[j, i])
    #   For fixed j, vector over i: (a[:, j] - a[j, j]) * (Bp[:, j] - Bp[j, j])
    #   This is C2[:, j].
    
    # Let's compute C1 and C2.
    # C1[i, j] = (a[i, i] - a[i, j]) * (Bp[i, j] - Bp[i, i])
    # Let A_ii_col = a_ii[:, None]
    # Let A_ii_row = a_ii[None, :]
    # C1 = (A_ii_col - a) * (Bp - Bp_ii[:, None])
    # C2 = (a - A_ii_row) * (Bp_ii[None, :] - Bp)
    
    C1 = (a_ii[:, None] - a) * (Bp - Bp_ii[:, None])
    C2 = (a - a_ii[None, :]) * (Bp_ii[None, :] - Bp)
    
    T1 = T1_raw - C1 - C2
    
    # Term 2: sum_{k != i, j} (a[i, k] - a[j, k]) * (Bp[j, k] - Bp[i, k])
    # By symmetry of the problem structure (swapping i and j is symmetric in the 
    # interaction with k), T2 is the transpose of T1?
    # Let's check.
    # T1[i, j] involves a[:, i] - a[:, j] and Bp[:, j] - Bp[:, i].
    # T2[i, j] involves a[i, :] - a[j, :] and Bp[j, :] - Bp[i, :].
    # If A and B are symmetric, T2[i, j] = T1[j, i].
    # But A is not symmetric in general QAP, though wil100 A is symmetric.
    # Let's just compute it to be safe, or assume symmetry if valid.
    # For wil100, A is symmetric.
    # If A is symmetric, a[:, i] - a[:, j] is the same as a[i, :] - a[j, :] (transposed).
    # And Bp is symmetric if B is symmetric.
    # So T2 = T1.T.
    
    # Let's verify with the formula.
    # T2_raw[i, j] = sum_k (a[i, k] - a[j, k]) * (Bp[j, k] - Bp[i, k])
    # If A is symmetric, a[i, k] = a[k, i].
    # Then (a[i, k] - a[j, k]) = (a[k, i] - a[k, j]).
    # And (Bp[j, k] - Bp[i, k]) = (Bp[k, j] - Bp[k, i]) if Bp is symmetric.
    # So T2_raw[i, j] = T1_raw[j, i].
    # The corrections C1 and C2 also transpose accordingly.
    # So T2 = T1.T.
    
    T2 = T1.T
    
    # Term 3: Direct interaction between i and j.
    # Delta = (a[i, j] - a[j, i]) * (b[p[j], p[i]] - b[p[i], p[j]])
    # If B is symmetric, b[p[j], p[i]] = b[p[i], p[j]], so Term 3 is 0.
    # For wil100, B is symmetric.
    # So T3 is 0.
    
    D = T1 + T2
    
    # The diagonal D[i, i] is irrelevant (swap with self).
    np.fill_diagonal(D, 0)
    
    return D

def _local_search(a, b, p, current_cost, deadline):
    """
    Perform greedy local search.
    """
    n = len(p)
    
    while time.time() < deadline:
        D = _compute_all_deltas(a, b, p)
        
        # Find the best improving move
        best_val = np.min(D)
        
        if best_val >= 0:
            break
            
        # Find indices of best move
        # There might be multiple, pick one.
        i, j = np.unravel_index(np.argmin(D), D.shape)
        
        # Apply swap
        p[i], p[j] = p[j], p[i]
        current_cost += best_val
        
    return p, current_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0 # Leave 5s buffer for the 120s limit
    start_time = time.time()
    
    # 1. Initial Solution
    p = _greedy_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Iterated Local Search
    while time.time() - start_time < total_deadline - 2:
        # Local Search
        p, current_cost = _local_search(a, b, p, current_cost, min(time.time() + 5, total_deadline - 2))
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
        # Perturbation
        # Shuffle a random block of size 5-10
        block_size = np.random.randint(5, 11)
        start_idx = np.random.randint(0, n - block_size)
        
        # Shuffle the block
        block = p[start_idx:start_idx + block_size].copy()
        np.random.shuffle(block)
        p[start_idx:start_idx + block_size] = block
        
        # Update cost
        current_cost = _qap_cost(a, b, p)
        
        # If perturbation was too disruptive, we might want to revert if it's much worse?
        # Standard ILS just proceeds to local search from the perturbed state.
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
