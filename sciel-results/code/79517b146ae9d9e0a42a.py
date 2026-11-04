"""
Strategy: Sequence-Partitioning Heuristic (Linear Ordering)
Models QAP as a 1D ordering problem based on structural tension (centrality).
Uses a greedy insertion algorithm to find a low-inversion linear ordering,
then polishes with fast vectorized local search.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j, a_sym=None):
    """
    Compute the change in cost when swapping facilities at positions i and j in permutation p.
    p is a numpy array where p[k] is the location assigned to facility k.
    """
    if a_sym is None:
        a_sym = a + a.T
    
    x = p[i]
    y = p[j]
    
    diff_a = a_sym[i, :] - a_sym[j, :]
    diff_b = b[y, :] - b[x, :]
    
    temp = diff_a * diff_b[p]
    delta = np.sum(temp) - diff_a[i] * diff_b[x] - diff_a[j] * diff_b[y]
    
    return delta


def _local_search_2opt(a, b, p, time_limit, a_sym=None, max_passes=10):
    """
    Perform 2-opt local search on permutation p.
    Returns the improved permutation and its cost.
    """
    n = len(p)
    if a_sym is None:
        a_sym = a + a.T
    
    best_p = p.copy()
    best_cost = _qap_cost(a, b, best_p)
    
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        # Vectorized evaluation of all swaps for a fixed i is expensive in memory 
        # but fast in numpy. However, for n=256, O(n^2) python loop with numpy delta 
        # is often faster than allocating huge arrays repeatedly.
        # We stick to the python loop but optimize the inner check.
        for i in range(n):
            if time.time() > time_limit:
                return best_p, best_cost
            # Precompute parts of delta for fixed i
            ai = a_sym[i, :]
            pi = best_p[i]
            # For each j, delta = sum_k (a_sym[i,k]-a_sym[j,k]) * (b[pi,k]-b[pj,k]) ... wait, formula is specific.
            # Let's just use the existing _swap_delta which is reasonably fast for n=256.
            for j in range(i + 1, n):
                if time.time() > time_limit:
                    return best_p, best_cost
                delta = _swap_delta(a, b, best_p, i, j, a_sym)
                if delta < 0:
                    best_p[i], best_p[j] = best_p[j], best_p[i]
                    best_cost += delta
                    improved = True
    
    return best_p, best_cost


def _linear_ordering_heuristic(a, b, a_sym=None):
    """
    Construct a permutation using a linear ordering heuristic.
    1. Compute 'structural tension' for facilities (row sums of a) and locations (row sums of b).
    2. Sort facilities by tension descending, locations by tension ascending (central to peripheral).
    3. Use a greedy insertion algorithm to build the sequence, minimizing the 'inversion cost'
       relative to the ideal rank ordering.
    """
    n = len(a)
    if a_sym is None:
        a_sym = a + a.T
    
    # 1. Compute centrality measures
    # Facilities: high flow sum should be central (low distance sum)
    # Locations: low distance sum is central
    fac_centrality = a_sym.sum(axis=1)  # Higher is more 'important'
    loc_centrality = b.sum(axis=1)      # Lower is more 'central'
    
    # Ideal ordering: High fac_centrality should map to Low loc_centrality.
    # Let's define a target rank for each facility based on its centrality.
    # Rank 0 = most important facility.
    fac_ranks = np.argsort(-fac_centrality, kind='stable') # Index of facility at rank r
    # Let's define loc_ranks: Index of location at rank r (0 = most central)
    loc_ranks = np.argsort(loc_centrality, kind='stable')
    
    # We want to assign facility fac_ranks[r] to location loc_ranks[r] ideally.
    # But QAP is not just rank matching. It involves pairwise interactions.
    # The "Linear Ordering" formulation suggests that the cost is dominated by inversions.
    # We construct a sequence of facilities. The position in the sequence determines the location.
    # Let sequence S be an ordering of facilities.
    # Location of S[k] is loc_ranks[k].
    
    # Greedy Insertion:
    # Start with empty sequence.
    # Add facilities one by one.
    # Which one to add next? The one with highest centrality? Or random?
    # Let's try a "Spectral" or "Rank-based" greedy.
    
    # A simpler approach: Use the rank matching as a base, then refine using a "best insertion" 
    # that considers the actual QAP cost contribution.
    
    # Base permutation from rank matching
    p_base = np.empty(n, dtype=np.int64)
    for r in range(n):
        f = fac_ranks[r]
        l = loc_ranks[r]
        p_base[f] = l
        
    # Now, refine this base permutation using a sequence-based insertion.
    # We view the permutation as a sequence of facilities assigned to locations in order of loc_centrality.
    # Let seq be the list of facilities such that seq[k] is assigned to loc_ranks[k].
    # Initially, seq = fac_ranks (since p_base[fac_ranks[r]] = loc_ranks[r]).
    
    seq = fac_ranks.copy()
    
    # Refine: Try to move facilities to better positions in the sequence.
    # This is essentially a local search on the sequence order.
    # We use a "best improvement" strategy on the sequence.
    
    current_seq = seq.copy()
    current_cost = _qap_cost(a, b, p_base)
    
    # To speed up, we don't do full 2-opt yet. We do a "sequence polish".
    # For each facility, try moving it to the best position in the sequence.
    
    for _ in range(5): # 5 passes of sequence polish
        for idx in range(n):
            if time.time() > time.time() + 0: # dummy check, we use external limit later
                pass
            fac = current_seq[idx]
            # Remove fac from seq
            seq_no_fac = np.delete(current_seq, idx)
            
            # Find best insertion position for fac
            best_pos = idx
            best_cost = current_cost
            
            # We can't easily compute delta for insertion without recomputing cost.
            # For n=256, recomputing cost is O(n^2) = 65k ops. 256*256*5 = 327k ops. Very fast.
            # Let's just recompute cost for candidate permutations.
            
            # Optimization: Only check positions near current idx?
            # Or check all? Let's check all for the first pass, then narrow.
            
            for new_idx in range(n):
                if new_idx == idx:
                    continue
                # Create candidate seq
                # Insert fac at new_idx
                # If new_idx > idx, inserting at new_idx in the list of length n-1 
                # corresponds to original index new_idx.
                # If new_idx < idx, inserting at new_idx in list of length n-1
                # corresponds to original index new_idx.
                
                # Efficient construction:
                # candidate_seq = np.concatenate([seq_no_fac[:new_idx], [fac], seq_no_fac[new_idx:]])
                
                # Compute cost
                # p_candidate: facility fac is at loc_ranks[new_idx]
                # Other facilities: their locations shift?
                # Wait. The mapping is:
                # current_seq[k] is at loc_ranks[k].
                # If we move fac from idx to new_idx:
                # If new_idx > idx:
                #   Facs at idx..new_idx-1 shift down by 1 in location index.
                # If new_idx < idx:
                #   Facs at new_idx..idx-1 shift up by 1 in location index.
                
                # This is complex to delta. Recomputation is safer and fast enough.
                
                # Let's just generate the full permutation p_candidate.
                p_cand = np.empty(n, dtype=np.int64)
                for k in range(n):
                    if k < new_idx:
                        p_cand[seq_no_fac[k]] = loc_ranks[k]
                    else:
                        p_cand[seq_no_fac[k-1]] = loc_ranks[k]
                p_cand[fac] = loc_ranks[new_idx]
                
                # Actually, simpler:
                # candidate_seq = list(seq_no_fac)
                # candidate_seq.insert(new_idx, fac)
                # p_cand = np.zeros(n, dtype=np.int64)
                # for k, f in enumerate(candidate_seq):
                #     p_cand[f] = loc_ranks[k]
                
                # Let's use numpy for speed
                cand_seq = np.empty(n, dtype=np.int64)
                cand_seq[:new_idx] = seq_no_fac[:new_idx]
                cand_seq[new_idx] = fac
                cand_seq[new_idx+1:] = seq_no_fac[new_idx:]
                
                p_cand = np.empty(n, dtype=np.int64)
                p_cand[cand_seq] = loc_ranks
                
                cost_cand = _qap_cost(a, b, p_cand)
                
                if cost_cand < best_cost:
                    best_cost = cost_cand
                    best_pos = new_idx
                    best_p = p_cand
                    best_seq = cand_seq
            
            if best_pos != idx:
                current_seq = best_seq
                current_cost = best_cost
    
    # Convert best_seq back to permutation p
    p_final = np.empty(n, dtype=np.int64)
    p_final[current_seq] = loc_ranks
    
    return p_final, current_cost


def _solve_tai256c(a, b, time_limit=145.0):
    n = len(a)
    start_time = time.time()
    
    # Precompute
    a_sym = a + a.T
    
    # 1. Generate initial solution using Linear Ordering Heuristic
    p_init, cost_init = _linear_ordering_heuristic(a, b, a_sym)
    
    best_p = p_init.copy()
    best_cost = cost_init
    
    # 2. Polish with Local Search
    # Use a tight time limit for the first polish
    t_limit_1 = time_limit * 0.3
    best_p, best_cost = _local_search_2opt(a, b, best_p, start_time + t_limit_1, a_sym, max_passes=20)
    
    # 3. Iterated Local Search (ILS)
    # Diversify by perturbing the sequence order
    while time.time() - start_time < time_limit - 10.0:
        # Perturb: Randomly swap a few elements in the sequence representation
        # Or just random swaps in p
        p_pert = best_p.copy()
        num_swaps = random.randint(10, 30)
        for _ in range(num_swaps):
            i, j = random.sample(range(n), 2)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
        
        # Polish perturbed solution
        # Give it a smaller time slice
        t_slice = 5.0
        p_polished, cost_polished = _local_search_2opt(
            a, b, p_pert, time.time() + t_slice, a_sym, max_passes=5
        )
        
        if cost_polished < best_cost:
            best_cost = cost_polished
            best_p = p_polished.copy()
            
        # Occasionally try a fresh Linear Ordering restart with random noise
        if random.random() < 0.1 and time.time() - start_time < time_limit - 30.0:
            # Add noise to centrality to get a different starting point
            noise_a = np.random.normal(0, 0.1 * a_sym.std(), (n, n))
            a_noisy = a_sym + noise_a
            # Re-run heuristic with noisy a? 
            # The heuristic uses a_sym for ranks. We can just randomize the ranks slightly.
            # Or just skip this for simplicity and rely on ILS swaps.
            pass

    # 4. Final Polish with remaining time
    remaining = time_limit - (time.time() - start_time)
    if remaining > 2.0:
        best_p, best_cost = _local_search_2opt(
            a, b, best_p, time.time() + remaining - 1.0, a_sym, max_passes=10
        )
    
    return best_p.tolist()


def solve(a, b):
    return _solve_tai256c(a, b)
# EVOLVE-BLOCK-END
