"""
Strategy: Constructive Algebraic Initialization with Multi-Start Local Search.

The current program uses random initialization for Iterated Local Search (ILS). 
This approach posits that the search landscape for minimizing max discrepancy is rugged, 
and random starts are far from the global optimum. 
By using the Legendre symbol sequence (a quadratic residue character modulo a prime) 
as the initial state, we leverage number-theoretic properties that provide low discrepancy 
on arithmetic progressions (specifically related to the Pólya-Vinogradov theorem). 
This provides a much stronger starting point, allowing the subsequent local search 
to refine the solution rather than explore the entire space from scratch.

We implement a multi-start strategy where:
1. We generate the Legendre symbol sequence for a prime near n (or handle non-primes by extension).
2. We perform a local search (greedy hill-climbing) starting from this algebraic sequence.
3. We also perform local search from a few other structured starts (e.g., shifted Legendre, 
   or a few high-quality random starts) to ensure robustness.
4. We keep the best result found across all starts within the time budget.

The local search itself remains similar to the current program (direct max-discrepancy minimization 
with incremental updates) but benefits significantly from the better initialization.
"""

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _build_ap_structure(n):
    """
    Pre-compute all arithmetic progressions and map indices to the progressions they belong to.
    """
    ap_indices = []
    index_to_aps = [[] for _ in range(n)]
    
    ap_id = 0
    for step in range(1, n + 1):
        for start in range(n):
            indices = np.arange(start, n, step, dtype=np.int32)
            if len(indices) == 0:
                continue
            
            ap_indices.append(indices)
            
            for pos, idx in enumerate(indices):
                index_to_aps[idx].append((ap_id, pos))
            
            ap_id += 1
            
    return ap_indices, index_to_aps

def _init_prefix_sums(signs, ap_indices):
    """
    Compute initial prefix sums for all APs.
    """
    prefix_sums = []
    signs_arr = np.array(signs, dtype=np.int64)
    for indices in ap_indices:
        subseq = signs_arr[indices]
        ps = np.cumsum(subseq)
        prefix_sums.append(ps)
    return prefix_sums

def _legendre_symbol(n, p):
    """
    Compute the Legendre symbol (n/p) using Euler's criterion.
    Returns 0 if n % p == 0, 1 if n is a quadratic residue mod p, -1 otherwise.
    """
    if n % p == 0:
        return 0
    return pow(n, (p - 1) // 2, p)

def _generate_legendre_sequence(n):
    """
    Generate a sequence of signs based on the Legendre symbol for a prime near n.
    We find the smallest prime >= n.
    """
    def is_prime(k):
        if k < 2:
            return False
        for i in range(2, int(np.sqrt(k)) + 1):
            if k % i == 0:
                return False
        return True

    # Find prime p >= n
    p = n
    while not is_prime(p):
        p += 1

    seq = []
    for i in range(1, n + 1):
        ls = _legendre_symbol(i, p)
        if ls == 0:
            # Should not happen for i < p, but just in case
            seq.append(1)
        else:
            seq.append(ls)
    return seq

def _local_search(signs, ap_indices, index_to_aps, deadline, rng):
    """
    Perform greedy local search starting from the given signs.
    Returns the best signs and the global max discrepancy found.
    """
    n = len(signs)
    signs = list(signs)
    prefix_sums = _init_prefix_sums(signs, ap_indices)
    ap_max_disc = np.array([np.max(np.abs(ps)) for ps in prefix_sums], dtype=np.int64)
    current_global_max = np.max(ap_max_disc)
    
    best_signs = list(signs)
    best_global_max = current_global_max
    
    max_ls_iters = 10000
    ls_iter = 0
    
    while ls_iter < max_ls_iters and time.time() < deadline - 2:
        improved = False
        indices_to_try = list(range(n))
        rng.shuffle(indices_to_try)
        
        for i in indices_to_try:
            if time.time() > deadline - 2:
                break
                
            old_val = signs[i]
            delta = -2 * old_val
            affected_aps = index_to_aps[i]
            
            # Evaluate the flip
            max_new_aff = 0
            for ap_id, pos in affected_aps:
                ps = prefix_sums[ap_id]
                if pos > 0:
                    max_pref = np.max(np.abs(ps[:pos]))
                else:
                    max_pref = 0
                max_suff = np.max(np.abs(ps[pos:] + delta))
                new_ap_max = max(max_pref, max_suff)
                if new_ap_max > max_new_aff:
                    max_new_aff = new_ap_max
                    
            # If the new max for affected APs is worse, skip
            if max_new_aff > current_global_max:
                continue
                
            # Apply flip tentatively
            signs[i] = -old_val
            for ap_id, pos in affected_aps:
                prefix_sums[ap_id][pos:] += delta
                new_m = np.max(np.abs(prefix_sums[ap_id]))
                ap_max_disc[ap_id] = new_m
                
            new_global_max = np.max(ap_max_disc)
            
            if new_global_max < current_global_max:
                current_global_max = new_global_max
                improved = True
                if current_global_max < best_global_max:
                    best_global_max = current_global_max
                    best_signs = list(signs)
            else:
                # Revert
                signs[i] = old_val
                for ap_id, pos in affected_aps:
                    prefix_sums[ap_id][pos:] -= delta
                    ap_max_disc[ap_id] = np.max(np.abs(prefix_sums[ap_id]))
                    
        ls_iter += 1
        if not improved:
            break
            
    return best_signs, best_global_max

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Pre-compute structure
    ap_indices, index_to_aps = _build_ap_structure(n)
    
    rng = random.Random(42)
    
    best_signs_overall = None
    best_global_max_overall = float('inf')
    
    # Define starting configurations
    starts = []
    
    # 1. Legendre Symbol Sequence
    starts.append(_generate_legendre_sequence(n))
    
    # 2. Negated Legendre Symbol Sequence
    starts.append([-x for x in starts[0]])
    
    # 3. Shifted Legendre (shift by 1)
    leg_seq = _generate_legendre_sequence(n)
    starts.append([leg_seq[(i+1) % n] for i in range(n)])
    
    # 4. A few high-quality random starts (pre-seeded)
    for _ in range(3):
        starts.append([rng.choice((-1, 1)) for _ in range(n)])
    
    # 5. Alternating sequence (for step 1, this is optimal, but for others it's a baseline)
    starts.append([1 if i % 2 == 0 else -1 for i in range(n)])
    
    # Run local search from each start
    for start_signs in starts:
        if time.time() > deadline - 10:
            break
            
        best_signs, best_max = _local_search(start_signs, ap_indices, index_to_aps, deadline, rng)
        
        if best_max < best_global_max_overall:
            best_global_max_overall = best_max
            best_signs_overall = best_signs
            
        # If we hit the theoretical lower bound (which is 1, since singletons must have sum +/-1),
        # we can stop early if we are confident. But 1 is very hard to achieve for all APs.
        # Actually, for any sign, the AP of length 1 has sum +/-1. So max_disc >= 1.
        # If we reach 1, it's optimal.
        if best_global_max_overall == 1:
            break

    if best_signs_overall is None:
        best_signs_overall = [1] * n
        
    return best_signs_overall
# EVOLVE-BLOCK-END
