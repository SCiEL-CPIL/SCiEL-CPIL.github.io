"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _sieve_primes(limit):
    """Return list of primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _compute_counts(residues, primes, max_check):
    """Compute coverage counts for all positions 1..max_check."""
    counts = np.zeros(max_check + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
    return counts

def _score_from_counts(counts, max_check=100000):
    """Compute the largest L such that every integer 1..L satisfies at least two congruences."""
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return max_check
    return int(below_two[0])

def _score(residues, primes, max_check=100000):
    """Compute the largest L such that every integer 1..L satisfies at least two congruences."""
    counts = _compute_counts(residues, primes, max_check)
    return _score_from_counts(counts, max_check)

def _greedy_cover(primes, max_check=100000, deadline=None):
    """
    Greedy set-cover heuristic: process primes in increasing order, selecting for each
    prime the residue that maximizes the number of positions in [1, current_L] that are
    covered by exactly one congruence (converting them to double-covered).
    """
    n_primes = len(primes)
    residues = np.zeros(n_primes, dtype=np.int32)
    counts = np.zeros(max_check + 1, dtype=np.int32)
    
    current_L = 0
    
    for idx, p in enumerate(primes):
        if deadline is not None and time.time() > deadline:
            break
        
        best_r = 0
        best_gain = -1
        
        if p <= 80:
            candidates = range(p)
        else:
            candidates = [0]
            for offset in range(1, min(p, 25)):
                candidates.append(offset)
            if p > 100:
                rand_candidates = np.random.randint(0, p, size=min(40, p))
                candidates.extend(rand_candidates.tolist())
        
        for r in candidates:
            if r == residues[idx]:
                continue
            
            start = r if r > 0 else p
            
            if start > max_check:
                gain = 0
            else:
                positions = np.arange(start, max_check + 1, p)
                
                if current_L > 0:
                    in_prefix = positions[positions <= current_L]
                    if len(in_prefix) > 0:
                        gain_prefix = np.sum(counts[in_prefix] == 1)
                    else:
                        gain_prefix = 0
                else:
                    gain_prefix = 0
                
                past_prefix = positions[(positions > current_L) & (positions <= current_L + p)]
                if len(past_prefix) > 0:
                    gain_past = np.sum(counts[past_prefix] >= 1)
                else:
                    gain_past = 0
                
                gain = gain_prefix + 2 * gain_past
            
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        residues[idx] = best_r
        start = best_r if best_r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
        
        check_start = current_L + 1
        if check_start <= max_check:
            below_two = np.where(counts[check_start:] < 2)[0]
            if len(below_two) == 0:
                current_L = max_check
            else:
                current_L = check_start + int(below_two[0]) - 1
    
    return residues, current_L

def _hill_climb(residues, primes, max_check=100000, deadline=None, max_passes=20):
    """
    Hill-climbing refinement: for each prime, try changing its residue to improve the score.
    Uses incremental count updates for speed.
    """
    n_primes = len(primes)
    best_score = _score(residues, primes, max_check)
    best_residues = residues.copy()
    
    for iteration in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        counts = _compute_counts(residues, primes, max_check)
        
        order = np.random.permutation(n_primes)
        
        for i in order:
            if deadline is not None and time.time() > deadline:
                break
            
            p = primes[i]
            current_r = residues[i]
            best_r = current_r
            best_local_score = best_score
            
            old_start = current_r if current_r > 0 else p
            if old_start <= max_check:
                counts[old_start::p] -= 1
            
            if p <= 80:
                candidates = list(range(p))
            else:
                candidates = [current_r]
                for offset in range(1, min(p, 15)):
                    candidates.append(offset)
                if p > 100:
                    rand_candidates = np.random.randint(0, p, size=min(25, p))
                    candidates.extend(rand_candidates.tolist())
            
            for r in candidates:
                if r == current_r:
                    continue
                
                new_start = r if r > 0 else p
                if new_start <= max_check:
                    counts[new_start::p] += 1
                
                new_score = _score_from_counts(counts, max_check)
                
                if new_start <= max_check:
                    counts[new_start::p] -= 1
                
                if new_score > best_local_score:
                    best_local_score = new_score
                    best_r = r
            
            if best_r != current_r:
                new_start = best_r if best_r > 0 else p
                if new_start <= max_check:
                    counts[new_start::p] += 1
                residues[i] = best_r
                best_score = best_local_score
                best_residues = residues.copy()
                improved = True
            else:
                if old_start <= max_check:
                    counts[old_start::p] += 1
        
        if not improved:
            break
    
    return best_residues, best_score

def _targeted_gap_fix(residues, primes, max_check=100000, deadline=None, max_rounds=10):
    """
    Targeted gap fixing: find the first position not covered twice, then try to
    adjust residues of primes that could cover that position to fix the gap.
    """
    n_primes = len(primes)
    best_score = _score(residues, primes, max_check)
    best_residues = residues.copy()
    
    for round_num in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        
        counts = _compute_counts(residues, primes, max_check)
        current_score = _score_from_counts(counts, max_check)
        
        if current_score >= max_check:
            break
        
        gap_pos = current_score + 1
        
        improved_this_round = False
        order = np.random.permutation(n_primes)
        
        for i in order:
            if deadline is not None and time.time() > deadline:
                break
            
            p = primes[i]
            current_r = residues[i]
            target_r = gap_pos % p
            
            if target_r == current_r:
                continue
            
            old_start = current_r if current_r > 0 else p
            if old_start <= max_check:
                counts[old_start::p] -= 1
            
            new_start = target_r if target_r > 0 else p
            if new_start <= max_check:
                counts[new_start::p] += 1
            
            new_score = _score_from_counts(counts, max_check)
            
            if new_score > best_score:
                residues[i] = target_r
                best_score = new_score
                best_residues = residues.copy()
                improved_this_round = True
            else:
                if new_start <= max_check:
                    counts[new_start::p] -= 1
                if old_start <= max_check:
                    counts[old_start::p] += 1
            
            if improved_this_round:
                counts = _compute_counts(residues, primes, max_check)
                break
        
        if not improved_this_round:
            break
    
    return best_residues, best_score

def _pair_swap(residues, primes, max_check=100000, deadline=None, max_rounds=5):
    """
    Pair-swap: try swapping residues between pairs of primes to escape local optima.
    """
    n_primes = len(primes)
    best_score = _score(residues, primes, max_check)
    best_residues = residues.copy()
    
    for round_num in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        counts = _compute_counts(residues, primes, max_check)
        
        # Try a random set of pairs
        num_swaps = min(50, n_primes * (n_primes - 1) // 2)
        pairs = np.random.choice(n_primes, size=(num_swaps, 2), replace=False)
        
        for i, j in pairs:
            if deadline is not None and time.time() > deadline:
                break
            
            if i == j:
                continue
            
            p_i = primes[i]
            p_j = primes[j]
            r_i = residues[i]
            r_j = residues[j]
            
            # Remove both
            start_i = r_i if r_i > 0 else p_i
            start_j = r_j if r_j > 0 else p_j
            if start_i <= max_check:
                counts[start_i::p_i] -= 1
            if start_j <= max_check:
                counts[start_j::p_j] -= 1
            
            # Swap
            new_start_i = r_j if r_j > 0 else p_i
            new_start_j = r_i if r_i > 0 else p_j
            if new_start_i <= max_check:
                counts[new_start_i::p_i] += 1
            if new_start_j <= max_check:
                counts[new_start_j::p_j] += 1
            
            new_score = _score_from_counts(counts, max_check)
            
            if new_score > best_score:
                residues[i], residues[j] = r_j, r_i
                best_score = new_score
                best_residues = residues.copy()
                improved = True
            else:
                # Revert
                if new_start_i <= max_check:
                    counts[new_start_i::p_i] -= 1
                if new_start_j <= max_check:
                    counts[new_start_j::p_j] -= 1
                if start_i <= max_check:
                    counts[start_i::p_i] += 1
                if start_j <= max_check:
                    counts[start_j::p_j] += 1
        
        if not improved:
            break
    
    return best_residues, best_score

def _random_restart_local(residues, primes, max_check=100000, deadline=None, n_restarts=3):
    """
    Random restarts with local search: perturb a random subset of residues,
    then run hill climbing. Keep the best result.
    """
    n_primes = len(primes)
    best_residues = residues.copy()
    best_score = _score(residues, primes, max_check)
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb: change 10-20% of residues randomly
        perturbed = best_residues.copy()
        n_perturb = max(5, n_primes // 10)
        indices = np.random.choice(n_primes, size=n_perturb, replace=False)
        for idx in indices:
            p = primes[idx]
            perturbed[idx] = np.random.randint(0, p)
        
        # Run a short hill climb on the perturbed version
        hc_residues, hc_score = _hill_climb(perturbed, primes, max_check, 
                                             deadline=deadline, max_passes=3)
        
        if hc_score > best_score:
            best_residues = hc_residues
            best_score = hc_score
    
    return best_residues, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    
    # Phase 1: Greedy set-cover initialization
    residues_greedy, L_greedy = _greedy_cover(primes, max_check=100000, deadline=deadline)
    score_greedy = _score(residues_greedy, primes)
    
    # Phase 2: Also try the seed initialization
    residues_seed = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int32)
    score_seed = _score(residues_seed, primes)
    
    # Choose the better starting point
    if score_greedy >= score_seed:
        residues = residues_greedy.copy()
        best_score = score_greedy
    else:
        residues = residues_seed.copy()
        best_score = score_seed
    
    best_residues = residues.copy()
    
    # Phase 3: Hill-climbing refinement
    if time.time() < deadline:
        residues_hc, score_hc = _hill_climb(residues, primes, max_check=100000, 
                                             deadline=deadline, max_passes=15)
        if score_hc > best_score:
            best_residues = residues_hc
            best_score = score_hc
    
    # Phase 4: Targeted gap fixing
    if time.time() < deadline:
        residues_tf, score_tf = _targeted_gap_fix(best_residues, primes, max_check=100000,
                                                   deadline=deadline, max_rounds=8)
        if score_tf > best_score:
            best_residues = residues_tf
            best_score = score_tf
    
    # Phase 5: Pair swaps to escape local optima
    if time.time() < deadline - 10:
        residues_swap, score_swap = _pair_swap(best_residues, primes, max_check=100000,
                                                deadline=deadline - 5, max_rounds=5)
        if score_swap > best_score:
            best_residues = residues_swap
            best_score = score_swap
    
    # Phase 6: Random restarts with local search
    if time.time() < deadline - 10:
        residues_rr, score_rr = _random_restart_local(best_residues, primes, max_check=100000,
                                                       deadline=deadline - 5, n_restarts=3)
        if score_rr > best_score:
            best_residues = residues_rr
            best_score = score_rr
    
    # Phase 7: Final hill-climbing on the best result
    if time.time() < deadline - 5:
        residues_final, score_final = _hill_climb(best_residues, primes, max_check=100000,
                                                   deadline=deadline - 2, max_passes=10)
        if score_final > best_score:
            best_residues = residues_final
            best_score = score_final
    
    # Phase 8: Final targeted gap fix
    if time.time() < deadline - 2:
        residues_final2, score_final2 = _targeted_gap_fix(best_residues, primes, max_check=100000,
                                                           deadline=deadline - 1, max_rounds=5)
        if score_final2 > best_score:
            best_residues = residues_final2
            best_score = score_final2
    
    # Final verification
    final_score = _score(best_residues, primes)
    if final_score > best_score:
        best_score = final_score
    
    return best_residues.tolist()
# EVOLVE-BLOCK-END
