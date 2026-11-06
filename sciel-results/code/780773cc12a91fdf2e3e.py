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
    
    # Track current doubly-covered prefix length
    current_L = 0
    
    for idx, p in enumerate(primes):
        if deadline is not None and time.time() > deadline:
            break
        
        best_r = 0
        best_gain = -1
        
        # For efficiency, we focus on positions up to current_L + p (since extending
        # the prefix beyond current_L requires covering positions just past current_L)
        # But we need to consider all positions up to max_check for the gain calculation
        
        # Try candidate residues
        # For small primes, try all; for large primes, sample
        if p <= 60:
            candidates = range(p)
        else:
            # Try a set of candidates: 0, and positions near the current gap
            candidates = [0]
            # Add some strategic candidates
            for offset in range(1, min(p, 20)):
                candidates.append(offset)
            # Add random candidates
            if p > 100:
                rand_candidates = np.random.randint(0, p, size=min(30, p))
                candidates.extend(rand_candidates.tolist())
        
        for r in candidates:
            if r == residues[idx]:
                continue
            
            # Compute gain: how many positions in [1, current_L] go from count=1 to count=2
            # and how many positions just past current_L get covered
            start = r if r > 0 else p
            
            if start > max_check:
                gain = 0
            else:
                # Positions covered by this residue in range [1, max_check]
                positions = np.arange(start, max_check + 1, p)
                
                # Gain from converting single-covered to double-covered in [1, current_L]
                if current_L > 0:
                    in_prefix = positions[positions <= current_L]
                    if len(in_prefix) > 0:
                        gain_prefix = np.sum(counts[in_prefix] == 1)
                    else:
                        gain_prefix = 0
                else:
                    gain_prefix = 0
                
                # Gain from covering positions just past current_L (to extend prefix)
                # A position x > current_L helps extend the prefix if all positions
                # current_L+1..x are covered at least twice after adding this
                # Simplified: count positions in (current_L, current_L + p] that get covered
                past_prefix = positions[(positions > current_L) & (positions <= current_L + p)]
                if len(past_prefix) > 0:
                    # These positions need to be covered by at least 2 total
                    # Currently they have some count; adding this residue adds 1
                    # So they become covered if their current count >= 1
                    gain_past = np.sum(counts[past_prefix] >= 1)
                else:
                    gain_past = 0
                
                # Weight: extending the prefix is more valuable
                gain = gain_prefix + 2 * gain_past
            
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        # Apply the best residue
        residues[idx] = best_r
        start = best_r if best_r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
        
        # Update current_L
        # Find the first position > current_L where count < 2
        # Check positions from current_L+1 onward
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
        
        # Process primes in random order for diversity
        order = np.random.permutation(n_primes)
        
        for i in order:
            if deadline is not None and time.time() > deadline:
                break
            
            p = primes[i]
            current_r = residues[i]
            best_r = current_r
            best_local_score = best_score
            
            # Remove current residue's contribution from counts
            old_start = current_r if current_r > 0 else p
            if old_start <= max_check:
                counts[old_start::p] -= 1
            
            # Try candidate residues
            if p <= 80:
                candidates = list(range(p))
            else:
                candidates = [current_r]
                # Strategic candidates
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
            
            # Restore: remove current, add best
            # We already removed current above
            if best_r != current_r:
                new_start = best_r if best_r > 0 else p
                if new_start <= max_check:
                    counts[new_start::p] += 1
                residues[i] = best_r
                best_score = best_local_score
                best_residues = residues.copy()
                improved = True
            else:
                # Restore current residue
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
        
        # Find the first gap position
        gap_pos = current_score + 1
        
        # Find which primes could cover gap_pos (i.e., gap_pos mod p == r_p for some r_p)
        # We can adjust any prime's residue to cover gap_pos
        # For prime p, the residue that covers gap_pos is gap_pos % p
        
        # Try adjusting each prime to cover gap_pos
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
            
            # Try changing this prime's residue to target_r
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
                # Revert
                if new_start <= max_check:
                    counts[new_start::p] -= 1
                if old_start <= max_check:
                    counts[old_start::p] += 1
            
            if improved_this_round:
                # Recompute counts and score after improvement
                counts = _compute_counts(residues, primes, max_check)
                break
        
        if not improved_this_round:
            break
    
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
    
    # Phase 5: Final hill-climbing on the best result
    if time.time() < deadline - 5:
        residues_final, score_final = _hill_climb(best_residues, primes, max_check=100000,
                                                   deadline=deadline - 2, max_passes=10)
        if score_final > best_score:
            best_residues = residues_final
            best_score = score_final
    
    # Final verification
    final_score = _score(best_residues, primes)
    if final_score > best_score:
        best_score = final_score
    
    return best_residues.tolist()
# EVOLVE-BLOCK-END
