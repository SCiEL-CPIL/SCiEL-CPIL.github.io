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

def _score(residues, primes, max_check=100000):
    """Compute the largest L such that every integer 1..L satisfies at least two congruences."""
    counts = np.zeros(max_check + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return max_check
    return int(below_two[0])

def _compute_counts(residues, primes, max_check=100000):
    """Compute coverage counts for all positions 1..max_check."""
    counts = np.zeros(max_check + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
    return counts

def _find_first_gap(counts):
    """Find the first position where count < 2."""
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return -1
    return int(below_two[0])

def _targeted_gap_fix(residues, primes, max_check=100000, deadline=None, max_attempts=500):
    """
    Targeted strategy: find the first gap (position with <2 coverage),
    then try to fix it by changing residues of primes that can cover that position.
    """
    residues = residues.copy()
    counts = _compute_counts(residues, primes, max_check)
    current_L = _find_first_gap(counts)
    
    if deadline is not None:
        deadline = min(deadline, time.time() + 140)
    
    attempt = 0
    while current_L < max_check and attempt < max_attempts:
        if deadline is not None and time.time() > deadline:
            break
        
        gap_pos = current_L  # 1-indexed position with <2 coverage
        
        # Find which primes can cover this position
        # x ≡ r_p (mod p) covers gap_pos if r_p ≡ gap_pos (mod p)
        # So we need to change r_p to gap_pos % p for some prime p
        # But we want to find which single change gives the best improvement
        
        best_gain = 0
        best_change = None  # (index, new_r)
        
        for i, p in enumerate(primes):
            target_r = gap_pos % p
            if target_r == residues[i]:
                continue  # already covers this position
            
            # Simulate: remove old residue, add new residue
            old_r = residues[i]
            old_start = old_r if old_r > 0 else p
            new_start = target_r if target_r > 0 else p
            
            # Count how many positions in [1, current_L + some_window] improve
            # A position improves if it goes from count==1 to count==2
            # Focus on the critical region: [1, current_L + 200]
            window_end = min(max_check, current_L + 500)
            
            # Positions that would gain coverage: those hit by new congruence but not old
            # that currently have count == 1
            gain = 0
            if new_start <= window_end:
                new_positions = np.arange(new_start, window_end + 1, p)
                # Subtract positions also hit by old congruence
                if old_start <= window_end:
                    old_positions = np.arange(old_start, window_end + 1, p)
                    # Positions in new but not old
                    mask = np.ones(len(new_positions), dtype=bool)
                    # Check which new positions are also in old positions
                    for op in old_positions:
                        if op in new_positions:
                            mask[np.where(new_positions == op)[0]] = False
                    new_only = new_positions[mask]
                else:
                    new_only = new_positions
                
                gain = np.sum(counts[new_only] == 1)
            
            if gain > best_gain:
                best_gain = gain
                best_change = (i, target_r)
        
        if best_change is None:
            break
        
        i, new_r = best_change
        old_r = residues[i]
        old_start = old_r if old_r > 0 else p
        new_start = new_r if new_r > 0 else p
        
        # Apply change
        if old_start <= max_check:
            counts[old_start::p] -= 1
        if new_start <= max_check:
            counts[new_start::p] += 1
        residues[i] = new_r
        
        new_L = _find_first_gap(counts)
        if new_L > current_L:
            current_L = new_L
        else:
            # Revert
            if new_start <= max_check:
                counts[new_start::p] -= 1
            if old_start <= max_check:
                counts[old_start::p] += 1
            residues[i] = old_r
            break
        
        attempt += 1
    
    return residues, current_L

def _greedy_cover(primes, max_check=100000):
    """
    Greedy set-cover heuristic: process primes in increasing order, selecting
    for each prime the residue that maximizes the number of integers in [1, max_check]
    that are covered by exactly one congruence (i.e., converting single-covered to double-covered).
    """
    counts = np.zeros(max_check + 1, dtype=np.int32)
    residues = np.zeros(len(primes), dtype=np.int32)
    
    for idx, p in enumerate(primes):
        best_r = 0
        best_gain = -1
        
        # For small primes, try all residues; for large, sample
        if p <= 200:
            candidates = list(range(p))
        else:
            candidates = list(range(0, p, max(1, p // 50)))
            candidates.append(0)
        
        for r in candidates:
            # Compute gain: how many positions in [1, max_check] go from count==1 to count==2
            start = r if r > 0 else p
            if start > max_check:
                gain = 0
            else:
                # Positions hit by this congruence
                positions = np.arange(start, max_check + 1, p)
                # Gain = number of positions where count == 1 (will become 2)
                gain = np.sum(counts[positions] == 1)
                # Penalty = number of positions where count >= 2 (will become >=3, but that's fine)
                # Actually we only care about extending the prefix, so focus on count==1
        
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        residues[idx] = best_r
        # Update counts
        start = best_r if best_r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
    
    return residues

def _hill_climb(residues, primes, max_check=100000, deadline=None):
    """Local search: hill-climbing on individual residues with fast incremental scoring."""
    residues = residues.copy()
    counts = _compute_counts(residues, primes, max_check)
    best_score = _find_first_gap(counts)
    n_primes = len(primes)
    max_passes = 50
    
    for iteration in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        for i in range(n_primes):
            if deadline is not None and time.time() > deadline:
                break
            
            p = primes[i]
            current_r = residues[i]
            best_r = current_r
            best_local_score = best_score
            
            if p <= 100:
                candidates = list(range(p))
            else:
                candidates = [current_r]
                if p <= 500:
                    candidates.extend(np.random.randint(0, p, size=min(30, p)).tolist())
                else:
                    candidates.extend(np.random.randint(0, p, size=15).tolist())
            
            for r in candidates:
                if r == current_r:
                    continue
                
                old_r = residues[i]
                old_start = old_r if old_r > 0 else p
                new_start = r if r > 0 else p
                
                # Incremental update
                if old_start <= max_check:
                    counts[old_start::p] -= 1
                if new_start <= max_check:
                    counts[new_start::p] += 1
                residues[i] = r
                
                new_score = _find_first_gap(counts)
                
                # Revert
                if new_start <= max_check:
                    counts[new_start::p] -= 1
                if old_start <= max_check:
                    counts[old_start::p] += 1
                residues[i] = old_r
                
                if new_score > best_local_score:
                    best_local_score = new_score
                    best_r = r
            
            if best_r != current_r:
                old_start = current_r if current_r > 0 else p
                new_start = best_r if best_r > 0 else p
                if old_start <= max_check:
                    counts[old_start::p] -= 1
                if new_start <= max_check:
                    counts[new_start::p] += 1
                residues[i] = best_r
                best_score = best_local_score
                improved = True
        
        if not improved:
            break
    
    return residues, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    max_check = 100000
    
    best_residues = None
    best_score = 0
    
    # Phase 1: Greedy set-cover heuristic
    greedy_residues = _greedy_cover(primes, max_check)
    greedy_score = _score(greedy_residues, primes, max_check)
    
    # Phase 2: Targeted gap fixing
    gap_fixed_residues, gap_fixed_score = _targeted_gap_fix(greedy_residues, primes, max_check, deadline)
    
    # Phase 3: Hill-climbing refinement from both starting points
    refined_from_greedy, refined_score_1 = _hill_climb(greedy_residues, primes, max_check, deadline)
    refined_from_gap, refined_score_2 = _hill_climb(gap_fixed_residues, primes, max_check, deadline)
    
    # Track best
    candidates = [
        (greedy_residues, greedy_score),
        (gap_fixed_residues, gap_fixed_score),
        (refined_from_greedy, refined_score_1),
        (refined_from_gap, refined_score_2),
    ]
    
    # Also try from seed
    seed_residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int32)
    seed_refined, seed_score = _hill_climb(seed_residues, primes, max_check, deadline)
    candidates.append((seed_refined, seed_score))
    
    # Pick the best
    for res, score in candidates:
        if score > best_score:
            best_score = score
            best_residues = res
    
    if best_residues is None:
        best_residues = greedy_residues
    
    return [int(r) for r in best_residues.tolist()]
# EVOLVE-BLOCK-END
