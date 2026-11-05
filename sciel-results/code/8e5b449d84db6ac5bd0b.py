"""
Structural greedy + goal-oriented repair + simulated annealing hybrid.
Combines the backward refinement structure of the current program with the
SA machinery from the inspiration, plus a repair phase that targets the
current bottleneck (first gap) by modifying large primes first.
"""
import time
import numpy as np

def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _build_count_array(residues, primes, max_L):
    """Build coverage count array for positions 1..max_L."""
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for i in range(len(primes)):
        r = residues[i]
        p = primes[i]
        start = r if r > 0 else p
        if start <= max_L:
            counts[start::p] += 1
    return counts

def _find_L(counts, current_L, max_L):
    """Find the largest L such that all positions 1..L have count >= 2."""
    if current_L > 0:
        if np.min(counts[1:current_L + 1]) < 2:
            bad = np.where(counts[1:current_L + 1] < 2)[0]
            return int(bad[0])
        L = current_L
        if L < max_L:
            rest = counts[L + 1:]
            bad_idx = np.where(rest < 2)[0]
            if len(bad_idx) == 0:
                return max_L
            else:
                return L + int(bad_idx[0])
    else:
        bad_idx = np.where(counts[1:] < 2)[0]
        if len(bad_idx) == 0:
            return max_L
        else:
            return int(bad_idx[0])

def _score(residues, primes, max_L=100000):
    """Compute the largest L such that every integer 1..L is covered at least twice."""
    counts = _build_count_array(residues, primes, max_L)
    return _find_L(counts, 0, max_L)

def _greedy_construct(primes, max_L_check=5000):
    """
    Forward greedy: process primes in order, for each prime choose the residue
    that maximizes coverage of currently-deficient positions.
    """
    n = len(primes)
    residues = [0] * n
    counts = np.zeros(max_L_check + 1, dtype=np.int32)
    
    for idx in range(n):
        p = primes[idx]
        best_r = 0
        best_score = -1
        
        # For small primes, try all residues; for large primes, sample
        if p <= 200:
            candidates = range(p)
        else:
            candidates = np.random.default_rng(idx).choice(p, size=min(p, 60), replace=False)
        
        for r in candidates:
            start = r if r > 0 else p
            if start > max_L_check:
                continue
            positions = np.arange(start, max_L_check + 1, p)
            deficient = np.sum(counts[positions] < 2)
            already_covered = np.sum(counts[positions] >= 2)
            score = deficient - 0.5 * already_covered
            if score > best_score:
                best_score = score
                best_r = int(r)
        
        residues[idx] = best_r
        start = best_r if best_r > 0 else p
        if start <= max_L_check:
            counts[start::p] += 1
    
    return residues

def _repair_phase(residues, primes, counts, current_L, max_L, deadline):
    """
    Goal-oriented repair: identify the first gap (position where count < 2)
    and try to fix it by modifying large primes first (to minimize collateral).
    Returns updated residues, counts, current_L.
    """
    n = len(primes)
    improved = True
    repair_rounds = 0
    
    while improved and time.time() < deadline - 5:
        improved = False
        repair_rounds += 1
        
        # Find the first deficient position
        if current_L >= max_L:
            break
        
        # Find first position after current_L with count < 2
        if current_L + 1 <= max_L:
            remaining = counts[current_L + 1:]
            bad_idx = np.where(remaining < 2)[0]
            if len(bad_idx) == 0:
                current_L = max_L
                break
            target_pos = current_L + 1 + int(bad_idx[0])
        else:
            break
        
        # Try to fix target_pos by changing large primes (they have sparse coverage)
        # Process primes in descending order of p (large primes first)
        prime_order = sorted(range(n), key=lambda i: primes[i], reverse=True)
        
        fixed = False
        for idx in prime_order:
            if time.time() > deadline - 5:
                break
            if fixed:
                break
                
            p = primes[idx]
            old_r = residues[idx]
            
            # What residue would cover target_pos?
            needed_r = target_pos % p
            
            if needed_r == old_r:
                continue  # Already covers it, problem is elsewhere
            
            # Try changing to needed_r
            # Remove old coverage
            old_start = old_r if old_r > 0 else p
            if old_start <= max_L:
                counts[old_start::p] -= 1
            
            # Add new coverage
            new_start = needed_r if needed_r > 0 else p
            if new_start <= max_L:
                counts[new_start::p] += 1
            
            # Check if this helps (doesn't break the prefix)
            new_L = _find_L(counts, min(current_L, target_pos - 1), max_L)
            
            if new_L > current_L:
                # Accept
                residues[idx] = needed_r
                current_L = new_L
                improved = True
                fixed = True
            else:
                # Revert
                counts[new_start::p] -= 1
                counts[old_start::p] += 1
                residues[idx] = old_r
        
        # Also try small primes for the target
        if not fixed:
            small_order = sorted(range(n), key=lambda i: primes[i])
            for idx in small_order[:30]:  # Only first 30 small primes
                if time.time() > deadline - 5:
                    break
                if fixed:
                    break
                    
                p = primes[idx]
                old_r = residues[idx]
                needed_r = target_pos % p
                
                if needed_r == old_r:
                    continue
                
                old_start = old_r if old_r > 0 else p
                if old_start <= max_L:
                    counts[old_start::p] -= 1
                new_start = needed_r if needed_r > 0 else p
                if new_start <= max_L:
                    counts[new_start::p] += 1
                
                new_L = _find_L(counts, min(current_L, target_pos - 1), max_L)
                
                if new_L > current_L:
                    residues[idx] = needed_r
                    current_L = new_L
                    improved = True
                    fixed = True
                else:
                    counts[new_start::p] -= 1
                    counts[old_start::p] += 1
                    residues[idx] = old_r
    
    return residues, counts, current_L

def _simulated_annealing(residues, primes, counts, current_L, max_L, deadline):
    """
    Simulated annealing to escape local optima.
    """
    n = len(primes)
    best_L = current_L
    best_residues = residues.copy()
    
    T = 5.0
    cooling = 0.99995
    min_T = 0.05
    rng = np.random.default_rng(12345)
    
    while time.time() < deadline:
        # Pick random prime
        i = rng.integers(n)
        p = primes[i]
        old_r = residues[i]
        new_r = int(rng.integers(p))
        
        if new_r == old_r:
            continue
        
        # Apply change
        old_start = old_r if old_r > 0 else p
        if old_start <= max_L:
            counts[old_start::p] -= 1
        
        new_start = new_r if new_r > 0 else p
        if new_start <= max_L:
            counts[new_start::p] += 1
        
        # Calculate new L
        new_L = _find_L(counts, current_L, max_L)
        
        # Acceptance
        if new_L >= current_L:
            residues[i] = new_r
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
        else:
            delta = new_L - current_L
            if delta > -200:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    residues[i] = new_r
                    current_L = new_L
                else:
                    # Revert
                    residues[i] = old_r
                    counts[new_start::p] -= 1
                    counts[old_start::p] += 1
            else:
                # Revert
                residues[i] = old_r
                counts[new_start::p] -= 1
                counts[old_start::p] += 1
        
        T *= cooling
        if T < min_T:
            T = min_T
    
    return best_residues, best_L

def _backward_refine(residues, primes, max_L_check, deadline):
    """
    Backward refinement: for each prime, try to find a better residue
    by looking at deficiencies caused by other primes.
    """
    n = len(primes)
    improved = False
    
    for idx in range(n - 1, -1, -1):
        if time.time() > deadline:
            break
            
        p = primes[idx]
        old_r = residues[idx]
        
        # Build counts for all OTHER primes
        base_counts = np.zeros(max_L_check + 1, dtype=np.int32)
        for j in range(n):
            if j == idx:
                continue
            r = residues[j]
            pj = primes[j]
            start = r if r > 0 else pj
            if start <= max_L_check:
                base_counts[start::pj] += 1
        
        # Find first few deficient positions
        deficient_pos = []
        for pos in range(1, min(max_L_check + 1, 100)):
            if base_counts[pos] < 2:
                deficient_pos.append(pos)
                if len(deficient_pos) >= 20:
                    break
        
        if not deficient_pos:
            continue
        
        # Candidate residues
        candidates = set()
        for pos in deficient_pos:
            candidates.add(pos % p)
        candidates.add(old_r)
        
        # Limit candidates
        if len(candidates) > 40:
            cand_scores = {}
            for c in candidates:
                score = sum(1 for pos in deficient_pos if pos % p == c)
                cand_scores[c] = score
            sorted_cands = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
            candidates = {c for c, _ in sorted_cands[:40]}
        
        best_r = old_r
        best_L = -1
        
        # Evaluate old residue
        r_old_eff = p if old_r == 0 else old_r
        cur_counts = base_counts.copy()
        if r_old_eff <= max_L_check:
            cur_counts[r_old_eff::p] += 1
        best_L = _find_L(cur_counts, 0, max_L_check)
        
        # Evaluate candidates
        for new_r in candidates:
            if time.time() > deadline:
                break
            if new_r == old_r:
                continue
                
            r_new_eff = p if new_r == 0 else new_r
            new_counts = base_counts.copy()
            if r_new_eff <= max_L_check:
                new_counts[r_new_eff::p] += 1
            new_L = _find_L(new_counts, 0, max_L_check)
            
            if new_L > best_L:
                best_L = new_L
                best_r = new_r
        
        if best_r != old_r:
            residues[idx] = best_r
            improved = True
    
    return residues, improved

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    primes = _get_primes(1000)
    n = len(primes)
    MAX_L = 100000
    
    # Phase 1: Greedy construction
    residues = _greedy_construct(primes, max_L_check=5000)
    
    # Build full count array
    counts = _build_count_array(residues, primes, MAX_L)
    current_L = _find_L(counts, 0, MAX_L)
    
    # Phase 2: Backward refinement at multiple scales
    for max_L_stage in [2000, 5000, 10000, 20000]:
        if time.time() > deadline - 20:
            break
        for _ in range(2):
            if time.time() > deadline - 15:
                break
            residues, improved = _backward_refine(residues, primes, max_L_stage, deadline - 10)
            if not improved:
                break
    
    # Rebuild counts after refinement
    counts = _build_count_array(residues, primes, MAX_L)
    current_L = _find_L(counts, 0, MAX_L)
    
    # Phase 3: Goal-oriented repair
    if time.time() < deadline - 15:
        residues, counts, current_L = _repair_phase(residues, primes, counts, current_L, MAX_L, deadline - 10)
    
    # Phase 4: Simulated annealing
    if time.time() < deadline - 5:
        residues, current_L = _simulated_annealing(residues, primes, counts, current_L, MAX_L, deadline)
    
    # Final evaluation
    final_score = _score(residues, primes, MAX_L)
    
    return residues
