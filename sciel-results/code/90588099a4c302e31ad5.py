"""Combined structural greedy + optimized refinement for covering problem."""
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
        if r == 0:
            r = p
        if r <= max_L:
            positions = np.arange(r, max_L + 1, p)
            counts[positions] += 1
    return counts

def _find_L(counts, max_L):
    """Find largest L such that all positions 1..L have count >= 2."""
    deficient = np.where(counts[1:max_L+1] < 2)[0]
    if len(deficient) == 0:
        return max_L
    return deficient[0]

def _score(residues, primes, max_L=100000):
    """Compute the largest L such that every integer 1..L is covered at least twice."""
    counts = _build_count_array(residues, primes, max_L)
    return _find_L(counts, max_L)

def _greedy_construct(primes, max_L_check=2000):
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
        best_gain = -1
        
        # Find the first few deficient positions
        deficient_positions = []
        for pos in range(1, min(max_L_check + 1, 500)):
            if counts[pos] < 2:
                deficient_positions.append(pos)
                if len(deficient_positions) >= 150:
                    break
        
        if not deficient_positions:
            # No deficient positions in early range, pick residue to maximize overall coverage
            best_r = 1
            best_count = 0
            for r in range(min(p, 80)):
                if r == 0:
                    r_eff = p
                else:
                    r_eff = r
                if r_eff > max_L_check:
                    continue
                num_positions = (max_L_check - r_eff) // p + 1
                if num_positions > best_count:
                    best_count = num_positions
                    best_r = r
        else:
            # For each candidate residue, count how many deficient positions it covers
            residue_weighted = np.zeros(p, dtype=np.int32)
            for pos in deficient_positions:
                r_needed = pos % p
                weight = 3 if counts[pos] == 0 else 1
                residue_weighted[r_needed] += weight
            
            best_r = int(np.argmax(residue_weighted))
            best_gain = residue_weighted[best_r]
        
        residues[idx] = best_r
        
        # Update counts
        r = best_r
        if r == 0:
            r_eff = p
        else:
            r_eff = r
        if r_eff <= max_L_check:
            positions = np.arange(r_eff, max_L_check + 1, p)
            counts[positions] += 1
    
    return residues

def _compute_L_from_base(base_counts, new_r, p, max_L_check):
    """Compute L given base_counts (without prime p) and adding prime p with residue new_r."""
    # A position x is covered if base_counts[x] + (1 if x%p==new_r else 0) >= 2
    # First failure: base_counts[x] == 0 (can't be fixed) or base_counts[x] == 1 and x%p != new_r
    
    # Find first position where base_counts == 0
    zero_mask = base_counts[1:max_L_check+1] == 0
    if np.any(zero_mask):
        first_zero = np.argmax(zero_mask) + 1
    else:
        first_zero = max_L_check + 1
    
    # For positions where base_counts == 1, they're OK if x%p == new_r
    # Find first position x <= first_zero-1 where base_counts[x] == 1 and x%p != new_r
    one_mask = base_counts[1:max_L_check+1] == 1
    one_positions = np.where(one_mask)[0] + 1  # positions 1..max_L_check
    
    new_L = first_zero - 1
    for x in one_positions:
        if x >= first_zero:
            break
        if x % p != new_r:
            new_L = x - 1
            break
    
    return max(0, new_L)

def _refine_pass(residues, primes, max_L_check, deadline):
    """
    Optimized backward refinement with incremental base_counts update.
    """
    n = len(primes)
    improved = False
    
    # Build full counts first
    full_counts = _build_count_array(residues, primes, max_L_check)
    
    # Process in reverse order
    for idx in range(n - 1, -1, -1):
        if time.time() > deadline:
            break
            
        p = primes[idx]
        old_r = residues[idx]
        
        # Compute base_counts = full_counts - contribution of prime idx
        base_counts = full_counts.copy()
        r_old_eff = p if old_r == 0 else old_r
        if r_old_eff <= max_L_check:
            positions = np.arange(r_old_eff, max_L_check + 1, p)
            base_counts[positions] -= 1
        
        # Find current L with old residue
        current_L = _compute_L_from_base(base_counts, old_r, p, max_L_check)
        
        # Generate candidates: all residues that cover at least one position with base_counts == 1
        one_positions = np.where(base_counts[1:max_L_check+1] == 1)[0] + 1
        
        if len(one_positions) == 0:
            # No positions that need this prime's help, keep old
            continue
        
        # Candidate residues are those that cover at least one position with base_counts == 1
        candidates = set()
        for pos in one_positions:
            candidates.add(pos % p)
        
        # Also include the old residue
        candidates.add(old_r)
        
        # Limit candidates to 120 max for speed, prioritizing those covering more one_positions
        if len(candidates) > 120:
            cand_scores = {}
            for c in candidates:
                score = 0
                for pos in one_positions:
                    if pos % p == c:
                        score += 1
                cand_scores[c] = score
            sorted_cands = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
            candidates = {c for c, _ in sorted_cands[:120]}
        
        best_r = old_r
        best_L = current_L
        
        # Evaluate candidates
        for new_r in candidates:
            if time.time() > deadline:
                break
            
            new_L = _compute_L_from_base(base_counts, new_r, p, max_L_check)
            
            if new_L > best_L:
                best_L = new_L
                best_r = new_r
        
        if best_r != old_r:
            residues[idx] = best_r
            improved = True
            
            # Update full_counts incrementally
            r_new_eff = p if best_r == 0 else best_r
            if r_old_eff <= max_L_check:
                positions_old = np.arange(r_old_eff, max_L_check + 1, p)
                full_counts[positions_old] -= 1
            if r_new_eff <= max_L_check:
                positions_new = np.arange(r_new_eff, max_L_check + 1, p)
                full_counts[positions_new] += 1
    
    return residues, improved

def _local_search(residues, primes, max_L_check, deadline):
    """
    Local search: try changing pairs of residues to escape local optima.
    """
    n = len(primes)
    best_residues = residues[:]
    best_L = _score(residues, primes, max_L_check)
    
    # Try single changes
    for idx in range(n):
        if time.time() > deadline:
            break
        
        p = primes[idx]
        old_r = residues[idx]
        
        # Build base counts
        base_counts = _build_count_array(residues, primes, max_L_check)
        r_old_eff = p if old_r == 0 else old_r
        if r_old_eff <= max_L_check:
            positions = np.arange(r_old_eff, max_L_check + 1, p)
            base_counts[positions] -= 1
        
        # Try a few random candidates
        import random
        random.seed(idx * 7919 + int(time.time() * 1000) % 10000)
        test_candidates = set([old_r])
        if p <= 50:
            test_candidates.update(range(p))
        else:
            # Try residues that cover early deficient positions
            one_positions = np.where(base_counts[1:min(200, max_L_check+1)] == 1)[0] + 1
            for pos in one_positions[:30]:
                test_candidates.add(pos % p)
            # Add some random
            for _ in range(20):
                test_candidates.add(random.randint(0, p-1))
        
        for new_r in test_candidates:
            if time.time() > deadline:
                break
            new_L = _compute_L_from_base(base_counts, new_r, p, max_L_check)
            if new_L > best_L:
                best_L = new_L
                best_residues[idx] = new_r
    
    return best_residues, best_L

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    primes = _get_primes(1000)
    n = len(primes)
    
    # Phase 1: Greedy construction with small max_L
    max_L_small = 800
    residues = _greedy_construct(primes, max_L_small)
    
    # Phase 2: Increase max_L and do backward refinement
    max_L_medium = 2500
    residues, _ = _refine_pass(residues, primes, max_L_medium, deadline)
    
    # Phase 3: Further refinement with larger max_L if time permits
    if time.time() < deadline - 30:
        max_L_large = 5000
        residues, _ = _refine_pass(residues, primes, max_L_large, deadline)
    
    # Phase 4: Final full-range refinement
    if time.time() < deadline - 20:
        max_L_final = 10000
        residues, _ = _refine_pass(residues, primes, max_L_final, deadline)
    
    # Phase 5: One more pass with even larger range if time allows
    if time.time() < deadline - 12:
        max_L_very_large = 20000
        residues, _ = _refine_pass(residues, primes, max_L_very_large, deadline)
    
    # Phase 6: Local search to escape local optima
    if time.time() < deadline - 8:
        residues, _ = _local_search(residues, primes, 10000, deadline)
    
    # Phase 7: Final refinement after local search
    if time.time() < deadline - 5:
        max_L_final2 = 15000
        residues, _ = _refine_pass(residues, primes, max_L_final2, deadline)
    
    # Final evaluation
    final_score = _score(residues, primes, 100000)
    
    return residues
