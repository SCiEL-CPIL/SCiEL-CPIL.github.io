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
        for pos in range(1, min(max_L_check + 1, 300)):
            if counts[pos] < 2:
                deficient_positions.append(pos)
                if len(deficient_positions) >= 80:
                    break
        
        if not deficient_positions:
            # No deficient positions in early range, pick residue to maximize overall coverage
            best_r = 1
            best_count = 0
            for r in range(min(p, 60)):
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

def _refine_pass(residues, primes, max_L_check, deadline):
    """
    Optimized backward refinement.
    For each prime, we determine the best residue by looking at the deficiencies
    caused by the other primes.
    """
    n = len(primes)
    improved = False
    
    # Process in reverse order
    for idx in range(n - 1, -1, -1):
        if time.time() > deadline:
            break
            
        p = primes[idx]
        old_r = residues[idx]
        
        # Build base counts (all primes except idx)
        base_counts = np.zeros(max_L_check + 1, dtype=np.int32)
        for j in range(n):
            if j == idx:
                continue
            r = residues[j]
            pj = primes[j]
            if r == 0:
                r_eff = pj
            else:
                r_eff = r
            if r_eff <= max_L_check:
                positions = np.arange(r_eff, max_L_check + 1, pj)
                base_counts[positions] += 1
        
        # Find current L with old residue
        current_counts = base_counts.copy()
        r_old_eff = p if old_r == 0 else old_r
        if r_old_eff <= max_L_check:
            positions = np.arange(r_old_eff, max_L_check + 1, p)
            current_counts[positions] += 1
        current_L = _find_L(current_counts, max_L_check)
        
        # Identify candidate residues that cover deficient positions
        # A position x needs this prime's help if base_counts[x] < 2
        # If base_counts[x] == 0, this prime alone can't fix it (max +1 = 1 < 2)
        # If base_counts[x] == 1, this prime can fix it if x % p == r
        # So we only care about positions where base_counts[x] == 1
        
        deficient_pos = []
        for pos in range(1, max_L_check + 1):
            if base_counts[pos] == 1:
                deficient_pos.append(pos)
                if len(deficient_pos) >= 50:
                    break
        
        if not deficient_pos:
            # No positions that need this prime's help, keep old
            continue
            
        # Candidate residues are those that cover at least one position with base_counts == 1
        candidates = set()
        for pos in deficient_pos:
            candidates.add(pos % p)
        
        # Also include the old residue
        candidates.add(old_r)
        
        # Limit candidates to 80 max for speed
        if len(candidates) > 80:
            cand_scores = {}
            for c in candidates:
                score = 0
                for pos in deficient_pos:
                    if pos % p == c:
                        score += 1
                cand_scores[c] = score
            sorted_cands = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
            candidates = {c for c, _ in sorted_cands[:80]}
        
        best_r = old_r
        best_L = current_L
        
        # Evaluate candidates
        for new_r in candidates:
            if time.time() > deadline:
                break
                
            r_new_eff = p if new_r == 0 else new_r
            
            # Calculate new L efficiently
            # A position x is covered if base_counts[x] + (1 if x%p==new_r else 0) >= 2
            # So we need the first x where base_counts[x] + (1 if x%p==new_r else 0) < 2
            # i.e., base_counts[x] == 0 (can't be fixed by this prime)
            # or base_counts[x] == 1 and x%p != new_r
            
            # Find first position where this fails
            # We can do this by scanning, but let's be smarter
            # Precompute positions where base_counts == 0
            zero_positions = np.where(base_counts[1:max_L_check+1] == 0)[0] + 1
            
            # The first zero position is a hard limit
            if len(zero_positions) > 0:
                hard_limit = zero_positions[0]
            else:
                hard_limit = max_L_check
            
            # For positions where base_counts == 1, they're OK if x%p == new_r
            # So we need the first position x <= hard_limit where base_counts[x] == 1 and x%p != new_r
            
            # Efficient check: iterate through positions 1..hard_limit
            # But this is O(hard_limit) per candidate, which could be slow
            # Let's precompute the positions where base_counts == 1
            one_positions = np.where(base_counts[1:max_L_check+1] == 1)[0] + 1
            
            # Find first one_position where x%p != new_r
            # This is the first x in one_positions such that x % p != new_r
            # Equivalently, find the first x in one_positions such that x % p != new_r
            
            new_L = hard_limit
            for x in one_positions:
                if x > hard_limit:
                    break
                if x % p != new_r:
                    new_L = x - 1
                    break
            
            if new_L > best_L:
                best_L = new_L
                best_r = new_r
        
        if best_r != old_r:
            residues[idx] = best_r
            improved = True
    
    return residues, improved

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
    if time.time() < deadline - 15:
        max_L_final = 10000
        residues, _ = _refine_pass(residues, primes, max_L_final, deadline)
    
    # Phase 5: One more pass with even larger range if time allows
    if time.time() < deadline - 10:
        max_L_very_large = 20000
        residues, _ = _refine_pass(residues, primes, max_L_very_large, deadline)
    
    # Final evaluation
    final_score = _score(residues, primes, 100000)
    
    return residues
