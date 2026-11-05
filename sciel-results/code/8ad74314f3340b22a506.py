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
        
        # Precompute positions where base_counts == 0 and base_counts == 1
        zero_positions = np.where(base_counts[1:max_L_check+1] == 0)[0] + 1
        one_positions = np.where(base_counts[1:max_L_check+1] == 1)[0] + 1
        
        # The first zero position is a hard limit
        if len(zero_positions) > 0:
            hard_limit = zero_positions[0]
        else:
            hard_limit = max_L_check
        
        # Evaluate candidates
        for new_r in candidates:
            if time.time() > deadline:
                break
            
            # Find first one_position where x%p != new_r
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

def _pairwise_swap_refine(residues, primes, max_L_check, deadline):
    """
    Pairwise swap refinement: try swapping residues of two primes to improve coverage.
    This can escape local optima that single-prime refinement cannot.
    """
    n = len(primes)
    improved = False
    
    # Get current L
    current_L = _score(residues, primes, max_L_check)
    
    # Build count array
    counts = _build_count_array(residues, primes, max_L_check)
    
    # Focus on the first few hundred positions where deficiencies occur
    search_range = min(300, max_L_check)
    
    # Find deficient positions
    deficient = []
    for pos in range(1, search_range + 1):
        if counts[pos] < 2:
            deficient.append(pos)
            if len(deficient) >= 30:
                break
    
    if not deficient:
        return residues, False
    
    # Try pairs of primes - focus on larger primes first (more flexibility)
    # Process pairs in reverse order
    for i in range(n - 1, 0, -1):
        if time.time() > deadline:
            break
        
        for j in range(i - 1, -1, -1):
            if time.time() > deadline:
                break
            
            pi = primes[i]
            pj = primes[j]
            ri = residues[i]
            rj = residues[j]
            
            # Only try swaps if primes are not too large (to keep candidate space manageable)
            if pi > 200 or pj > 200:
                continue
            
            # Build base counts without these two primes
            base_counts = np.zeros(max_L_check + 1, dtype=np.int32)
            for k in range(n):
                if k == i or k == j:
                    continue
                r = residues[k]
                pk = primes[k]
                if r == 0:
                    r_eff = pk
                else:
                    r_eff = r
                if r_eff <= max_L_check:
                    positions = np.arange(r_eff, max_L_check + 1, pk)
                    base_counts[positions] += 1
            
            # Find current L with old residues
            current_counts = base_counts.copy()
            ri_eff = pi if ri == 0 else ri
            rj_eff = pj if rj == 0 else rj
            if ri_eff <= max_L_check:
                positions = np.arange(ri_eff, max_L_check + 1, pi)
                current_counts[positions] += 1
            if rj_eff <= max_L_check:
                positions = np.arange(rj_eff, max_L_check + 1, pj)
                current_counts[positions] += 1
            old_L = _find_L(current_counts, max_L_check)
            
            # Find candidate residues for each prime based on deficient positions
            # For prime i: residues that cover positions where base_counts + (1 if x%pi==r_i else 0) + (1 if x%pj==r_j else 0) < 2
            # Simplify: focus on positions where base_counts < 2
            
            cand_i = set()
            cand_j = set()
            for pos in deficient:
                if base_counts[pos] <= 1:
                    cand_i.add(pos % pi)
                    cand_j.add(pos % pj)
            
            cand_i.add(ri)
            cand_j.add(rj)
            
            # Limit candidates
            if len(cand_i) > 30:
                cand_i = list(cand_i)[:30]
            if len(cand_j) > 30:
                cand_j = list(cand_j)[:30]
            
            best_L = old_L
            best_ri = ri
            best_rj = rj
            
            for new_ri in cand_i:
                if time.time() > deadline:
                    break
                
                ri_eff = pi if new_ri == 0 else new_ri
                add_i = np.zeros(max_L_check + 1, dtype=np.int32)
                if ri_eff <= max_L_check:
                    positions = np.arange(ri_eff, max_L_check + 1, pi)
                    add_i[positions] = 1
                
                for new_rj in cand_j:
                    if time.time() > deadline:
                        break
                    
                    rj_eff = pj if new_rj == 0 else new_rj
                    add_j = np.zeros(max_L_check + 1, dtype=np.int32)
                    if rj_eff <= max_L_check:
                        positions = np.arange(rj_eff, max_L_check + 1, pj)
                        add_j[positions] = 1
                    
                    # Compute new L
                    # A position x is covered if base_counts[x] + add_i[x] + add_j[x] >= 2
                    # Find first x where this fails
                    new_L = max_L_check
                    for x in range(1, min(search_range + 1, max_L_check + 1)):
                        if base_counts[x] + add_i[x] + add_j[x] < 2:
                            new_L = x - 1
                            break
                    
                    if new_L > best_L:
                        best_L = new_L
                        best_ri = new_ri
                        best_rj = new_rj
            
            if best_ri != ri or best_rj != rj:
                residues[i] = best_ri
                residues[j] = best_rj
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
    
    # Phase 4: Pairwise swap refinement to escape local optima
    if time.time() < deadline - 25:
        max_L_pair = 3000
        residues, _ = _pairwise_swap_refine(residues, primes, max_L_pair, deadline)
    
    # Phase 5: Another backward refinement pass after pairwise swaps
    if time.time() < deadline - 20:
        max_L_large2 = 5000
        residues, _ = _refine_pass(residues, primes, max_L_large2, deadline)
    
    # Phase 6: Final full-range refinement
    if time.time() < deadline - 15:
        max_L_final = 10000
        residues, _ = _refine_pass(residues, primes, max_L_final, deadline)
    
    # Phase 7: One more pass with even larger range if time allows
    if time.time() < deadline - 10:
        max_L_very_large = 20000
        residues, _ = _refine_pass(residues, primes, max_L_very_large, deadline)
    
    # Final evaluation
    final_score = _score(residues, primes, 100000)
    
    return residues
