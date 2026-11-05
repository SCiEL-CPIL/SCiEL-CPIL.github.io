"""Combined program: local search with targeted gap repair and diverse restarts."""
import numpy as np
import time
import random
import math

def _get_primes():
    """Return list of primes <= 1000."""
    sieve = [True] * 1001
    sieve[0] = sieve[1] = False
    for i in range(2, int(1000**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(2, 1001) if sieve[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    primes = _get_primes()
    n = len(primes)
    MAX_L = 100000
    
    def build_coverage(residues):
        """Build the coverage array from scratch."""
        cov = np.zeros(MAX_L + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            if start <= MAX_L:
                cov[start::p] += 1
        return cov
    
    def find_first_gap(cov):
        """Find the first x >= 1 where cov[x] < 2."""
        bad = np.where(cov[1:] < 2)[0]
        if len(bad) == 0:
            return MAX_L
        return int(bad[0])
    
    def update_coverage(cov, idx, old_r, new_r):
        """Update coverage array in-place when changing residue at idx."""
        p = primes[idx]
        if old_r == 0:
            old_start = p
        else:
            old_start = old_r
        if old_start <= MAX_L:
            cov[old_start::p] -= 1
            
        if new_r == 0:
            new_start = p
        else:
            new_start = new_r
        if new_start <= MAX_L:
            cov[new_start::p] += 1

    best_L_global = 0
    best_residues_global = [0] * n
    
    def init_seed():
        return [((i + 2) // 2) % p for i, p in enumerate(primes)]
        
    def init_random():
        return [random.randint(0, p-1) for p in primes]

    def init_perturbed():
        r = init_seed()
        for _ in range(15):
            idx = random.randint(0, n-1)
            r[idx] = random.randint(0, primes[idx]-1)
        return r

    def init_targeted():
        """Greedy initialization targeting double coverage of early integers."""
        r = [0] * n
        # Track which primes are "used" for each position
        for x in range(1, 601):
            count = 0
            for i, p in enumerate(primes):
                if (x - r[i]) % p == 0:
                    count += 1
            if count < 2:
                # Find primes that can cover x and aren't already assigned to cover it
                candidates = []
                for i, p in enumerate(primes):
                    if (x - r[i]) % p != 0:  # This prime doesn't currently cover x
                        candidates.append(i)
                # Try to assign to cover x
                for idx in random.sample(candidates, min(len(candidates), 5)):
                    p = primes[idx]
                    new_r = x % p
                    # Check if this assignment helps
                    r[idx] = new_r
                    count += 1
                    if count >= 2:
                        break
        # Fill any remaining with random
        for i in range(n):
            if r[i] == 0:
                r[i] = random.randint(0, primes[i]-1)
        return r

    def init_double_targeted():
        """More aggressive targeted initialization for up to 1000."""
        r = [0] * n
        for x in range(1, 1001):
            count = 0
            for i, p in enumerate(primes):
                if (x - r[i]) % p == 0:
                    count += 1
            if count < 2:
                candidates = [i for i in range(n) if (x - r[i]) % primes[i] != 0]
                random.shuffle(candidates)
                for idx in candidates[:3]:
                    r[idx] = x % primes[idx]
                    count += 1
                    if count >= 2:
                        break
        for i in range(n):
            if r[i] == 0:
                r[i] = random.randint(0, primes[i]-1)
        return r

    def init_concentrated():
        """Initialize with residues concentrated to cover small numbers densely."""
        r = [0] * n
        # Assign residues to cover 1..200 with at least 2 each
        for x in range(1, 201):
            needed = 2
            for i, p in enumerate(primes):
                if (x - r[i]) % p == 0:
                    needed -= 1
                    if needed <= 0:
                        break
            if needed > 0:
                # Assign to primes that don't cover x yet
                for i in random.sample(range(n), min(n, 10)):
                    if (x - r[i]) % primes[i] != 0:
                        r[i] = x % primes[i]
                        needed -= 1
                        if needed <= 0:
                            break
        for i in range(n):
            if r[i] == 0:
                r[i] = random.randint(0, primes[i]-1)
        return r

    initializers = [init_seed, init_perturbed, init_random, init_targeted, 
                    init_double_targeted, init_concentrated,
                    init_seed, init_perturbed, init_random, init_targeted,
                    init_double_targeted, init_concentrated]
    
    def local_search(residues, coverage, current_L, time_limit):
        """Perform local search to improve coverage."""
        improved = True
        pass_count = 0
        max_passes = 60
        
        while improved and pass_count < max_passes and time.time() < time_limit:
            improved = False
            pass_count += 1
            shuffled_indices = list(range(n))
            random.shuffle(shuffled_indices)
            
            for idx in shuffled_indices:
                if time.time() > time_limit:
                    break
                
                p = primes[idx]
                old_r = residues[idx]
                best_r_for_idx = old_r
                best_L_for_idx = current_L
                
                # Generate candidates
                candidates = set()
                candidates.add(old_r)
                
                # Target the next gap
                target = current_L + 1
                if target <= MAX_L:
                    candidates.add(target % p)
                    
                # Target positions where coverage is exactly 1
                check_range = min(current_L + 300, MAX_L)
                if check_range >= 1:
                    one_cov_positions = np.where(coverage[1:check_range+1] == 1)[0] + 1
                    if len(one_cov_positions) > 0:
                        sample_size = min(80, len(one_cov_positions))
                        sampled = random.sample(list(one_cov_positions), sample_size)
                        for pos in sampled:
                            candidates.add(pos % p)
                
                # Add random candidates
                num_random = min(60, p)
                for _ in range(num_random):
                    candidates.add(random.randint(0, p-1))
                    
                # Add neighbors
                candidates.add((old_r + 1) % p)
                candidates.add((old_r - 1) % p)
                
                # For small primes, try all
                if p <= 50:
                    candidates = set(range(p))
                
                for r in candidates:
                    if r == old_r:
                        continue
                    if time.time() > time_limit:
                        break
                        
                    update_coverage(coverage, idx, old_r, r)
                    new_L = find_first_gap(coverage)
                    update_coverage(coverage, idx, r, old_r)
                    
                    if new_L > best_L_for_idx:
                        best_L_for_idx = new_L
                        best_r_for_idx = r
                    
                if best_r_for_idx != old_r:
                    update_coverage(coverage, idx, old_r, best_r_for_idx)
                    residues[idx] = best_r_for_idx
                    current_L = best_L_for_idx
                    improved = True
                    if current_L > best_L_global:
                        best_L_global = current_L
                        best_residues_global = residues[:]
        
        return residues, coverage, current_L
    
    def gap_repair(residues, coverage, current_L, time_limit):
        """Targeted repair of gaps by focusing on deficient positions."""
        improved = True
        repair_passes = 0
        max_repair_passes = 30
        
        while improved and repair_passes < max_repair_passes and time.time() < time_limit:
            improved = False
            repair_passes += 1
            
            # Find all deficient positions up to current_L + 500
            check_end = min(current_L + 500, MAX_L)
            deficient = np.where(coverage[1:check_end+1] < 2)[0] + 1
            
            if len(deficient) == 0:
                break
            
            # Process deficient positions in order
            for pos in deficient[:100]:  # Limit to first 100 for time
                if time.time() > time_limit:
                    break
                
                # Find primes that can cover this position
                covering_primes = []
                for i, p in enumerate(primes):
                    if (pos - residues[i]) % p != 0:
                        covering_primes.append(i)
                
                if not covering_primes:
                    continue
                
                # Try to assign a residue to one of these primes to cover pos
                best_gain = 0
                best_idx = -1
                best_r = -1
                
                for idx in covering_primes[:10]:  # Limit candidates
                    p = primes[idx]
                    new_r = pos % p
                    
                    # Estimate gain: how many positions would this help?
                    # Check a window around pos
                    window_start = max(1, pos - 20)
                    window_end = min(MAX_L, pos + 20)
                    
                    old_cov_window = coverage[window_start:window_end+1].copy()
                    
                    # Apply change temporarily
                    update_coverage(coverage, idx, residues[idx], new_r)
                    
                    # Count how many positions in window now have >= 2
                    new_cov_window = coverage[window_start:window_end+1]
                    gain = np.sum((new_cov_window >= 2) - (old_cov_window >= 2))
                    
                    # Revert
                    update_coverage(coverage, idx, new_r, residues[idx])
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                        best_r = new_r
                
                if best_idx >= 0 and best_gain > 0:
                    update_coverage(coverage, best_idx, residues[best_idx], best_r)
                    residues[best_idx] = best_r
                    current_L = find_first_gap(coverage)
                    improved = True
                    if current_L > best_L_global:
                        best_L_global = current_L
                        best_residues_global = residues[:]
        
        return residues, coverage, current_L
    
    # Main loop over initializations
    for init_idx, init_func in enumerate(initializers):
        if time.time() > deadline - 10:
            break
        
        residues = init_func()
        coverage = build_coverage(residues)
        current_L = find_first_gap(coverage)
        
        if current_L > best_L_global:
            best_L_global = current_L
            best_residues_global = residues[:]
        
        # Phase 1: Standard local search
        time_limit_phase1 = min(deadline - 5, time.time() + 12)
        residues, coverage, current_L = local_search(residues, coverage, current_L, time_limit_phase1)
        
        # Phase 2: Gap repair
        time_limit_phase2 = min(deadline - 5, time.time() + 15)
        residues, coverage, current_L = gap_repair(residues, coverage, current_L, time_limit_phase2)
        
        # Phase 3: Another local search pass
        time_limit_phase3 = min(deadline - 5, time.time() + 10)
        residues, coverage, current_L = local_search(residues, coverage, current_L, time_limit_phase3)
        
        # Update global best
        if current_L > best_L_global:
            best_L_global = current_L
            best_residues_global = residues[:]
        
        # If we've reached a good score, do a final polish
        if best_L_global >= 50 and time.time() < deadline - 30:
            # Final polish: try all residues for small primes
            for idx in range(min(20, n)):  # First 20 primes (small ones)
                if time.time() > deadline - 10:
                    break
                p = primes[idx]
                old_r = residues[idx]
                best_r = old_r
                best_L = find_first_gap(coverage)
                
                for r in range(p):
                    if r == old_r:
                        continue
                    if time.time() > deadline - 10:
                        break
                    update_coverage(coverage, idx, old_r, r)
                    new_L = find_first_gap(coverage)
                    update_coverage(coverage, idx, r, old_r)
                    if new_L > best_L:
                        best_L = new_L
                        best_r = r
                
                if best_r != old_r:
                    update_coverage(coverage, idx, old_r, best_r)
                    residues[idx] = best_r
                    current_L = best_L
                    if current_L > best_L_global:
                        best_L_global = current_L
                        best_residues_global = residues[:]
    
    return [int(r) for r in best_residues_global]
