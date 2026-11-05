import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i:limit + 1:i] = [False] * len(sieve[i * i:limit + 1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)
    MAX_L = 100000
    
    # Helper to calculate counts for a given residue set
    def get_counts(residues):
        counts = np.zeros(MAX_L + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            start = r if r > 0 else p
            if start <= MAX_L:
                counts[start::p] += 1
        return counts

    def find_L(counts, check_limit=None):
        if check_limit is None:
            check_limit = MAX_L
        check_limit = min(check_limit, MAX_L)
        if check_limit <= 0:
            return 0
        bad_idx = np.where(counts[1:check_limit+1] < 2)[0]
        if len(bad_idx) == 0:
            return check_limit
        return bad_idx[0]

    # --- Phase 1: Multiple Greedy Initializations ---
    best_init_L = 0
    best_init_res = [0] * n_primes
    best_init_counts = np.zeros(MAX_L + 1, dtype=np.int32)
    
    for attempt in range(10):
        residues = [0] * n_primes
        counts = np.zeros(MAX_L + 1, dtype=np.int32)
        
        for idx in range(n_primes):
            p = primes[idx]
            best_r = 0
            best_score = -10**9
            
            if p <= 40:
                candidates = list(range(p))
            else:
                candidates = set()
                for _ in range(min(100, p)):
                    candidates.add(np.random.randint(p))
                for target in range(1, 50):
                    candidates.add(target % p)
                candidates = list(candidates)
            
            for r in candidates:
                start = r if r > 0 else p
                if start > MAX_L:
                    continue
                pos = np.arange(start, MAX_L + 1, p)
                current_counts = counts[pos]
                gained = np.sum(current_counts == 1)
                wasted = np.sum(current_counts >= 2)
                score = gained - 0.5 * wasted
                if score > best_score:
                    best_score = score
                    best_r = r
            
            residues[idx] = best_r
            r = best_r
            start = r if r > 0 else p
            if start <= MAX_L:
                counts[start::p] += 1
        
        L = find_L(counts)
        if L > best_init_L:
            best_init_L = L
            best_init_res = residues.copy()
            best_init_counts = counts.copy()

    # Start from the best greedy result
    residues = best_init_res
    counts = best_init_counts
    current_L = best_init_L
    best_L = current_L
    best_residues = residues.copy()
    
    # --- Phase 2: Directed Backtracking for Coverage Debt ---
    # This phase identifies positions with count < 2 and tries to fix them
    # by adjusting residues of primes that can cover those positions
    
    def find_deficient_positions(counts, limit):
        """Find positions 1..limit where count < 2"""
        deficient = np.where(counts[1:limit+1] < 2)[0] + 1
        return deficient
    
    def can_cover_position(pos, p, current_residues, counts):
        """Check if prime p can cover position pos with some residue"""
        r = pos % p
        # Check if this residue would help (i.e., position is currently deficient)
        return counts[pos] < 2
    
    def try_fix_position(pos, counts, residues):
        """Try to fix a single deficient position by adjusting one prime's residue"""
        p_candidates = []
        for i, p in enumerate(primes):
            if p > pos:
                continue
            r_needed = pos % p
            # Check if changing this prime to cover pos would help
            # and not hurt other positions too much
            old_r = residues[i]
            if old_r == r_needed:
                continue
            
            # Estimate impact: how many positions with count==1 would become 2
            start_old = old_r if old_r > 0 else p
            start_new = r_needed if r_needed > 0 else p
            
            # Count positions that would gain coverage
            positions_new = np.arange(start_new, MAX_L + 1, p)
            positions_old = np.arange(start_old, MAX_L + 1, p)
            
            # Positions covered by new but not old
            new_only = np.setdiff1d(positions_new, positions_old)
            old_only = np.setdiff1d(positions_old, positions_new)
            
            gained = np.sum(counts[new_only] == 1) if len(new_only) > 0 else 0
            lost = np.sum(counts[old_only] == 2) if len(old_only) > 0 else 0
            
            if gained > lost:
                p_candidates.append((i, r_needed, gained - lost))
        
        if not p_candidates:
            return False
        
        # Pick the best candidate
        p_candidates.sort(key=lambda x: -x[2])
        i, r_new, _ = p_candidates[0]
        p = primes[i]
        old_r = residues[i]
        
        # Apply the change
        start_old = old_r if old_r > 0 else p
        start_new = r_new if r_new > 0 else p
        
        if start_old <= MAX_L:
            counts[start_old::p] -= 1
        if start_new <= MAX_L:
            counts[start_new::p] += 1
        
        residues[i] = r_new
        return True
    
    # Run directed fixing until deadline
    stuck_iterations = 0
    max_stuck = 50
    
    while time.time() < deadline:
        # Find the first deficient position
        check_limit = min(current_L + 10000, MAX_L)
        deficient = find_deficient_positions(counts, check_limit)
        
        if len(deficient) == 0:
            # No deficient positions in range, try to extend
            new_L = find_L(counts)
            if new_L > current_L:
                current_L = new_L
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues.copy()
            stuck_iterations += 1
            if stuck_iterations > max_stuck:
                break
            continue
        
        stuck_iterations = 0
        
        # Try to fix the first few deficient positions
        fixed_any = False
        for pos in deficient[:5]:
            if try_fix_position(pos, counts, residues):
                fixed_any = True
                break
        
        if not fixed_any:
            # Try random perturbation as fallback
            idx = np.random.randint(n_primes)
            p = primes[idx]
            old_r = residues[idx]
            new_r = np.random.randint(p)
            
            if old_r != new_r:
                start_old = old_r if old_r > 0 else p
                start_new = new_r if new_r > 0 else p
                
                if start_old <= MAX_L:
                    counts[start_old::p] -= 1
                if start_new <= MAX_L:
                    counts[start_new::p] += 1
                
                residues[idx] = new_r
            
            stuck_iterations += 1
            if stuck_iterations > max_stuck:
                break
        
        # Recalculate L
        new_L = find_L(counts)
        if new_L > current_L:
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
    
    return best_residues
