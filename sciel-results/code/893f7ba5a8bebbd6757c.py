"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _get_primes(limit):
    """Sieve of Eratosthenes to get primes up to limit."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin before 150s kill
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    
    # Search window large enough to potentially exceed 1000.
    # 5000 is a good balance between coverage and speed.
    max_check = 5000
    
    # Initialize residues to 0
    residues = np.zeros(n_primes, dtype=np.int64)
    
    # counts[x] = number of primes p such that x == residues[idx_of_p] (mod p)
    counts = np.zeros(max_check + 1, dtype=np.int32)
    
    def update_counts(idx, old_r, new_r, delta):
        """
        Update counts array.
        delta = +1 means adding the residue, -1 means removing.
        """
        p = primes[idx]
        if delta == 1:
            r = new_r
            start = r if r > 0 else p
            if start <= max_check:
                counts[start::p] += 1
        else: # delta == -1
            r = old_r
            start = r if r > 0 else p
            if start <= max_check:
                counts[start::p] -= 1

    # Initial setup: all residues 0
    for i in range(n_primes):
        update_counts(i, 0, 0, 1)
        
    def get_prefix_score():
        """
        Find the largest L such that counts[1..L] >= 2.
        """
        # We want the first index in 1..max_check where count < 2
        below_two = np.where(counts[1:max_check+1] < 2)[0]
        if len(below_two) == 0:
            return max_check
        return below_two[0]

    def get_global_weakness():
        """
        Count the number of positions in [1, max_check] with count < 2.
        Lower is better.
        """
        return np.sum(counts[1:max_check+1] < 2)

    best_score = get_prefix_score()
    best_residues = residues.copy()
    best_weakness = get_global_weakness()
    
    # Phase 1: Global Iterative Improvement
    # Goal: Minimize the number of "weak" positions (count < 2) in the window.
    # This helps build a robust coverage structure that can support a long prefix.
    
    # We will iterate through primes and try to find a residue change that reduces global weakness.
    # To speed up, we focus on primes that are currently "underperforming" or random samples.
    
    max_iterations_phase1 = 20000
    iter_count = 0
    
    # Precompute prime indices for faster access
    prime_indices = list(range(n_primes))
    
    while time.time() < deadline - 20.0 and iter_count < max_iterations_phase1:
        iter_count += 1
        
        # Pick a random prime to modify
        i = np.random.randint(n_primes)
        p = primes[i]
        old_r = residues[i]
        
        # Try a few random new residues for this prime
        # Or, better: try the residue that covers the most "weak" points?
        # That's expensive. Let's try random perturbations first.
        # If random fails, we can try targeted moves later.
        
        # Actually, a smarter move: 
        # Identify the most frequent "gap" residue.
        # But let's stick to simple random search for now to escape local minima.
        
        # Try 5 random residues
        best_new_r = -1
        best_delta_weakness = 0
        
        for _ in range(5):
            new_r = np.random.randint(p)
            if new_r == old_r:
                continue
            
            # Calculate delta in weakness
            # Weakness = sum(1 for x in 1..max_check if counts[x] < 2)
            # Change affects positions where x % p == old_r (lose 1) and x % p == new_r (gain 1)
            
            # Positions losing coverage:
            # x in [1, max_check] such that x % p == old_r
            # Positions gaining coverage:
            # x in [1, max_check] such that x % p == new_r
            
            # We only care about positions where the change crosses the threshold 2.
            # Lose coverage (count -> count-1):
            #   If count was 2, it becomes 1 (weakness +1)
            #   If count was 1, it becomes 0 (weakness +1)
            #   If count was 0, it becomes -1 (invalid, but counts shouldn't go negative if we track correctly? 
            #      Wait, counts are non-negative. If count is 0, it can't lose coverage. 
            #      So we only consider positions currently covered by old_r.)
            
            # Gain coverage (count -> count+1):
            #   If count was 1, it becomes 2 (weakness -1)
            #   If count was 0, it becomes 1 (weakness 0)
            #   If count was 2+, it becomes 3+ (weakness 0)
            
            # So:
            # Delta Weakness = 
            #   (Number of x covered by old_r with count[x] <= 2)  [These become < 2 or stay < 2 but were weak? 
            #      Wait. If count[x] == 2, it was NOT weak. Becomes 1 (weak). Delta +1.
            #      If count[x] == 1, it WAS weak. Becomes 0 (weak). Delta 0.
            #      If count[x] == 0, it WAS weak. But it wasn't covered, so it doesn't lose coverage.
            #      So: Positions covered by old_r with count[x] == 2 contribute +1 to weakness.
            #      Positions covered by old_r with count[x] == 1 contribute 0.
            #   MINUS
            #   (Number of x covered by new_r with count[x] == 1) [These become 2, so weak -> strong. Delta -1]
            #      Positions covered by new_r with count[x] == 0 contribute 0 (weak -> weak).
            
            # Let's compute these counts efficiently using numpy.
            
            # Positions covered by old_r
            if old_r > 0:
                pos_old = np.arange(old_r, max_check + 1, p)
            else:
                pos_old = np.arange(p, max_check + 1, p)
            
            # Positions covered by new_r
            if new_r > 0:
                pos_new = np.arange(new_r, max_check + 1, p)
            else:
                pos_new = np.arange(p, max_check + 1, p)
            
            # Count of pos_old with count == 2
            # Note: pos_old and pos_new are disjoint if old_r != new_r
            num_old_critical = np.sum(counts[pos_old] == 2)
            
            # Count of pos_new with count == 1
            num_new_weak = np.sum(counts[pos_new] == 1)
            
            delta_weakness = num_old_critical - num_new_weak
            
            if delta_weakness < best_delta_weakness:
                best_delta_weakness = delta_weakness
                best_new_r = new_r
        
        if best_new_r != -1 and best_delta_weakness < 0:
            # Apply change
            update_counts(i, old_r, best_new_r, -1)
            residues[i] = best_new_r
            update_counts(i, best_new_r, best_new_r, 1)
            
            # Check prefix score
            current_score = get_prefix_score()
            if current_score > best_score:
                best_score = current_score
                best_residues = residues.copy()
                
            # Check global weakness
            current_weakness = get_global_weakness()
            if current_weakness < best_weakness:
                best_weakness = current_weakness
    
    # Phase 2: Prefix Repair
    # Now that we have a better global structure, try to extend the prefix strictly.
    
    # Restore the best found residues so far? 
    # The current 'residues' might have a worse prefix than 'best_residues' but better global structure.
    # Let's continue from the current state, as it might be closer to a solution.
    # But if the prefix is very short, we might want to reset to best_residues.
    # Let's just continue with current state, as the global improvement likely helped.
    
    max_iterations_phase2 = 10000
    iter_count = 0
    
    while time.time() < deadline and iter_count < max_iterations_phase2:
        iter_count += 1
        
        current_score = get_prefix_score()
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
            
        if current_score >= max_check:
            break
            
        x_target = current_score + 1
        current_count_x = counts[x_target]
        
        # We need to increase count[x_target] to at least 2.
        # Try all primes, pick the one that is valid (doesn't break prefix) and has smallest p.
        applied = False
        
        # Optimization: Only try primes that are NOT currently covering x_target?
        # If a prime is already covering x_target, changing it might remove coverage.
        # We want to ADD coverage. So we prefer primes where new_r = x_target % p != old_r.
        
        for i in range(n_primes):
            p = primes[i]
            old_r = residues[i]
            new_r = x_target % p
            
            if old_r == new_r:
                continue
                
            # Check validity: Does changing old_r to new_r break any prefix point?
            # Affected positions: x in [1, x_target-1] where x % p == old_r
            # If any such x has count[x] == 2, it becomes 1, breaking prefix.
            
            start_pos = old_r if old_r > 0 else p
            if start_pos >= x_target:
                valid = True
            else:
                positions = np.arange(start_pos, x_target, p)
                if np.any(counts[positions] == 2):
                    valid = False
                else:
                    valid = True
            
            if valid:
                update_counts(i, old_r, new_r, -1)
                residues[i] = new_r
                update_counts(i, new_r, new_r, 1)
                applied = True
                break
        
        if not applied:
            # Stuck. Try a random "kick" to escape?
            # Pick a random prime, random residue, apply if it doesn't drastically reduce score?
            # For now, break to save time.
            break
            
    return [int(r) for r in best_residues]
# EVOLVE-BLOCK-END
