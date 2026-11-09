import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)
    
    # Max L to check for score
    M = 100000
    
    def calc_L(counts):
        """Find the largest L such that every integer 1..L has count >= 2."""
        if counts[1] < 2:
            return 0
        mask = ~(counts[1:] >= 2)
        if not np.any(mask):
            return M
        first_false_idx = np.argmax(mask)
        return first_false_idx

    def greedy_init(seed_val):
        """Greedy initialization: try to cover positions sequentially."""
        rng = random.Random(seed_val)
        residues = np.zeros(n_primes, dtype=np.int64)
        # Start with a simple pattern, then refine
        for i, p in enumerate(primes):
            residues[i] = rng.randint(0, p - 1)
        return residues

    def conflict_driven_repair(residues, counts, L_target, time_limit):
        """Try to fix positions in [1, L_target] that have coverage < 2."""
        if time.time() > time_limit:
            return residues, counts, False
        
        # Find positions with coverage < 2 in [1, min(L_target, 5000)]
        check_range = min(L_target, 5000)
        weak_positions = np.where(counts[1:check_range+1] < 2)[0] + 1
        
        if len(weak_positions) == 0:
            return residues, counts, True
        
        # For each weak position, find primes that could cover it
        # A prime p covers position x if x % p == r_p, i.e., r_p = x % p
        fixed_any = False
        for pos in weak_positions[:50]:  # Limit to 50 positions to save time
            if time.time() > time_limit:
                break
            current_count = counts[pos]
            if current_count >= 2:
                continue
            
            # Find primes where we could change residue to cover this position
            candidates = []
            for i, p in enumerate(primes):
                if time.time() > time_limit:
                    break
                needed_r = pos % p
                if needed_r == residues[i]:
                    continue  # Already covers this position
                # Check if changing this residue would help
                # The new residue would cover pos, but might uncover other positions
                candidates.append((i, p, needed_r))
            
            # Try the most promising candidates (smaller primes affect more positions)
            candidates.sort(key=lambda x: x[1])
            
            for i, p, new_r in candidates[:10]:
                if time.time() > time_limit:
                    break
                old_r = residues[i]
                if old_r == new_r:
                    continue
                
                # Try the change
                counts[old_r::p] -= 1
                counts[new_r::p] += 1
                residues[i] = new_r
                
                new_L = calc_L(counts)
                if new_L >= L_target:
                    fixed_any = True
                    break
                else:
                    # Revert
                    counts[new_r::p] -= 1
                    counts[old_r::p] += 1
                    residues[i] = old_r
            
            if fixed_any:
                break
        
        return residues, counts, fixed_any

    def simulated_annealing(residues, counts, initial_L, time_limit, seed_val):
        """Run simulated annealing with the given initial state."""
        rng = random.Random(seed_val)
        current_L = initial_L
        best_L = initial_L
        best_residues = residues.copy()
        
        T = 5.0
        T_min = 0.1
        cooling_rate = 0.9995
        
        while time.time() < time_limit:
            # Pick random prime index
            i = rng.randint(0, n_primes - 1)
            p = primes[i]
            r_old = residues[i]
            
            # Pick new random residue
            r_new = rng.randint(0, p - 1)
            if r_new == r_old:
                continue
            
            # Update counts
            counts[r_old::p] -= 1
            counts[r_new::p] += 1
            
            new_L = calc_L(counts)
            
            if new_L >= current_L:
                current_L = new_L
                residues[i] = r_new
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues.copy()
            else:
                # Accept with probability exp(-delta / T)
                delta = current_L - new_L
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_L = new_L
                    residues[i] = r_new
                else:
                    # Reject, revert
                    counts[r_new::p] -= 1
                    counts[r_old::p] += 1
                    residues[i] = r_old
            
            # Cool down
            T = max(T_min, T * cooling_rate)
            
            # Periodic conflict-driven repair
            if best_L > 1000 and rng.random() < 0.05:
                residues, counts, _ = conflict_driven_repair(
                    residues, counts, best_L, time_limit - 2.0
                )
                current_L = calc_L(counts)
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
        
        return best_residues, best_L

    def polish_pass(residues, L_current, time_limit):
        """Exhaustive local search on small primes to polish the solution."""
        counts = np.zeros(M + 1, dtype=np.int64)
        for i, p in enumerate(primes):
            r = residues[i]
            counts[r::p] += 1
        
        best_L = L_current
        best_residues = residues.copy()
        
        # Focus on small primes (first 30 primes) which affect many positions
        num_small = min(30, n_primes)
        
        for i in range(num_small):
            if time.time() > time_limit:
                break
            p = primes[i]
            current_r = residues[i]
            old_idx = current_r
            best_r = current_r
            best_local_L = best_L
            
            for r in range(p):
                if time.time() > time_limit:
                    break
                if r == current_r:
                    continue
                
                # Try changing residue i to r
                counts[old_idx::p] -= 1
                counts[r::p] += 1
                
                new_L = calc_L(counts)
                
                if new_L > best_local_L:
                    best_local_L = new_L
                    best_r = r
                else:
                    # Revert
                    counts[r::p] -= 1
                    counts[old_idx::p] += 1
            
            # Apply best
            if best_r != current_r:
                counts[old_idx::p] -= 1
                counts[best_r::p] += 1
                residues[i] = best_r
                best_residues[i] = best_r
                best_L = best_local_L
            else:
                # Ensure counts are correct
                counts[old_idx::p] -= 1
                counts[current_r::p] += 1
        
        return best_residues, best_L

    # Main search: multiple restarts with different strategies
    global_best_L = 0
    global_best_residues = None
    
    # Phase 1: Multiple SA restarts with different initializations
    num_restarts = 3
    time_per_restart = (deadline - start_time - 10.0) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        restart_start = time.time()
        restart_end = restart_start + time_per_restart - 1.0
        
        # Different initialization strategies
        if restart == 0:
            # Greedy-like init
            residues = greedy_init(42)
        elif restart == 1:
            # Random init
            residues = np.array([random.randint(0, p - 1) for p in primes], dtype=np.int64)
        else:
            # Pattern-based init
            residues = np.array([((i + 2) // 2) % primes[i] for i in range(n_primes)], dtype=np.int64)
        
        # Initialize counts
        counts = np.zeros(M + 1, dtype=np.int64)
        for i, p in enumerate(primes):
            r = residues[i]
            counts[r::p] += 1
        
        initial_L = calc_L(counts)
        
        # Run SA
        best_res, best_L = simulated_annealing(
            residues, counts, initial_L, restart_end, seed_val=100 + restart
        )
        
        if best_L > global_best_L:
            global_best_L = best_L
            global_best_residues = best_res.copy()
    
    # Phase 2: Polish pass on the best solution found
    if time.time() < deadline - 5.0 and global_best_residues is not None:
        final_res, final_L = polish_pass(
            global_best_residues, global_best_L, deadline - 2.0
        )
        if final_L > global_best_L:
            global_best_L = final_L
            global_best_residues = final_res
    
    # Phase 3: Final conflict-driven repair attempt
    if time.time() < deadline - 2.0 and global_best_residues is not None:
        counts = np.zeros(M + 1, dtype=np.int64)
        for i, p in enumerate(primes):
            r = global_best_residues[i]
            counts[r::p] += 1
        
        global_best_residues, counts, _ = conflict_driven_repair(
            global_best_residues, counts, global_best_L, deadline - 1.0
        )
        final_L = calc_L(counts)
        if final_L > global_best_L:
            global_best_L = final_L
    
    if global_best_residues is None:
        # Fallback: return a simple pattern
        return [((i + 2) // 2) % primes[i] for i in range(n_primes)]
    
    return [int(r) for r in global_best_residues]
