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

    def init_residues(seed_val):
        """Greedy-ish random initialization."""
        rng = random.Random(seed_val)
        residues = np.array([rng.randint(0, p - 1) for p in primes], dtype=np.int64)
        return residues

    def build_counts(residues):
        """Build coverage counts array from residues."""
        counts = np.zeros(M + 1, dtype=np.int64)
        for i, p in enumerate(primes):
            r = residues[i]
            counts[r::p] += 1
        return counts

    def repair_phase(residues, counts, time_limit):
        """
        Repair-first strategy: find the first gap and try to fix it.
        Returns updated residues, counts, and current L.
        """
        current_L = calc_L(counts)
        
        while time.time() < time_limit:
            if current_L >= M:
                break
            
            gap_pos = current_L + 1
            if counts[gap_pos] >= 2:
                # Recalculate (shouldn't happen if current_L is accurate, but safety)
                current_L = calc_L(counts)
                continue

            # Find candidates: primes that can cover gap_pos
            # We want to change a prime's residue to gap_pos % p
            # Prefer smaller primes as they have more frequent occurrences? 
            # Or prefer primes that currently have low coverage elsewhere?
            # Let's try all primes that are NOT already covering gap_pos
            
            candidates = []
            for i, p in enumerate(primes):
                r_needed = gap_pos % p
                if residues[i] != r_needed:
                    candidates.append((i, p, r_needed))
            
            if not candidates:
                break # Should not happen, but safety

            # Sort by prime size (smaller primes first)
            candidates.sort(key=lambda x: x[1])
            
            improved = False
            # Try the first few candidates
            for i, p, r_new in candidates[:20]:
                if time.time() > time_limit:
                    break
                
                r_old = residues[i]
                
                # Apply change
                counts[r_old::p] -= 1
                counts[r_new::p] += 1
                residues[i] = r_new
                
                new_L = calc_L(counts)
                
                if new_L > current_L:
                    current_L = new_L
                    improved = True
                    break
                else:
                    # Revert
                    counts[r_new::p] -= 1
                    counts[r_old::p] += 1
                    residues[i] = r_old
            
            if not improved:
                # If we can't fix the immediate gap with a single move, 
                # we might need a more global search (annealing)
                break
        
        return residues, counts, current_L

    def simulated_annealing(residues, counts, initial_L, time_limit, seed_val):
        """
        Simulated Annealing to escape local minima.
        """
        rng = random.Random(seed_val)
        current_L = initial_L
        best_L = initial_L
        best_residues = residues.copy()
        
        T = 5.0
        T_min = 0.1
        cooling_rate = 0.9995
        
        while time.time() < time_limit:
            # Pick a random prime
            i = rng.randint(0, n_primes - 1)
            p = primes[i]
            r_old = residues[i]
            
            # Pick a new random residue
            # Bias towards residues that cover the current gap?
            # For now, pure random to explore broadly
            r_new = rng.randint(0, p - 1)
            
            if r_new == r_old:
                continue
            
            # Apply
            counts[r_old::p] -= 1
            counts[r_new::p] += 1
            residues[i] = r_new
            
            new_L = calc_L(counts)
            
            if new_L >= current_L:
                current_L = new_L
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues.copy()
            else:
                # Accept with probability exp(-delta / T)
                delta = current_L - new_L
                # If delta is large, prob is small
                if delta > 1000: # Too bad, reject
                    counts[r_new::p] -= 1
                    counts[r_old::p] += 1
                    residues[i] = r_old
                else:
                    prob = np.exp(-delta / T)
                    if rng.random() < prob:
                        current_L = new_L
                    else:
                        # Revert
                        counts[r_new::p] -= 1
                        counts[r_old::p] += 1
                        residues[i] = r_old
            
            T = max(T_min, T * cooling_rate)
            
            # Occasionally run a quick repair to get back on track if we've degraded
            if rng.random() < 0.01:
                residues, counts, current_L = repair_phase(residues, counts, time.time() + 0.1)
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
            
        return best_residues, best_L

    # Main logic
    # 1. Initial random start
    rng_main = random.Random(42)
    residues = init_residues(42)
    counts = build_counts(residues)
    current_L = calc_L(counts)
    
    # 2. Initial Repair Phase (fast, greedy)
    # Give it a generous chunk of time, say 20 seconds
    repair_deadline = start_time + 20.0
    residues, counts, current_L = repair_phase(residues, counts, repair_deadline)
    
    best_L = current_L
    best_residues = residues.copy()
    
    # 3. Simulated Annealing Phase
    # Use remaining time
    anneal_deadline = deadline - 2.0
    if time.time() < anneal_deadline:
        best_residues, best_L = simulated_annealing(
            residues, counts, current_L, anneal_deadline, seed_val=123
        )
        
        # 4. Final Polish with Repair
        # Take the best found and try to repair it further
        final_residues = best_residues.copy()
        final_counts = build_counts(final_residues)
        final_L = calc_L(final_counts)
        
        if final_L < best_L:
            # Rebuild from best_residues
            final_counts = build_counts(best_residues)
            final_L = calc_L(final_counts)
            final_residues = best_residues.copy()
        
        final_residues, final_counts, final_L = repair_phase(
            final_residues, final_counts, deadline - 0.5
        )
        
        if final_L > best_L:
            best_L = final_L
            best_residues = final_residues

    return [int(r) for r in best_residues]
