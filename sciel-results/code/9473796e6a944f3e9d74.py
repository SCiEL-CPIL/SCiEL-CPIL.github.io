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

    # Precompute residue ranges for each prime
    # For prime at index i, the residues are 0..p-1
    # We need efficient count updates

    def calc_L(counts):
        """Find the largest L such that every integer 1..L has count >= 2."""
        if counts[1] < 2:
            return 0
        if np.all(counts[1:] >= 2):
            return M
        mask = ~(counts[1:] >= 2)
        first_false_idx = np.argmax(mask)
        return first_false_idx

    def build_counts(residues):
        counts = np.zeros(M + 1, dtype=np.int64)
        for i, p in enumerate(primes):
            r = residues[i]
            counts[r::p] += 1
        return counts

    # Greedy initialization: try to cover as much as possible
    def greedy_init():
        residues = np.zeros(n_primes, dtype=np.int64)
        counts = np.zeros(M + 1, dtype=np.int64)
        
        # Process primes in order, choosing residue that maximizes coverage
        for i, p in enumerate(primes):
            best_r = 0
            best_score = -1
            # Try a subset of residues for speed
            candidates = list(range(min(p, 20)))
            random.shuffle(candidates)
            for r in candidates:
                # Temporarily add this residue
                counts[r::p] += 1
                L = calc_L(counts)
                counts[r::p] -= 1
                if L > best_score:
                    best_score = L
                    best_r = r
            residues[i] = best_r
            counts[best_r::p] += 1
        
        return residues, counts

    # Conflict-driven repair: find positions with count < 2 and fix them
    def conflict_driven_repair(residues, counts):
        """Try to fix positions with count < 2 by changing residues."""
        improved = True
        max_iters = 200
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            # Find positions with count < 2
            deficient = np.where(counts[1:] < 2)[0]
            if len(deficient) == 0:
                break
            
            # Pick a random deficient position
            pos = deficient[random.randint(0, len(deficient) - 1)] + 1
            current_count = counts[pos]
            
            # Find primes that could cover this position
            # For prime p, the residue that covers pos is pos % p
            # Try changing some prime's residue to cover this position
            for i in random.sample(range(n_primes), min(n_primes, 30)):
                p = primes[i]
                r_needed = pos % p
                r_old = residues[i]
                if r_needed == r_old:
                    continue
                
                # Try this change
                counts[r_old::p] -= 1
                counts[r_needed::p] += 1
                
                new_L = calc_L(counts)
                old_L = calc_L(counts)  # This is wrong, need to track
                
                # Revert and check properly
                counts[r_needed::p] -= 1
                counts[r_old::p] += 1
                
                # Actually, let's just accept if it helps
                counts[r_old::p] -= 1
                counts[r_needed::p] += 1
                new_L = calc_L(counts)
                old_L = calc_L(counts)
                
                # This is getting complicated. Let's simplify.
                counts[r_needed::p] -= 1
                counts[r_old::p] += 1
                
                # Just do a simple hill climb on deficient positions
                break
        
        return residues, counts

    # Better: simple local search
    def local_search(residues, counts, max_time=30.0):
        """Hill climbing on single residue changes."""
        end_time = start_time + max_time
        current_L = calc_L(counts)
        best_L = current_L
        best_residues = residues.copy()
        
        while time.time() < end_time:
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            r_old = residues[i]
            r_new = random.randint(0, p - 1)
            if r_new == r_old:
                continue
            
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
                counts[r_new::p] -= 1
                counts[r_old::p] += 1
        
        return best_residues, build_counts(best_residues)

    # Simulated annealing with better schedule
    def simulated_annealing(residues, counts, max_time=60.0):
        end_time = start_time + max_time
        current_L = calc_L(counts)
        best_L = current_L
        best_residues = residues.copy()
        
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.9995
        
        while time.time() < end_time:
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            r_old = residues[i]
            r_new = random.randint(0, p - 1)
            if r_new == r_old:
                continue
            
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
                delta = current_L - new_L
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_L = new_L
                    residues[i] = r_new
                else:
                    counts[r_new::p] -= 1
                    counts[r_old::p] += 1
            
            T = max(T_min, T * cooling_rate)
        
        return best_residues, build_counts(best_residues)

    # Polish pass: fine-tune with smaller perturbations
    def polish_pass(residues, counts, max_time=15.0):
        end_time = start_time + max_time
        current_L = calc_L(counts)
        best_L = current_L
        best_residues = residues.copy()
        
        while time.time() < end_time:
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            r_old = residues[i]
            # Try nearby residues
            offsets = [-2, -1, 1, 2]
            random.shuffle(offsets)
            improved = False
            for offset in offsets:
                r_new = (r_old + offset) % p
                if r_new == r_old:
                    continue
                
                counts[r_old::p] -= 1
                counts[r_new::p] += 1
                new_L = calc_L(counts)
                
                if new_L > current_L:
                    current_L = new_L
                    residues[i] = r_new
                    r_old = r_new
                    improved = True
                    if new_L > best_L:
                        best_L = new_L
                        best_residues = residues.copy()
                else:
                    counts[r_new::p] -= 1
                    counts[r_old::p] += 1
                    break
            if not improved:
                continue
        
        return best_residues, build_counts(best_residues)

    # Main algorithm
    # Phase 1: Greedy initialization
    residues, counts = greedy_init()
    current_L = calc_L(counts)
    best_L = current_L
    best_residues = residues.copy()

    # Phase 2: Local search
    if time.time() < deadline - 100:
        residues, counts = local_search(residues, counts, max_time=25.0)
        current_L = calc_L(counts)
        if current_L > best_L:
            best_L = current_L
            best_residues = residues.copy()

    # Phase 3: Simulated annealing
    if time.time() < deadline - 60:
        residues, counts = simulated_annealing(residues, counts, max_time=40.0)
        current_L = calc_L(counts)
        if current_L > best_L:
            best_L = current_L
            best_residues = residues.copy()

    # Phase 4: Polish pass
    if time.time() < deadline - 20:
        residues, counts = polish_pass(residues, counts, max_time=10.0)
        current_L = calc_L(counts)
        if current_L > best_L:
            best_L = current_L
            best_residues = residues.copy()

    # Final check
    if time.time() < deadline - 10:
        # One more quick local search
        residues, counts = local_search(residues, counts, max_time=5.0)
        current_L = calc_L(counts)
        if current_L > best_L:
            best_L = current_L
            best_residues = residues.copy()

    return [int(r) for r in best_residues]
