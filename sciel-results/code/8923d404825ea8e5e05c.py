# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import math

def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin (150s limit, leave buffer)

    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    n_primes = len(primes)
    primes_list = primes
    primes_arr = np.array(primes_list, dtype=np.int64)

    # Evaluation bound
    EVAL_BOUND = 10000

    def compute_counts(residues):
        """Compute coverage counts for positions 1..EVAL_BOUND."""
        counts = np.zeros(EVAL_BOUND + 1, dtype=np.int32)
        for i, p in enumerate(primes_list):
            r = residues[i]
            if r == 0:
                first = p
            else:
                first = r
            if first >= 1 and first <= EVAL_BOUND:
                idx = np.arange(first, EVAL_BOUND + 1, p)
                counts[idx] += 1
        return counts

    def get_L_from_counts(counts):
        """Find largest L such that counts[1..L] >= 2."""
        below = np.where(counts[1:] < 2)[0]
        if len(below) == 0:
            return EVAL_BOUND
        return int(below[0])

    def evaluate(residues):
        counts = compute_counts(residues)
        return get_L_from_counts(counts)

    def evaluate_local(residues, L_target):
        """Evaluate only up to L_target for speed when L is small."""
        bound = min(L_target + 100, EVAL_BOUND)
        counts = np.zeros(bound + 1, dtype=np.int32)
        for i, p in enumerate(primes_list):
            r = residues[i]
            if r == 0:
                first = p
            else:
                first = r
            if first >= 1 and first <= bound:
                idx = np.arange(first, bound + 1, p)
                counts[idx] += 1
        below = np.where(counts[1:] < 2)[0]
        if len(below) == 0:
            return bound
        return int(below[0])

    # Initial best
    best_L = 0
    best_residues = [0] * n_primes

    random.seed(42)

    def greedy_fill_gaps(current, current_L):
        """
        Priority-based gap filling: repeatedly find the first gap,
        and choose the prime whose change causes the least damage to nearby positions.
        """
        max_steps = 5000
        steps = 0
        L = current_L
        
        while L < EVAL_BOUND and steps < max_steps:
            if time.time() > deadline - 2.0:
                break
            
            gap_pos = L + 1
            if gap_pos > EVAL_BOUND:
                break
            
            # Find all primes that do NOT cover gap_pos
            candidates = []
            for i, p in enumerate(primes_list):
                if current[i] != gap_pos % p:
                    candidates.append(i)
            
            if not candidates:
                break
            
            # For each candidate, compute how many positions in [L+1, L+50] 
            # would lose coverage if we change this prime's residue
            # We want to pick the one that minimizes damage
            best_candidate = None
            best_damage = float('inf')
            best_new_L = -1
            
            # Sample a subset of candidates for speed
            if len(candidates) > 30:
                sampled = random.sample(candidates, 30)
            else:
                sampled = candidates
            
            for i in sampled:
                p = primes_list[i]
                old_r = current[i]
                new_r = gap_pos % p
                
                if new_r == old_r:
                    continue
                
                # Simpler: just try the change and evaluate locally
                current[i] = new_r
                new_L = evaluate_local(current, L)
                
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_candidate = i
                    best_damage = 0  # We'll keep this one
                
                current[i] = old_r
            
            if best_candidate is None:
                break
            
            # Apply the best change
            p = primes_list[best_candidate]
            current[best_candidate] = gap_pos % p
            L = evaluate_local(current, L)
            
            if L > current_L:
                current_L = L
                steps += 1
            else:
                # No improvement, break
                break
        
        return current, current_L

    def simulated_annealing(current, current_L, max_iterations=2000):
        """
        Simulated annealing to escape local minima.
        Accepts worse solutions with probability exp(-delta/T) to explore the search space.
        """
        T = 10.0  # Initial temperature
        T_min = 0.1
        cooling_rate = 0.995
        
        best_in_sa = current_L
        best_in_sa_residues = current[:]
        current_SA = current[:]
        current_SA_L = current_L
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 3.0:
                break
            
            # Choose a random prime to modify
            idx = random.randint(0, n_primes - 1)
            p = primes_list[idx]
            old_r = current_SA[idx]
            
            # Choose a new random residue
            if random.random() < 0.3:
                # Targeted: try to cover a gap
                gap_pos = current_SA_L + 1
                new_r = gap_pos % p
            else:
                # Random perturbation
                new_r = random.randint(0, p - 1)
            
            if new_r == old_r:
                continue
            
            current_SA[idx] = new_r
            new_L = evaluate(current_SA)
            
            # Accept or reject
            if new_L > current_SA_L:
                # Always accept improvement
                current_SA_L = new_L
                if new_L > best_in_sa:
                    best_in_sa = new_L
                    best_in_sa_residues = current_SA[:]
            else:
                # Accept worse solution with probability exp(-delta/T)
                delta = current_SA_L - new_L
                if delta == 0:
                    accept = True
                else:
                    prob = math.exp(-delta / T)
                    accept = random.random() < prob
                
                if accept:
                    current_SA_L = new_L
                else:
                    current_SA[idx] = old_r
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        return best_in_sa_residues, best_in_sa

    # Phase 1: Simulated annealing with greedy refinement
    while time.time() < deadline - 15.0:
        # Initialize current residues
        if best_L > 0 and random.random() < 0.5:
            current = best_residues[:]
            # Perturb 10-20% of residues
            n_perturb = n_primes // 10
            for _ in range(n_perturb):
                idx = random.randint(0, n_primes - 1)
                p = primes_list[idx]
                current[idx] = random.randint(0, p - 1)
        else:
            current = [random.randint(0, p - 1) for p in primes_list]
        
        current_L = evaluate(current)
        
        if current_L > best_L:
            best_L = current_L
            best_residues = current[:]
        
        # Run simulated annealing
        if time.time() < deadline - 8.0:
            current, current_L = simulated_annealing(current, current_L, max_iterations=1500)
            
            if current_L > best_L:
                best_L = current_L
                best_residues = current[:]
        
        # Greedy gap filling
        if time.time() < deadline - 5.0:
            current, current_L = greedy_fill_gaps(current, current_L)
            
            if current_L > best_L:
                best_L = current_L
                best_residues = current[:]
        
        if best_L >= 1000:
            break

    # Phase 2: Local search on best solution
    if time.time() < deadline - 3.0 and best_L < EVAL_BOUND:
        current = best_residues[:]
        current_L = best_L
        no_improve_count = 0
        max_no_improve = 50
        
        while current_L < EVAL_BOUND and no_improve_count < max_no_improve:
            if time.time() > deadline - 2.0:
                break
            
            gap_pos = current_L + 1
            if gap_pos > EVAL_BOUND:
                break
            
            candidates = []
            for i, p in enumerate(primes_list):
                if current[i] != gap_pos % p:
                    candidates.append(i)
            
            if not candidates:
                break
            
            improved = False
            random.shuffle(candidates)
            
            for i in candidates[:40]:
                if time.time() > deadline - 2.0:
                    break
                
                p = primes_list[i]
                old_r = current[i]
                new_r = gap_pos % p
                
                if new_r == old_r:
                    continue
                
                current[i] = new_r
                new_L = evaluate(current)
                
                if new_L > current_L:
                    current_L = new_L
                    improved = True
                    no_improve_count = 0
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = current[:]
                    break
                else:
                    current[i] = old_r
            
            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0

    # Ensure Python integers
    return [int(r) for r in best_residues]
# EVOLVE-BLOCK-END
