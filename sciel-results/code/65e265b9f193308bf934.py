# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

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

    # Evaluation bound: we want to maximize L, cap at 10000 for safety/speed
    # Actually, the task says L >= 1000 is the target. Let's evaluate up to 10000 to be safe.
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

    # Initial best
    best_L = 0
    best_residues = [0] * n_primes

    # --- NEW STRATEGY: STRUCTURED CONSTRUCTIVE APPROACH + LOCAL SEARCH ---
    
    # Idea: The problem is to cover [1, L] with at least 2 congruences.
    # This is related to "covering systems".
    # A simple constructive heuristic:
    # 1. Assign residues for small primes to cover the initial segment densely.
    # 2. For each prime p, we want to choose r_p such that it helps cover gaps.
    
    # Let's try a different initialization:
    # Instead of random, let's try to maximize coverage for the first, say, 500 numbers.
    # We can do this greedily or by local search on a smaller scale.
    
    # However, the previous program already does local search.
    # The inspiration suggests that the bottleneck is evaluation.
    # Let's try to make the local search more effective by:
    # 1. Using a better initialization (e.g., all zeros, or specific patterns).
    # 2. Using "pairwise swaps" or "multi-move" steps to escape local minima.
    # 3. Focusing the search on the "frontier" (the first uncovered point).
    
    # Let's implement a "Greedy Frontier" approach:
    # Start with all residues 0.
    # Find the first gap L+1.
    # For each prime p, check if changing r_p to (L+1) % p improves the score.
    # If multiple primes can cover it, pick the one that gives the best improvement.
    # Repeat until no improvement.
    
    # This is similar to the existing code but let's refine the "improvement" check.
    # The existing code checks `new_L > current_L`.
    # We can also check if `new_L == current_L` but the "next gap" is further? No, L is the metric.
    
    # Let's try a different angle: **Simulated Annealing with a better schedule**.
    # The current code uses random restarts and greedy repair.
    # Let's try pure Simulated Annealing with a cooling schedule.
    
    # BUT, the inspiration says "The previous program failed because get_L() was called inside the main loop...".
    # Wait, the current program DOES call evaluate() inside the loop.
    # `evaluate` calls `compute_counts` which is O(N * P) roughly.
    # N=10000, P=168. 10000 * 168 = 1.68M operations per eval.
    # If we do 1000 evals, that's 1.68B ops. Might be slow in Python/NumPy overhead.
    # NumPy vectorization helps, but `np.arange` in a loop is not fully vectorized across primes.
    
    # Let's optimize `compute_counts` to be faster.
    # Instead of looping over primes and doing `np.arange`, we can use a difference array or similar?
    # No, the intervals are periodic.
    # `counts[idx] += 1` is the bottleneck.
    
    # Alternative: Pre-compute the indices for each prime?
    # `indices[p] = np.arange(r, EVAL_BOUND+1, p)`
    # But r changes.
    
    # Let's stick to the current evaluation but optimize the search strategy.
    # The key insight from "inspiration 3" is that small primes are more important.
    # Let's weight the random selection of primes in the repair step by 1/p or similar.
    
    # Also, let's try **Multi-Start Local Search** with **Diversity**.
    
    random.seed(42)
    
    # Helper: Try to improve a residue set
    def local_search(current, current_L):
        improved = True
        steps = 0
        max_steps = 5000
        while improved and steps < max_steps and time.time() < deadline - 5.0:
            improved = False
            steps += 1
            gap_pos = current_L + 1
            if gap_pos > EVAL_BOUND:
                break
            
            # Find candidates that cover gap_pos
            # We want to change a prime p such that new_r = gap_pos % p
            # And current[i] != new_r (otherwise no change)
            
            # To be efficient, let's try primes in order of size (smaller primes first? or larger?)
            # Smaller primes have more "leverage" in the first 1000 numbers.
            # Let's try all primes, but prioritize smaller ones in the random shuffle?
            
            candidates = []
            for i, p in enumerate(primes_list):
                new_r = gap_pos % p
                if current[i] != new_r:
                    candidates.append((i, new_r))
            
            if not candidates:
                break
            
            # Shuffle, but put smaller primes first?
            # Let's just shuffle.
            random.shuffle(candidates)
            
            best_candidate = None
            best_new_L = current_L
            
            # Try top K candidates
            K = 50
            for i, new_r in candidates[:K]:
                old_r = current[i]
                current[i] = new_r
                new_L = evaluate(current)
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_candidate = (i, new_r)
                current[i] = old_r
            
            if best_candidate:
                i, new_r = best_candidate
                current[i] = new_r
                current_L = best_new_L
                improved = True
                
        return current, current_L

    # Multi-start
    for restart in range(100):
        if time.time() > deadline - 10.0:
            break
        
        # Different initialization strategies
        if restart == 0:
            current = [0] * n_primes
        elif restart == 1:
            current = [random.randint(0, p - 1) for p in primes_list]
        elif restart % 2 == 0:
            # Perturb best
            current = best_residues[:]
            for _ in range(n_primes // 5):
                idx = random.randint(0, n_primes - 1)
                p = primes_list[idx]
                current[idx] = random.randint(0, p - 1)
        else:
            # Random
            current = [random.randint(0, p - 1) for p in primes_list]
        
        current_L = evaluate(current)
        
        if current_L > best_L:
            best_L = current_L
            best_residues = current[:]
        
        # Local search
        current, current_L = local_search(current, current_L)
        
        if current_L > best_L:
            best_L = current_L
            best_residues = current[:]
            
        if best_L >= 1000:
            break

    return [int(r) for r in best_residues]
# EVOLVE-BLOCK-END
