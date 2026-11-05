"""Optimized solver for the covering problem."""
import time
import random
import numpy as np

def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    
    # We'll work with a limit that's large enough
    limit = 5000
    
    def evaluate_counts(residues):
        """Compute coverage counts for all integers up to limit."""
        counts = np.zeros(limit + 2, dtype=np.int32)
        for r, p in zip(residues, primes):
            start = r if r > 0 else p
            if start <= limit:
                counts[start:limit + 1:p] += 1
        return counts
    
    def find_L(counts):
        """Find largest L such that all 1..L have count >= 2."""
        L = 0
        for i in range(1, limit + 1):
            if counts[i] >= 2:
                L = i
            else:
                break
        return L
    
    # Initial heuristic: spread residues to cover early integers
    # Use the seed approach as baseline, then improve
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    counts = evaluate_counts(residues)
    L = find_L(counts)
    
    best_L = L
    best_residues = residues[:]
    
    # Greedy extension: try to extend L by modifying one prime at a time
    while time.time() < deadline - 5:
        target = L + 1
        if target > limit:
            break
        
        # Try each prime to see if we can change its residue to cover target
        # without breaking coverage of 1..L
        found_improvement = False
        # Shuffle to break symmetry
        indices = list(range(n_primes))
        random.shuffle(indices)
        
        for idx in indices:
            p = primes[idx]
            old_r = residues[idx]
            new_r = target % p
            
            if old_r == new_r:
                continue
            
            # Determine which positions in [1, L] are affected
            old_start = old_r if old_r > 0 else p
            new_start = new_r if new_r > 0 else p
            
            # Positions covered by old_r in [1, L]
            old_positions = set()
            if old_start <= L:
                old_positions = set(range(old_start, L + 1, p))
            
            # Positions covered by new_r in [1, L]
            new_positions = set()
            if new_start <= L:
                new_positions = set(range(new_start, L + 1, p))
            
            # Check safety: for positions in old_positions but not new_positions,
            # count must be >= 3 (so removing one still leaves >= 2)
            safe = True
            for pos in old_positions:
                if pos not in new_positions:
                    if counts[pos] < 3:
                        safe = False
                        break
            
            if not safe:
                continue
            
            # Also need to ensure target gets enough coverage
            # target is covered by new_r. Check if old_r also covered target
            old_covers_target = (target % p == old_r)
            
            # After change, target count = current count + (1 if not old_covers_target else 0)
            current_target_count = counts[target]
            new_target_count = current_target_count + (0 if old_covers_target else 1)
            
            if new_target_count < 2:
                continue
            
            # Apply the change
            residues[idx] = new_r
            
            # Update counts
            for pos in old_positions:
                if pos not in new_positions:
                    counts[pos] -= 1
            for pos in new_positions:
                if pos not in old_positions:
                    counts[pos] += 1
            if not old_covers_target:
                counts[target] += 1
            
            L = target
            found_improvement = True
            break
        
        if not found_improvement:
            # Try a random perturbation to escape local minima
            if random.random() < 0.3:
                # Pick a random prime and change its residue randomly
                idx = random.randint(0, n_primes - 1)
                p = primes[idx]
                old_r = residues[idx]
                new_r = random.randint(0, p - 1)
                
                if new_r != old_r:
                    # Check if this helps
                    old_start = old_r if old_r > 0 else p
                    new_start = new_r if new_r > 0 else p
                    
                    old_positions = set()
                    if old_start <= L:
                        old_positions = set(range(old_start, L + 1, p))
                    
                    new_positions = set()
                    if new_start <= L:
                        new_positions = set(range(new_start, L + 1, p))
                    
                    safe = True
                    for pos in old_positions:
                        if pos not in new_positions:
                            if counts[pos] < 3:
                                safe = False
                                break
                    
                    if safe:
                        residues[idx] = new_r
                        for pos in old_positions:
                            if pos not in new_positions:
                                counts[pos] -= 1
                        for pos in new_positions:
                            if pos not in old_positions:
                                counts[pos] += 1
                        
                        # Re-evaluate L
                        new_L = find_L(counts)
                        if new_L > L:
                            L = new_L
                        elif new_L < L:
                            # Revert
                            residues[idx] = old_r
                            for pos in old_positions:
                                if pos not in new_positions:
                                    counts[pos] += 1
                            for pos in new_positions:
                                if pos not in old_positions:
                                    counts[pos] -= 1
            
            # If stuck, try a different initialization
            if random.random() < 0.1:
                # Try a completely different random initialization
                residues = [random.randint(0, p - 1) for p in primes]
                counts = evaluate_counts(residues)
                L = find_L(counts)
        
        if L > best_L:
            best_L = L
            best_residues = residues[:]
    
    return best_residues
