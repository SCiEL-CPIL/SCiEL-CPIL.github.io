"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _sieve_primes(limit):
    """Return list of primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _get_L(counts, max_check=100000):
    """Find the largest L such that counts[1..L] >= 2."""
    below_two = np.where(counts[1:max_check+1] < 2)[0]
    if len(below_two) == 0:
        return max_check
    return int(below_two[0])

def _compute_counts(residues, primes, max_check=100000):
    """Compute coverage counts for all integers 1..max_check."""
    counts = np.zeros(max_check + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
    return counts

def _greedy_construct(primes, order, max_check=100000):
    """
    Greedy set-cover heuristic.
    For each prime in 'order', pick residue to cover the first gap.
    """
    counts = np.zeros(max_check + 1, dtype=np.int32)
    residues = np.zeros(len(primes), dtype=np.int32)
    current_L = 0
    
    for p in order:
        if current_L < max_check:
            start_search = current_L + 1
            if start_search > max_check:
                g1 = max_check + 1
            else:
                segment = counts[start_search:]
                below_two = np.where(segment < 2)[0]
                if len(below_two) == 0:
                    g1 = max_check + 1
                else:
                    g1 = start_search + int(below_two[0])
        else:
            g1 = max_check + 1
            
        if g1 > max_check:
            r = 0
        else:
            r = g1 % p
        
        idx = primes.index(p)
        residues[idx] = r
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
            
        new_L = _get_L(counts, max_check)
        current_L = new_L
        
    return residues, current_L

def _targeted_gap_fix(residues, primes, counts, current_L, max_check=100000, deadline=None):
    """
    Targeted gap-fixing: identify uncovered positions and find primes whose
    residue changes can cover multiple gaps simultaneously using CRT-inspired logic.
    
    Key insight: Instead of random hill-climbing, systematically identify the
    first few uncovered positions and for each prime, compute which residue
    would cover the most uncovered positions. Apply the best improvement.
    """
    n_primes = len(primes)
    improved = True
    passes = 0
    max_passes = 50
    
    while improved and passes < max_passes:
        if deadline and time.time() > deadline:
            return residues, counts, current_L
        passes += 1
        improved = False
        
        # Find the first K uncovered positions
        if current_L >= max_check:
            break
            
        start_search = current_L + 1
        segment = counts[start_search:start_search + min(1000, max_check - start_search + 1)]
        below_two = np.where(segment < 2)[0]
        
        if len(below_two) == 0:
            break
            
        # Get up to 20 uncovered positions to work with
        uncovered_positions = [start_search + int(x) for x in below_two[:20]]
        
        # For each prime, find the best residue that covers the most uncovered positions
        best_improvement = None
        best_new_L = current_L
        
        for i in range(n_primes):
            if deadline and time.time() > deadline:
                return residues, counts, current_L
                
            p = primes[i]
            current_r = residues[i]
            
            # For each uncovered position, compute what residue p would need
            # to cover it: r_needed = pos % p
            # But we want to find a single residue that covers multiple positions
            
            # Group uncovered positions by their required residue mod p
            residue_counts = {}
            for pos in uncovered_positions:
                r_needed = pos % p
                residue_counts[r_needed] = residue_counts.get(r_needed, 0) + 1
            
            # Also consider residues that are "close" to current_r (less disruption)
            # and residues that cover the first gap
            candidate_residues = set(residue_counts.keys())
            
            # Add current_r and nearby residues
            for delta in range(-5, 6):
                candidate_residues.add((current_r + delta) % p)
            
            # For each candidate, estimate how many uncovered positions it covers
            best_r_for_prime = current_r
            best_coverage = -1
            
            for r in candidate_residues:
                if r == current_r:
                    continue
                    
                # Count how many uncovered positions this residue covers
                coverage = 0
                for pos in uncovered_positions:
                    if pos % p == r:
                        coverage += 1
                
                # Bonus for covering the first uncovered position
                if uncovered_positions and uncovered_positions[0] % p == r:
                    coverage += 10  # Heavy bonus for fixing the first gap
                
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_r_for_prime = r
            
            # If we found a better residue, evaluate it
            if best_r_for_prime != current_r and best_coverage > 0:
                # Apply the change temporarily
                old_start = current_r if current_r > 0 else p
                new_start = best_r_for_prime if best_r_for_prime > 0 else p
                
                if old_start <= max_check:
                    counts[old_start::p] -= 1
                if new_start <= max_check:
                    counts[new_start::p] += 1
                    
                new_L = _get_L(counts, max_check)
                
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_improvement = (i, best_r_for_prime)
                else:
                    # Revert
                    if new_start <= max_check:
                        counts[new_start::p] -= 1
                    if old_start <= max_check:
                        counts[old_start::p] += 1
            else:
                # No change for this prime
                pass
        
        # Apply the best improvement found
        if best_improvement is not None:
            i, new_r = best_improvement
            p = primes[i]
            old_r = residues[i]
            
            old_start = old_r if old_r > 0 else p
            new_start = new_r if new_r > 0 else p
            
            if old_start <= max_check:
                counts[old_start::p] -= 1
            if new_start <= max_check:
                counts[new_start::p] += 1
                
            residues[i] = new_r
            current_L = best_new_L
            improved = True
        else:
            # No improvement found; try a more aggressive random perturbation
            if passes % 5 == 0:
                # Pick a random prime and try a random residue
                i = np.random.randint(0, n_primes)
                p = primes[i]
                old_r = residues[i]
                new_r = np.random.randint(0, p)
                
                if new_r != old_r:
                    old_start = old_r if old_r > 0 else p
                    new_start = new_r if new_r > 0 else p
                    
                    if old_start <= max_check:
                        counts[old_start::p] -= 1
                    if new_start <= max_check:
                        counts[new_start::p] += 1
                        
                    new_L = _get_L(counts, max_check)
                    
                    if new_L > current_L:
                        residues[i] = new_r
                        current_L = new_L
                        improved = True
                    else:
                        # Revert
                        if new_start <= max_check:
                            counts[new_start::p] -= 1
                        if old_start <= max_check:
                            counts[old_start::p] += 1
        
    return residues, counts, current_L

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    max_check = 100000
    
    best_residues = None
    best_score = 0
    
    # Try multiple different greedy orders
    orders = [
        primes,  # Increasing
        primes[::-1],  # Decreasing
        list(np.random.permutation(primes)),
        sorted(primes, key=lambda p: -np.log(p)),  # Decreasing log
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
            
        residues, L = _greedy_construct(primes, order, max_check)
        
        # Refine with targeted gap fixing
        counts = _compute_counts(residues, primes, max_check)
        
        residues, counts, L = _targeted_gap_fix(residues, primes, counts, L, max_check, deadline)
        
        if L > best_score:
            best_score = L
            best_residues = residues.copy()
            
    if best_residues is None:
        # Fallback
        best_residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int32)
        
    return best_residues.tolist()
# EVOLVE-BLOCK-END
