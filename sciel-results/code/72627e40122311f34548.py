"""Combined program: vectorized greedy set-reduction with NumPy for speed."""
import time
import numpy as np

N = 5000
DEADLINE = 300.0  # seconds, safety margin below 330s kill


def _primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _is_admissible(points, all_primes, max_prime_check=None):
    """Check if points form an admissible tuple."""
    n = len(points)
    limit = max_prime_check if max_prime_check is not None else n
    for p in all_primes:
        if p > limit:
            break
        residues = points % p
        class_counts = np.bincount(residues, minlength=p)
        if np.all(class_counts > 0):
            return False
    return True


def _greedy_repair(points, all_primes, start_time, deadline):
    """Greedily remove least-populated residue classes to achieve admissibility."""
    changed = True
    while changed:
        if time.time() - start_time > deadline - 10:
            break
        changed = False
        current_len = len(points)
        
        for p in all_primes:
            if p > current_len:
                break
            
            residues = points % p
            class_counts = np.bincount(residues, minlength=p)
            
            if np.all(class_counts > 0):
                min_class = np.argmin(class_counts)
                points = points[points % p != min_class]
                changed = True
                break
    
    # Final polish pass
    current_len = len(points)
    for p in all_primes:
        if p > current_len:
            break
        residues = points % p
        class_counts = np.bincount(residues, minlength=p)
        if np.all(class_counts > 0):
            min_class = np.argmin(class_counts)
            points = points[points % p != min_class]
            current_len = len(points)
    
    return points


def _local_improve(points, all_primes, start_time, deadline):
    """Try to add back removed points or swap to increase set size while maintaining admissibility."""
    points = np.sort(points)
    n = len(points)
    # Build a boolean mask of which points are present
    present = np.zeros(N + 1, dtype=bool)
    present[points] = True
    
    # Candidates to try adding: points not in current set
    candidates = np.where(~present)[0]
    
    improved = True
    while improved:
        if time.time() - start_time > deadline - 15:
            break
        improved = False
        current_len = len(points)
        
        # Try adding each candidate (in order of smallest first for determinism)
        for cand in candidates:
            if time.time() - start_time > deadline - 15:
                break
            
            # Tentatively add cand
            new_points = np.concatenate([points, [cand]])
            new_points = np.sort(new_points)
            
            # Check admissibility
            if _is_admissible(new_points, all_primes, max_prime_check=current_len + 1):
                points = new_points
                present[cand] = True
                improved = True
                # Restart candidate loop with updated set
                candidates = np.where(~present)[0]
                break
    
    return points


def _try_swaps(points, all_primes, start_time, deadline):
    """Try swapping one point for another to maintain or increase size."""
    # This is more complex; skip for now if time is tight
    if time.time() - start_time > deadline - 20:
        return points
    
    points = np.sort(points)
    n = len(points)
    present = np.zeros(N + 1, dtype=bool)
    present[points] = True
    
    improved = True
    while improved:
        if time.time() - start_time > deadline - 20:
            break
        improved = False
        current_len = len(points)
        
        # Try removing one point and adding a different one
        # Only worth trying if we can find a net gain, but swaps keep size same
        # So this is mainly for enabling future additions
        # Skip for now - focus on additions
        pass
    
    return points


def construct():
    start_time = time.time()
    
    # Start with all integers in [0, N]
    points = np.arange(N + 1, dtype=np.int32)
    
    # Precompute primes up to N+1 (max possible len is N+1)
    all_primes = _primes(N + 1)
    
    # Phase 1: Greedy repair
    points = _greedy_repair(points, all_primes, start_time, DEADLINE)
    
    # Phase 2: Local improvement - try to add back points
    if time.time() - start_time < DEADLINE - 20:
        points = _local_improve(points, all_primes, start_time, DEADLINE)
    
    # Final verification
    result = points.tolist()
    n_final = len(result)
    for p in all_primes:
        if p > n_final:
            break
        residues = set(v % p for v in result)
        if len(residues) == p:
            counts = [0] * p
            for v in result:
                counts[v % p] += 1
            min_c = min(counts)
            min_r = counts.index(min_c)
            result = [v for v in result if v % p != min_r]
            n_final = len(result)
    
    return result
