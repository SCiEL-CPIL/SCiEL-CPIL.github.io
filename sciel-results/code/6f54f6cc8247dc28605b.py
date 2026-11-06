"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 5000

# EVOLVE-BLOCK-START
def _primes_upto(limit):
    """Return list of all primes <= limit."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]


def _check_admissible(points, all_primes):
    """Check if points is admissible for all primes p <= len(points)."""
    m = len(points)
    for p in all_primes:
        if p > m:
            break
        residues = set()
        for v in points:
            residues.add(v % p)
            if len(residues) == p:
                return False
    return True


def _greedy_repair_numpy(points, all_primes, deadline, rng=None):
    """
    Vectorized greedy repair using numpy.
    For each prime p <= len(points), if all residue classes are covered,
    remove the least-populated class (with random tie-breaking if rng provided).
    """
    points = np.array(points, dtype=np.int32)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        m = len(points)
        for p in all_primes:
            if p > m:
                break
            # Compute residue counts vectorized
            residues = points % p
            # Use bincount to get counts
            counts = np.bincount(residues, minlength=p)
            if counts.min() > 0:
                # All classes covered - violation
                if rng is not None:
                    # Random tie-breaking among minimum count classes
                    min_count = counts.min()
                    min_indices = np.where(counts == min_count)[0]
                    chosen = rng.choice(min_indices)
                else:
                    chosen = np.argmin(counts)
                # Remove all elements with residue == chosen
                mask = residues != chosen
                points = points[mask]
                changed = True
                break
    return points


def _local_improve_numpy(points, all_primes, deadline, rng=None):
    """
    Fast local improvement: try to add elements back.
    Maintain residue counts for each relevant prime.
    A candidate can be added if for no prime p <= len(points)+1 does adding it
    cause all p residue classes to be covered.
    """
    points = np.array(points, dtype=np.int32)
    points_set = set(points.tolist())
    m = len(points)
    
    # Get relevant primes
    relevant_primes = [p for p in all_primes if p <= m + 1]
    
    # Build residue count structures
    residue_counts = {}
    for p in relevant_primes:
        residues = points % p
        residue_counts[p] = np.bincount(residues, minlength=p)
    
    nonempty_counts = {p: int((residue_counts[p] > 0).sum()) for p in relevant_primes}
    
    improved = True
    candidates = list(range(N + 1))
    if rng is not None:
        rng.shuffle(candidates)
    
    while improved and time.time() < deadline:
        improved = False
        m = len(points)
        
        # Update relevant primes
        new_relevant = [p for p in all_primes if p <= m + 1]
        for p in new_relevant:
            if p not in residue_counts:
                residues = points % p
                residue_counts[p] = np.bincount(residues, minlength=p)
                nonempty_counts[p] = int((residue_counts[p] > 0).sum())
        relevant_primes = new_relevant
        
        # Try candidates
        for candidate in candidates:
            if time.time() >= deadline:
                break
            if candidate in points_set:
                continue
            
            # Check if adding candidate would violate admissibility
            admissible = True
            for p in relevant_primes:
                counts = residue_counts[p]
                ne = nonempty_counts[p]
                r = candidate % p
                if counts[r] == 0 and ne + 1 == p:
                    admissible = False
                    break
            
            if admissible:
                # Add the candidate
                points = np.append(points, candidate)
                points_set.add(candidate)
                m += 1
                for p in relevant_primes:
                    r = candidate % p
                    if residue_counts[p][r] == 0:
                        nonempty_counts[p] += 1
                    residue_counts[p][r] += 1
                
                # Add new prime if m+1 is prime
                for p in all_primes:
                    if p == m + 1:
                        residues = points % p
                        residue_counts[p] = np.bincount(residues, minlength=p)
                        nonempty_counts[p] = int((residue_counts[p] > 0).sum())
                        relevant_primes.append(p)
                        break
                improved = True
                break  # Restart candidate iteration
    
    return points


def _greedy_repair_python(points, all_primes, deadline):
    """
    Python greedy repair: repeatedly find a prime p <= len(points) whose residue classes
    are all covered, and remove the least-populated class. Continue until admissible
    or deadline.
    """
    points = list(points)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        m = len(points)
        for p in all_primes:
            if p > m:
                break
            class_counts = [0] * p
            for v in points:
                class_counts[v % p] += 1
            if all(c > 0 for c in class_counts):
                min_count = min(class_counts)
                to_remove = set()
                for v in points:
                    if class_counts[v % p] == min_count:
                        to_remove.add(v)
                points = [v for v in points if v not in to_remove]
                changed = True
                break
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s kill

    # Precompute primes up to N+1
    all_primes = _primes_upto(N + 1)

    best_points = []
    best_len = 0

    rng = np.random.default_rng(seed=12345)

    # Strategy 1: Start from all integers and repair with numpy
    points_all = np.arange(N + 1, dtype=np.int32)
    points_all = _greedy_repair_numpy(points_all, all_primes, deadline, rng)
    if len(points_all) > best_len:
        best_len = len(points_all)
        best_points = points_all.copy()

    # Strategy 2: Start from odd numbers
    if time.time() < deadline - 30:
        points_odd = np.arange(1, N + 1, 2, dtype=np.int32)
        points_odd = _greedy_repair_numpy(points_odd, all_primes, deadline, rng)
        if len(points_odd) > best_len:
            best_len = len(points_odd)
            best_points = points_odd.copy()

    # Strategy 3: Start from numbers not divisible by 2 or 3
    if time.time() < deadline - 30:
        points_coprime = np.array([v for v in range(N + 1) if v % 2 != 0 and v % 3 != 0], dtype=np.int32)
        points_coprime = _greedy_repair_numpy(points_coprime, all_primes, deadline, rng)
        if len(points_coprime) > best_len:
            best_len = len(points_coprime)
            best_points = points_coprime.copy()

    # Local improvement on best result
    if time.time() < deadline and len(best_points) > 0:
        best_points = _local_improve_numpy(best_points, all_primes, deadline, rng)
        best_len = len(best_points)

    # Try another round of local improvement with fresh randomization
    if time.time() < deadline - 20 and len(best_points) > 0:
        best_points2 = _local_improve_numpy(best_points, all_primes, deadline, rng)
        if len(best_points2) > best_len:
            best_len = len(best_points2)
            best_points = best_points2

    # Final verification and repair if needed
    result = np.sort(best_points).tolist()
    if not _check_admissible(result, all_primes):
        result = _greedy_repair_python(result, all_primes, deadline)

    # Final safety: verify and repair if still invalid
    if not _check_admissible(result, all_primes):
        # Aggressive repair: remove elements until admissible
        m = len(result)
        for p in all_primes:
            if p > m:
                break
            counts = [0] * p
            for v in result:
                counts[v % p] += 1
            if all(c > 0 for c in counts):
                min_c = min(counts)
                min_r = counts.index(min_c)
                result = [v for v in result if v % p != min_r]
                m = len(result)

    return result
# EVOLVE-BLOCK-END
