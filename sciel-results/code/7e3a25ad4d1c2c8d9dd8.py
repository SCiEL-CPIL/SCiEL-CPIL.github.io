import time
import numpy as np

N = 5000
_DEADLINE = time.time() + 320.0

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, ok in enumerate(sieve) if ok]

def construct():
    # Start with all integers in [0, N]
    points = np.arange(N + 1, dtype=np.int64)
    primes = _primes(N)
    
    while True:
        if time.time() > _DEADLINE:
            return points.tolist()
            
        k = len(points)
        if k == 0:
            return points.tolist()
            
        # Find the smallest prime p <= k such that points covers all residues mod p
        # If no such prime exists, points is admissible.
        removed = False
        for p in primes:
            if p > k:
                break
                
            # Count residues using numpy
            residues = points % p
            counts = np.bincount(residues, minlength=p)
            
            min_count = counts.min()
            if min_count > 0:
                # All residues are present. We must remove one residue class.
                # Look-ahead: for each candidate residue to remove, estimate
                # how many future primes will still be fully covered.
                # Choose the residue whose removal minimizes the number of
                # "conflicting" primes (primes that would still need handling).
                
                # Quick heuristic: among residues with minimum count,
                # pick the one that removes elements that appear in more
                # other residue classes for small primes.
                # Simpler effective heuristic: pick min-count residue,
                # but if tie, prefer the residue that is 0 mod p (boundary effect)
                # or use a weighted score.
                
                min_indices = np.where(counts == min_count)[0]
                if len(min_indices) == 1:
                    target_residue = min_indices[0]
                else:
                    # Among tied min-count residues, prefer the one whose
                    # removal leaves the most "spread" for next primes.
                    # Heuristic: remove the residue closest to the edge (0 or p-1)
                    # as it tends to be less "central" in future mod operations.
                    # Actually, a better heuristic: simulate one more step.
                    # For each candidate, count how many of the next few primes
                    # would still have all residues covered after removal.
                    
                    next_primes = [q for q in primes if q > p and q <= k]
                    best_residue = min_indices[0]
                    best_score = float('inf')
                    
                    for cand in min_indices:
                        # Simulate removal of residue cand mod p
                        mask = residues != cand
                        new_points = points[mask]
                        if len(new_points) == 0:
                            score = 0
                        else:
                            # Count how many of next_primes still fully cover
                            covered = 0
                            for q in next_primes[:3]:  # limit to 3 for speed
                                if q > len(new_points):
                                    break
                                new_residues = new_points % q
                                new_counts = np.bincount(new_residues, minlength=q)
                                if new_counts.min() > 0:
                                    covered += 1
                            score = covered
                        if score < best_score:
                            best_score = score
                            best_residue = cand
                    
                    target_residue = best_residue
                
                points = points[points % p != target_residue]
                removed = True
                break
                
        if not removed:
            return points.tolist()
