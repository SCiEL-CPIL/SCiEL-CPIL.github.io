"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def _greedy_construct(all_primes, N, seed=0):
    random.seed(seed)
    points = list(range(N + 1))
    
    primes_list = [p for p in all_primes if p <= N+1]
    
    best_points = []
    best_len = -1
    
    start_time = time.time()
    deadline = start_time + 280
    
    # Run multiple greedy restarts
    num_restarts = 20 # Adjust based on time
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Randomize order
        ps = primes_list[:]
        random.shuffle(ps)
        
        pts = list(range(N + 1))
        
        # Greedy Deletion
        for p in ps:
            if p > len(pts):
                continue
                
            # Check if fully covered
            residues = np.array([pt % p for pt in pts])
            counts = np.bincount(residues, minlength=p)
            
            if counts.min() > 0:
                # Fully covered
                # Choose class to remove: least populated
                min_class = np.argmin(counts)
                
                # Remove
                mask = (np.array(pts) % p) != min_class
                pts = [pt for pt, m in zip(pts, mask) if m]
        
        # Re-addition Phase
        # Current set is pts
        # Removed set is set(range(N+1)) - set(pts)
        # This is slow to compute. Better to track removed during greedy?
        # Or just compute set difference once.
        # set(range(N+1)) is large.
        # Let's just generate candidates from the complement.
        # Actually, we can just iterate through all points in [0, N] and check if they are in pts?
        # No, we need to add points NOT in pts.
        
        # Efficient way:
        # present = set(pts)
        # candidates = [x for x in range(N+1) if x not in present]
        
        # This is O(N). N=5000. Fast.
        
        present_set = set(pts)
        candidates = [x for x in range(N + 1) if x not in present_set]
        random.shuffle(candidates)
        
        # Track missing residues for primes <= len(pts)
        L = len(pts)
        # Initialize M
        M = {}
        tracked_primes = []
        
        # We need to initialize M for all primes <= L
        # This is expensive if done naively.
        # But we only do it once per restart.
        
        # Optimization: Only track primes that are "critical"?
        # No, we need to ensure we don't break admissibility.
        
        # Let's compute M for all primes <= L
        # This is O(pi(L) * L) roughly?
        # For each p, we need residues of pts mod p.
        # Sum of L for p <= L is L * pi(L) ~ 2000 * 300 = 600,000. Fast.
        
        for p in primes_list:
            if p > L:
                break
            residues = set(pt % p for pt in pts)
            M[p] = set(range(p)) - residues
            tracked_primes.append(p)
        
        # Re-add loop
        for x in candidates:
            # Check if x can be added
            safe = True
            
            # Check existing tracked primes
            for p in tracked_primes:
                if M[p] == {x % p}:
                    safe = False
                    break
            
            if not safe:
                continue
                
            new_L = L + 1
            
            # Check if new_L is prime and adding x breaks it
            if _is_prime(new_L):
                # Check if pts U {x} covers all residues mod new_L
                # pts has size L.
                # We need to know if x fills the last missing residue.
                # i.e., is x % new_L the ONLY residue missing in pts?
                # Compute missing residues for new_L in pts
                residues_in_pts = set(pt % new_L for pt in pts)
                if len(residues_in_pts) == new_L - 1:
                    missing = set(range(new_L)) - residues_in_pts
                    if x % new_L in missing:
                        safe = False
            
            if safe:
                # Add x
                pts.append(x)
                L = new_L
                present_set.add(x)
                
                # Update M for existing tracked primes
                for p in tracked_primes:
                    r = x % p
                    if r in M[p]:
                        M[p].remove(r)
                        # M[p] should not become empty if logic is correct
                        # But if it does, we have a problem.
                        # However, we checked M[p] != {r}.
                        # So if M[p] was {r, ...}, it becomes {...}.
                        # If M[p] was {}, it stays {}.
                        # But M[p] is never {} for tracked primes in an admissible set.
                        pass
                
                # If new_L is prime, add to tracked
                if _is_prime(new_L):
                    # Compute M[new_L]
                    residues = set(pt % new_L for pt in pts)
                    M[new_L] = set(range(new_L)) - residues
                    tracked_primes.append(new_L)
                    
        # Update best
        if len(pts) > best_len:
            best_len = len(pts)
            best_points = pts[:]
            
    if not best_points:
        best_points = list(range(N+1))
        
    # Final verification and repair if needed
    final_primes = [p for p in all_primes if p <= len(best_points)]
    for p in final_primes:
        residues = set(pt % p for pt in best_points)
        if len(residues) == p:
            counts = np.zeros(p, dtype=int)
            for pt in best_points:
                counts[pt % p] += 1
            min_class = np.argmin(counts)
            best_points = [pt for pt in best_points if pt % p != min_class]
            
    return best_points

def construct():
    return _greedy_construct(_primes(N+1), N)
# EVOLVE-BLOCK-END
