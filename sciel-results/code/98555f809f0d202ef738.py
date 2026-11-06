import time
import math
import random
import numpy as np
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 145  # Safety margin for 150s limit

    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.sqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(1000)
    
    L_MAX = 10000  # Search space cap for speed
    
    # Bitset for coverage: use numpy array of uint8 for speed
    # covered[n] = 1 if n is covered, 0 otherwise
    covered = np.zeros(L_MAX + 1, dtype=np.uint8)
    
    # Track which primes are used and their residues
    used_primes = {}  # p -> r
    
    # Precompute: for each prime p, the list of positions covered by residue r
    # We'll compute this on-the-fly for speed
    
    def get_L(covered_arr):
        """Find largest L such that all 1..L are covered."""
        # Find first uncovered position
        uncovered = np.where(covered_arr[1:] == 0)[0]
        if len(uncovered) == 0:
            return L_MAX
        return uncovered[0]
    
    def add_class(p, r):
        """Add prime p with residue r to coverage."""
        if r == 0:
            positions = np.arange(p, L_MAX + 1, p)
        else:
            positions = np.arange(r, L_MAX + 1, p)
        if len(positions) > 0:
            covered[positions] = 1
        used_primes[p] = r
    
    def remove_class(p, r):
        """Remove prime p with residue r from coverage."""
        if r == 0:
            positions = np.arange(p, L_MAX + 1, p)
        else:
            positions = np.arange(r, L_MAX + 1, p)
        if len(positions) > 0:
            # Only set to 0 if no other class covers it
            # For simplicity, we'll recompute coverage after removal
            pass
        del used_primes[p]
    
    def recompute_coverage():
        """Recompute coverage from scratch based on used_primes."""
        covered[:] = 0
        for p, r in used_primes.items():
            if r == 0:
                positions = np.arange(p, L_MAX + 1, p)
            else:
                positions = np.arange(r, L_MAX + 1, p)
            if len(positions) > 0:
                covered[positions] = 1
    
    # Greedy set-cover: repeatedly pick the class that covers the most new positions
    best_L = 0
    best_used = {}
    
    # Available primes: start with all primes up to 1000
    # But we need sum(1/p) <= 1, so we'll track cost
    available_primes = set(primes)
    current_cost = Fraction(0, 1)
    
    # Phase 1: Greedy construction
    while time.time() < deadline - 30:
        current_L = get_L(covered)
        if current_L >= L_MAX:
            break
        
        # Find the next uncovered position
        uncovered_positions = np.where(covered[1:] == 0)[0]
        if len(uncovered_positions) == 0:
            break
        next_uncovered = uncovered_positions[0] + 1
        
        # For each available prime, find the best residue that covers next_uncovered
        # and maximizes additional coverage
        best_gain = -1
        best_p = None
        best_r = None
        
        for p in list(available_primes):
            # Check if adding this prime would exceed cost budget
            if current_cost + Fraction(1, p) > 1:
                continue
            
            # The residue that covers next_uncovered is r = next_uncovered % p
            r = next_uncovered % p
            
            # Compute how many new positions this would cover
            if r == 0:
                positions = np.arange(p, L_MAX + 1, p)
            else:
                positions = np.arange(r, L_MAX + 1, p)
            
            if len(positions) == 0:
                continue
            
            # Count positions not currently covered
            new_positions = positions[covered[positions] == 0]
            gain = len(new_positions)
            
            if gain > best_gain:
                best_gain = gain
                best_p = p
                best_r = r
        
        if best_p is None:
            break
        
        # Add the best class
        add_class(best_p, best_r)
        available_primes.remove(best_p)
        current_cost += Fraction(1, best_p)
        
        # Update best
        current_L = get_L(covered)
        if current_L > best_L:
            best_L = current_L
            best_used = dict(used_primes)
    
    # Phase 2: Local search refinement
    # Try to improve by changing residues of used primes
    if time.time() < deadline - 10:
        T = 10.0
        T_min = 0.1
        cooling = 0.999
        
        while time.time() < deadline - 5:
            # Pick a random used prime
            if not used_primes:
                break
            p = random.choice(list(used_primes.keys()))
            r_old = used_primes[p]
            
            # Generate new residue
            r_new = random.randint(0, p - 1)
            if r_new == r_old:
                continue
            
            # Temporarily change and measure
            used_primes[p] = r_new
            recompute_coverage()
            new_L = get_L(covered)
            
            # Acceptance
            if new_L > best_L:
                best_L = new_L
                best_used = dict(used_primes)
            else:
                delta = new_L - get_L(covered)  # This is 0 since we just computed it
                # Actually, we need to compare to current state
                current_L = get_L(covered)
                if new_L >= current_L:
                    # Accept
                    pass
                else:
                    # Reject with some probability
                    prob = math.exp((new_L - current_L) / T)
                    if random.random() < prob:
                        pass  # Accept
                    else:
                        # Reject: revert
                        used_primes[p] = r_old
                        recompute_coverage()
            
            T *= cooling
            if T < T_min:
                T = T_min
    
    # Return best result
    if not best_used:
        # Fallback
        primes_seed = [p for p in primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]
    
    return [[p, r] for p, r in sorted(best_used.items())]

