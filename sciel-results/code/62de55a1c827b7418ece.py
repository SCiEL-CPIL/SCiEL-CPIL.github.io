"""
Strategy: Simulated Annealing over joint prime selection and residues.
1. Represent the solution as a set of active primes (subset of primes <= 1000) and their residues.
2. Use NumPy for fast coverage evaluation: instead of marking a boolean array for each prime,
   we can check coverage of a range [1, L] efficiently, or better, maintain a coverage count array
   and update it incrementally? No, updating counts for a prime swap is O(L/p) which is fine, 
   but finding the first gap requires scanning.
   
   Actually, the most efficient way to find the "first uncovered" number is to maintain a boolean 
   array `covered` of size CAP+1. 
   Updating `covered` for a prime change (p, r) -> (p', r') involves:
   - Unmarking p: covered[n] = False for n in range(start, CAP+1, p) IF no other prime covers n.
     This is hard to do incrementally without a count array.
   
   Alternative: Use a count array `counts` of size CAP+1.
   - When adding/updating prime (p, r): increment counts for n in range(start, CAP+1, p).
   - When removing/updating prime (p, r): decrement counts for n in range(start, CAP+1, p).
   - The first uncovered number is the first index i where counts[i] == 0.
   
   Updating counts for a prime p takes O(CAP/p) operations. For small p (e.g., 2), this is 50,000 ops.
   For large p (e.g., 1000), this is 100 ops.
   Finding the first gap takes O(L) in worst case, but we can cache the current L and only scan forward.
   If the new L is close to the old L, this is fast.
   
   Total cost per move: O(CAP/p_new + CAP/p_old + (L_new - L_old)).
   This is much faster than recomputing from scratch (O(N_primes * CAP/avg_p)).
   
   Let's implement this incremental update strategy.
"""

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill
CAP = 100000

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _get_primes(limit=1000):
    return [p for p in range(2, limit + 1) if _is_prime(p)]

ALL_PRIMES = _get_primes(1000)
PRIME_SET = set(ALL_PRIMES)

class CoverageTracker:
    def __init__(self, cap):
        self.cap = cap
        self.counts = np.zeros(cap + 1, dtype=np.int32)
        self.current_L = 0
        
    def _update_counts(self, p, r, delta):
        # delta is +1 for add/increment, -1 for remove/decrement
        # Range of n in [1, cap] such that n % p == r
        if r == 0:
            start = p
        else:
            start = r
            
        if start > self.cap:
            return
            
        # Generate indices: start, start+p, start+2p, ... <= cap
        # Using numpy arange for speed
        indices = np.arange(start, self.cap + 1, p)
        self.counts[indices] += delta
        
    def _find_first_gap(self, start_check):
        # Find first i >= start_check such that counts[i] == 0
        # We know counts[1..start_check-1] are > 0 (if start_check > 1)
        # Actually, we should verify from 1 if start_check is 0 or 1.
        # For efficiency, we assume previous state was valid up to current_L.
        # If we changed a prime, the gap might have moved.
        
        # Optimization: Only scan from max(1, min(start_check, self.current_L))
        # But if we removed coverage, the gap might be earlier.
        # Safe approach: Scan from 1? No, too slow if L is large.
        
        # Heuristic: If we added coverage, L might increase. Scan from current_L + 1.
        # If we removed coverage, L might decrease. We might need to scan from 1.
        # However, in SA, we usually try moves that improve or are close.
        # Let's just scan from 1 to be safe, but break early.
        # With numpy, we can do:
        # zeros = np.where(self.counts == 0)[0]
        # first_gap = zeros[0] if len(zeros) > 0 and zeros[0] > 0 else cap+1
        # But this scans the whole array (100k). 100k ops is ~0.1ms.
        # 10,000 moves * 0.1ms = 1s. This is feasible!
        
        # Let's use the numpy where approach for simplicity and robustness.
        # counts[0] is always 0 (we don't cover 0). We care about 1..CAP.
        # Find first index > 0 where counts == 0.
        
        # To avoid scanning 0, we can check counts[1:]
        # If all are > 0, L = CAP.
        # Else, find first 0.
        
        # np.where on 100k array is fast.
        zero_indices = np.where(self.counts[1:] == 0)[0]
        if len(zero_indices) == 0:
            return self.cap
        else:
            return zero_indices[0] + 1

    def set_prime(self, p, r):
        # Remove old coverage for p? No, this method assumes p is new or we handle removal separately.
        # Let's have add_prime and remove_prime.
        pass

    def add_prime(self, p, r):
        self._update_counts(p, r, 1)
        
    def remove_prime(self, p, r):
        self._update_counts(p, r, -1)
        
    def get_L(self):
        # Recalculate L
        self.current_L = self._find_first_gap(1)
        return self.current_L

def construct():
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback
        return [[13, 1], [17, 1]]

    rng = random.Random(42)
    
    # Initial State: 
    # Start with a set of small primes that fits the budget.
    # Budget: sum(1/p) <= 1.
    # Let's start with {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}
    # Sum 1/p for first 25 primes:
    # 1/2+1/3+1/5+1/7+1/11+1/13+1/17+1/19+1/23+1/29+1/31+1/37+1/41+1/43+1/47+1/53+1/59+1/61+1/67+1/71+1/73+1/79+1/83+1/89+1/97
    # Approx: 0.5+0.33+0.2+0.14+0.09+0.077+0.059+0.053+0.043+0.034+0.032+0.027+0.024+0.023+0.021+0.019+0.017+0.016+0.015+0.014+0.0137+0.0127+0.012+0.0112+0.0103
    # Sum ~ 1.5. Too high.
    
    # We need sum <= 1.
    # Greedy selection of smallest primes?
    # 1/2 = 0.5
    # 1/3 = 0.333 (Sum 0.833)
    # 1/5 = 0.2 (Sum 1.033) -> Too high.
    # So {2, 3} is max for smallest primes?
    # 1/2 + 1/3 + 1/7 = 0.5 + 0.333 + 0.142 = 0.975.
    # 1/2 + 1/3 + 1/11 = 0.5 + 0.333 + 0.09 = 0.923.
    # 1/2 + 1/5 + 1/7 = 0.5 + 0.2 + 0.142 = 0.842.
    # 1/2 + 1/5 + 1/7 + 1/11 = 0.842 + 0.09 = 0.932.
    # 1/2 + 1/5 + 1/7 + 1/11 + 1/13 = 0.932 + 0.077 = 1.009 -> Too high.
    # So {2, 5, 7, 11} is a good dense set.
    
    # Let's start with a set of primes that is dense but fits.
    # We will use SA to swap primes.
    
    initial_primes = [2, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    # Check sum:
    # 0.5 + 0.2 + 0.142 + 0.09 + 0.077 + 0.059 + 0.053 + 0.043 + 0.034 + 0.032
    # 0.5+0.2=0.7
    # +0.142=0.842
    # +0.09=0.932
    # +0.077=1.009 -> Too high.
    
    # Drop 13.
    initial_primes = [2, 5, 7, 11, 17, 19, 23, 29, 31]
    # 0.932 (up to 11)
    # + 1/17 (0.059) = 0.991
    # + 1/19 (0.053) = 1.044 -> Too high.
    
    # So {2, 5, 7, 11, 17} is valid. Sum ~ 0.991.
    # Let's add 19? No.
    # How about dropping 2?
    # 1/3 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19 + 1/23 + 1/29 + 1/31 + 1/37 + 1/41 + 1/43 + 1/47 + 1/53
    # 0.333+0.2+0.142+0.09+0.077+0.059+0.053+0.043+0.034+0.032+0.027+0.024+0.023+0.021+0.019
    # Sum ~ 1.2. Too high.
    
    # The seed used primes >= 13.
    # Let's start with the seed's primes but allow swapping in smaller ones if budget allows.
    
    seed_primes = [p for p in range(13, 1001) if _is_prime(p)]
    # Sum is ~ 0.92.
    # We have ~ 0.08 budget left.
    # Can we swap a large prime for a small one?
    # E.g., replace 997 (1/997 ~ 0.001) with 11 (1/11 ~ 0.09)?
    # Cost increase: 0.09 - 0.001 = 0.089.
    # Budget left 0.08. Not enough.
    # Replace 997 and 991 with 11?
    # Cost: 0.09 - 0.001 - 0.001 = 0.088. Still tight.
    
    # Let's initialize with seed primes and random residues.
    primes = seed_primes[:]
    residues = [rng.randrange(p) for p in primes]
    
    # Build tracker
    tracker = CoverageTracker(CAP)
    for p, r in zip(primes, residues):
        tracker.add_prime(p, r)
        
    current_L = tracker.get_L()
    best_L = current_L
    best_primes = primes[:]
    best_residues = residues[:]
    
    # Track budget
    # Use Fraction for exactness? Or float with epsilon?
    # Float is faster. Sum of 1/p for 168 primes is ~ 0.92.
    # Precision of float is ~ 1e-16. Sum error is ~ 1e-14.
    # Constraint is <= 1.0.
    # If sum is 0.999999999999999, it's valid.
    # If sum is 1.000000000000001, it's invalid.
    # To be safe, we can use a slightly stricter bound, e.g., 0.999999.
    # Or use Fraction. Fraction is slower but exact.
    # Let's use float for speed in SA loop, and validate at the end with Fraction.
    # If invalid, we can try to remove the largest prime.
    
    current_budget = sum(1.0 / p for p in primes)
    
    # SA Parameters
    T = 1.0
    cooling = 0.9995
    min_T = 0.001
    
    while time.time() - _START_TIME < _TIME_LIMIT:
        if T < min_T:
            break
            
        # Move: Pick a random prime index
        idx = rng.randrange(len(primes))
        p_old = primes[idx]
        r_old = residues[idx]
        
        # Move Type 1: Change Residue
        if rng.random() < 0.7:
            r_new = rng.randrange(p_old)
            if r_new == r_old:
                continue
                
            # Apply Move
            tracker.remove_prime(p_old, r_old)
            tracker.add_prime(p_old, r_new)
            new_L = tracker.get_L()
            
            # Evaluate
            delta_L = new_L - current_L
            if delta_L > 0 or rng.random() < np.exp(delta_L / T):
                # Accept
                residues[idx] = r_new
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_primes = primes[:]
                    best_residues = residues[:]
            else:
                # Reject: Revert
                tracker.remove_prime(p_old, r_new)
                tracker.add_prime(p_old, r_old)
                
        # Move Type 2: Swap Prime
        else:
            # Pick a new prime from ALL_PRIMES
            p_new = rng.choice(ALL_PRIMES)
            if p_new == p_old:
                continue
                
            # Check Budget
            delta_budget = (1.0 / p_new) - (1.0 / p_old)
            new_budget = current_budget + delta_budget
            
            if new_budget > 1.0:
                # Invalid, skip
                continue
                
            # Apply Move
            r_new = rng.randrange(p_new)
            
            tracker.remove_prime(p_old, r_old)
            tracker.add_prime(p_new, r_new)
            new_L = tracker.get_L()
            
            delta_L = new_L - current_L
            if delta_L > 0 or rng.random() < np.exp(delta_L / T):
                # Accept
                primes[idx] = p_new
                residues[idx] = r_new
                current_budget = new_budget
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_primes = primes[:]
                    best_residues = residues[:]
            else:
                # Reject: Revert
                tracker.remove_prime(p_new, r_new)
                tracker.add_prime(p_old, r_old)
                
        T *= cooling
        
    # Final Validation with Fraction
    # Ensure sum(1/p) <= 1 exactly
    budget_frac = sum(Fraction(1, p) for p in best_primes)
    if budget_frac > 1:
        # Remove primes until valid
        # Sort by 1/p descending (largest primes first to remove? No, remove smallest 1/p to keep budget low? 
        # Actually, we want to keep the set valid. If it's invalid, we must remove some.
        # Removing a prime decreases the sum.
        # To maximize coverage, we should probably keep the smaller primes.
        # So remove the largest primes (smallest 1/p).
        sorted_indices = sorted(range(len(best_primes)), key=lambda i: 1.0/best_primes[i], reverse=True)
        # Wait, 1/p is smaller for larger p.
        # We want to remove primes that contribute least to budget? No, we want to reduce budget.
        # Any removal reduces budget.
        # To preserve coverage, we should remove primes that cover the fewest unique numbers?
        # Simple heuristic: Remove largest primes.
        for i in sorted_indices:
            if budget_frac <= 1:
                break
            p_to_remove = best_primes[i]
            budget_frac -= Fraction(1, p_to_remove)
            best_primes.pop(i)
            best_residues.pop(i)
            # Note: This invalidates indices in sorted_indices, but we break out or re-sort?
            # Since we pop, indices shift. Let's just do a loop while invalid.
            # Actually, just remove the last one (largest prime) repeatedly.
            
    # Re-verify
    budget_frac = sum(Fraction(1, p) for p in best_primes)
    if budget_frac > 1:
        # Fallback to seed if still invalid (should not happen)
        best_primes = [p for p in range(13, 1001) if _is_prime(p)]
        best_residues = [(i + 1) % p for i, p in enumerate(best_primes)]
        
    return [[p, r] for p, r in zip(best_primes, best_residues)]
# EVOLVE-BLOCK-END
