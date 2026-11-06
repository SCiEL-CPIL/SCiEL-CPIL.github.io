import math
import numpy as np
from fractions import Fraction

def _primes_upto(n):
    """Return list of primes up to n."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

def construct():
    CAP = 10**5
    primes = _primes_upto(1000)
    
    covered = np.zeros(CAP + 1, dtype=bool)
    # Precompute indices for reuse
    all_indices = np.arange(1, CAP + 1)
    
    used_primes = set()
    budget_used = Fraction(0, 1)
    classes = []
    
    L_cur = 0
    
    while True:
        best_p = None
        best_r = None
        best_new_L = L_cur
        
        candidates = []
        for p in primes:
            if p in used_primes:
                continue
            cost = Fraction(1, p)
            if budget_used + cost <= Fraction(1, 1):
                candidates.append(p)
                
        if not candidates:
            break
            
        for p in candidates:
            r = (L_cur + 1) % p
            
            # Calculate new prefix length
            # We look at n from L_cur + 1 to CAP
            # Condition: covered[n] is True OR n % p == r
            
            # Slice of covered
            # We need to handle the case where L_cur + 1 > CAP
            if L_cur >= CAP:
                new_L = CAP
            else:
                n_start = L_cur + 1
                # Create mask for this specific p, r
                # We can do this on the slice of indices
                slice_indices = all_indices[n_start - 1:CAP] # 0-indexed slice of all_indices (which is 1..CAP)
                # all_indices is 1..CAP. 
                # n_start corresponds to index n_start - 1 in 0-based array?
                # all_indices[0] = 1.
                # So index for value v is v - 1.
                # We want values n_start .. CAP.
                # Indices: n_start - 1 .. CAP - 1.
                
                # Actually, simpler:
                # slice_cov = covered[n_start:CAP+1]
                # slice_idx = np.arange(n_start, CAP+1)
                
                # To avoid arange allocation, we can use the precomputed all_indices slice
                slice_idx = all_indices[n_start - 1:CAP]
                slice_cov = covered[n_start:CAP+1]
                
                # Check if all are covered by existing OR new class
                # new_class_mask = (slice_idx % p) == r
                # combined = slice_cov | new_class_mask
                
                # We want the first False in combined.
                # If no False, new_L = CAP.
                # Else, new_L = n_start + index_of_first_false - 1.
                
                # Optimization: 
                # If slice_cov is all True, then new_L = CAP (since new class adds nothing negative).
                # Wait, if slice_cov is all True, the prefix is already CAP.
                # But we are trying to EXTEND from L_cur.
                # If L_cur < CAP, slice_cov is not all True.
                
                # Let's just compute it.
                # This allocates memory for slice_idx and new_class_mask.
                # 10^5 integers is 800KB. 168 times is 130MB total allocated over time.
                # This is acceptable.
                
                new_class_mask = (slice_idx % p) == r
                combined = slice_cov | new_class_mask
                
                if combined.all():
                    new_L = CAP
                else:
                    # Find first False
                    # ~combined gives True where False
                    first_false_idx = np.argmax(~combined)
                    # The value is n_start + first_false_idx
                    # The covered prefix ends at value - 1
                    new_L = n_start + first_false_idx - 1
                    
            if new_L > best_new_L:
                best_new_L = new_L
                best_p = p
                best_r = r
                
        if best_p is None:
            break
            
        # Commit
        used_primes.add(best_p)
        budget_used += Fraction(1, best_p)
        classes.append([best_p, best_r])
        
        # Update covered array
        # Mark n where n % best_p == best_r
        mask = (all_indices % best_p) == best_r
        covered[1:] |= mask # covered[1:] corresponds to values 1..CAP
        
        L_cur = best_new_L
        
        if L_cur == CAP:
            break
            
    # Final L
    # Find first False in covered[1:]
    # If all True, L = CAP
    # Else L = index + 1 - 1 = index?
    # covered[1:] is values 1..CAP.
    # If covered[1] is False, L=0.
    # If covered[1] True, covered[2] False, L=1.
    
    if covered[1:].all():
        L_final = CAP
    else:
        first_false_idx = np.argmax(~covered[1:])
        # The value is 1 + first_false_idx
        # The prefix ends at value - 1
        L_final = first_false_idx
        
    return classes
