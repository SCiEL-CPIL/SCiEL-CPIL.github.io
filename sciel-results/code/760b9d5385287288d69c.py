import time
from fractions import Fraction
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    random.seed(42)

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    
    # Helper to build a prime set given a mandatory base and a skip set
    def build_set(base, skip):
        base_set = set(base)
        skip_set = set(skip)
        s = sum(Fraction(1, p) for p in base_set)
        if s > 1:
            return []
        
        final_set = list(base_set)
        # Fill with smallest available primes not in base or skip
        for p in primes:
            if p in base_set or p in skip_set:
                continue
            if s + Fraction(1, p) <= 1:
                final_set.append(p)
                s += Fraction(1, p)
        return sorted(final_set)

    # Generate diverse candidate sets
    candidates = []
    
    # 1. Standard greedy fill (smallest primes first)
    candidates.append(build_set([], []))
    
    # 2. Skip 3
    candidates.append(build_set([], {3}))
    
    # 3. Skip 5
    candidates.append(build_set([], {5}))
    
    # 4. Skip 3, 5
    candidates.append(build_set([], {3, 5}))
    
    # 5. Force 2
    candidates.append(build_set([2], []))
    
    # 6. Force 2, skip 3
    candidates.append(build_set([2], {3}))
    
    # 7. Force 2, skip 5
    candidates.append(build_set([2], {5}))
    
    # 8. Force 2, 5
    candidates.append(build_set([2, 5], []))
    
    # 9. Force 3
    candidates.append(build_set([3], []))
    
    # 10. Force 3, skip 5
    candidates.append(build_set([3], {5}))
    
    # 11. Force 5
    candidates.append(build_set([5], []))
    
    # 12. Force 2, 7
    candidates.append(build_set([2, 7], []))
    
    # 13. Force 2, 11
    candidates.append(build_set([2, 11], []))
    
    # 14. Skip 2, 3
    candidates.append(build_set([], {2, 3}))
    
    # 15. Skip 2, 5
    candidates.append(build_set([], {2, 5}))
    
    # 16. Skip 3, 7
    candidates.append(build_set([], {3, 7}))
    
    # 17. Skip 2
    candidates.append(build_set([], {2}))
    
    # 18. Force 2, 3, 7
    candidates.append(build_set([2, 3, 7], []))

    # Remove duplicates
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(c)
        if key not in seen and c:
            seen.add(key)
            unique_candidates.append(c)
    candidates = unique_candidates

    best_L = 0
    best_classes = []
    
    # Evaluation limit for the prefix.
    # We check up to 10,000 to be safe, but the target is 999.
    # Using NumPy for fast coverage updates.
    eval_limit = 10000
    
    def get_L(counts):
        # Find first False (count == 0)
        # counts is a numpy array of ints
        # We care about indices 0 to eval_limit-1 corresponding to numbers 1 to eval_limit
        # If all are > 0, return eval_limit
        if np.all(counts[:eval_limit] > 0):
            return eval_limit
        # Find first index where count is 0
        first_zero = np.argmax(counts[:eval_limit] == 0)
        # The number is first_zero + 1
        # L is the largest number such that 1..L are covered.
        # So if index k (number k+1) is uncovered, L = k.
        return first_zero

    for P in candidates:
        if time.time() > deadline - 20:
            break
        if not P:
            continue
        
        # Local Search for Residues
        # Start with a greedy initialization
        counts = np.zeros(eval_limit + 1, dtype=np.int32)
        residues = {}
        used_primes = set()
        
        # Greedy Init
        P_sorted = sorted(P)
        for n in range(1, eval_limit + 1):
            if time.time() > deadline - 5:
                break
            # n is the number we want to cover. Index in array is n-1.
            if counts[n-1] > 0:
                continue
            
            assigned = False
            # Try to find a prime that covers n.
            # For init, just pick smallest available.
            for p in P_sorted:
                if p in used_primes:
                    continue
                r = n % p
                residues[p] = r
                used_primes.add(p)
                
                # Update covered
                # Indices i such that (i+1) % p == r
                # i = r - 1, r - 1 + p, ...
                start_idx = r - 1
                indices = np.arange(start_idx, eval_limit, p)
                indices = indices[indices >= 0]
                counts[indices] += 1
                assigned = True
                break
            
            if not assigned:
                break
        
        current_L = get_L(counts)
        
        # Hill Climbing
        # Try to change residue of one prime to improve L.
        # Use count array for efficient updates.
        
        for iteration in range(50): # Max 50 passes
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Shuffle primes to avoid bias
            P_shuffled = P_sorted.copy()
            random.shuffle(P_shuffled)
            
            for p in P_shuffled:
                if time.time() > deadline - 5:
                    break
                
                old_r = residues.get(p)
                if old_r is None:
                    continue
                    
                best_new_r = old_r
                best_L_candidate = current_L
                
                # Try all residues
                for new_r in range(p):
                    if new_r == old_r:
                        continue
                    
                    # Update counts
                    # Remove old
                    start_old = old_r - 1
                    idx_old = np.arange(start_old, eval_limit, p)
                    idx_old = idx_old[idx_old >= 0]
                    counts[idx_old] -= 1
                    
                    # Add new
                    start_new = new_r - 1
                    idx_new = np.arange(start_new, eval_limit, p)
                    idx_new = idx_new[idx_new >= 0]
                    counts[idx_new] += 1
                    
                    # Check L
                    # Optimization: If the change affects indices < current_L, check if any become 0.
                    # If so, L drops.
                    # If not, L is at least current_L. Then check if it extended.
                    
                    # Check if any count in [0, current_L-1] is 0.
                    # Note: current_L is the number of covered integers.
                    # Indices 0 to current_L-1 correspond to numbers 1 to current_L.
                    if current_L > 0 and np.any(counts[:current_L] == 0):
                        # L dropped. Revert.
                        counts[idx_old] += 1
                        counts[idx_new] -= 1
                        continue
                    
                    # If prefix is safe, check if it extended.
                    # Calculate new L.
                    new_L = get_L(counts)
                    
                    if new_L > best_L_candidate:
                        best_L_candidate = new_L
                        best_new_r = new_r

                    # Revert
                    counts[idx_old] += 1
                    counts[idx_new] -= 1
                
                if best_new_r != old_r:
                    # Apply best change
                    start_old = old_r - 1
                    idx_old = np.arange(start_old, eval_limit, p)
                    idx_old = idx_old[idx_old >= 0]
                    counts[idx_old] -= 1
                    
                    start_new = best_new_r - 1
                    idx_new = np.arange(start_new, eval_limit, p)
                    idx_new = idx_new[idx_new >= 0]
                    counts[idx_new] += 1
                    
                    residues[p] = best_new_r
                    current_L = best_L_candidate
                    improved = True
            
            if not improved:
                break
        
        # Final check
        if current_L > best_L:
            best_L = current_L
            best_classes = [[p, residues[p]] for p in P if p in residues]
            
    return best_classes

# Test
if __name__ == "__main__":
    classes = construct()
    print(classes)
