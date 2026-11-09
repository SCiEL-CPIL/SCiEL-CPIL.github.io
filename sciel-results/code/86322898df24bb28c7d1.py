import time
import random
import numpy as np
from fractions import Fraction

def _get_primes(limit=1000):
    """Generate primes up to limit using simple sieve."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    primes = _get_primes(1000)
    L_cap = 100000
    
    # 1. Greedy Initialization
    # We want to cover 1..L. 
    # Strategy: Iteratively pick the prime-residue pair that covers the most 
    # currently uncovered numbers in 1..L_cap, subject to budget.
    # Since we have a fixed budget sum(1/p) <= 1, we can't just pick all.
    # A common heuristic for covering systems is to use small primes first 
    # because they cover more numbers per unit budget? 
    # Actually, 1/p is the cost. Small p -> high cost, high coverage density.
    # Large p -> low cost, low coverage density.
    # To maximize L, we likely want a mix.
    # However, the "standard" covering for x=1000 uses primes >= 13.
    # Let's start with a set of primes and optimize residues.
    
    # Initialize with primes >= 13, as this is a known good starting point 
    # for the Erdos-Ruzsa context at x=1000.
    # Sum of reciprocals of primes >= 13 up to 1000 is approx 0.92.
    # This leaves little room for small primes (2,3,5,7,11).
    # If we include 2, cost is 0.5. Remaining budget 0.5.
    # Let's try two strategies:
    # A) Primes >= 13 (Low cost, high count)
    # B) Include 2,3,5,7 (High cost, low count) and fill rest with large primes.
    
    # Let's stick to Strategy A for now, but allow the optimizer to swap primes.
    # Initial set: all primes p such that sum(1/p for p in set) <= 1.
    # Let's just take all primes >= 13.
    
    current_primes = [p for p in primes if p >= 13]
    # Random residues initially
    current_residues = [random.randint(0, p-1) for p in current_primes]
    
    # Check budget
    current_sum = sum(Fraction(1, p) for p in current_primes)
    
    # If budget is exceeded (shouldn't be for >=13), remove largest primes?
    # No, sum of 1/p for p>=13 is < 1.
    
    # 2. Vectorized Local Search
    # We maintain a coverage count array of length L_cap+1.
    # counts[n] = number of classes covering n.
    # We want to maximize the first index where counts[n] == 0.
    
    # To speed up evaluation:
    # Instead of recalculating the whole L from scratch every move (O(L)),
    # we can maintain the counts and just check the "frontier".
    # But updating counts for a move (p, r) -> (p, r') requires:
    # 1. Decrement counts for old residue class.
    # 2. Increment counts for new residue class.
    # This is O(L/p) for each.
    # With NumPy, this is very fast.
    
    counts = np.zeros(L_cap + 1, dtype=np.int8)
    
    # Initialize counts
    for i, p in enumerate(current_primes):
        r = current_residues[i]
        # Indices: r, r+p, r+2p...
        # Note: if r==0, covers 0, p, 2p... 
        # The problem says 1..L. So 0 is irrelevant.
        # If r > 0, covers r, r+p...
        # If r == 0, covers p, 2p... (since 0 is not in 1..L)
        # Wait, residue 0 mod p means multiples of p.
        # 1..L does not include 0.
        # So for r=0, indices are p, 2p, ...
        # For r>0, indices are r, r+p, ...
        
        if r == 0:
            idxs = np.arange(p, L_cap + 1, p)
        else:
            idxs = np.arange(r, L_cap + 1, p)
        counts[idxs] += 1
            
    def get_L(counts):
        # Find first 0 in counts[1:]
        # np.argmax returns index of first True.
        # If all covered, argmax returns 0 (first element of empty slice? No).
        # Let's use:
        # zeros = np.where(counts[1:] == 0)[0]
        # if len(zeros) > 0: return zeros[0]
        # else: return L_cap
        
        # Faster: 
        # If counts[1] is 0, L=0? No, L is largest L such that 1..L covered.
        # If 1 is not covered, L=0.
        # The score is L.
        
        # Let's just find the first index i >= 1 where counts[i] == 0.
        # L = i - 1.
        
        # np.nonzero(counts[1:] == 0)[0] gives indices relative to 1.
        # If empty, L = L_cap.
        
        # To avoid creating a boolean array every time (memory allocation),
        # we can just loop? No, NumPy is faster.
        # But allocation is slow.
        
        # Alternative: maintain a variable `current_L` and update it.
        # This is tricky with vectorized updates.
        
        # Let's stick to `np.where` but be mindful of cost.
        # L_cap = 100,000. Array size 100k.
        # `counts[1:] == 0` creates a 100k bool array.
        # `np.where` scans it.
        # This is ~1-2 microseconds in NumPy? Maybe 10us.
        # 10,000 moves * 10us = 0.1s. Very fast.
        
        zeros = np.where(counts[1:] == 0)[0]
        if len(zeros) > 0:
            return zeros[0] # This is the index in the slice, so actual index is zeros[0] + 1? 
            # Wait. counts[1:] starts at index 1.
            # If counts[1] == 0, zeros[0] is 0. Actual index is 1. L = 1 - 1 = 0.
            # If counts[5] == 0, zeros[0] is 4. Actual index is 5. L = 5 - 1 = 4.
            # So L = zeros[0].
            return zeros[0]
        return L_cap

    current_L = get_L(counts)
    best_L = current_L
    best_primes = current_primes[:]
    best_residues = current_residues[:]
    
    # Precompute prime indices for quick lookup
    prime_set = set(current_primes)
    
    # Local Search Loop
    # We will perform moves:
    # 1. Change residue of an existing prime.
    # 2. Swap a prime for another (if budget allows).
    
    # To make it fast, we'll do a hill-climbing with random restarts or simulated annealing.
    # Given the vectorized speed, we can do many iterations.
    
    T = 1.0
    cooling = 0.9999
    T_min = 0.01
    
    while time.time() < deadline:
        # Pick a random prime index
        idx = random.randint(0, len(current_primes) - 1)
        p = current_primes[idx]
        old_r = current_residues[idx]
        
        # Generate new residue
        # Bias towards covering the first gap?
        # First gap is current_L + 1.
        # We want new_r such that (current_L + 1) % p == new_r.
        # This guarantees we cover the immediate next number IF we don't break previous coverage.
        # But changing residue might uncover something else.
        
        # Strategy: 
        # 50% chance: target the gap.
        # 50% chance: random.
        
        if random.random() < 0.5:
            target = current_L + 1
            new_r = target % p
        else:
            new_r = random.randint(0, p - 1)
            
        if new_r == old_r:
            continue
            
        # Apply move
        # 1. Remove old contribution
        if old_r == 0:
            old_idxs = np.arange(p, L_cap + 1, p)
        else:
            old_idxs = np.arange(old_r, L_cap + 1, p)
        counts[old_idxs] -= 1
        
        # 2. Add new contribution
        if new_r == 0:
            new_idxs = np.arange(p, L_cap + 1, p)
        else:
            new_idxs = np.arange(new_r, L_cap + 1, p)
        counts[new_idxs] += 1
        
        # 3. Evaluate
        new_L = get_L(counts)
        delta_L = new_L - current_L
        
        if delta_L >= 0:
            # Accept
            current_L = new_L
            current_residues[idx] = new_r
            if current_L > best_L:
                best_L = current_L
                best_primes = current_primes[:]
                best_residues = current_residues[:]
        else:
            # Accept with probability exp(delta_L / T)
            # delta_L is negative integer.
            # If T is small, prob is ~0.
            if random.random() < np.exp(delta_L / T):
                current_L = new_L
                current_residues[idx] = new_r
            else:
                # Revert
                # Remove new, add old
                counts[new_idxs] -= 1
                counts[old_idxs] += 1
                # current_residues[idx] remains old_r
            
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            
        # Occasionally try a prime swap?
        # Prime swaps are harder to vectorize efficiently if we don't keep a pool.
        # And changing residue is likely sufficient to find good L if the prime set is good.
        # The prime set {13..997} is fixed in this phase.
        
        # If stuck for too long, reheat?
        # For now, simple SA.
        
        # Check if we can improve by adding a small prime?
        # Budget check.
        # If current_sum + 1/2 <= 1, we could add 2.
        # But we initialized with >= 13.
        # Let's allow swapping a large prime for a small one if it helps.
        # This is complex. Let's stick to residue optimization for the fixed set first.
        # If L is low, maybe the set is bad.
        
        # Simple heuristic: If L < 100, try swapping a random prime for a smaller one.
        if current_L < 100 and random.random() < 0.1:
            # Swap prime
            new_p = random.choice([p for p in primes if p < current_primes[idx]])
            if new_p in current_primes:
                continue
            # Check budget
            new_sum = current_sum - Fraction(1, current_primes[idx]) + Fraction(1, new_p)
            if new_sum > 1:
                continue
            
            # Apply swap
            old_p = current_primes[idx]
            old_r = current_residues[idx]
            
            # Remove old
            if old_r == 0:
                old_idxs = np.arange(old_p, L_cap + 1, old_p)
            else:
                old_idxs = np.arange(old_r, L_cap + 1, old_p)
            counts[old_idxs] -= 1
            
            # Add new (with same residue mod new_p? Or random?)
            # Let's use random residue for new prime
            new_r = random.randint(0, new_p - 1)
            if new_r == 0:
                new_idxs = np.arange(new_p, L_cap + 1, new_p)
            else:
                new_idxs = np.arange(new_r, L_cap + 1, new_p)
            counts[new_idxs] += 1
            
            new_L = get_L(counts)
            delta_L = new_L - current_L
            
            if delta_L >= 0:
                current_L = new_L
                current_primes[idx] = new_p
                current_residues[idx] = new_r
                current_sum = new_sum
                prime_set.remove(old_p)
                prime_set.add(new_p)
                
                if current_L > best_L:
                    best_L = current_L
                    best_primes = current_primes[:]
                    best_residues = current_residues[:]
            else:
                if random.random() < np.exp(delta_L / T):
                    current_L = new_L
                    current_primes[idx] = new_p
                    current_residues[idx] = new_r
                    current_sum = new_sum
                    prime_set.remove(old_p)
                    prime_set.add(new_p)
                else:
                    # Revert
                    counts[new_idxs] -= 1
                    counts[old_idxs] += 1
                    # State remains old

    # Final validation and return
    # Ensure best result is valid
    final_sum = sum(Fraction(1, p) for p in best_primes)
    if final_sum > 1:
        # Fallback
        return [[p, (i + 1) % p] for i, p in enumerate([p for p in primes if p >= 13])]
        
    return [[p, r] for p, r in zip(best_primes, best_residues)]

if __name__ == "__main__":
    # Quick test
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end - start:.2f}s")
    print(f"Result size: {len(res)}")
    # Calculate L for verification
    L_cap = 100000
    counts = np.zeros(L_cap + 1, dtype=np.int8)
    for p, r in res:
        if r == 0:
            idxs = np.arange(p, L_cap + 1, p)
        else:
            idxs = np.arange(r, L_cap + 1, p)
        counts[idxs] += 1
    zeros = np.where(counts[1:] == 0)[0]
    if len(zeros) > 0:
        L = zeros[0]
    else:
        L = L_cap
    print(f"Verified L: {L}")
    budget = sum(Fraction(1, p) for p, r in res)
    print(f"Budget: {budget}")
