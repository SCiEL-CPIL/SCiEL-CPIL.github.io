import time
import random
import math
from fractions import Fraction
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 115.0

    def _primes_upto(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]

    all_primes = _primes_upto(1000)
    selected_primes = [p for p in all_primes if p >= 13]
    N = len(selected_primes)

    cost = sum(Fraction(1, p) for p in selected_primes)
    while cost > Fraction(1, 1) and selected_primes:
        min_idx = 0
        min_p = selected_primes[0]
        for i, p in enumerate(selected_primes):
            if p < min_p:
                min_p = p
                min_idx = i
        cost -= Fraction(1, selected_primes[min_idx])
        selected_primes.pop(min_idx)
        N -= 1

    MAX_L = 15000
    counts = np.zeros(MAX_L + 2, dtype=np.int32)
    primes_arr = np.array(selected_primes, dtype=np.int32)

    residues = [0] * N
    for i in range(N):
        residues[i] = (i + 1) % selected_primes[i]

    for i in range(N):
        p = selected_primes[i]
        r = residues[i]
        start = r if r != 0 else p
        idxs = np.arange(start, MAX_L + 1, p)
        counts[idxs] += 1

    def calc_L():
        # Find the first index where counts is 0
        # counts[1] corresponds to integer 1
        nonzero = np.nonzero(counts[1:MAX_L + 1] > 0)[0]
        if len(nonzero) == 0:
            return 0
        # The first gap is at index nonzero[0] + 1 if nonzero[0] > 0
        # Actually, if counts[1] > 0, then 1 is covered.
        # We want the largest L such that 1..L are covered.
        # This is the first index k (1-indexed) where counts[k] == 0, minus 1.
        # nonzero contains 0-indexed positions in the slice [1:MAX_L+1].
        # So position 0 corresponds to index 1.
        # If nonzero[0] > 0, then counts[1] is 0, so L=0.
        # If nonzero[0] == 0, then counts[1] > 0.
        # We need the first index j in 1..MAX_L where counts[j] == 0.
        # Let's just scan from the start.
        # For efficiency, we can use the nonzero array.
        # If nonzero is empty, L=0.
        # If nonzero[0] > 0, then index 1 (offset 0) is 0. So L=0.
        # Wait, np.nonzero returns indices where array != 0.
        # If counts[1] == 0, then 0 is NOT in nonzero.
        # So if 0 is not in nonzero, L=0.
        # If 0 is in nonzero, then 1 is covered.
        # We need to find the first index j >= 1 such that counts[j] == 0.
        # This is equivalent to finding the first index k in the range [0, MAX_L-1] such that k is NOT in nonzero.
        # Since nonzero is sorted, we can iterate.
        
        if len(nonzero) == 0:
            return 0
            
        if nonzero[0] != 0:
            return 0
            
        # Iterate through nonzero to find the first gap
        prev = -1
        for idx in nonzero:
            if idx != prev + 1:
                # Gap found at idx
                # The covered length is the index of the gap (1-indexed) minus 1
                # idx is 0-indexed offset. So actual index is idx + 1.
                # L is (idx + 1) - 1 = idx.
                return idx
            prev = idx
            
        # If no gap found, all 1..MAX_L are covered
        return MAX_L

    def apply_change(i, old_r, new_r):
        p = selected_primes[i]
        if old_r != 0:
            idxs = np.arange(old_r, MAX_L + 1, p)
            counts[idxs] -= 1
        else:
            idxs = np.arange(p, MAX_L + 1, p)
            counts[idxs] -= 1
            
        if new_r != 0:
            idxs = np.arange(new_r, MAX_L + 1, p)
            counts[idxs] += 1
        else:
            idxs = np.arange(p, MAX_L + 1, p)
            counts[idxs] += 1

    L = calc_L()
    best_L = L
    best_residues = residues[:]

    def greedy_extend(current_L):
        extended = True
        L_local = current_L
        # Use a simple loop. 
        # To speed up, we can precompute which primes can cover L+1.
        # But checking safety is the bottleneck.
        
        while extended and time.time() < deadline:
            extended = False
            # Find candidates that can cover L+1
            # Candidates are primes p where (L+1) % p is a valid residue (always true)
            # But we only change if current residue is NOT (L+1) % p
            # And if it's safe to change.
            
            # To avoid iterating all N every time, we can shuffle.
            # But we want to prioritize primes that are "close" to covering more?
            # No, just random is fine.
            
            order = list(range(N))
            random.shuffle(order)
            
            for i in order:
                if time.time() > deadline:
                    return L_local
                    
                p = selected_primes[i]
                r_old = residues[i]
                target_r = (L_local + 1) % p
                
                if r_old == target_r:
                    continue
                
                # Safety check
                # We need counts[n] > 1 for all n in old_arithmetic_progression up to L_local
                # n = r_old, r_old+p, ... <= L_local
                if r_old != 0:
                    idxs = np.arange(r_old, L_local + 1, p)
                else:
                    idxs = np.arange(p, L_local + 1, p)
                    
                # If idxs is empty, it's safe (no overlap with 1..L_local)
                if len(idxs) == 0:
                    safe = True
                else:
                    # Check if min count in these indices is > 1
                    # np.min is fast
                    if np.min(counts[idxs]) > 1:
                        safe = True
                    else:
                        safe = False

                if safe:
                    apply_change(i, r_old, target_r)
                    residues[i] = target_r
                    
                    # Extend L
                    # Instead of while loop, use calc_L?
                    # calc_L is O(MAX_L) in worst case but usually fast.
                    # Or we can just check L+1, L+2...
                    # Since we only extended by 1 potentially, we can check incrementally.
                    # But the change might have uncovered a gap earlier? No, we checked safety.
                    # So 1..L_local are still covered.
                    # We just need to extend from L_local+1.
                    
                    while L_local < MAX_L and counts[L_local + 1] > 0:
                        L_local += 1
                        
                    extended = True
                    break # Restart with new L
                
        return L_local

    # Main Loop
    while time.time() < deadline:
        # Try to extend
        new_L = greedy_extend(L)
        
        if new_L > L:
            L = new_L
            if L > best_L:
                best_L = L
                best_residues = residues[:]
        else:
            # Stuck. Kick.
            # Save state
            save_residues = residues[:]
            save_counts = counts.copy()
            
            # Kick
            K = random.randint(2, 4)
            for _ in range(K):
                i_kick = random.randint(0, N - 1)
                p_kick = selected_primes[i_kick]
                old_r_kick = residues[i_kick]
                new_r_kick = random.randint(0, p_kick - 1)
                
                if old_r_kick != new_r_kick:
                    apply_change(i_kick, old_r_kick, new_r_kick)
                    residues[i_kick] = new_r_kick
            
            # Repair
            L_temp = calc_L()
            L_repaired = greedy_extend(L_temp)
            
            if L_repaired > best_L:
                L = L_repaired
                best_L = L
                best_residues = residues[:]
            elif L_repaired >= L:
                L = L_repaired
                if L > best_L:
                    best_L = L
                    best_residues = residues[:]
            else:
                # Revert
                residues[:] = save_residues
                counts[:] = save_counts
                L = save_L

    return [[selected_primes[i], best_residues[i]] for i in range(N)]
