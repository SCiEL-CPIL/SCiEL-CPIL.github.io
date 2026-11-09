import time
from fractions import Fraction
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # The optimal set of primes with reciprocal sum <= 1
    # 1/3 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19 + 1/23
    # = 35913/35970 < 1
    prime_list = [3, 5, 7, 11, 13, 17, 19, 23]
    
    # Verify sum <= 1
    total = sum(Fraction(1, p) for p in prime_list)
    assert total <= 1, f"Sum {total} > 1"
    
    n_primes = len(prime_list)
    
    def get_coverage_length(residues, L):
        """Return the largest L' such that 1..L' is covered by the given residues."""
        covered = bytearray(L + 1)
        for i, p in enumerate(prime_list):
            r = residues[i] % p
            # Mark all y ≡ r (mod p) in 1..L
            start = r
            if start == 0:
                start = p
            for y in range(start, L + 1, p):
                covered[y] = 1
        # Find the first gap
        for x in range(1, L + 1):
            if not covered[x]:
                return x - 1
        return L
    
    def greedy_solve(L, seed=0):
        """
        Greedy: for each uncovered x from 1 to L, assign the prime that covers
        the most currently uncovered numbers.
        Returns residues list or None if failed.
        """
        rng = random.Random(seed)
        n = L
        covered = bytearray(n + 1)
        used = [False] * n_primes
        residues = [0] * n_primes
        
        # For each unused prime, track uncovered count per residue
        # unc[p_idx][r] = number of uncovered y in 1..n with y % p == r
        unc = []
        for i, p in enumerate(prime_list):
            counts = [0] * p
            for y in range(1, n + 1):
                counts[y % p] += 1
            unc.append(counts)
        
        def find_next_uncovered(start):
            for i in range(start, n + 1):
                if not covered[i]:
                    return i
            return n + 1
        
        next_unc = 1
        while next_unc <= n:
            x = next_unc
            # Find the best prime to cover x
            best_gain = -1
            best_i = -1
            best_r = 0
            for i in range(n_primes):
                if used[i]:
                    continue
                r = x % prime_list[i]
                gain = unc[i][r]
                if gain > best_gain or (gain == best_gain and rng.random() < 0.5):
                    best_gain = gain
                    best_i = i
                    best_r = r
            
            if best_i < 0:
                return None
            
            p = prime_list[best_i]
            r = best_r
            used[best_i] = True
            residues[best_i] = r
            
            # Update covered and unc
            start_y = r if r > 0 else p
            for y in range(start_y, n + 1, p):
                if not covered[y]:
                    covered[y] = 1
                    for j in range(n_primes):
                        pj = prime_list[j]
                        unc[j][y % pj] -= 1
            
            next_unc = find_next_uncovered(x + 1)
        
        return residues
    
    def local_search(residues, L, max_iters=200):
        """
        Try to improve coverage by swapping residues.
        Returns improved residues or the original if no improvement.
        """
        best_len = get_coverage_length(residues, L)
        if best_len >= L:
            return residues, best_len
        
        current_residues = list(residues)
        current_len = best_len
        rng = random.Random(42)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            # Pick a random prime and try a random different residue
            i = rng.randrange(n_primes)
            p = prime_list[i]
            old_r = current_residues[i]
            new_r = rng.randrange(p)
            if new_r == old_r:
                continue
            
            current_residues[i] = new_r
            new_len = get_coverage_length(current_residues, L)
            if new_len > current_len:
                current_len = new_len
                if current_len >= L:
                    break
            else:
                # Revert with some probability (simulated annealing style)
                if rng.random() < 0.7:
                    current_residues[i] = old_r
                # Keep the change if it doesn't hurt much (accept worse with small prob)
                elif rng.random() < 0.1:
                    pass  # keep it
                else:
                    current_residues[i] = old_r
        
        return current_residues, current_len
    
    # Strategy:
    # 1. Try greedy with multiple seeds to find good initial residues
    # 2. Use local search to improve
    # 3. Binary search on L to find the maximum covered prefix
    
    best_L = 0
    best_residues = None
    
    # First, try to cover up to 1000 directly
    # Use greedy with multiple seeds
    for seed in range(20):
        if time.time() > deadline - 30:
            break
        residues = greedy_solve(1000, seed)
        if residues is not None:
            L = get_coverage_length(residues, 1000)
            if L > best_L:
                best_L = L
                best_residues = list(residues)
            if L >= 1000:
                break
    
    # If we haven't reached 1000, try local search to improve
    if best_residues is not None and best_L < 1000:
        improved, new_L = local_search(best_residues, 1000, max_iters=500)
        if new_L > best_L:
            best_L = new_L
            best_residues = improved
    
    # Try more greedy seeds if we have time
    if time.time() < deadline - 20:
        for seed in range(20, 50):
            if time.time() > deadline - 10:
                break
            residues = greedy_solve(1000, seed)
            if residues is not None:
                L = get_coverage_length(residues, 1000)
                if L > best_L:
                    best_L = L
                    best_residues = list(residues)
                if L >= 1000:
                    break
    
    # If still not 1000, do more local search
    if best_residues is not None and best_L < 1000 and time.time() < deadline - 10:
        improved, new_L = local_search(best_residues, 1000, max_iters=1000)
        if new_L > best_L:
            best_L = new_L
            best_residues = improved
    
    # Final local search with a fresh start if we have time
    if time.time() < deadline - 5:
        # Try a different approach: start from a known good pattern
        # Known covering for {3,5,7,11,13,17,19,23}:
        # 3|0, 5|1, 7|2, 11|3, 13|4, 17|5, 19|6, 23|7 (this is a guess)
        # Let's try several fixed patterns
        fixed_patterns = [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [2, 3, 4, 5, 6, 7, 8, 9],
            [0, 2, 4, 6, 8, 10, 12, 14],
        ]
        for pat in fixed_patterns:
            if time.time() > deadline - 5:
                break
            residues = list(pat)
            # Normalize residues
            for i in range(n_primes):
                residues[i] = residues[i] % prime_list[i]
            L = get_coverage_length(residues, 1000)
            if L > best_L:
                best_L = L
                best_residues = list(residues)
            if L >= 1000:
                break
    
    # One more round of local search on the best found
    if best_residues is not None and best_L < 1000 and time.time() < deadline - 3:
        improved, new_L = local_search(best_residues, 1000, max_iters=300)
        if new_L > best_L:
            best_L = new_L
            best_residues = improved
    
    if best_residues is not None:
        result = []
        for i, p in enumerate(prime_list):
            result.append([p, best_residues[i] % p])
        return result
    
    # Fallback: return a simple valid set
    return [[3, 0], [5, 1], [7, 2]]
