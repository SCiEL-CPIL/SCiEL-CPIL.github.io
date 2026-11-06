"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt

def construct():
    deadline = time.time() + 50.0  # safety margin before 60s hard kill

    # ---- Sieve of primes up to 10000 ----
    N = 10000
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start::step] = [False] * (((N - start) // step) + 1)
    primes = [i for i, v in enumerate(is_prime) if v]
    n = len(primes)
    prime_index = {p: i for i, p in enumerate(primes)}
    primes_arr = np.array(primes, dtype=np.int32)

    def validate(chain):
        if not chain:
            return False
        if chain[-1] > N:
            return False
        for p in chain:
            if not is_prime[p]:
                return False
        if len(chain) < 2:
            return True
        for a, b in zip(chain, chain[1:]):
            if b <= a:
                return False
        for i in range(1, len(chain) - 1):
            if chain[i + 1] - chain[i] < chain[i] - chain[i - 1]:
                return False
        return True

    # ---- Phase 1: Greedy from each prime (quick baseline) ----
    best = []

    def greedy_from(start_idx):
        if start_idx >= n:
            return [primes[start_idx]]
        chain = [primes[start_idx]]
        last_gap = 0
        for j in range(start_idx + 1, n):
            p = primes[j]
            gap = p - chain[-1]
            if gap >= last_gap:
                chain.append(p)
                last_gap = gap
        return chain

    for i in range(n):
        if time.time() > deadline:
            break
        chain = greedy_from(i)
        if validate(chain) and len(chain) > len(best):
            best = chain

    # ---- Phase 2: DP with numpy for exact longest nondecreasing-gap subsequence ----
    # dp[j][g] = max chain length ending at prime j with last gap exactly g
    # We want max over all j, g of dp[j][g]
    # Transition: dp[j][g] = 1 + max_{h <= g} dp[i][h] where i = prime_index[p_j - g]
    #
    # Key insight: for a fixed j and gap g, there is at most one i (if p_j - g is prime).
    # So we iterate over valid (i, j) pairs where gap = p_j - p_i is "small enough".
    #
    # Strategy: cap max gap to keep the DP table manageable.
    # For long chains, gaps tend to be small. Let's try max_gap = 2000.
    # Actually, let's think: the sum of gaps in a chain of length L starting from ~2
    # and ending at <= 10000 is at most 10000. With nondecreasing gaps, the average
    # gap is ~10000/L. For L=15, avg gap ~667. For L=20, avg gap ~500.
    # So gaps up to ~2000 should cover most cases. Let's use max_gap = 3000 to be safe.
    # But that makes the table 1229 * 3001 * 2 bytes (int16) = ~7.4 MB, which is fine.
    
    # Actually, let's use a smarter approach. Instead of a full 2D array,
    # for each prime j, only store gaps that are achievable (i.e., p_j - p_i for some i < j).
    # And use a dictionary or sparse structure.
    
    # But the numpy approach with cum_max is cleaner. Let's use max_gap = 2000.
    # If the optimal chain has a gap > 2000, we'd miss it, but that's unlikely for long chains.
    
    max_g = 2000
    dp = np.zeros((n, max_g + 1), dtype=np.int16)
    cum_max = np.zeros((n, max_g + 1), dtype=np.int16)
    
    # Initialize: each prime alone
    dp[:, 0] = 1
    cum_max[:, 0] = 1
    # For g > 0, dp is 0 initially, cum_max will be computed
    
    # Precompute cum_max for all rows: cum_max[j][g] = max(dp[j][0..g])
    # We'll update this incrementally.
    
    best_len = 1
    best_j = 0
    best_g = 0

    for j in range(n):
        if time.time() > deadline:
            break
        p_j = primes[j]
        
        # For each possible gap g, check if p_j - g is a prime
        # g ranges from 1 to min(max_g, p_j - 2)
        max_g_for_j = min(max_g, p_j - 2)
        
        for g in range(1, max_g_for_j + 1):
            p_prev = p_j - g
            i = prime_index.get(p_prev)
            if i is None or i >= j:
                continue
            
            # dp[j][g] = 1 + cum_max[i][g]
            val = 1 + cum_max[i][g]
            if val > dp[j][g]:
                dp[j][g] = val
                if val > best_len:
                    best_len = val
                    best_j = j
                    best_g = g
        
        # Update cum_max for row j: cum_max[j][g] = max(dp[j][0..g])
        for g in range(1, max_g_for_j + 1):
            cm = cum_max[j][g - 1]
            if dp[j][g] > cm:
                cum_max[j][g] = dp[j][g]
            else:
                cum_max[j][g] = cm

    # Reconstruct chain from best_j, best_g
    if best_len > len(best):
        chain = []
        j = best_j
        g = best_g
        while j is not None:
            chain.append(primes[j])
            if g == 0:
                break
            p_prev = primes[j] - g
            i = prime_index.get(p_prev)
            if i is None:
                break
            # Find the h that achieved cum_max[i][g]
            # cum_max[i][g] = max(dp[i][0..g])
            # We need to find which h in [0, g] achieved this max
            # For reconstruction, we just need any h that works
            # Let's find the largest h <= g such that dp[i][h] == cum_max[i][g]
            # Actually, for the chain to be valid, we just need h <= g
            # The actual previous gap in the chain is h, so we need to track it
            # Let's search for h
            found_h = 0
            for h in range(g, -1, -1):
                if dp[i][h] == cum_max[i][g]:
                    found_h = h
                    break
            j = i
            g = found_h
        chain.reverse()
        if validate(chain) and len(chain) > len(best):
            best = chain

    # ---- Phase 3: Local search refinement ----
    # Try to extend or modify the best chain
    
    if best and len(best) >= 2:
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            last = best[-1]
            last_gap = best[-1] - best[-2]
            start_idx = prime_index.get(last)
            if start_idx is None:
                break
            # Try all valid next primes
            best_next = None
            best_next_len = 0
            for j in range(start_idx + 1, n):
                p = primes[j]
                gap = p - last
                if gap < last_gap:
                    continue
                # Greedily extend from p
                test_chain = best + [p]
                lg = gap
                for k in range(j + 1, n):
                    pk = primes[k]
                    gk = pk - test_chain[-1]
                    if gk >= lg:
                        test_chain.append(pk)
                        lg = gk
                if len(test_chain) > best_next_len:
                    best_next_len = len(test_chain)
                    best_next = test_chain
            if best_next and best_next_len > len(best):
                best = best_next
                improved = True

        # Try replacing the last element
        if len(best) >= 3:
            last_gap = best[-1] - best[-2]
            prev = best[-2]
            prev_idx = prime_index.get(prev)
            if prev_idx is not None:
                for j in range(prev_idx + 1, n):
                    p = primes[j]
                    gap = p - prev
                    if gap < last_gap:
                        continue
                    test_chain = best[:-1] + [p]
                    lg = gap
                    for k in range(j + 1, n):
                        pk = primes[k]
                        gk = pk - test_chain[-1]
                        if gk >= lg:
                            test_chain.append(pk)
                            lg = gk
                    if len(test_chain) > len(best):
                        best = test_chain
                        break

    if not best or not validate(best):
        best = [2, 3, 5]

    return best
# EVOLVE-BLOCK-END
