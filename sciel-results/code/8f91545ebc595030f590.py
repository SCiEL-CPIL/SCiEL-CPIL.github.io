"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt

def construct():
    deadline = time.time() + 55.0  # safety margin before 90s hard kill

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

    best = []

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

    # ---- Phase 1: Greedy from each prime ----
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

    # ---- Phase 2: DP with numpy ----
    # dp[j][g] = max length of chain ending at prime j with last gap g
    max_g = 2000  # increased from 500 to allow more gap variations
    dp = np.zeros((n, max_g + 1), dtype=np.int16)
    cum_max = np.zeros((n, max_g + 1), dtype=np.int16)

    # Initialize: each prime alone has chain length 1, gap 0
    dp[:, 0] = 1
    cum_max[:, 0] = 1

    best_len = 1
    best_j = 0
    best_g = 0

    # Process primes in order
    for j in range(n):
        if time.time() > deadline:
            break
        p_j = primes[j]

        # For each possible gap g, check if p_j - g is a prime
        # and update dp[j][g] accordingly
        for g in range(1, max_g + 1):
            p_prev = p_j - g
            if p_prev < 2:
                break
            i = prime_index.get(p_prev)
            if i is None or i >= j:
                continue
            # dp[j][g] = 1 + max_{h=0..g} dp[i][h] = 1 + cum_max[i][g]
            val = 1 + cum_max[i, g]
            if val > dp[j, g]:
                dp[j, g] = val

        # Compute cumulative max for row j
        cum_max[j, :] = np.maximum.accumulate(dp[j, :])

        # Update best
        row_max = dp[j, :].max()
        if row_max > best_len:
            best_len = int(row_max)
            best_j = j
            best_g = int(np.argmax(dp[j, :]))

    # Reconstruct chain from DP
    if best_len > len(best):
        chain = []
        j = best_j
        g = best_g
        while j is not None and g is not None:
            chain.append(primes[j])
            if g == 0:
                break
            p_prev = primes[j] - g
            i = prime_index.get(p_prev)
            if i is None:
                break
            # Find the h in [0, g] that maximized dp[i][h]
            sub = dp[i, 0:g + 1]
            h = int(np.argmax(sub))
            j = i
            g = h
        chain.reverse()
        if validate(chain) and len(chain) > len(best):
            best = chain

    # ---- Phase 3: Beam search refinement (inspired by inspiration 1) ----
    # Use beam search to try to find longer chains, starting from the best found so far
    # and also from other promising starting points

    def beam_search_from(start_chain, max_depth=8, beam_width=50, gap_window=4):
        if time.time() > deadline:
            return start_chain

        # State: (chain, last_gap)
        states = [(list(start_chain), start_chain[-1] - start_chain[-2] if len(start_chain) >= 2 else 0)]
        best_local = list(start_chain)

        for depth in range(max_depth):
            if time.time() > deadline:
                break
            next_states = []
            for chain, lg in states:
                if len(chain) >= max_depth:
                    continue
                last = chain[-1]
                idx = prime_index.get(last)
                if idx is None:
                    continue
                # Try gaps from lg to lg + gap_window*2
                lo = lg
                hi = lg + gap_window * 2
                for j in range(idx + 1, n):
                    p = primes[j]
                    if p > N:
                        break
                    gap = p - last
                    if gap < lo:
                        continue
                    if gap > hi:
                        break
                    next_states.append((chain + [p], gap))
                    if len(next_states) > beam_width * 5:
                        break
                if len(next_states) > beam_width * 5:
                    break

            if not next_states:
                break

            # Prune: keep top beam_width states
            next_states.sort(key=lambda s: (-len(s[0]), s[1], s[0][-1]))
            seen = set()
            pruned = []
            for s in next_states:
                key = tuple(s[0])
                if key in seen:
                    continue
                seen.add(key)
                pruned.append(s)
                if len(pruned) >= beam_width:
                    break
            states = pruned

            # Update best
            for chain, _ in states:
                if len(chain) > len(best_local):
                    best_local = list(chain)

        return best_local

    # Try beam search from the current best chain
    if best and len(best) >= 2:
        bc = beam_search_from(best, max_depth=10, beam_width=100, gap_window=3)
        if validate(bc) and len(bc) > len(best):
            best = bc

    # Try beam search from a few other promising starting points
    for start_p in [2, 3, 5, 7, 11, 13]:
        if time.time() > deadline:
            break
        start_chain = [start_p]
        # Greedily extend a bit to get a decent starting chain
        last_gap = 0
        idx = prime_index.get(start_p)
        if idx is not None:
            for j in range(idx + 1, min(idx + 10, n)):
                p = primes[j]
                gap = p - start_chain[-1]
                if gap >= last_gap:
                    start_chain.append(p)
                    last_gap = gap
        if len(start_chain) >= 2:
            bc = beam_search_from(start_chain, max_depth=10, beam_width=50, gap_window=3)
            if validate(bc) and len(bc) > len(best):
                best = bc

    # ---- Phase 4: Final greedy extension of best chain ----
    if best and len(best) > 0:
        last = best[-1]
        if len(best) >= 2:
            last_gap = best[-1] - best[-2]
        else:
            last_gap = 0
        start_idx = prime_index.get(last)
        if start_idx is not None:
            for j in range(start_idx + 1, n):
                p = primes[j]
                gap = p - last
                if gap >= last_gap:
                    best.append(p)
                    last_gap = gap
                    last = p

    if not validate(best):
        best = [2, 3, 5]

    return best
# EVOLVE-BLOCK-END
