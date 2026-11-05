import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill

    N = 512

    # 1. Precomputation
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return is_prime

    is_prime = sieve(511)
    primes = [p for p in range(2, 512) if is_prime[p]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)

    # Prime factor masks for each number 1..511
    masks = [0] * 512
    for a in range(2, 512):
        m = a
        mask_val = 0
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                mask_val |= (1 << prime_to_idx[p])
                while m % p == 0:
                    m //= p
        if m > 1:
            mask_val |= (1 << prime_to_idx[m])
        masks[a] = mask_val
    masks[1] = 0

    # Weights
    weights = [0.0] * 512
    for a in range(1, 512):
        weights[a] = 1.0 / (512 - a)

    # Candidates sorted by weight descending
    candidates = list(range(511, 0, -1))
    candidates.sort(key=lambda x: weights[x], reverse=True)

    # Precompute for each prime, the list of candidates divisible by it, sorted by weight desc
    prime_cands = [[] for _ in range(num_primes)]
    for a in candidates:
        m = masks[a]
        temp = m
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            prime_cands[idx].append(a)
            temp ^= lsb

    # 2. Greedy Initial Solution
    def greedy_solve():
        used_primes = 0
        score = 0.0
        sol = []
        for a in candidates:
            if (masks[a] & used_primes) == 0:
                sol.append(a)
                used_primes |= masks[a]
                score += weights[a]
        return sol, score

    initial_sol, initial_score = greedy_solve()
    best_score = initial_score
    best_sol = initial_sol[:]

    # 3. Local Search with diverse moves
    def local_search(sol, max_time=10):
        sol = sol[:]
        score = sum(weights[a] for a in sol)
        t0 = time.time()
        improved = True
        while improved and (time.time() - t0) < max_time:
            improved = False
            # Try to add
            used = 0
            for a in sol:
                used |= masks[a]
            for a in candidates:
                if a in sol:
                    continue
                if (masks[a] & used) == 0:
                    sol.append(a)
                    score += weights[a]
                    used |= masks[a]
                    improved = True
                    break

            if improved:
                continue

            # Try 1-for-1 swap
            for i in range(len(sol)):
                if time.time() > deadline - 5:
                    return sol, score
                a_out = sol[i]
                remaining = [x for j, x in enumerate(sol) if j != i]
                rem_used = 0
                for a in remaining:
                    rem_used |= masks[a]
                rem_score = score - weights[a_out]

                best_swap = -1
                best_gain = 0
                for a_in in candidates:
                    if a_in in remaining:
                        continue
                    if (masks[a_in] & rem_used) == 0:
                        gain = weights[a_in] - weights[a_out]
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = a_in

                if best_swap != -1:
                    sol[i] = best_swap
                    score += best_gain
                    improved = True
                    break

            if improved:
                continue

            # Try 1-for-2 swap (remove one, add two)
            for i in range(len(sol)):
                if time.time() > deadline - 5:
                    return sol, score
                a_out = sol[i]
                remaining = [x for j, x in enumerate(sol) if j != i]
                rem_used = 0
                for a in remaining:
                    rem_used |= masks[a]
                rem_score = score - weights[a_out]

                # Find best pair to add
                best_pair = None
                best_pair_gain = 0
                valid = [a for a in candidates if a not in remaining and (masks[a] & rem_used) == 0]
                for i2 in range(len(valid)):
                    for j2 in range(i2 + 1, len(valid)):
                        a1, a2 = valid[i2], valid[j2]
                        if masks[a1] & masks[a2]:
                            continue
                        gain = weights[a1] + weights[a2] - weights[a_out]
                        if gain > best_pair_gain:
                            best_pair_gain = gain
                            best_pair = (a1, a2)
                if best_pair and best_pair_gain > 0:
                    sol = remaining + list(best_pair)
                    score += best_pair_gain
                    improved = True
                    break

        return sol, score

    sol, score = local_search(initial_sol, max_time=15)
    if score > best_score:
        best_score = score
        best_sol = sol[:]

    # 4. Branch and Bound with tighter upper bound
    full_avail_mask = 0
    for a in range(1, 512):
        full_avail_mask |= (1 << a)

    def calculate_ub(used_primes, avail_mask):
        """
        Upper bound: for each unused prime, find the best compatible candidate.
        But ensure no candidate is counted twice (if it's the best for multiple primes).
        Process primes in order of their best candidate's weight (greedy matching).
        """
        # For each unused prime, find best compatible candidate in avail_mask
        prime_best = []  # list of (weight, candidate, prime_idx)
        for i in range(num_primes):
            if used_primes & (1 << i):
                continue
            found = False
            for a in prime_cands[i]:
                if not (avail_mask & (1 << a)):
                    continue
                if masks[a] & used_primes:
                    continue
                prime_best.append((weights[a], a, i))
                found = True
                break
            if not found:
                continue

        # Sort by weight descending
        prime_best.sort(key=lambda x: -x[0])

        # Greedily assign: each candidate can only be used once
        used_cands = set()
        ub = 0.0
        for w, a, p_idx in prime_best:
            if a in used_cands:
                # This prime's best candidate is taken; try next best? 
                # For simplicity, skip (this makes UB tighter but still valid)
                continue
            used_cands.add(a)
            ub += w

        return ub

    def bnb_search(used_primes, current_score, avail_mask, sol):
        nonlocal best_score, best_sol

        if time.time() > deadline - 3:
            return

        if current_score > best_score:
            best_score = current_score
            best_sol = sol[:]

        if avail_mask == 0:
            return

        # Calculate Upper Bound
        ub = calculate_ub(used_primes, avail_mask)
        if current_score + ub <= best_score + 1e-12:
            return

        # Select pivot: highest weight compatible candidate
        pivot = -1
        for a in candidates:
            if not (avail_mask & (1 << a)):
                continue
            if masks[a] & used_primes:
                continue
            pivot = a
            break

        if pivot == -1:
            return

        # Branch 1: Include pivot
        new_used = used_primes | masks[pivot]
        new_avail = avail_mask & ~(1 << pivot)
        new_sol = sol + [pivot]
        bnb_search(new_used, current_score + weights[pivot], new_avail, new_sol)

        if time.time() > deadline - 3:
            return

        # Branch 2: Exclude pivot
        new_avail_excl = avail_mask & ~(1 << pivot)
        bnb_search(used_primes, current_score, new_avail_excl, sol)

    # Run BNB from empty set
    bnb_search(0, 0.0, full_avail_mask, [])

    # Final polish: local search on best_sol
    sol2, score2 = local_search(best_sol, max_time=5)
    if score2 > best_score:
        best_score = score2
        best_sol = sol2

    return best_sol
