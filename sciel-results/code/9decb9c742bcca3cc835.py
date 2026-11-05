import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (150s limit)

    N = 512

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

    weights = [0.0] * 512
    for a in range(1, 512):
        weights[a] = 1.0 / (512 - a)

    # Candidates sorted by weight descending (higher a = higher weight)
    candidates = list(range(511, 0, -1))
    candidates.sort(key=lambda x: weights[x], reverse=True)

    # For each prime, list of candidates divisible by it, sorted by weight desc
    prime_cands = [[] for _ in range(num_primes)]
    for a in candidates:
        temp = masks[a]
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            prime_cands[idx].append(a)
            temp ^= lsb

    # Greedy initial solution
    def greedy_solve():
        used_primes = 0
        sol = []
        for a in candidates:
            if (masks[a] & used_primes) == 0:
                sol.append(a)
                used_primes |= masks[a]
        return sol

    # Local search: add, 1-for-1 swap, 2-for-1, 1-for-2
    def local_search(sol):
        sol = sol[:]
        score = sum(weights[a] for a in sol)
        improved = True
        while improved:
            improved = False
            used = 0
            for a in sol:
                used |= masks[a]
            sol_set = set(sol)

            # Try to add
            for a in candidates:
                if a in sol_set:
                    continue
                if (masks[a] & used) == 0:
                    sol.append(a)
                    sol_set.add(a)
                    used |= masks[a]
                    score += weights[a]
                    improved = True
                    break

            if time.time() > deadline - 3:
                break

            # Try 1-for-1 swap
            for i in range(len(sol)):
                a_out = sol[i]
                rem_used = 0
                for j, x in enumerate(sol):
                    if j != i:
                        rem_used |= masks[x]
                rem_score = score - weights[a_out]
                best_swap = -1
                best_gain = 0.0
                for a_in in candidates:
                    if a_in in sol_set:
                        continue
                    if (masks[a_in] & rem_used) == 0:
                        gain = weights[a_in] - weights[a_out]
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = a_in
                if best_swap != -1:
                    sol[i] = best_swap
                    sol_set.discard(a_out)
                    sol_set.add(best_swap)
                    score += best_gain
                    improved = True
                    break

            if time.time() > deadline - 3:
                break

            # Try 1-for-2 swap (remove one, add two)
            for i in range(len(sol)):
                a_out = sol[i]
                rem_used = 0
                for j, x in enumerate(sol):
                    if j != i:
                        rem_used |= masks[x]
                rem_score = score - weights[a_out]
                best_pair = None
                best_gain = 0.0
                # Find best pair to add
                cands_rem = [a for a in candidates if a not in sol_set and (masks[a] & rem_used) == 0]
                for k1 in range(len(cands_rem)):
                    a1 = cands_rem[k1]
                    for k2 in range(k1+1, len(cands_rem)):
                        a2 = cands_rem[k2]
                        if (masks[a1] & masks[a2]) == 0:
                            gain = weights[a1] + weights[a2] - weights[a_out]
                            if gain > best_gain:
                                best_gain = gain
                                best_pair = (a1, a2)
                if best_pair is not None:
                    sol[i] = best_pair[0]
                    sol.append(best_pair[1])
                    sol_set.discard(a_out)
                    sol_set.add(best_pair[0])
                    sol_set.add(best_pair[1])
                    score += best_gain
                    improved = True
                    break

            if time.time() > deadline - 3:
                break

        return sol, score

    # Multi-start: different greedy orderings
    best_score = 0.0
    best_sol = []

    # Standard greedy
    sol0 = greedy_solve()
    sol0, s0 = local_search(sol0)
    best_score = s0
    best_sol = sol0[:]

    # Greedy with random perturbations
    import random
    random.seed(42)
    for trial in range(20):
        if time.time() > deadline - 10:
            break
        # Perturb: remove a random element, re-greedy
        sol = sol0[:]
        if len(sol) > 3:
            idx = random.randint(0, len(sol)-1)
            sol.pop(idx)
        # Re-greedy from perturbed
        used = 0
        for a in sol:
            used |= masks[a]
        for a in candidates:
            if a not in set(sol) and (masks[a] & used) == 0:
                sol.append(a)
                used |= masks[a]
        sol, s = local_search(sol)
        if s > best_score:
            best_score = s
            best_sol = sol[:]

    # Branch and bound with Lagrangian dual upper bound
    def lagrangian_ub(used_primes, avail_set, dual):
        """Upper bound using Lagrangian relaxation.
        For each available candidate, effective weight = weight - sum(dual[p] for p in prime_factors(a) not in used)
        UB = current + sum of max(0, effective_weight) for compatible candidates, but we need a valid bound.
        Simpler valid bound: for each free prime, take the best candidate using it.
        """
        ub = 0.0
        for i in range(num_primes):
            if used_primes & (1 << i):
                continue
            # Best available candidate using prime i
            for a in prime_cands[i]:
                if a not in avail_set:
                    continue
                if masks[a] & used_primes:
                    continue
                # Effective weight with dual pricing
                eff = weights[a]
                temp = masks[a]
                while temp:
                    lsb = temp & -temp
                    pi = lsb.bit_length() - 1
                    if not (used_primes & (1 << pi)):
                        eff -= dual[pi]
                    temp ^= lsb
                ub += max(0.0, eff)
                break
        return ub

    def bnb(used_primes, current_score, avail_mask, sol):
        nonlocal best_score, best_sol

        if time.time() > deadline - 2:
            return

        if current_score > best_score:
            best_score = current_score
            best_sol = sol[:]

        if avail_mask == 0:
            return

        # Find pivot: highest weight compatible candidate
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

        # Upper bound: greedy sum of best per free prime
        ub = 0.0
        for i in range(num_primes):
            if used_primes & (1 << i):
                continue
            for a in prime_cands[i]:
                if not (avail_mask & (1 << a)):
                    continue
                if masks[a] & used_primes:
                    continue
                ub += weights[a]
                break

        if current_score + ub <= best_score:
            return

        # Branch: include pivot
        new_used = used_primes | masks[pivot]
        new_avail = avail_mask & ~(1 << pivot)
        bnb(new_used, current_score + weights[pivot], new_avail, sol + [pivot])

        if time.time() > deadline - 2:
            return

        # Branch: exclude pivot
        bnb(used_primes, current_score, avail_mask & ~(1 << pivot), sol)

    # Run BNB from the best solution found
    # Start BNB trying to improve on best_sol
    # Build avail_mask excluding best_sol elements (they're already "used")
    # Actually, let's run BNB fresh but with good initial bound
    full_avail = 0
    for a in range(1, 512):
        full_avail |= (1 << a)

    # Limit BNB depth/time: only explore promising branches
    # Use a node counter to limit
    node_count = [0]
    max_nodes = 500000

    def bnb_limited(used_primes, current_score, avail_mask, sol):
        nonlocal best_score, best_sol

        if time.time() > deadline - 2:
            return
        node_count[0] += 1
        if node_count[0] > max_nodes:
            return

        if current_score > best_score:
            best_score = current_score
            best_sol = sol[:]

        if avail_mask == 0:
            return

        # Find pivot
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

        # Upper bound
        ub = 0.0
        for i in range(num_primes):
            if used_primes & (1 << i):
                continue
            for a in prime_cands[i]:
                if not (avail_mask & (1 << a)):
                    continue
                if masks[a] & used_primes:
                    continue
                ub += weights[a]
                break

        if current_score + ub <= best_score + 1e-12:
            return

        # Branch include
        new_used = used_primes | masks[pivot]
        new_avail = avail_mask & ~(1 << pivot)
        bnb_limited(new_used, current_score + weights[pivot], new_avail, sol + [pivot])

        if time.time() > deadline - 2:
            return

        # Branch exclude
        bnb_limited(used_primes, current_score, avail_mask & ~(1 << pivot), sol)

    bnb_limited(0, 0.0, full_avail, [])

    # Final local search polish
    best_sol, final_score = local_search(best_sol)

    return best_sol
