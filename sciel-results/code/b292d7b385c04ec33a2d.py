import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin (limit is 90s)

    N = 256

    # 1. Generate primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False

    primes = [i for i in range(2, N + 1) if is_prime[i]]

    # 2. Define base solution: Largest prime power <= N for each prime, plus 1.
    # This is a valid, non-empty, pairwise coprime set.
    # Let S_p be the largest power of prime p <= N.
    prime_power_map = {}
    base_sum = 1
    selected = {1: True}  # 1 is selected, value 1

    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_power_map[p] = power
        base_sum += power
        selected[power] = True

    best_sum = base_sum
    best_set = list(selected.keys())

    # 3. Identify potential "swaps"
    # A swap is replacing the prime powers of the prime factors of a composite C
    # with C itself, provided:
    # a) The prime powers are currently in the solution (they always are in base, unless swapped out)
    # b) C is not currently in the solution.
    # c) The new set remains pairwise coprime (which is guaranteed if we only pick disjoint sets of primes).
    # d) The sum increases.

    candidates = []
    # For each composite number c from 2 to N
    for c in range(4, N + 1):
        if not is_prime[c] and c > 1:
            # Calculate the set of primes dividing c
            div_primes = [p for p in primes if c % p == 0]
            # Calculate base value of these primes' powers
            base_val = sum(prime_power_map[p] for p in div_primes)
            gain = c - base_val
            if gain > 0:
                candidates.append((c, div_primes, gain))

    # Sort candidates by gain descending to find good solutions early
    candidates.sort(key=lambda x: x[2], reverse=True)

    num_candidates = len(candidates)

    # 4. Solve the Weighted Set Packing problem using backtracking with strong pruning.
    # We want to pick a subset of candidates such that no prime is used more than once,
    # maximizing total gain.

    # Pre-compute suffix sums of gains for upper bound pruning
    # suffix_gain[i] = sum of gains from i to end
    suffix_gain = [0] * (num_candidates + 1)
    for i in range(num_candidates - 1, -1, -1):
        suffix_gain[i] = suffix_gain[i + 1] + candidates[i][2]

    def backtrack(idx, current_gain, selected_primes, current_selections):
        nonlocal best_sum, best_set
        nonlocal deadline, start_time

        # Prune if time is up
        if time.time() > deadline - 2:
            return

        # Upper bound: current gain + sum of all remaining gains
        if current_gain + suffix_gain[idx] <= best_sum - base_sum:
            return

        if idx == num_candidates:
            if current_gain > 0:
                new_sum = base_sum + current_gain
                if new_sum > best_sum:
                    # Construct set
                    new_set_dict = {k: True for k in selected}
                    for c, div_primes, g in current_selections:
                        for p in div_primes:
                            lp = prime_power_map[p]
                            if lp in new_set_dict:
                                del new_set_dict[lp]
                        new_set_dict[c] = True
                    best_set = list(new_set_dict.keys())
                    best_sum = new_sum
            return

        c, div_primes, gain = candidates[idx]

        # Option 1: Take candidate at idx
        can_take = True
        for p in div_primes:
            if p in selected_primes:
                can_take = False
                break

        if can_take:
            new_primes = selected_primes | set(div_primes)
            backtrack(idx + 1, current_gain + gain, new_primes, current_selections + [(c, div_primes, gain)])

        # Option 2: Skip candidate at idx
        backtrack(idx + 1, current_gain, selected_primes, current_selections)

    # Run backtracking
    backtrack(0, 0, set(), [])

    # 5. Final validation
    if time.time() < deadline:
        s = set(best_set)
        if len(s) != len(best_set):
            return [1] + [prime_power_map[p] for p in primes]

        # Check pairwise coprime
        valid = True
        for i in range(len(best_set)):
            for j in range(i + 1, len(best_set)):
                if math.gcd(best_set[i], best_set[j]) != 1:
                    valid = False
                    break
            if not valid:
                break

        if not valid:
            return [1] + [prime_power_map[p] for p in primes]

        # Ensure types are int
        result = [int(x) for x in best_set]
        return result
    else:
        # Fallback to base solution if time runs out
        result = [int(x) for x in [1] + [prime_power_map[p] for p in primes]]
        return result
