"""Combined: prime-factor bitmask for fast coprime checks + greedy with multiple orderings + local search."""
import math
import random
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    N = 512

    # Precompute prime factor bitmasks for all numbers 1..511
    # This allows O(1) coprime check via bitwise AND
    prime_factors_mask = [0] * N

    # Find all primes < 512
    is_prime = [True] * N
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N, i):
                is_prime[j] = False

    primes = [i for i in range(2, N) if is_prime[i]]
    prime_index = {p: i for i, p in enumerate(primes)}

    # Compute bitmask for each number
    for a in range(2, N):
        x = a
        mask = 0
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                mask |= (1 << prime_index[p])
                while x % p == 0:
                    x //= p
        if x > 1:
            mask |= (1 << prime_index[x])
        prime_factors_mask[a] = mask
    prime_factors_mask[1] = 0

    def score(arr):
        return sum(1.0 / (N - a) for a in arr)

    # Greedy from largest to smallest (baseline)
    def greedy_desc():
        chosen = []
        used_mask = 0
        for a in range(511, 0, -1):
            if (prime_factors_mask[a] & used_mask) == 0:
                chosen.append(a)
                used_mask |= prime_factors_mask[a]
        return chosen

    best = greedy_desc()
    best_score = score(best)

    # Precompute all numbers 1..511
    all_nums = list(range(1, 512))

    # Try different random orderings with greedy selection
    rng = random.Random(42)

    def greedy_with_order(order):
        """Greedy: iterate through order, add a if coprime with all chosen."""
        chosen = []
        used_mask = 0
        for a in order:
            if (prime_factors_mask[a] & used_mask) == 0:
                chosen.append(a)
                used_mask |= prime_factors_mask[a]
        return chosen

    # Try several random permutations (sampled, not full)
    # To be efficient, try orders biased toward large values
    for trial in range(500):
        if time.time() > deadline:
            break

        # Create a biased random order: large values more likely early
        # Use exponential weighting
        weights = [1.0 / (N - a) for a in all_nums]
        # Sample without replacement using weighted selection
        remaining = list(all_nums)
        rem_weights = list(weights)
        order = []
        for _ in range(len(remaining)):
            total_w = sum(rem_weights)
            if total_w <= 0:
                order.extend(remaining)
                break
            r = rng.random() * total_w
            cumulative = 0
            for i, w in enumerate(rem_weights):
                cumulative += w
                if cumulative >= r:
                    order.append(remaining.pop(i))
                    rem_weights.pop(i)
                    break
            else:
                order.extend(remaining)
                break

        cand = greedy_with_order(order)
        s = score(cand)
        if s > best_score:
            best_score = s
            best = cand

    # Local search: try to swap or add/remove elements
    def local_search(chosen, max_iters=200):
        chosen_set = set(chosen)
        current_score = sum(1.0 / (N - a) for a in chosen)
        used_mask = 0
        for a in chosen:
            used_mask |= prime_factors_mask[a]

        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() > deadline:
                break
            improved = False
            iters += 1

            # Try adding a new element
            # Find candidates not in chosen that are coprime with all chosen
            for a in range(511, 0, -1):
                if a in chosen_set:
                    continue
                if (prime_factors_mask[a] & used_mask) == 0:
                    # Adding improves score (always positive)
                    chosen.append(a)
                    chosen_set.add(a)
                    used_mask |= prime_factors_mask[a]
                    current_score += 1.0 / (N - a)
                    improved = True
                    break

            if improved:
                continue

            # Try multi-element swaps: remove 1-2 elements, add 1-3 elements
            # Sort chosen by contribution (ascending) so we try removing low-value elements first
            sorted_chosen = sorted(chosen, key=lambda x: 1.0 / (N - x))
            
            # First try single removal
            for a in sorted_chosen:
                if time.time() > deadline:
                    break
                # Try removing a
                remaining = [x for x in chosen if x != a]
                rem_set = set(remaining)
                rem_mask = 0
                for x in remaining:
                    rem_mask |= prime_factors_mask[x]

                # Greedily add best available
                new_chosen = list(remaining)
                new_set = set(remaining)
                new_mask = rem_mask
                new_score = sum(1.0 / (N - x) for x in new_chosen)

                for c in range(511, 0, -1):
                    if c in new_set:
                        continue
                    if (prime_factors_mask[c] & new_mask) == 0:
                        new_chosen.append(c)
                        new_set.add(c)
                        new_mask |= prime_factors_mask[c]
                        new_score += 1.0 / (N - c)

                if new_score > current_score + 1e-12:
                    chosen = new_chosen
                    chosen_set = new_set
                    used_mask = new_mask
                    current_score = new_score
                    improved = True
                    break

            if improved:
                continue

            # Try double removal: remove 2 elements, add up to 4
            if len(chosen) >= 2:
                for i in range(len(sorted_chosen)):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(sorted_chosen)):
                        a, b = sorted_chosen[i], sorted_chosen[j]
                        remaining = [x for x in chosen if x != a and x != b]
                        rem_set = set(remaining)
                        rem_mask = 0
                        for x in remaining:
                            rem_mask |= prime_factors_mask[x]

                        # Greedily add best available (up to 4 elements)
                        new_chosen = list(remaining)
                        new_set = set(remaining)
                        new_mask = rem_mask
                        new_score = sum(1.0 / (N - x) for x in new_chosen)
                        added_count = 0

                        for c in range(511, 0, -1):
                            if c in new_set:
                                continue
                            if (prime_factors_mask[c] & new_mask) == 0:
                                new_chosen.append(c)
                                new_set.add(c)
                                new_mask |= prime_factors_mask[c]
                                new_score += 1.0 / (N - c)
                                added_count += 1
                                if added_count >= 4:
                                    break

                        if new_score > current_score + 1e-12:
                            chosen = new_chosen
                            chosen_set = new_set
                            used_mask = new_mask
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break

        return chosen, current_score

    best, best_score = local_search(best)

    # More random restarts with local search
    for restart in range(200):
        if time.time() > deadline - 5:
            break

        rng_seed = 1000 + restart * 7919
        rng2 = random.Random(rng_seed)

        # Randomized greedy with more diverse strategies
        strategy = restart % 5
        
        if strategy == 0:
            # High-value range
            seed_candidates = list(range(511, 300, -1))
        elif strategy == 1:
            # Broader range
            seed_candidates = list(range(511, 200, -1))
        elif strategy == 2:
            # Even broader
            seed_candidates = list(range(511, 100, -1))
        elif strategy == 3:
            # Random subset of high-value
            seed_candidates = [a for a in range(511, 400, -1) if rng2.random() < 0.5]
        else:
            # Mixed: some high, some random
            seed_candidates = list(range(511, 450, -1)) + [rng2.randint(1, 511) for _ in range(20)]

        rng2.shuffle(seed_candidates)

        # Pick first few as initial set (vary the count)
        num_seeds = 2 + (restart % 5)  # 2-6 seeds
        chosen = []
        used_mask = 0
        for a in seed_candidates:
            if (prime_factors_mask[a] & used_mask) == 0:
                chosen.append(a)
                used_mask |= prime_factors_mask[a]
            if len(chosen) >= num_seeds:
                break

        if not chosen:
            continue

        # Complete with greedy from largest
        for a in range(511, 0, -1):
            if a in chosen:
                continue
            if (prime_factors_mask[a] & used_mask) == 0:
                chosen.append(a)
                used_mask |= prime_factors_mask[a]

        # Local search with more iterations for better convergence
        chosen, s = local_search(chosen, max_iters=150)
        if s > best_score:
            best_score = s
            best = chosen

    # One more pass: try to add any remaining coprime numbers
    best_set = set(best)
    used_mask = 0
    for a in best:
        used_mask |= prime_factors_mask[a]
    for a in range(511, 0, -1):
        if a not in best_set:
            if (prime_factors_mask[a] & used_mask) == 0:
                best.append(a)
                best_set.add(a)
                used_mask |= prime_factors_mask[a]

    # Final polish: run local search one more time with tighter tolerance
    if time.time() < deadline - 10:
        best, final_score = local_search(best, max_iters=50)
        if final_score > best_score:
            best_score = final_score

    return best

# EVOLVE-BLOCK-END
