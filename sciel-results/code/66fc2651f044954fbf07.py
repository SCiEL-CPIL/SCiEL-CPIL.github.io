"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random

    deadline = time.time() + 110

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S_set:
            for b in S_set:
                if a + b in S_set:
                    return False
        return True

    def can_add(S_set, x):
        """Check if x can be added to S_set while maintaining sum-free."""
        if 2 * x in S_set:
            return False
        for a in S_set:
            if a + x in S_set or x - a in S_set:
                return False
        return True

    def greedy_build(S, key_func):
        """Greedy construction using key_func for ordering."""
        S_set = set()
        result = []
        for x in sorted(S, key=key_func):
            if can_add(S_set, x):
                result.append(x)
                S_set.add(x)
        return result

    def local_search_add(S, candidates, time_budget=5.0):
        """Try to add elements from candidates to S."""
        S_set = set(S)
        result = list(S)
        start = time.time()
        improved = True
        while improved and time.time() - start < time_budget:
            improved = False
            for x in candidates:
                if x in S_set:
                    continue
                if can_add(S_set, x):
                    result.append(x)
                    S_set.add(x)
                    improved = True
                    break
        return result

    def swap_improve(S, all_elements, time_budget=5.0):
        """Try swapping elements to improve the set."""
        S_set = set(S)
        result = list(S)
        start = time.time()
        improved = True
        while improved and time.time() - start < time_budget:
            improved = False
            for i, x in enumerate(result):
                for y in all_elements:
                    if y in S_set:
                        continue
                    # Try removing x and adding y
                    new_set = S_set - {x}
                    if can_add(new_set, y):
                        result[i] = y
                        S_set = new_set | {y}
                        improved = True
                        break
                if improved:
                    break
        return result

    def solve_one(S, seed):
        n = len(S)
        rng = Random(seed)
        best = []
        best_set = set()

        # Strategy 1: Odds
        odds = [x for x in S if x % 2 != 0]
        if len(odds) > len(best):
            best = odds
            best_set = set(odds)

        # Strategy 2: Greedy by absolute value
        greedy_abs = greedy_build(S, key_func=abs)
        if len(greedy_abs) > len(best):
            best = greedy_abs
            best_set = set(greedy_abs)

        # Strategy 3: Greedy by value (smallest first)
        greedy_val = greedy_build(S, key_func=lambda x: x)
        if len(greedy_val) > len(best):
            best = greedy_val
            best_set = set(greedy_val)

        # Strategy 4: Greedy by -value (largest first)
        greedy_neg = greedy_build(S, key_func=lambda x: -x)
        if len(greedy_neg) > len(best):
            best = greedy_neg
            best_set = set(greedy_neg)

        # Strategy 5: Local search from best
        if time.time() < deadline - 10:
            candidates = [x for x in S if x not in best_set]
            improved = local_search_add(best, candidates, time_budget=3.0)
            if len(improved) > len(best):
                best = improved
                best_set = set(improved)

        # Strategy 6: Swap improvement
        if time.time() < deadline - 10:
            improved = swap_improve(best, S, time_budget=3.0)
            if len(improved) > len(best):
                best = improved
                best_set = set(improved)

        # Strategy 7: Random restarts with local search
        for _ in range(15):
            if time.time() > deadline - 5:
                break
            # Start with random subset biased toward odds
            random_subset = []
            for x in S:
                if x % 2 != 0:
                    if rng.random() < 0.7:
                        random_subset.append(x)
                else:
                    if rng.random() < 0.3:
                        random_subset.append(x)
            random_set = set(random_subset)
            candidates = [x for x in S if x not in random_set]
            improved = local_search_add(random_subset, candidates, time_budget=2.0)
            if len(improved) > len(best):
                best = improved
                best_set = set(improved)
            # Try swap on the improved set
            if time.time() < deadline - 5:
                improved2 = swap_improve(improved, S, time_budget=2.0)
                if len(improved2) > len(best):
                    best = improved2
                    best_set = set(improved2)

        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline - 5:
            S = instance(size, seed)
            results.append([x for x in S if x % 2 != 0])
        else:
            S = instance(size, seed)
            result = solve_one(S, seed)
            if not is_sum_free(result):
                result = [x for x in S if x % 2 != 0]
            results.append(result)

    return results
# EVOLVE-BLOCK-END
