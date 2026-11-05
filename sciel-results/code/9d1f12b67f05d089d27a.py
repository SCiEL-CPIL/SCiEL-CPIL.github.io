import time
import random

def construct():
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S_set:
            for b in S_set:
                if a + b in S_set:
                    return False
        return True

    def can_add(x, S_set):
        for a in S_set:
            if (x - a) in S_set:
                return False
        for a in S_set:
            if (a + x) in S_set:
                return False
        if (2 * x) in S_set:
            return False
        return True

    def greedy_build(S, order=None):
        """Build a sum-free set greedily from a given order."""
        result = set()
        for x in order:
            if can_add(x, result):
                result.add(x)
        return result

    def get_suffix(S):
        n = len(S)
        best_k = 1
        for k in range(1, n + 1):
            if 2 * S[n - k] > S[n - 1]:
                best_k = k
            else:
                break
        return S[n - best_k: n]

    def greedy_improve(S_set, all_elements, time_limit):
        """Iteratively try to add elements from the complement."""
        S_set = set(S_set)
        remaining = [x for x in all_elements if x not in S_set]
        start_time = time.time()
        while True:
            if time.time() - start_time > time_limit:
                break
            added_any = False
            remaining.sort(key=lambda x: sum(1 for a in S_set if (x - a) in S_set or (a + x) in S_set or (2 * x) in S_set))
            for x in remaining[:]:
                if x in S_set:
                    continue
                if can_add(x, S_set):
                    S_set.add(x)
                    remaining.remove(x)
                    added_any = True
            if not added_any:
                break
        return S_set

    def local_search_single(S_set, all_elements, time_limit):
        """Try to improve by removing one element and adding others."""
        S_set = set(S_set)
        start_time = time.time()
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for remove_candidate in list(S_set):
                if time.time() - start_time > time_limit:
                    break
                S_set.remove(remove_candidate)
                candidates = [x for x in all_elements if x not in S_set]
                candidates.sort(key=lambda x: sum(1 for a in S_set if (x - a) in S_set or (a + x) in S_set or (2 * x) in S_set))
                added = 0
                for x in candidates:
                    if can_add(x, S_set):
                        S_set.add(x)
                        added += 1
                if added > 1:
                    improved = True
                else:
                    S_set.add(remove_candidate)
        return S_set

    def local_search_pair(S_set, all_elements, time_limit):
        """Try to improve by removing two elements and adding others."""
        S_set = set(S_set)
        start_time = time.time()
        improved = True
        elements = list(S_set)
        while improved and time.time() - start_time < time_limit:
            improved = False
            elements = list(S_set)
            n = len(elements)
            for i in range(n):
                if time.time() - start_time > time_limit:
                    break
                for j in range(i + 1, n):
                    if time.time() - start_time > time_limit:
                        break
                    trial = set(S_set)
                    trial.discard(elements[i])
                    trial.discard(elements[j])
                    candidates = [x for x in all_elements if x not in trial]
                    candidates.sort(key=lambda x: sum(1 for a in trial if (x - a) in trial or (a + x) in trial or (2 * x) in trial))
                    added = 0
                    for x in candidates:
                        if can_add(x, trial):
                            trial.add(x)
                            added += 1
                    if len(trial) > len(S_set):
                        S_set = trial
                        improved = True
                        break
                if improved:
                    break
        return S_set

    def perturbation_restart(S_set, all_elements, time_limit, num_restarts=15):
        """Try random perturbations to escape local optima."""
        S_set = set(S_set)
        start_time = time.time()
        for _ in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
            perturbed = set(S_set)
            for _ in range(random.randint(1, 3)):
                if perturbed:
                    perturbed.discard(random.choice(list(perturbed)))
            remaining = [x for x in all_elements if x not in perturbed]
            random.shuffle(remaining)
            for x in remaining:
                if can_add(x, perturbed):
                    perturbed.add(x)
            if len(perturbed) > len(S_set):
                S_set = perturbed
        return S_set

    def solve_instance(S, time_limit):
        start_time = time.time()
        n = len(S)
        all_elements = S
        S_set = set(S)

        # Generate multiple initializations
        inits = []
        inits.append(greedy_build(S, S))
        inits.append(greedy_build(S, list(reversed(S))))
        inits.append(greedy_build(S, sorted(S, key=lambda x: -abs(x))))
        inits.append(greedy_build(S, sorted(S, key=lambda x: abs(x))))
        inits.append(greedy_build(S, sorted(S, key=lambda x: x)))
        inits.append(greedy_build(S, sorted(S, key=lambda x: -x)))
        odds = [x for x in S if x % 2 != 0]
        inits.append(set(odds))
        suffix = get_suffix(S)
        inits.append(set(suffix))

        # Add random shuffles
        rng_local = random.Random(42)
        for _ in range(8):
            shuffled = list(S)
            rng_local.shuffle(shuffled)
            inits.append(greedy_build(S, shuffled))

        best = max(inits, key=len)

        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 2
        if remaining_time <= 0:
            return list(best)

        # Greedy improve
        best = greedy_improve(best, all_elements, min(remaining_time, 5))
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 2
        if remaining_time <= 0:
            return list(best)

        # Single-removal local search
        best = local_search_single(best, all_elements, min(remaining_time, 8))
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 2
        if remaining_time <= 0:
            return list(best)

        # Pair-removal local search
        best = local_search_pair(best, all_elements, min(remaining_time, 10))
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 2
        if remaining_time <= 0:
            return list(best)

        # Perturbation restarts if time remains
        if remaining_time > 3:
            best = perturbation_restart(best, all_elements, remaining_time - 2)

        # Final greedy improve
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 1
        if remaining_time > 0:
            best = greedy_improve(best, all_elements, remaining_time)

        return list(best)

    # Instances
    instances = [
        instance(60, 801),
        instance(120, 802),
        instance(240, 803)
    ]

    results = []
    total_time = 140  # Safety margin before 150s kill
    start = time.time()

    # Allocate time: smaller instances get less, larger get more
    time_alloc = [25, 40, 70]

    for i, S in enumerate(instances):
        elapsed = time.time() - start
        remaining_time = total_time - elapsed

        if remaining_time <= 0:
            results.append([x for x in S if x % 2 != 0])
            continue

        time_limit = min(time_alloc[i], remaining_time - 5)
        if time_limit <= 0:
            results.append([x for x in S if x % 2 != 0])
            continue

        res = solve_instance(S, time_limit)
        results.append(res)

    # Final verification
    for i, res in enumerate(results):
        if not is_sum_free(res):
            results[i] = [x for x in instances[i] if x % 2 != 0]

    return results
