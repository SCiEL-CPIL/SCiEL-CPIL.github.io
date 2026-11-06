"""Rewritten program: local search for large sum-free subsets."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 110.0  # safety margin before 120s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c with a,b,c in S (a=b allowed)."""
        s = set(S)
        for i, a in enumerate(S):
            for b in S:
                if a + b in s:
                    return False
        return True

    def conflicts(S):
        """Return set of elements that cannot be added to S without violating sum-free."""
        s = set(S)
        blocked = set()
        for a in S:
            for b in S:
                c = a + b
                if c in s:
                    # c is already in S, so a+b=c is a conflict within S
                    # but for adding new elements, we need to know which new x would conflict
                    pass
        # For a candidate x, x conflicts with S if:
        #   x + a in S for some a in S, OR
        #   a + b = x for some a,b in S
        return blocked

    def greedy_construct(S, order):
        """Greedy: add elements in given order if sum-free is maintained."""
        result = []
        s = set()
        for x in order:
            # Check if adding x keeps sum-free
            ok = True
            for a in result:
                if a + x in s or x + a in s:
                    ok = False
                    break
                if x + x in s:
                    ok = False
                    break
            if ok:
                result.append(x)
                s.add(x)
        return result

    def greedy_with_conflicts(S, rng):
        """Greedy construction with random tie-breaking and conflict-aware ordering."""
        remaining = list(S)
        rng.shuffle(remaining)
        result = []
        s = set()
        for x in remaining:
            # Check if adding x keeps sum-free
            ok = True
            for a in result:
                if (a + x) in s:
                    ok = False
                    break
            if x + x in s:
                ok = False
            if ok:
                result.append(x)
                s.add(x)
        return result

    def local_search(S, rng, max_iter=5000):
        """Hill-climbing: try to improve a sum-free subset by removing one element and re-adding blocked ones."""
        current = list(S)
        current_set = set(current)
        best = list(current)
        best_size = len(current)

        for _ in range(max_iter):
            if time.time() - start_time > TIME_LIMIT:
                break
            if not current:
                break
            # Pick a random element to remove
            idx = rng.randint(0, len(current) - 1)
            removed = current[idx]
            current.pop(idx)
            current_set.discard(removed)
            new_size = len(current)

            # Try to re-add elements that were previously blocked
            # We don't track blocked explicitly, so try all elements not in current
            # that could potentially be added
            added_any = False
            for x in S:
                if x in current_set:
                    continue
                # Check if adding x keeps sum-free
                ok = True
                for a in current:
                    if (a + x) in current_set:
                        ok = False
                        break
                if x + x in current_set:
                    ok = False
                if ok:
                    current.append(x)
                    current_set.add(x)
                    new_size += 1
                    added_any = True

            if new_size > best_size:
                best = list(current)
                best_size = new_size
            elif new_size < len(best):
                # Revert: put removed back
                current.append(removed)
                current_set.add(removed)
                # Remove any newly added that made it worse? No, we keep best separately.
                # Actually we should revert to the state before this move if it's worse
                # But we already modified current... Let's just track best and continue.
                # The current state is now smaller, which is bad. Let's revert.
                # Undo the additions
                for x in current:
                    if x not in best and x != removed:
                        # This was added in this step
                        pass
                # Simpler: just revert to before this iteration
                # This is getting complicated. Let me restructure.
                pass

        return best

    def improved_local_search(S, rng, max_iter=10000):
        """Better local search: maintain state properly."""
        current = list(S)
        current_set = set(current)
        best = list(current)
        best_size = len(current)
        n = len(S)
        S_set = set(S)

        for iteration in range(max_iter):
            if time.time() - start_time > TIME_LIMIT:
                break
            if not current:
                break

            # Save state
            old_current = list(current)
            old_current_set = set(current_set)
            old_size = len(current)

            # Pick a random element to remove
            idx = rng.randint(0, len(current) - 1)
            removed = current[idx]
            current.pop(idx)
            current_set.discard(removed)

            # Try to re-add elements not in current
            for x in S:
                if x in current_set:
                    continue
                ok = True
                for a in current:
                    if (a + x) in current_set:
                        ok = False
                        break
                if x + x in current_set:
                    ok = False
                if ok:
                    current.append(x)
                    current_set.add(x)

            new_size = len(current)
            if new_size > best_size:
                best = list(current)
                best_size = new_size
            else:
                # Revert
                current = old_current
                current_set = old_current_set

        return best

    def solve_instance(S, seed):
        """Find a large sum-free subset of S."""
        rng = random.Random(seed + 1000)
        S_list = list(S)
        n = len(S_list)
        best = []
        best_size = 0

        # Multiple greedy starts
        for trial in range(50):
            if time.time() - start_time > TIME_LIMIT:
                break
            # Different orderings
            perm = list(S_list)
            rng.shuffle(perm)
            # Also try sorted by absolute value, or random
            if trial % 3 == 0:
                perm = sorted(perm, key=lambda x: abs(x))
            elif trial % 3 == 1:
                perm = sorted(perm, key=lambda x: -abs(x))
            else:
                pass  # already shuffled

            result = greedy_with_conflicts(S_list, rng)
            if len(result) > best_size:
                best = result
                best_size = len(result)

        # Local search from best
        if best_size > 0:
            improved = improved_local_search(S_list, rng, max_iter=20000)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)

        # More local search iterations if time permits
        if time.time() - start_time < TIME_LIMIT - 10:
            improved2 = improved_local_search(S_list, rng, max_iter=50000)
            if len(improved2) > best_size:
                best = improved2
                best_size = len(best)

        return best

    # Solve each instance
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        subset = solve_instance(S, seed)
        # Verify sum-free
        if not is_sum_free(subset):
            # Fallback to odd numbers
            subset = [x for x in S if x % 2]
        results.append(subset)

    return results
# EVOLVE-BLOCK-END
