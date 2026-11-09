import time
import random

def construct():
    start = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    def is_valid(subset):
        reachable = 1
        for x in subset:
            new_reachable = reachable | (reachable << x)
            if (new_reachable >> 200) & 1:
                return False
            reachable = new_reachable
        return True

    def greedy(order, existing=None):
        if existing is None:
            existing = set()
        reachable = 1
        selected = set(existing)
        for x in order:
            if x in selected:
                continue
            new_reachable = reachable | (reachable << x)
            if not ((new_reachable >> 200) & 1):
                selected.add(x)
                reachable = new_reachable
        return selected

    def local_search(best, deadline):
        """Improve via swap and add moves"""
        best = set(best)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding elements
            for x in range(1, 101):
                if x in best:
                    continue
                candidate = best | {x}
                if is_valid(candidate):
                    best = candidate
                    improved = True
                    break
            if time.time() >= deadline:
                break
            # Try swapping: remove one, add one
            for remove_x in sorted(best):
                if time.time() >= deadline:
                    break
                remaining = best - {remove_x}
                for add_x in range(1, 101):
                    if add_x in remaining:
                        continue
                    candidate = remaining | {add_x}
                    if is_valid(candidate):
                        best = candidate
                        improved = True
                        break
                if improved:
                    break
        return best

    # Generate diverse initial solutions
    random.seed(42)
    candidates = []

    # Strategy 1: Greedy from top
    candidates.append(greedy(range(100, 0, -1)))

    # Strategy 2: Greedy from bottom
    candidates.append(greedy(range(1, 101)))

    # Strategy 3: Greedy from middle out
    mid_order = list(range(50, 0, -1)) + list(range(51, 101))
    candidates.append(greedy(mid_order))

    # Strategy 4: Greedy prioritizing large numbers first (to avoid small subset sums)
    candidates.append(greedy(sorted(range(1, 101), key=lambda x: -x)))

    # Strategy 5-10: Random orderings
    for i in range(10):
        order = list(range(1, 101))
        random.shuffle(order)
        candidates.append(greedy(order))

    # Strategy 11-20: Weighted random (favor larger numbers)
    for i in range(10):
        weights = [1.0 / x for x in range(1, 101)]  # favor small... wait, let's favor large
        weights = [float(x) for x in range(1, 101)]  # favor large
        order = random.sample(range(1, 101), 100, weights=weights)
        candidates.append(greedy(order))

    # Pick best initial
    best = max(candidates, key=len)
    best_size = len(best)

    # Local search to improve
    best = local_search(best, DEADLINE - 2.0)

    # Try multiple local search restarts from different initial solutions
    if time.time() < DEADLINE - 5.0:
        for cand in sorted(candidates, key=len, reverse=True)[:5]:
            if time.time() >= DEADLINE - 3.0:
                break
            if len(cand) < best_size:
                continue
            improved = local_search(set(cand), DEADLINE - 2.0)
            if len(improved) > len(best):
                best = improved
                best_size = len(best)

    # Final attempt: try to add any remaining elements
    for x in range(1, 101):
        if time.time() >= DEADLINE - 1.0:
            break
        if x in best:
            continue
        candidate = best | {x}
        if is_valid(candidate):
            best = candidate

    # Final verification
    assert len(best) > 0
    assert all(1 <= x <= 100 for x in best)
    assert len(best) == len(set(best))

    # Verify no subset sums to 200
    reachable = 1
    for x in sorted(best):
        new_reachable = reachable | (reachable << x)
        if (new_reachable >> 200) & 1:
            # Should not happen - return best effort
            return list(best)
        reachable = new_reachable

    return list(best)
