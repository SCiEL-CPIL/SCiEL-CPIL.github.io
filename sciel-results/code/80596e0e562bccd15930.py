"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    t0 = time.time()
    TIME_LIMIT = 140.0  # safety margin under 150s

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        Sset = set(S)
        for i, a in enumerate(S):
            for b in S:
                if a + b in Sset:
                    return False
        return True

    def conflicts_with(x, Sset):
        """Check if adding x to Sset creates a sum-free violation."""
        for a in Sset:
            if a + x in Sset:
                return True
            if x + a in Sset:
                return True
        # Also check x + x in Sset
        if x + x in Sset:
            return True
        return False

    def greedy_build(S, rng, start_set=None):
        """Greedily build a sum-free subset."""
        if start_set is not None:
            Sset = set(start_set)
        else:
            Sset = set()
        remaining = list(S)
        rng.shuffle(remaining)
        for x in remaining:
            if x not in Sset and not conflicts_with(x, Sset):
                Sset.add(x)
        return Sset

    def local_search(S, rng, max_iters=50000):
        """Local search: try to add elements or swap to improve size."""
        Sset = set()
        # Start with greedy
        remaining = list(S)
        rng.shuffle(remaining)
        for x in remaining:
            if not conflicts_with(x, Sset):
                Sset.add(x)

        best = len(Sset)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try to add any element not in Sset
            for x in S:
                if x not in Sset and not conflicts_with(x, Sset):
                    Sset.add(x)
                    best = len(Sset)
                    improved = True
                    break
            if improved:
                continue
            # Try swap: remove one element, then try to add others
            Slist = list(Sset)
            rng.shuffle(Slist)
            for remove_x in Slist:
                trial = Sset - {remove_x}
                added = 0
                for x in S:
                    if x not in trial and not conflicts_with(x, trial):
                        trial.add(x)
                        added += 1
                if len(trial) > len(Sset):
                    Sset = trial
                    best = len(Sset)
                    improved = True
                    break
        return Sset

    def solve_instance(S, rng, time_budget):
        """Solve one instance with local search within time budget."""
        best_set = set()
        best_size = 0

        # Multiple initial strategies
        strategies = []

        # Strategy 1: Greedy from random order
        s1 = greedy_build(S, rng)
        strategies.append(s1)

        # Strategy 2: Greedy preferring large absolute values
        s2 = set()
        for x in sorted(S, key=lambda v: -abs(v)):
            if not conflicts_with(x, s2):
                s2.add(x)
        strategies.append(s2)

        # Strategy 3: Greedy preferring positive values
        s3 = set()
        for x in sorted(S, key=lambda v: -v):
            if not conflicts_with(x, s3):
                s3.add(x)
        strategies.append(s3)

        # Strategy 4: Greedy preferring odd values
        s4 = set()
        for x in [v for v in S if v % 2 != 0]:
            if not conflicts_with(x, s4):
                s4.add(x)
        # Then try to add evens
        for x in [v for v in S if v % 2 == 0]:
            if not conflicts_with(x, s4):
                s4.add(x)
        strategies.append(s4)

        # Strategy 5: Random restarts
        for _ in range(3):
            s = greedy_build(S, rng)
            strategies.append(s)

        # Keep the best initial
        for s in strategies:
            if len(s) > best_size:
                best_size = len(s)
                best_set = set(s)

        # Local search on the best
        if time.time() - t0 < time_budget:
            result = local_search(S, rng, max_iters=100000)
            if len(result) > best_size:
                best_set = result
                best_size = len(result)

        # Additional local search iterations if time permits
        while time.time() - t0 < time_budget:
            Slist = list(best_set)
            rng.shuffle(Slist)
            improved = False
            for remove_x in Slist:
                trial = best_set - {remove_x}
                for x in S:
                    if x not in trial and not conflicts_with(x, trial):
                        trial.add(x)
                if len(trial) > len(best_set):
                    best_set = trial
                    improved = True
                    break
            if not improved:
                # Try adding
                for x in S:
                    if x not in best_set and not conflicts_with(x, best_set):
                        best_set.add(x)
                        improved = True
                        break
            if not improved:
                break

        return list(best_set)

    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    for size, seed in instances:
        S = instance(size, seed)
        rng = Random(seed * 7919 + 12345)  # separate RNG for search
        # Allocate time: give more to larger instances
        remaining = TIME_LIMIT - (time.time() - t0)
        time_budget = remaining * (size / 420.0)  # proportional allocation
        results.append(solve_instance(S, rng, time_budget))

    return results
# EVOLVE-BLOCK-END
