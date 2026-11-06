"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256
import random
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55  # 55s safety margin for 60s limit
    best_result = None
    best_size = float('inf')

    def is_sidon(s, sums):
        """Verify s is Sidon using precomputed sums."""
        s_set = set(s)
        if len(s_set) != len(s):
            return False
        # Check all pair sums are distinct
        seen = set()
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                sm = s[i] + s[j]
                if sm in seen:
                    return False
                seen.add(sm)
        return True

    def compute_sums(s):
        s_set = set(s)
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.add(s[i] + s[j])
        return sums

    def is_maximal(s, sums):
        """Check if s is inclusion-maximal Sidon in [1,N]."""
        s_set = set(s)
        for x in range(1, N + 1):
            if x in s_set:
                continue
            # Check if x can be added
            new_sums = {2 * x}
            for a in s:
                new_sums.add(x + a)
            if sums.isdisjoint(new_sums):
                return False
        return True

    def try_remove_element(s, sums, x):
        """Try removing x from s. Returns (new_s, new_sums) or (None, None) if not maximal."""
        if x not in s:
            return None, None
        new_s = [a for a in s if a != x]
        new_sums = compute_sums(new_s)
        # Check if still Sidon (should be, since removing can't create conflicts)
        # Check if still maximal
        if is_maximal(new_s, new_sums):
            return new_s, new_sums
        return None, None

    def local_search_minimize(s, sums, max_iters=200):
        """Try to minimize size by removing elements that maintain maximality."""
        s = list(s)
        sums = set(sums)
        improved = True
        iterations = 0
        while improved and iterations < max_iters:
            improved = False
            iterations += 1
            # Try removing each element
            for x in list(s):
                new_s, new_sums = try_remove_element(s, sums, x)
                if new_s is not None:
                    s = new_s
                    sums = new_sums
                    improved = True
                    break
        return s, sums

    def greedy_construct_minimal(seed_rng, target_size):
        """Construct a Sidon set of approximately target_size using greedy with blocking."""
        s = []
        sums = set()
        candidates = list(range(1, N + 1))
        seed_rng.shuffle(candidates)

        for x in candidates:
            if len(s) >= target_size:
                break
            new_sums = {2 * x}
            for a in s:
                new_sums.add(x + a)
            if sums.isdisjoint(new_sums):
                s.append(x)
                sums.update(new_sums)
        return s, sums

    def make_maximal_optimal(s, sums):
        """Make s maximal by adding minimal elements. Uses look-ahead to minimize additions."""
        s = list(s)
        sums = set(sums)
        s_set = set(s)

        while True:
            # Find all unblocked elements
            unblocked = []
            for x in range(1, N + 1):
                if x not in s_set:
                    new_sums_x = {2 * x}
                    for a in s:
                        new_sums_x.add(x + a)
                    if sums.isdisjoint(new_sums_x):
                        unblocked.append(x)

            if not unblocked:
                break

            # For each candidate, compute how many unblocked elements it blocks
            best_x = None
            best_block_count = -1
            for x in unblocked:
                new_sums_x = {2 * x}
                for a in s:
                    new_sums_x.add(x + a)

                # Count how many other unblocked elements would be blocked
                block_count = 0
                for y in unblocked:
                    if y == x:
                        continue
                    # Check if y would be blocked after adding x
                    # y is blocked if 2y in (sums ∪ new_sums_x) or y+a in (sums ∪ new_sums_x) for some a in s∪{x}
                    blocked = False
                    if 2 * y in sums or 2 * y in new_sums_x:
                        blocked = True
                    else:
                        for a in s:
                            if y + a in sums:
                                blocked = True
                                break
                        if not blocked:
                            if y + x in sums or y + x in new_sums_x:
                                blocked = True
                    if blocked:
                        block_count += 1

                if block_count > best_block_count:
                    best_block_count = block_count
                    best_x = x

            if best_x is None:
                break

            # Add best_x
            new_sums = {2 * best_x}
            for a in s:
                new_sums.add(best_x + a)
            if sums.isdisjoint(new_sums):
                s.append(best_x)
                s_set.add(best_x)
                sums.update(new_sums)
            else:
                break

        return s, sums

    # Main search: try various strategies
    random.seed(42)

    # Strategy 1: Random Sidon sets of various sizes, then make maximal, then minimize
    for trial in range(800):
        if time.time() > deadline:
            break

        target_size = random.randint(6, 20)
        s, sums = greedy_construct_minimal(random, target_size)

        if len(s) < 3:
            continue

        # Make maximal
        s_max, sums_max = make_maximal_optimal(s, sums)

        # Verify maximality
        if not is_maximal(s_max, sums_max):
            continue

        # Local search to minimize
        s_min, sums_min = local_search_minimize(s_max, sums_max)

        # Final verification
        if not is_sidon(s_min, sums_min):
            continue
        if not is_maximal(s_min, sums_min):
            continue

        size = len(s_min)
        if size < best_size:
            best_size = size
            best_result = s_min
            if size <= 10:
                break

    # Strategy 2: Direct construction trying to keep size small
    for trial in range(300):
        if time.time() > deadline:
            break

        # Build a small Sidon set, then aggressively make it maximal with minimal additions
        s = []
        sums = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)

        # Start with 3-5 random elements
        start_size = random.randint(3, 5)
        for x in candidates[:start_size * 3]:
            if len(s) >= start_size:
                break
            new_sums = {2 * x}
            for a in s:
                new_sums.add(x + a)
            if sums.isdisjoint(new_sums):
                s.append(x)
                sums.update(new_sums)

        if len(s) < 3:
            continue

        # Make maximal with minimal additions
        s_max, sums_max = make_maximal_optimal(s, sums)

        if not is_maximal(s_max, sums_max):
            continue

        # Local search
        s_min, sums_min = local_search_minimize(s_max, sums_max)

        if not is_sidon(s_min, sums_min) or not is_maximal(s_min, sums_min):
            continue

        size = len(s_min)
        if size < best_size:
            best_size = size
            best_result = s_min
            if size <= 10:
                break

    # Fallback: deterministic ascending greedy
    if best_result is None:
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        # Make maximal
        s_max, sums_max = make_maximal_optimal(selected, sums)
        # Local search
        s_min, sums_min = local_search_minimize(s_max, sums_max)
        if is_sidon(s_min, sums_min) and is_maximal(s_min, sums_min):
            return s_min
        return s_max

    return best_result
# EVOLVE-BLOCK-END
