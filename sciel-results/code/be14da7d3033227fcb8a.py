import time
import random

N = 512

# Precompute conflicts: for each x, list of triples (u,v,w) such that {x,u,v,w} is a 4-AP
_conflicts = [[] for _ in range(N + 1)]

for a in range(1, N - 2):
    for d in range(1, (N - a) // 3 + 1):
        e = a + 3 * d
        if e > N:
            break
        b = a + d
        c = a + 2 * d
        ap = (a, b, c, e)
        for i in range(4):
            x = ap[i]
            others = tuple(ap[j] for j in range(4) if j != i)
            _conflicts[x].append(others)


def _can_add(x, S):
    """Check if x can be added to set S without creating a 4-AP."""
    for u, v, w in _conflicts[x]:
        if u in S and v in S and w in S:
            return False
    return True


def _greedy(order, S=None):
    """Greedy construction given an order of elements."""
    if S is None:
        S = set()
    for x in order:
        if x not in S and _can_add(x, S):
            S.add(x)
    return S


def _local_search(S, deadline, rng):
    """
    Local search using Set representation.
    Attempts to add elements, then performs swaps to escape local optima.
    """
    current = set(S)
    current_size = len(current)
    all_elems = list(range(1, N + 1))

    while time.time() < deadline:
        improved = False

        # 1. Try to add any element
        for x in all_elems:
            if x not in current and _can_add(x, current):
                current.add(x)
                current_size += 1
                improved = True

        if improved:
            continue

        # 2. If no additions, try swaps
        attempts = 0
        max_attempts = 200

        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            x = rng.choice(all_elems)
            if x in current:
                continue

            # Check if x is blocked
            blocking_elements = set()
            is_blocked = False
            for u, v, w in _conflicts[x]:
                if u in current and v in current and w in current:
                    is_blocked = True
                    blocking_elements.update([u, v, w])

            if not is_blocked:
                # Should have been added in step 1
                current.add(x)
                current_size += 1
                improved = True
                break

            if not blocking_elements:
                continue

            # Try to remove one of the blocking elements and add x
            rem = rng.choice(list(blocking_elements))

            # Check if removing rem allows adding x
            valid_swap = True
            for u, v, w in _conflicts[x]:
                if u in current and v in current and w in current:
                    if rem not in (u, v, w):
                        valid_swap = False
                        break

            if valid_swap:
                current.discard(rem)
                current.add(x)
                improved = True
                break

        if not improved:
            # Stuck. Try a "kick": remove a few random elements and re-greedy
            if current_size > 5:
                for _ in range(3):
                    elem = rng.choice(list(current))
                    current.discard(elem)
                current_size = len(current)
                # Re-greedy fill
                for x in all_elems:
                    if x not in current and _can_add(x, current):
                        current.add(x)
                current_size = len(current)

    return current


def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    rng = random.Random(42)

    # 1. Generate multiple seeds
    seeds = []

    # Forward greedy
    seeds.append(_greedy(range(1, N + 1)))

    # Reverse greedy
    seeds.append(_greedy(range(N, 0, -1)))

    # Random greedy (a few times)
    for _ in range(5):
        order = list(range(1, N + 1))
        rng.shuffle(order)
        seeds.append(_greedy(order))

    # Find best seed
    best = max(seeds, key=len)
    best_size = len(best)

    # 2. Local search on best seed
    if time.time() < deadline - 10:
        improved = _local_search(best, deadline - 5, rng)
        if len(improved) > best_size:
            best = improved
            best_size = len(improved)

    # 3. Randomized restarts with local search
    restarts = 0
    while time.time() < deadline - 15 and restarts < 10:
        restarts += 1
        order = list(range(1, N + 1))
        rng.shuffle(order)
        seed = _greedy(order)

        if len(seed) < best_size:
            if best_size > 1:
                elem = rng.choice(list(best))
                perturbed = best - {elem}
                seed = _greedy(order, perturbed)

        if len(seed) > best_size:
            best = seed
            best_size = len(best)

        if time.time() < deadline - 5:
            improved = _local_search(seed, deadline - 2, rng)
            if len(improved) > best_size:
                best = improved
                best_size = len(improved)

    # 4. Final local search on the global best
    if time.time() < deadline - 1:
        improved = _local_search(best, deadline, rng)
        if len(improved) > best_size:
            best = improved

    # 5. Verify
    for x in best:
        for u, v, w in _conflicts[x]:
            if u in best and v in best and w in best:
                return list(_greedy(range(1, N + 1)))

    return sorted(list(best))


if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
