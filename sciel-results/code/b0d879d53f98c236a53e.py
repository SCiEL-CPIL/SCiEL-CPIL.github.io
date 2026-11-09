"""B2[2] set in 1..1000: maximize size via randomized greedy + local search with swaps."""
import time
import random

_START_TIME = time.time()
_DEADLINE = 110.0  # safety margin before 120s kill

def _elapsed():
    return time.time() - _START_TIME

def _valid_insertion(counts, x, chosen_set):
    """Return None if invalid, else the list of sums to update and their new counts (delta dict)."""
    for y in chosen_set:
        s = x + y
        if counts.get(s, 0) >= 2:
            return None
    if counts.get(2 * x, 0) >= 2:
        return None
    delta = {}
    for y in chosen_set:
        s = x + y
        delta[s] = delta.get(s, 0) + 1
    delta[2 * x] = delta.get(2 * x, 0) + 1
    return delta

def _remove_element(counts, x, chosen_set):
    """Remove x from chosen_set and update counts. Returns True on success."""
    if x not in chosen_set:
        return False
    # For each y in chosen_set (y != x), the pair (min(x,y), max(x,y)) contributed to sum x+y
    # Also 2*x contributed from (x,x)
    for y in chosen_set:
        if y == x:
            continue
        s = x + y
        counts[s] = counts.get(s, 0) - 1
        if counts[s] <= 0:
            del counts[s]
    # Remove 2*x
    s = 2 * x
    counts[s] = counts.get(s, 0) - 1
    if counts[s] <= 0:
        del counts[s]
    chosen_set.discard(x)
    return True

def _build_greedy(order, counts, chosen):
    """Greedy add elements in given order, updating counts and chosen in place."""
    chosen_set = set(chosen)
    for x in order:
        if _elapsed() > _DEADLINE:
            return
        delta = _valid_insertion(counts, x, chosen_set)
        if delta is not None:
            for s, d in delta.items():
                counts[s] = counts.get(s, 0) + d
            chosen.append(x)
            chosen_set.add(x)

def _local_search(counts, chosen, chosen_set, max_steps=200):
    """Local search: try swaps (remove one, add another) to increase size or maintain size with better structure."""
    steps = 0
    improved = True
    while improved and steps < max_steps and _elapsed() < _DEADLINE - 2.0:
        improved = False
        steps += 1
        # Try each element in chosen as a removal candidate
        # For each, try to add a random candidate from universe
        for i in range(len(chosen)):
            if _elapsed() > _DEADLINE - 2.0:
                return
            x_remove = chosen[i]
            # Try a few random candidates to add
            for _ in range(5):
                if _elapsed() > _DEADLINE - 2.0:
                    return
                x_add = random.randint(1, 1000)
                if x_add == x_remove or x_add in chosen_set:
                    continue
                # Check if we can remove x_remove and add x_add
                # First check if x_add can be inserted into (chosen - {x_remove})
                temp_set = set(chosen_set)
                temp_set.discard(x_remove)
                # Check validity of inserting x_add into temp_set with adjusted counts
                # We need to check: for all y in temp_set, counts[x_add+y] - (1 if x_remove+y was counted) < 2
                # And counts[2*x_add] - (1 if 2*x_add == 2*x_remove) < 2
                # This is complex; simpler: actually perform the swap tentatively
                # Save state
                saved_counts = dict(counts)
                saved_chosen = list(chosen)
                saved_set = set(chosen_set)
                
                # Remove x_remove
                _remove_element(counts, x_remove, chosen_set)
                # Try to add x_add
                delta = _valid_insertion(counts, x_add, chosen_set)
                if delta is not None:
                    # Valid swap: net size same, but we may have freed up constraints
                    # Actually this doesn't increase size. Let's try a different approach:
                    # Try to add x_add WITHOUT removing anything first (pure addition)
                    pass
                # Restore
                counts.clear()
                counts.update(saved_counts)
                chosen.clear()
                chosen.extend(saved_chosen)
                chosen_set.clear()
                chosen_set.update(saved_set)
            # Try pure addition of a random element
            for _ in range(3):
                if _elapsed() > _DEADLINE - 2.0:
                    return
                x_add = random.randint(1, 1000)
                if x_add in chosen_set:
                    continue
                delta = _valid_insertion(counts, x_add, chosen_set)
                if delta is not None:
                    for s, d in delta.items():
                        counts[s] = counts.get(s, 0) + d
                    chosen.append(x_add)
                    chosen_set.add(x_add)
                    improved = True
                    break

def _local_fill(counts, chosen, chosen_set):
    """Try to add remaining candidates in descending order."""
    for x in range(1000, 0, -1):
        if _elapsed() > _DEADLINE - 1.0:
            break
        if x in chosen_set:
            continue
        delta = _valid_insertion(counts, x, chosen_set)
        if delta is not None:
            for s, d in delta.items():
                counts[s] = counts.get(s, 0) + d
            chosen.append(x)
            chosen_set.add(x)

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = min(110.0, _DEADLINE)
    random.seed(0)
    best = None
    best_size = -1

    universe = list(range(1, 1001))

    iters = 0
    max_iters = 5000

    while iters < max_iters and _elapsed() < _DEADLINE - 3.0:
        iters += 1
        # Create a biased order
        r = random.random()
        if r < 0.3:
            order = sorted(universe, reverse=True)
        elif r < 0.6:
            order = universe[:]
            random.shuffle(order)
        else:
            order = universe[:]
            order.sort(key=lambda x: (-x + random.random() * 400))

        counts = {}
        chosen = []
        _build_greedy(order, counts, chosen)
        chosen_set = set(chosen)

        # Local fill
        _local_fill(counts, chosen, chosen_set)

        # Local search with random additions
        _local_search(counts, chosen, chosen_set, max_steps=100)

        sz = len(chosen)
        if sz > best_size:
            best_size = sz
            best = sorted(chosen)
            if best_size >= 1500:
                break

    if best is None:
        counts = {}
        chosen = []
        for x in range(1000, 0, -1):
            delta = _valid_insertion(counts, x, set(chosen))
            if delta is not None:
                for s, d in delta.items():
                    counts[s] = counts.get(s, 0) + d
                chosen.append(x)
        best = chosen

    return best
