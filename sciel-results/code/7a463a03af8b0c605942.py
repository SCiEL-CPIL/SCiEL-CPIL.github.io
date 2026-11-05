import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    n = 256
    max_per_set = 64

    def get_diffs(S):
        """Get set of nonzero differences for set S."""
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                diffs.add(S[i] - S[j])
                diffs.add(S[j] - S[i])
        return diffs

    def get_sums(S):
        """Get set of pair sums (including a+a) for set S."""
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        return sums

    def is_valid_addition(x, S, S_sums, S_diffs, other_diffs):
        """Check if x can be added to S maintaining Sidon property and disjoint diffs."""
        # Check Sidon: all pair sums including x+x must be unique
        new_sums = set()
        for a in S:
            s = x + a
            if s in S_sums or s in new_sums:
                return False
            new_sums.add(s)
        if 2 * x in S_sums or 2 * x in new_sums:
            return False
        new_sums.add(2 * x)

        # Check differences don't conflict with other set
        for a in S:
            d1 = x - a
            d2 = a - x
            if d1 in other_diffs or d2 in other_diffs:
                return False

        return True

    def add_to_set(x, S, S_sums, S_diffs):
        """Add x to S, updating sums and diffs incrementally."""
        for a in S:
            S_sums.add(x + a)
            S_diffs.add(x - a)
            S_diffs.add(a - x)
        S_sums.add(2 * x)
        S.append(x)

    def remove_from_set(x, S, S_sums, S_diffs):
        """Remove x from S, updating sums and diffs incrementally."""
        S.remove(x)
        S_sums.discard(2 * x)
        for a in S:
            S_sums.discard(x + a)
            S_diffs.discard(x - a)
            S_diffs.discard(a - x)

    def greedy_construct(order, max_time=None, balance_strategy="smaller"):
        """Greedy construction given a specific element order."""
        sets = [[], []]
        sets_sums = [set(), set()]
        sets_diffs = [set(), set()]

        for x in order:
            if max_time is not None and time.time() > max_time:
                break
            
            valid = []
            for side in (0, 1):
                if len(sets[side]) >= max_per_set:
                    continue
                if is_valid_addition(x, sets[side], sets_sums[side], sets_diffs[side], sets_diffs[1 - side]):
                    valid.append(side)
            
            if not valid:
                continue

            if len(valid) == 1:
                best_side = valid[0]
            else:
                if balance_strategy == "smaller":
                    best_side = valid[0] if len(sets[0]) <= len(sets[1]) else valid[1]
                elif balance_strategy == "random":
                    best_side = random.choice(valid)
                else: # default to 0
                    best_side = valid[0]
            
            add_to_set(x, sets[best_side], sets_sums[best_side], sets_diffs[best_side])

        return sets, sets_sums, sets_diffs

    def validate(sets):
        """Validate that both sets are Sidon and their nonzero differences are disjoint."""
        if not sets[0] or not sets[1]:
            return False
        if len(sets[0]) > max_per_set or len(sets[1]) > max_per_set:
            return False

        # Check Sidon property for each set
        for S in sets:
            sums = set()
            for i in range(len(S)):
                for j in range(i, len(S)):
                    s = S[i] + S[j]
                    if s in sums:
                        return False
                    sums.add(s)

        # Check disjoint nonzero differences
        diffs0 = get_diffs(sets[0])
        diffs1 = get_diffs(sets[1])
        if diffs0 & diffs1:
            return False

        return True

    def local_search_add(sets, sets_sums, sets_diffs, max_time=None):
        """Try to add elements from [1,256] that aren't in either set.
        Uses incremental data structures for speed."""
        if max_time is None:
            max_time = deadline - 3
        improved = True
        while improved and time.time() < max_time:
            improved = False
            all_elements = set(range(1, n + 1))
            in_sets = set(sets[0]) | set(sets[1])
            candidates = list(all_elements - in_sets)
            random.shuffle(candidates)

            for x in candidates:
                if time.time() > max_time:
                    break
                for side in (0, 1):
                    if len(sets[side]) >= max_per_set:
                        continue
                    if is_valid_addition(x, sets[side], sets_sums[side], sets_diffs[side], sets_diffs[1 - side]):
                        add_to_set(x, sets[side], sets_sums[side], sets_diffs[side])
                        improved = True
                        break

        return sets, sets_sums, sets_diffs

    def hill_climb(sets, sets_sums, sets_diffs, max_time=None):
        """More sophisticated local search: try additions and remove-one-add-two."""
        if max_time is None:
            max_time = deadline - 3

        # First, try pure additions
        sets, sets_sums, sets_diffs = local_search_add(sets, sets_sums, sets_diffs, max_time)

        # Try to remove one element from a set and add two or more others (net gain)
        for side in (0, 1):
            if time.time() > max_time:
                break
            for i in range(len(sets[side])):
                if time.time() > max_time:
                    break
                removed = sets[side][i]
                # Work on a copy
                new_set = list(sets[side])
                new_sums = set(sets_sums[side])
                new_diffs = set(sets_diffs[side])
                remove_from_set(removed, new_set, new_sums, new_diffs)

                other_diffs = sets_diffs[1 - side]

                # Try to add multiple elements to new_set
                all_elements = set(range(1, n + 1))
                in_other = set(sets[1 - side])
                candidates = list(all_elements - in_other - set(new_set))
                random.shuffle(candidates)

                for x in candidates:
                    if time.time() > max_time:
                        break
                    if len(new_set) >= max_per_set:
                        break
                    if is_valid_addition(x, new_set, new_sums, new_diffs, other_diffs):
                        add_to_set(x, new_set, new_sums, new_diffs)

                if len(new_set) > len(sets[side]):
                    sets[side] = new_set
                    sets_sums[side] = new_sums
                    sets_diffs[side] = new_diffs

        return sets, sets_sums, sets_diffs

    # Strategy: Try multiple orderings and keep the best valid result
    best_sets = [[], []]
    best_size = 0

    random.seed(42)

    # Try several different orderings
    orderings = []
    # Natural order
    orderings.append(list(range(1, n + 1)))
    # Reverse order
    orderings.append(list(range(n, 0, -1)))
    # Random orderings
    for _ in range(10):
        order = list(range(1, n + 1))
        random.shuffle(order)
        orderings.append(order)
    # Weighted orderings (favor certain ranges)
    for _ in range(5):
        order = list(range(1, n + 1))
        # Bias towards smaller numbers
        order.sort(key=lambda x: x * random.random())
        orderings.append(order)
    for _ in range(5):
        order = list(range(1, n + 1))
        # Bias towards larger numbers
        order.sort(key=lambda x: (n + 1 - x) * random.random())
        orderings.append(order)

    # Different balance strategies
    strategies = ["smaller", "random", "smaller", "random"]

    for i, order in enumerate(orderings):
        if time.time() > deadline - 5:
            break
        
        # Use different balance strategy for different runs
        strat = strategies[i % len(strategies)]
        
        sets, sets_sums, sets_diffs = greedy_construct(order, deadline - 3, strat)
        if validate(sets):
            sets, sets_sums, sets_diffs = hill_climb(sets, sets_sums, sets_diffs, deadline - 2)
            if validate(sets):
                size = len(sets[0]) + len(sets[1])
                if size > best_size:
                    best_size = size
                    best_sets = [list(sets[0]), list(sets[1])]

    # If we didn't find a good solution, use a simple fallback
    if not best_sets[0] or not best_sets[1]:
        sets, _, _ = greedy_construct(list(range(1, n + 1)))
        if validate(sets):
            best_sets = [list(sets[0]), list(sets[1])]
        else:
            best_sets = [[1], [2]]

    return best_sets
