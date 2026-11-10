"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin
    
    def make_bitset():
        # bit i set means sum i is reachable, for i in 0..200
        return np.zeros(201, dtype=bool)
    
    def add_to_bitset(bs, x):
        # returns new bitset after adding x, or None if 200 would be reached
        # new sums: old sums + x
        # Vectorized: shift bs by x positions
        if x >= 201:
            return None
        new_sums = np.zeros(201, dtype=bool)
        new_sums[x:] = bs[:201-x]
        if new_sums[200]:
            return None
        return bs | new_sums
    
    # Greedy: try largest first
    def greedy(order):
        selected = []
        bs = make_bitset()
        bs[0] = True
        for x in order:
            new_bs = add_to_bitset(bs, x)
            if new_bs is not None:
                selected.append(x)
                bs = new_bs
        return selected
    
    # Try largest-first
    best = greedy(range(100, 0, -1))
    best_size = len(best)
    
    # Try smallest-first
    if time.time() < deadline:
        candidate = greedy(range(1, 101))
        if len(candidate) > best_size:
            best = candidate
            best_size = len(best)
    
    # Try various structured orderings
    if time.time() < deadline - 2.0:
        # Try alternating large/small
        order = []
        for i in range(50):
            order.append(100 - i)
            if i < 50:
                order.append(i + 1)
        candidate = greedy(order)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(best)
        
        # Try multiples of 3 first, then others
        order = [x for x in range(1, 101) if x % 3 != 0] + [x for x in range(1, 101) if x % 3 == 0]
        candidate = greedy(order)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(best)
        
        # Try even numbers first
        order = [x for x in range(1, 101) if x % 2 == 0] + [x for x in range(1, 101) if x % 2 != 0]
        candidate = greedy(order)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(best)
    
    # Try random orderings with different seeds
    for seed in [42, 123, 456, 789, 1011, 2023, 3141, 2718, 1618, 1414, 3333, 7777]:
        if time.time() > deadline - 5.0:
            break
        rng = random.Random(seed)
        for trial in range(80):
            if time.time() > deadline - 5.0:
                break
            order = list(range(1, 101))
            rng.shuffle(order)
            candidate = greedy(order)
            if len(candidate) > best_size:
                best = candidate
                best_size = len(best)
    
    # Local search: try to add elements not in best, with multiple insertion orders
    for pass_num in range(5):
        if time.time() < deadline - 5.0:
            best_set = set(best)
            bs = make_bitset()
            bs[0] = True
            for x in best:
                bs = add_to_bitset(bs, x)  # should not be None
            # Try adding each missing element in different orders
            added_any = False
            orders = [
                list(range(1, 101)),
                list(range(100, 0, -1)),
                [x for x in range(1, 101) if x not in best_set],
                sorted([x for x in range(1, 101) if x not in best_set], key=lambda v: -v),
            ]
            for order in orders:
                if time.time() > deadline - 5.0:
                    break
                for x in order:
                    if x not in best_set:
                        new_bs = add_to_bitset(bs, x)
                        if new_bs is not None:
                            best.append(x)
                            best_set.add(x)
                            bs = new_bs
                            best_size = len(best)
                            added_any = True
            if not added_any:
                break
    
    if time.time() > deadline:
        return best
    
    # Try swap-based local search: remove one element, try to add more
    # Use multiple insertion orders for the "add all missing" step
    for _ in range(50):
        if time.time() > deadline - 5.0:
            break
        best_set = set(best)
        improved = False
        for remove_x in list(best):
            # Try removing remove_x and see if we can add more
            remaining = [x for x in best if x != remove_x]
            bs2 = make_bitset()
            bs2[0] = True
            valid = True
            for x in remaining:
                new_bs2 = add_to_bitset(bs2, x)
                if new_bs2 is None:
                    valid = False
                    break
                bs2 = new_bs2
            if not valid:
                continue
            # Try adding all missing elements with multiple orders
            for order in [list(range(1, 101)), list(range(100, 0, -1))]:
                if time.time() > deadline - 5.0:
                    break
                new_best = list(remaining)
                new_set = set(remaining)
                bs3 = bs2.copy()
                for x in order:
                    if x not in new_set:
                        new_bs3 = add_to_bitset(bs3, x)
                        if new_bs3 is not None:
                            new_best.append(x)
                            new_set.add(x)
                            bs3 = new_bs3
                if len(new_best) > best_size:
                    best = new_best
                    best_size = len(best)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    # Try double-swap: remove two elements, try to add more
    if time.time() < deadline - 10.0:
        best_set = set(best)
        found_improvement = True
        while found_improvement:
            found_improvement = False
            for i in range(len(best)):
                if time.time() > deadline - 5.0:
                    break
                for j in range(i + 1, len(best)):
                    if time.time() > deadline - 5.0:
                        break
                    remove_x = best[i]
                    remove_y = best[j]
                    remaining = [x for x in best if x != remove_x and x != remove_y]
                    bs2 = make_bitset()
                    bs2[0] = True
                    valid = True
                    for x in remaining:
                        new_bs2 = add_to_bitset(bs2, x)
                        if new_bs2 is None:
                            valid = False
                            break
                        bs2 = new_bs2
                    if not valid:
                        continue
                    # Try adding all missing elements with multiple orders
                    for order in [list(range(1, 101)), list(range(100, 0, -1))]:
                        if time.time() > deadline - 5.0:
                            break
                        new_best = list(remaining)
                        new_set = set(remaining)
                        bs3 = bs2.copy()
                        for x in order:
                            if x not in new_set:
                                new_bs3 = add_to_bitset(bs3, x)
                                if new_bs3 is not None:
                                    new_best.append(x)
                                    new_set.add(x)
                                    bs3 = new_bs3
                        if len(new_best) > best_size:
                            best = new_best
                            best_size = len(best)
                            found_improvement = True
                            break
                    if found_improvement:
                        break
                if found_improvement or time.time() > deadline - 5.0:
                    break
    
    return best
# EVOLVE-BLOCK-END
