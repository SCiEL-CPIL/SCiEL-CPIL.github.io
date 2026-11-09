"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _greedy_construction(order):
    """Fast greedy: pick elements in given order, tracking reachable subset sums via bitset."""
    reachable = 1  # bit i set means sum i is reachable; start with sum 0
    selected = []
    for x in order:
        # Adding x creates sum 200 iff sum (200-x) is currently reachable
        if (200 - x) >= 0 and (reachable >> (200 - x)) & 1:
            continue
        selected.append(x)
        # Update reachable sums: new_sums = old_sums | (old_sums << x), masked to [0,200]
        mask = (1 << 201) - 1  # bits 0..200
        reachable = (reachable | (reachable << x)) & mask
    return selected

def _has_sum_200(elements):
    """Check if any subset of elements sums to exactly 200."""
    reachable = 1
    for x in elements:
        mask = (1 << 201) - 1
        reachable = (reachable | (reachable << x)) & mask
    return (reachable >> 200) & 1

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 90s kill

    # Phase 1: Try multiple deterministic and random orderings
    best = []
    best_size = 0

    # Deterministic orderings
    deterministic_orders = [
        list(range(100, 0, -1)),          # descending
        list(range(1, 101)),              # ascending
        list(range(51, 101)) + list(range(1, 51)),  # large then small
        list(range(1, 51)) + list(range(100, 50, -1)),  # small then large
        list(range(100, 50, -1)) + list(range(1, 51)),  # large desc then small asc
        list(range(1, 51)) + list(range(51, 101)),     # small asc then large asc
    ]

    for order in deterministic_orders:
        if time.time() > deadline:
            break
        result = _greedy_construction(order)
        if len(result) > best_size:
            best_size = len(result)
            best = result[:]

    # Random restarts
    rng = random.Random(42)
    trial = 0
    while time.time() < deadline - 10.0:  # leave 10s for local search
        order = list(range(1, 101))
        rng.shuffle(order)
        result = _greedy_construction(order)
        if len(result) > best_size:
            best_size = len(result)
            best = result[:]
        trial += 1

    # Phase 2: Local search - try to add elements not in best
    if time.time() < deadline - 2.0:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            best_set = set(best)
            available = [x for x in range(1, 101) if x not in best_set]
            
            # Try adding one element at a time
            for x in available:
                if time.time() > deadline - 2.0:
                    break
                test = best + [x]
                if not _has_sum_200(test):
                    best = test
                    best_size += 1
                    best_set = set(best)
                    available = [x for x in range(1, 101) if x not in best_set]
                    improved = True

            # Try swapping: remove one, add two
            if not improved and time.time() < deadline - 3.0:
                for remove in list(best_set):
                    if time.time() > deadline - 3.0:
                        break
                    remaining = [x for x in best if x != remove]
                    avail = [x for x in range(1, 101) if x not in set(remaining) and x != remove]
                    for i in range(len(avail)):
                        if time.time() > deadline - 3.0:
                            break
                        for j in range(i + 1, len(avail)):
                            test = remaining + [avail[i], avail[j]]
                            if not _has_sum_200(test):
                                best = test
                                best_size = len(best)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break

    # Final verification
    if _has_sum_200(best):
        best = list(range(66, 101))

    return best

# EVOLVE-BLOCK-END
