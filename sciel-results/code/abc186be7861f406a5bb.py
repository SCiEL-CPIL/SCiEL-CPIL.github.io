"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    N = 256

    # Precompute conflict as a set of (min, max) tuples
    conflict_set = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict_set.add((a, b))

    # Precompute conflict bitmask for each number (bit i set if conflicts with i+1)
    conflict_mask = [0] * (N + 1)
    for a in range(1, N + 1):
        mask = 0
        for b in range(1, N + 1):
            if a == b:
                continue
            lo, hi = (a, b) if a < b else (b, a)
            if (lo, hi) in conflict_set:
                mask |= (1 << b)
        conflict_mask[a] = mask

    # Precompute: for each num, list of numbers it conflicts with
    conflict_list = [[] for _ in range(N + 1)]
    for (a, b) in conflict_set:
        conflict_list[a].append(b)
        conflict_list[b].append(a)

    def can_add(num, current_mask):
        """Check if num can be added given current_mask (bitmask of selected numbers)."""
        return (conflict_mask[num] & current_mask) == 0

    def greedy_fill_from_mask(current_mask, candidates):
        """Try to add each candidate to current_mask; return (new_mask, added_count)."""
        added = 0
        for c in candidates:
            if current_mask & (1 << c):
                continue
            if can_add(c, current_mask):
                current_mask |= (1 << c)
                added += 1
        return current_mask, added

    # Start with all odd numbers
    odds_mask = 0
    for x in range(1, N + 1, 2):
        odds_mask |= (1 << x)
    
    best_mask = odds_mask
    best_size = bin(best_mask).count('1')

    # Greedy: try adding each even number
    evens = [x for x in range(2, N + 1, 2)]
    random.shuffle(evens)
    current_mask = odds_mask
    for e in evens:
        if time.time() > deadline:
            break
        if can_add(e, current_mask):
            current_mask |= (1 << e)

    cur_size = bin(current_mask).count('1')
    if cur_size > best_size:
        best_mask = current_mask
        best_size = cur_size

    # 1-swap local search: remove one element, greedily refill
    current_mask = best_mask
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Get list of set bits
        elements = [i for i in range(1, N + 1) if current_mask & (1 << i)]
        random.shuffle(elements)
        for rem in elements:
            if time.time() > deadline:
                break
            new_mask = current_mask & ~(1 << rem)
            candidates = [x for x in range(1, N + 1) if not (new_mask & (1 << x))]
            random.shuffle(candidates)
            new_mask, _ = greedy_fill_from_mask(new_mask, candidates)
            new_size = bin(new_mask).count('1')
            if new_size > bin(current_mask).count('1'):
                current_mask = new_mask
                best_mask = new_mask
                best_size = new_size
                improved = True
                break

    # Randomized perturbation + refill
    for restart in range(100):
        if time.time() > deadline:
            break

        trial_mask = best_mask
        n_remove = random.randint(1, 4)
        elements = [i for i in range(1, N + 1) if trial_mask & (1 << i)]
        to_remove = random.sample(elements, min(n_remove, len(elements)))
        for t in to_remove:
            trial_mask &= ~(1 << t)

        candidates = [x for x in range(1, N + 1) if not (trial_mask & (1 << x))]
        random.shuffle(candidates)
        trial_mask, _ = greedy_fill_from_mask(trial_mask, candidates)

        trial_size = bin(trial_mask).count('1')
        if trial_size > best_size:
            best_mask = trial_mask
            best_size = trial_size

    # Final 1-swap polish
    current_mask = best_mask
    improved = True
    while improved and time.time() < deadline:
        improved = False
        elements = [i for i in range(1, N + 1) if current_mask & (1 << i)]
        random.shuffle(elements)
        for rem in elements:
            if time.time() > deadline:
                break
            new_mask = current_mask & ~(1 << rem)
            candidates = [x for x in range(1, N + 1) if not (new_mask & (1 << x))]
            random.shuffle(candidates)
            new_mask, _ = greedy_fill_from_mask(new_mask, candidates)
            new_size = bin(new_mask).count('1')
            if new_size > bin(current_mask).count('1'):
                current_mask = new_mask
                best_mask = new_mask
                best_size = new_size
                improved = True
                break

    # Convert mask to list
    best_list = [i for i in range(1, N + 1) if best_mask & (1 << i)]

    # Final verification
    valid = True
    for i in range(len(best_list)):
        for j in range(i + 1, len(best_list)):
            a, b = best_list[i], best_list[j]
            if (a * b) % (a + b) == 0:
                valid = False
                break
        if not valid:
            break

    if not valid or len(best_list) == 0:
        return list(range(1, N + 1, 2))

    return best_list
# EVOLVE-BLOCK-END
