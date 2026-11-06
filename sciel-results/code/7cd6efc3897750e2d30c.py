import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    base = [1, 2, 5, 11]
    base_set = set(base)

    # Calculate initial sums
    initial_sums = set()
    for i in range(len(base)):
        for j in range(i, len(base)):
            initial_sums.add(base[i] + base[j])

    # Initial candidates: all numbers in 1..284 not in base
    all_numbers = list(range(1, 285))

    def is_valid_candidate(x, current_set, current_sums):
        """
        Check if x can be added to current_set without creating sum collisions.
        """
        new_sums = set()
        # Self sum
        s_self = 2 * x
        if s_self in current_sums:
            return False
        new_sums.add(s_self)

        # Sums with existing elements
        for y in current_set:
            s = x + y
            if s in current_sums:
                return False
            new_sums.add(s)

        return True

    # Filter initial candidates
    candidates = []
    for x in all_numbers:
        if x in base_set:
            continue
        if is_valid_candidate(x, base_set, initial_sums):
            candidates.append(x)

    candidates.sort()

    best_solution = list(base)
    best_size = len(base)

    # Greedy with randomized restarts
    def run_greedy_randomized():
        nonlocal best_solution, best_size

        # Randomize candidates slightly
        cands = candidates[:]
        random.shuffle(cands)

        current_set = set(base)
        current_sums = set(initial_sums)

        # Greedily add candidates
        i = 0
        while i < len(cands):
            x = cands[i]
            if is_valid_candidate(x, current_set, current_sums):
                current_set.add(x)
                current_sums.add(2 * x)
                for y in list(current_set):
                    if y != x:
                        current_sums.add(x + y)

                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_solution = list(current_set)

                # Filter out candidates that are no longer valid
                cands = [c for c in cands[i+1:] if is_valid_candidate(c, current_set, current_sums)]
                i = 0
            else:
                i += 1

        return best_solution

    # Run restarts until time limit
    restart_count = 0
    while time.time() < deadline:
        run_greedy_randomized()
        restart_count += 1
        # Optional: If we reached size 20, we can stop early?
        # The max is 20. If we hit 20, it's optimal (likely).
        if best_size >= 20:
            break

    return sorted(best_solution)

if __name__ == "__main__":
    res = construct()
    print(res)
    print(len(res))
