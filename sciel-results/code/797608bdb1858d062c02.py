"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START
def construct():
    n = 512
    start_time = time.time()
    time_limit = 85  # Safety margin before 90s kill

    def can_add(x, in_S):
        # x is 1-indexed, in_S is a list of booleans (1-indexed)
        # Check if x, x+d, x+2d, x+3d in S (x is first term)
        max_d1 = (n - x) // 3
        for d in range(1, max_d1 + 1):
            if in_S[x + d] and in_S[x + 2 * d] and in_S[x + 3 * d]:
                return False
        # x is second term: x-d, x, x+d, x+2d
        max_d2 = min(x - 1, (n - x) // 2)
        for d in range(1, max_d2 + 1):
            if in_S[x - d] and in_S[x + d] and in_S[x + 2 * d]:
                return False
        # x is third term: x-2d, x-d, x, x+d
        max_d3 = min((x - 1) // 2, n - x)
        for d in range(1, max_d3 + 1):
            if in_S[x - 2 * d] and in_S[x - d] and in_S[x + d]:
                return False
        # x is fourth term: x-3d, x-2d, x-d, x
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            if in_S[x - 3 * d] and in_S[x - 2 * d] and in_S[x - d]:
                return False
        return True

    best_size = 0
    best_in_S = None

    # Phase 1: Standard greedy (deterministic, fast baseline)
    in_S = [False] * (n + 1)
    count = 0
    for x in range(1, n + 1):
        if can_add(x, in_S):
            in_S[x] = True
            count += 1
    if count > best_size:
        best_size = count
        best_in_S = in_S.copy()

    # Phase 2: Randomized greedy with multiple restarts
    candidates = list(range(1, n + 1))
    while time.time() - start_time < time_limit * 0.6:
        random.shuffle(candidates)
        in_S = [False] * (n + 1)
        count = 0
        for x in candidates:
            if can_add(x, in_S):
                in_S[x] = True
                count += 1
        if count > best_size:
            best_size = count
            best_in_S = in_S.copy()

    # Phase 3: Local search with swaps on the best set found
    if best_in_S is not None:
        current = best_in_S.copy()
        current_size = best_size
        remaining = [x for x in range(1, n + 1) if not current[x]]
        selected_list = [x for x in range(1, n + 1) if current[x]]

        while time.time() - start_time < time_limit * 0.95:
            improved = False

            # Try to add any remaining element that fits
            random.shuffle(remaining)
            for x in remaining:
                if can_add(x, current):
                    current[x] = True
                    remaining.remove(x)
                    current_size += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_in_S = current.copy()

            # Try swap moves: remove one selected element, try to add two remaining
            if not improved and selected_list:
                # Try a few random swaps
                for _ in range(50):
                    if time.time() - start_time > time_limit * 0.95:
                        break
                    idx = random.randint(0, len(selected_list) - 1)
                    x_to_remove = selected_list[idx]
                    current[x_to_remove] = False
                    current_size -= 1
                    selected_list.pop(idx)

                    # Try to add two elements from remaining
                    added = 0
                    candidates_to_add = remaining.copy()
                    random.shuffle(candidates_to_add)
                    for y in candidates_to_add:
                        if added >= 2:
                            break
                        if can_add(y, current):
                            current[y] = True
                            remaining.remove(y)
                            current_size += 1
                            added += 1

                    if current_size > best_size:
                        best_size = current_size
                        best_in_S = current.copy()
                        selected_list = [x for x in range(1, n + 1) if current[x]]
                    else:
                        # Revert: put x_to_remove back
                        current[x_to_remove] = True
                        current_size += 1
                        selected_list.append(x_to_remove)
                        # Remove any elements we added
                        for y in candidates_to_add:
                            if current[y] and y in remaining:
                                # This shouldn't happen, but safety check
                                pass

            if not improved:
                # No additions possible, try a few more swap attempts
                break

    return sorted([x for x in range(1, n + 1) if best_in_S[x]])
# EVOLVE-BLOCK-END
