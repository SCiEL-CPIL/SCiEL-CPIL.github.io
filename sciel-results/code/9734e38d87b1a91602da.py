"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 9
MAX_VERTICES = 2 ** N

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s kill

    best_path = _greedy_construct()
    best_len = len(best_path) - 1

    # Try to improve via local search / backtracking with time budget
    if time.time() < deadline:
        improved = _improve_path(best_path, deadline)
        if improved is not None and len(improved) > len(best_path):
            best_path = improved
            best_len = len(best_path) - 1

    # If we have more time, try randomized restarts with different greedy orders
    if time.time() < deadline - 5.0:
        best_path = _randomized_search(deadline, best_path, best_len)

    return best_path


def _greedy_construct():
    """Greedy snake: always extend with the lowest bit flip that keeps the path induced."""
    path, used = [0], {0}
    while True:
        cur, nxt = path[-1], None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check non-consecutive constraint: v must differ from all path[:-1] by >= 2 bits
            valid = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    valid = False
                    break
            if valid:
                nxt = v
                break
        if nxt is None:
            return path
        path.append(nxt)
        used.add(nxt)


def _is_valid_extension(path, v, used):
    """Check if v can be appended to path as an induced path extension."""
    if v in used:
        return False
    cur = path[-1]
    if (v ^ cur).bit_count() != 1:
        return False
    # Non-consecutive vertices must differ by at least 2 bits
    for p in path[:-1]:
        if (v ^ p).bit_count() == 1:
            return False
    return True


def _improve_path(path, deadline):
    """Try to extend or modify the path using local search."""
    path = list(path)
    used = set(path)
    improved = False

    # Try extending from the end
    while time.time() < deadline - 2.0:
        cur = path[-1]
        best_next = None
        # Try all bit flips
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            if _is_valid_extension(path, v, used):
                best_next = v
                break

        if best_next is not None:
            path.append(best_next)
            used.add(best_next)
            improved = True
        else:
            break

    # Try extending from the beginning (prepend)
    if time.time() < deadline - 2.0:
        while True:
            cur = path[0]
            best_prev = None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                # Check: v must be adjacent to cur (1 bit diff) and not adjacent to path[1]
                if (v ^ cur).bit_count() != 1:
                    continue
                if (v ^ path[1]).bit_count() == 1:
                    continue
                # Check against all other non-consecutive vertices
                valid = True
                for p in path[2:]:
                    if (v ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    best_prev = v
                    break
            if best_prev is not None:
                path.insert(0, best_prev)
                used.add(best_prev)
                improved = True
            else:
                break

    return path if improved else None


def _randomized_search(deadline, current_best, current_best_len):
    """Randomized greedy with different bit orderings, keep best result."""
    best_path = list(current_best)
    best_len = current_best_len
    rng = random.Random(42)

    while time.time() < deadline - 3.0:
        # Shuffle bit order
        bit_order = list(range(N))
        rng.shuffle(bit_order)

        path, used = [0], {0}
        while True:
            cur, nxt = path[-1], None
            for b in bit_order:
                v = cur ^ (1 << b)
                if v in used:
                    continue
                if _is_valid_extension(path, v, used):
                    nxt = v
                    break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)

        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)

    return best_path
# EVOLVE-BLOCK-END
