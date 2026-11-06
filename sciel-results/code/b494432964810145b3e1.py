"""Two Sidon sets with disjoint difference sets – Bitset + multi-start local search."""
# EVOLVE-BLOCK-START
import time
import random

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # safety margin before 60s kill


def _can_add(elem, elems, sum_bits, diff_bits, other_diff_bits):
    """Check if elem can be added to elems maintaining Sidon + disjoint diffs."""
    # Check elem+elem sum
    s2 = elem + elem
    if s2 <= 512 and (sum_bits >> s2) & 1:
        return False
    # Check elem+a sums and |elem-a| diffs
    for a in elems:
        s = elem + a
        if (sum_bits >> s) & 1:
            return False
        d = abs(elem - a)
        if (diff_bits >> d) & 1 or (other_diff_bits >> d) & 1:
            return False
    return True


def _add_elem(state, side, elem):
    """Add elem to side; update bitsets. Returns None if invalid."""
    elems = state['sets'][side]
    sum_bits = state['sum_bits'][side]
    diff_bits = state['diff_bits'][side]
    other_diff = state['diff_bits'][1 - side]

    if not _can_add(elem, elems, sum_bits, diff_bits, other_diff):
        return None

    # Update sum bits
    s2 = elem + elem
    sum_bits |= (1 << s2)
    for a in elems:
        sum_bits |= (1 << (elem + a))

    # Update diff bits
    for a in elems:
        d = abs(elem - a)
        diff_bits |= (1 << d)

    state['sets'][side].append(elem)
    state['sum_bits'][side] = sum_bits
    state['diff_bits'][side] = diff_bits
    return state


def _remove_elem(state, side, idx):
    """Remove element at index idx from side; rebuild bitsets."""
    elems = state['sets'][side]
    elem = elems.pop(idx)

    # Rebuild sum bits
    from itertools import combinations_with_replacement
    new_sum = 0
    for a, b in combinations_with_replacement(elems, 2):
        new_sum |= (1 << (a + b))
    state['sum_bits'][side] = new_sum

    # Rebuild diff bits
    new_diff = 0
    from itertools import combinations
    for a, b in combinations(elems, 2):
        new_diff |= (1 << abs(a - b))
    state['diff_bits'][side] = new_diff


def _greedy_build(order, max_time):
    """Greedy build using given element order."""
    state = {
        'sets': [[], []],
        'sum_bits': [0, 0],
        'diff_bits': [0, 0]
    }
    for elem in order:
        if time.time() > max_time:
            break
        # Try adding to set 0 first, then set 1
        if len(state['sets'][0]) < MAX_EACH:
            r = _add_elem(state, 0, elem)
            if r is not None:
                continue
        if len(state['sets'][1]) < MAX_EACH:
            r = _add_elem(state, 1, elem)
            if r is not None:
                continue
    return state


def _local_search(state, max_time):
    """1-for-2 swap local search: try replacing one element with two new ones."""
    improved = True
    while improved and time.time() < max_time:
        improved = False
        for side in (0, 1):
            if len(state['sets'][side]) < MAX_EACH:
                # Try adding a new element not in either set
                all_used = set(state['sets'][0]) | set(state['sets'][1])
                for candidate in range(1, N + 1):
                    if candidate in all_used:
                        continue
                    if time.time() > max_time:
                        return state
                    r = _add_elem(state, side, candidate)
                    if r is not None:
                        improved = True
                        break
            # Try 1-for-2 swap: remove one, add two
            if len(state['sets'][side]) >= 1 and len(state['sets'][side]) < MAX_EACH:
                for i in range(len(state['sets'][side])):
                    if time.time() > max_time:
                        return state
                    removed = state['sets'][side][i]
                    _remove_elem(state, side, i)
                    # Try to add two new elements
                    added = 0
                    all_used = set(state['sets'][0]) | set(state['sets'][1])
                    for candidate in range(1, N + 1):
                        if candidate in all_used or candidate == removed:
                            continue
                        if time.time() > max_time:
                            # Restore
                            _add_elem(state, side, removed)
                            return state
                        r = _add_elem(state, side, candidate)
                        if r is not None:
                            added += 1
                            if added >= 2:
                                improved = True
                                break
                    if added < 2:
                        # Restore removed element
                        _add_elem(state, side, removed)
    return state


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    best_state = None
    best_size = 0

    # Multi-start with different random orderings
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 5.0:
        attempts += 1
        order = list(range(1, N + 1))
        random.shuffle(order)

        state = _greedy_build(order, deadline - 3.0)
        state = _local_search(state, deadline - 1.0)

        size = len(state['sets'][0]) + len(state['sets'][1])
        if size > best_size:
            best_size = size
            best_state = {
                'sets': [list(state['sets'][0]), list(state['sets'][1])],
                'sum_bits': [state['sum_bits'][0], state['sum_bits'][1]],
                'diff_bits': [state['diff_bits'][0], state['diff_bits'][1]]
            }

        if best_size >= 2 * MAX_EACH:
            break

    # Fallback: simple greedy if nothing found
    if best_state is None:
        order = list(range(1, N + 1))
        state = _greedy_build(order, deadline)
        best_state = {
            'sets': [list(state['sets'][0]), list(state['sets'][1])],
            'sum_bits': [state['sum_bits'][0], state['sum_bits'][1]],
            'diff_bits': [state['diff_bits'][0], state['diff_bits'][1]]
        }

    return [best_state['sets'][0], best_state['sets'][1]]
# EVOLVE-BLOCK-END
