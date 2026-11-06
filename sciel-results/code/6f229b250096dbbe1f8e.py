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
    if s2 <= 512 and ((sum_bits >> s2) & 1):
        return False
    # Check elem+a sums and |elem-a| diffs
    for a in elems:
        s = elem + a
        if s <= 512 and ((sum_bits >> s) & 1):
            return False
        d = abs(elem - a)
        if d > 0:
            if ((diff_bits >> d) & 1) or ((other_diff_bits >> d) & 1):
                return False
    return True


def _add_elem(state, side, elem):
    """Add elem to side. state = [elems0, elems1, sum_bits0, sum_bits1, diff_bits0, diff_bits1]"""
    elems = state[side]
    sum_bits = state[2 + side]
    diff_bits = state[4 + side]
    other_diff_bits = state[4 + (1 - side)]

    elems.append(elem)
    # Update sum bits
    for a in elems[:-1]:  # all except the newly added one
        s = elem + a
        if s <= 512:
            sum_bits |= (1 << s)
    s2 = elem + elem
    if s2 <= 512:
        sum_bits |= (1 << s2)
    # Update diff bits
    for a in elems[:-1]:
        d = abs(elem - a)
        if d > 0:
            diff_bits |= (1 << d)

    state[side] = elems
    state[2 + side] = sum_bits
    state[4 + side] = diff_bits


def _remove_elem(state, side, elem):
    """Remove elem from side. state = [elems0, elems1, sum_bits0, sum_bits1, diff_bits0, diff_bits1]"""
    elems = state[side]
    sum_bits = state[2 + side]
    diff_bits = state[4 + side]

    idx = elems.index(elem)
    elems.pop(idx)

    # Rebuild sum_bits and diff_bits from scratch (simpler and correct)
    new_sum_bits = 0
    new_diff_bits = 0
    for i in range(len(elems)):
        for j in range(i, len(elems)):
            s = elems[i] + elems[j]
            if s <= 512:
                new_sum_bits |= (1 << s)
            if i != j:
                d = abs(elems[i] - elems[j])
                if d > 0:
                    new_diff_bits |= (1 << d)

    state[side] = elems
    state[2 + side] = new_sum_bits
    state[4 + side] = new_diff_bits


def _greedy_build(deadline):
    """Greedy construction: add elements to both sets."""
    state = [[], [], 0, 0, 0, 0]  # elems0, elems1, sum_bits0, sum_bits1, diff_bits0, diff_bits1
    for x in range(1, N + 1):
        for side in (0, 1):
            if len(state[side]) >= MAX_EACH:
                continue
            if time.time() > deadline:
                break
            elems = state[side]
            sum_bits = state[2 + side]
            diff_bits = state[4 + side]
            other_diff_bits = state[4 + (1 - side)]
            if _can_add(x, elems, sum_bits, diff_bits, other_diff_bits):
                _add_elem(state, side, x)
        if time.time() > deadline:
            break
    return state


def _local_search(state, deadline):
    """1-for-1 and 1-for-2 swap local search."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for side in (0, 1):
            if time.time() > deadline:
                break
            elems = state[side]
            if len(elems) < 2:
                continue
            for i in range(len(elems)):
                if time.time() > deadline:
                    break
                elem_to_remove = elems[i]
                # Try removing elem_to_remove and adding a new element
                _remove_elem(state, side, elem_to_remove)
                elems_after = state[side]
                sum_bits = state[2 + side]
                diff_bits = state[4 + side]
                other_diff_bits = state[4 + (1 - side)]

                best_new_elem = None
                for x in range(1, N + 1):
                    if x in elems_after:
                        continue
                    if _can_add(x, elems_after, sum_bits, diff_bits, other_diff_bits):
                        best_new_elem = x
                        break

                if best_new_elem is not None:
                    # Check if this is actually an improvement (different element)
                    if best_new_elem != elem_to_remove:
                        _add_elem(state, side, best_new_elem)
                        improved = True
                    else:
                        _add_elem(state, side, elem_to_remove)
                else:
                    _add_elem(state, side, elem_to_remove)

    return state


def _multi_start_local_search(deadline):
    """Run multiple greedy builds with different orderings, then local search each."""
    best_state = None
    best_score = -1

    num_starts = 20
    for start in range(num_starts):
        if time.time() > deadline:
            break

        # Create a shuffled order for this start
        if start == 0:
            order = list(range(1, N + 1))
        else:
            order = list(range(1, N + 1))
            random.shuffle(order)

        # Greedy build with this order
        state = [[], [], 0, 0, 0, 0]
        for x in order:
            if time.time() > deadline:
                break
            for side in (0, 1):
                if len(state[side]) >= MAX_EACH:
                    continue
                elems = state[side]
                sum_bits = state[2 + side]
                diff_bits = state[4 + side]
                other_diff_bits = state[4 + (1 - side)]
                if _can_add(x, elems, sum_bits, diff_bits, other_diff_bits):
                    _add_elem(state, side, x)

        # Local search
        state = _local_search(state, deadline)

        score = len(state[0]) + len(state[1])
        if score > best_score:
            best_score = score
            best_state = [state[0][:], state[1][:], state[2], state[3], state[4], state[5]]

    if best_state is None:
        # Fallback: simple greedy
        best_state = _greedy_build(deadline)

    return best_state


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    # First, do a quick greedy build to have a valid answer
    state = _greedy_build(deadline - 5.0)
    best_state = [state[0][:], state[1][:], state[2], state[3], state[4], state[5]]
    best_score = len(state[0]) + len(state[1])

    # Then, try multi-start local search with remaining time
    if time.time() < deadline - 5.0:
        search_deadline = deadline - 5.0
        state2 = _multi_start_local_search(search_deadline)
        score2 = len(state2[0]) + len(state2[1])
        if score2 > best_score:
            best_score = score2
            best_state = state2

    # Return the best found
    return [best_state[0], best_state[1]]
# EVOLVE-BLOCK-END
