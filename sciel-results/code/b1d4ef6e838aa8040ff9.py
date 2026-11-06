"""Rewritten program: local search over comparator networks for 17 wires."""
import time
import numpy as np

N = 17
DEADLINE = 85.0  # safety margin before 90s kill
START_TIME = time.time()


def _time_left():
    return DEADLINE - (time.time() - START_TIME)


def _seed_network():
    """Odd-even transposition sorting network for N wires."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps


def _make_all_inputs():
    """All 2^N boolean inputs as uint8 array shape (2^N, N)."""
    n_inputs = 1 << N
    indices = np.arange(n_inputs, dtype=np.uint32)[:, None]
    bit_positions = np.arange(N, dtype=np.uint32)[None, :]
    return ((indices >> bit_positions) & 1).astype(np.uint8)


_ALL_INPUTS = None


def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        _ALL_INPUTS = _make_all_inputs()
    return _ALL_INPUTS


def simulate_network(comps):
    """Simulate network on all boolean inputs. Returns final uint8 array (2^N, N)."""
    data = _get_all_inputs().copy()
    for a, b in comps:
        lo = np.minimum(data[:, a], data[:, b])
        hi = np.maximum(data[:, a], data[:, b])
        data[:, a] = lo
        data[:, b] = hi
    return data


def is_sorted_network(comps):
    """Check if network sorts all boolean inputs (zero-one principle)."""
    final = simulate_network(comps)
    # Sorted means each row is non-decreasing: 0...0 1...1
    # Check: for each row, once we see a 1, no 0 should follow.
    # Equivalent: the row equals the sorted version.
    # Faster: check that the sum of each row is preserved (it is, since min/max preserve sum)
    # and that the array is sorted.
    # A row is sorted iff for all j: final[:,j] >= final[:,j+1] is FALSE... 
    # Actually sorted ascending: final[:,j] <= final[:,j+1]
    # Check: np.all(final[:, :-1] <= final[:, 1:])
    return np.all(final[:, :-1] <= final[:, 1:])


def _try_remove(comps, idx):
    """Try removing comparator at index idx; return new list if still valid, else None."""
    if idx < 0 or idx >= len(comps):
        return None
    new_comps = comps[:idx] + comps[idx + 1:]
    if is_sorted_network(new_comps):
        return new_comps
    return None


def _try_swap(comps, i, j):
    """Try swapping comparator positions i and j; return new list if valid, else None."""
    if i == j:
        return None
    new_comps = list(comps)
    new_comps[i], new_comps[j] = new_comps[j], new_comps[i]
    if is_sorted_network(new_comps):
        return new_comps
    return None


def _try_move(comps, src, dst):
    """Move comparator from position src to position dst; return new list if valid, else None."""
    if src == dst:
        return None
    new_comps = list(comps)
    elem = new_comps.pop(src)
    new_comps.insert(dst, elem)
    if is_sorted_network(new_comps):
        return new_comps
    return None


def improve_network(comps):
    """Local search: repeatedly try removals, then swaps/moves to reduce count."""
    import random
    rng = random.Random(42)
    best = list(comps)
    improved = True
    while improved and _time_left() > 1.0:
        improved = False
        # Try removals from the end (later comparators are more likely redundant)
        for idx in range(len(best) - 1, -1, -1):
            if _time_left() < 0.5:
                break
            new_comps = _try_remove(best, idx)
            if new_comps is not None:
                best = new_comps
                improved = True
                break
        if not improved and _time_left() > 1.0:
            # Try random swaps
            n = len(best)
            for _ in range(200):
                if _time_left() < 0.5:
                    break
                i = rng.randrange(n)
                j = rng.randrange(n)
                new_comps = _try_swap(best, i, j)
                if new_comps is not None:
                    # After a swap, try removals again
                    for idx in range(len(new_comps) - 1, -1, -1):
                        if _time_left() < 0.3:
                            break
                        nc2 = _try_remove(new_comps, idx)
                        if nc2 is not None:
                            best = nc2
                            improved = True
                            break
                    if improved:
                        break
    return best


def construct():
    """Construct a sorting network for N=17 wires, minimizing comparator count."""
    global START_TIME
    START_TIME = time.time()

    comps = _seed_network()
    # Verify seed is correct
    if not is_sorted_network(comps):
        # Fallback: return seed as-is (shouldn't happen)
        return comps

    # Phase 1: Greedy removal from the end
    improved = True
    while improved and _time_left() > 2.0:
        improved = False
        for idx in range(len(comps) - 1, -1, -1):
            if _time_left() < 1.0:
                break
            new_comps = _try_remove(comps, idx)
            if new_comps is not None:
                comps = new_comps
                improved = True
                break

    # Phase 2: Local search with swaps and moves
    if _time_left() > 2.0:
        comps = improve_network(comps)

    # Final verification
    if not is_sorted_network(comps):
        # Return the seed if something went wrong
        return _seed_network()

    return comps


if __name__ == "__main__":
    result = construct()
    print(f"Comparators: {len(result)}")
