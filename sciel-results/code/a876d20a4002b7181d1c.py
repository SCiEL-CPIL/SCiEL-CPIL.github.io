"""Size-efficient sorting network synthesis for 18 wires.

Strategy:
  - Fast bit-parallel 0-1 verification.
  - Batcher odd-even mergesort as a dense starting network.
  - Aggressive greedy removal passes.
  - Simulated annealing with diverse moves and periodic restarts.
  - Deadline guard with safety margin (returns best found before 90s limit).
"""
import time
import random
import math

N = 18
_M = 1 << N  # 262144


def _build_mask(w, n_bits):
    """Build bit mask where bit k is set iff (k >> w) & 1."""
    block = 1 << w
    pattern = (1 << block) - 1
    pattern <<= block
    num_periods = n_bits // (2 * block)
    m = 0
    shift = 0
    for _ in range(num_periods):
        m |= (pattern << shift)
        shift += 2 * block
    return m


_INITIAL_MASKS = [0] * N
for _w in range(N):
    _INITIAL_MASKS[_w] = _build_mask(_w, _M)


def _verify(comparators):
    """Verify comparator list sorts all 2**N boolean inputs via bitmask method."""
    state = _INITIAL_MASKS[:]
    for i, j in comparators:
        a = state[i]
        b = state[j]
        state[i] = a & b
        state[j] = a | b
    for i in range(N - 1):
        if (state[i] & ~state[i + 1]) != 0:
            return False
    return True


def _odd_even_transposition():
    """Odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def _batcher_odd_even_merge(n):
    """Odd-even merge of n already-sorted halves. Returns (i,j) list."""
    if n <= 1:
        return []
    result = []

    def _oem(k, offset):
        if k <= 1:
            return
        half_k = k // 2
        for i in range(offset, offset + k, 2):
            _oem(half_k, i)
        for i in range(offset + 1, offset + k, 2):
            _oem(half_k, i)
        for i in range(offset, offset + k - 1):
            result.append((i, i + 1))

    _oem(n, 0)
    return result


def _batcher_odd_even_mergesort(n):
    """Batcher's odd-even mergesort for n (power-of-2) wires."""
    if n <= 1:
        return []
    half = n // 2
    left = _batcher_odd_even_mergesort(half)
    right = _batcher_odd_even_mergesort(half)
    right = [(i + half, j + half) for i, j in right]
    merge = _batcher_odd_even_merge(n)
    return left + right + merge


def _build_initial_network():
    """Dense Batcher mergesort on next power of 2 >= N, restricted to N wires."""
    p = 1
    while p < N:
        p *= 2
    network = _batcher_odd_even_mergesort(p)
    network = [(i, j) for i, j in network if i < N and j < N]
    seen = set()
    unique = []
    for comp in network:
        if comp not in seen:
            seen.add(comp)
            unique.append(comp)
    return unique


def _greedy_remove_pass(current, deadline):
    """One greedy pass: scan from end, remove if still valid. Returns (new_list, removed_count)."""
    removed = 0
    for i in range(len(current) - 1, -1, -1):
        if time.time() > deadline:
            break
        candidate = current[:i] + current[i + 1:]
        if _verify(candidate):
            current = candidate
            removed += 1
    return current, removed


def construct():
    """Construct a compare-exchange network that sorts all 18-input sequences."""
    start_time = time.time()
    # The evaluator kills after 90s wall-clock. Use a conservative deadline.
    deadline = start_time + 80

    # Start from dense Batcher network
    current = _build_initial_network()
    if not _verify(current):
        current = _odd_even_transposition()

    best = current[:]
    best_len = len(best)

    # Phase 1: Greedy removal (multiple passes until no more removals)
    while time.time() < deadline - 15:
        current, removed = _greedy_remove_pass(current, deadline)
        if removed == 0:
            break
        if len(current) < best_len:
            best = current[:]
            best_len = len(current)

    # Phase 2: Simulated annealing with diverse moves
    if time.time() < deadline - 3:
        T = 8.0
        T_min = 0.01
        cooling_rate = 0.9995
        rng = random.Random(42)

        all_comparators = [(i, j) for i in range(N) for j in range(i + 1, N)]

        steps_since_improvement = 0
        restart_threshold = 1500

        while time.time() < deadline - 2 and T > T_min:
            steps_since_improvement += 1

            if steps_since_improvement > restart_threshold:
                # Restart from best with perturbation
                current = best[:]
                for _ in range(rng.randint(3, 7)):
                    comp = rng.choice(all_comparators)
                    pos = rng.randrange(len(current) + 1)
                    current.insert(pos, comp)
                # Quick greedy pass to clean up
                for _ in range(3):
                    current, _ = _greedy_remove_pass(current, deadline)
                steps_since_improvement = 0

            r = rng.random()
            if r < 0.30:
                move_type = 'remove'
            elif r < 0.50:
                move_type = 'add'
            elif r < 0.65:
                move_type = 'swap'
            elif r < 0.80:
                move_type = 'move'
            else:
                move_type = 'slide'

            neighbor = current[:]
            valid_move = True

            if move_type == 'remove' and len(neighbor) > 0:
                idx = rng.randrange(len(neighbor))
                neighbor.pop(idx)
            elif move_type == 'add':
                comp = rng.choice(all_comparators)
                pos = rng.randrange(len(neighbor) + 1)
                neighbor.insert(pos, comp)
            elif move_type == 'swap' and len(neighbor) >= 2:
                i = rng.randrange(len(neighbor))
                j = rng.randrange(len(neighbor))
                if i == j:
                    valid_move = False
                else:
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            elif move_type == 'move' and len(neighbor) >= 2:
                i = rng.randrange(len(neighbor))
                j = rng.randrange(len(neighbor))
                if i == j:
                    valid_move = False
                else:
                    comp = neighbor.pop(i)
                    if i < j:
                        j -= 1
                    neighbor.insert(j, comp)
            elif move_type == 'slide' and len(neighbor) >= 1:
                i = rng.randrange(len(neighbor))
                if rng.random() < 0.5:
                    if i < len(neighbor) - 1:
                        neighbor[i], neighbor[i + 1] = neighbor[i + 1], neighbor[i]
                    else:
                        valid_move = False
                else:
                    if i > 0:
                        neighbor[i], neighbor[i - 1] = neighbor[i - 1], neighbor[i]
                    else:
                        valid_move = False
            else:
                valid_move = False

            if not valid_move:
                continue

            if _verify(neighbor):
                delta = len(neighbor) - len(current)
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    current = neighbor
                    if len(current) < best_len:
                        best = current[:]
                        best_len = len(current)
                        steps_since_improvement = 0
                    else:
                        steps_since_improvement += 1
                else:
                    steps_since_improvement += 1
            else:
                steps_since_improvement += 1

            T *= cooling_rate

    # Final greedy cleanup on best
    if time.time() < deadline - 1:
        best, _ = _greedy_remove_pass(best, deadline)

    # Final safety check
    if not _verify(best):
        best = _odd_even_transposition()

    return best
