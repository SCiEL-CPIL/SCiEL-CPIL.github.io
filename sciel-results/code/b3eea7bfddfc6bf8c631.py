"""HP lattice protein folding: fragment re-insertion + reversal moves with fast energy."""
# EVOLVE-BLOCK-START
import time
import random

_T0 = time.time()
_DEADLINE = 195.0  # 210s wall-clock limit, 15s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _energy_fast(seq, chain):
    """Fast energy using a position->index map."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if seq[i] == "H":
            x, y = p
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                j = where.get((x + dx, y + dy))
                if j is not None and j > i + 1 and seq[j] == "H":
                    contacts += 1
    return -contacts


def _is_valid_chain(chain):
    """Check self-avoidance and unit steps."""
    n = len(chain)
    if n < 2:
        return True
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            dx = abs(chain[i][0] - chain[i - 1][0])
            dy = abs(chain[i][1] - chain[i - 1][1])
            if dx + dy != 1:
                return False
    return True


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    for i in range(1, n):
        options = []
        x, y = chain[-1]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            site = (x + dx, y + dy)
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                sx, sy = site
                for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q = (sx + adx, sy + ady)
                    if q in occupied:
                        idx = occupied[q]
                        if idx < i - 1 and sequence[idx] == "H":
                            gain += 1
            free = 0
            sx, sy = site
            for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (sx + adx, sy + ady) not in occupied:
                    free += 1
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _try_reinsert_fragments(chain, sequence, rng, max_frag_len=5):
    """
    Attempt to re-insert a fragment: cut out chain[a:b], then try to regrow it
    from chain[a-1] (if a>0) connecting to chain[b] (if b<n).
    Returns (new_chain, energy_delta) or (None, 0).
    """
    n = len(chain)
    if n < 5:
        return None, 0

    # Pick fragment length and position
    L = rng.randint(2, min(max_frag_len, n - 3))
    a = rng.randint(1, n - L - 1)  # a >= 1 so there's a left neighbor
    b = a + L  # b <= n-1 so there's a right neighbor

    # The fragment to remove: chain[a:b]
    # Left neighbor: chain[a-1], Right neighbor: chain[b]
    left_neighbor = chain[a - 1]
    right_neighbor = chain[b]

    # The rest of the chain without the fragment
    rest = chain[:a] + chain[b:]
    rest_occupied = {p: i for i, p in enumerate(rest)}

    # Try to regrow the fragment from left_neighbor, must end at right_neighbor
    # Fragment length is L, so we need L steps from left_neighbor to right_neighbor
    # The fragment residues are sequence[a:b]
    frag_seq = sequence[a:b]

    # Check if right_neighbor is at Manhattan distance L from left_neighbor
    lx, ly = left_neighbor
    rx, ry = right_neighbor
    manhattan = abs(rx - lx) + abs(ry - ly)
    if manhattan > L or (L - manhattan) % 2 != 0:
        return None, 0

    # Try to grow the fragment from left_neighbor to right_neighbor
    # This is a constrained path problem
    def _grow_constrained(start, end, length, seq, occupied, rng, steps_left=50):
        """Grow a chain of given length from start to end, avoiding occupied sites."""
        if length == 0:
            if start == end:
                return [start]
            return None

        x, y = start
        options = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            site = (x + dx, y + dy)
            if site in occupied:
                continue
            # Heuristic: prefer moving toward end
            ex, ey = end
            dist_to_end = abs(ex - site[0]) + abs(ey - site[1])
            options.append((dist_to_end + rng.random() * 0.5, site))

        if not options:
            return None

        # Sort by distance to end, try a few
        options.sort(key=lambda t: t[0])
        for _, site in options[:3]:
            new_occupied = dict(occupied)
            new_occupied[site] = True
            result = _grow_constrained(site, end, length - 1, seq, new_occupied, rng, steps_left - 1)
            if result is not None:
                return [start] + result
        return None

    # Grow fragment
    fragment = _grow_constrained(left_neighbor, right_neighbor, L, frag_seq, rest_occupied, rng)
    if fragment is None:
        return None, 0

    # The fragment includes the start point, so actual new positions are fragment[1:]
    # Check that fragment[-1] == right_neighbor
    if fragment[-1] != right_neighbor:
        return None, 0

    # Build new chain
    new_chain = chain[:a] + fragment[1:] + chain[b:]

    # Validate
    if not _is_valid_chain(new_chain):
        return None, 0

    return new_chain, 0


def _reversal_move(chain, rng, max_len=None):
    """Random reversal move."""
    n = len(chain)
    if n < 4:
        return None
    if max_len is None:
        max_len = min(8, n - 2)
    L = rng.randint(2, max_len)
    a = rng.randint(0, n - L - 1)
    b = a + L
    new_chain = chain[:a] + chain[a:b + 1][::-1] + chain[b + 1:]
    if _is_valid_chain(new_chain):
        return new_chain
    return None


def _kink_move(chain, rng):
    """
    Kink move: pick a residue i (1..n-2) and try to move it to an adjacent
    free position, if the result is a valid self-avoiding walk.
    """
    n = len(chain)
    if n < 3:
        return None
    i = rng.randint(1, n - 2)
    x, y = chain[i]
    options = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        site = (x + dx, y + dy)
        # Must not be occupied by any other residue
        if site in chain and site != chain[i]:
            continue
        # Must maintain unit step with neighbors
        # Check with chain[i-1]
        px, py = chain[i - 1]
        if abs(site[0] - px) + abs(site[1] - py) != 1:
            continue
        # Check with chain[i+1]
        nx, ny = chain[i + 1]
        if abs(site[0] - nx) + abs(site[1] - ny) != 1:
            continue
        options.append(site)

    if not options:
        return None
    site = options[rng.randint(0, len(options) - 1)]
    new_chain = chain[:]
    new_chain[i] = site
    if _is_valid_chain(new_chain):
        return new_chain
    return None


def _local_search(sequence, chain, rng, max_iters=500):
    """Local search with fragment re-insertion, reversals, and kinks."""
    best_chain = list(chain)
    best_energy = _energy_fast(sequence, best_chain)
    n = len(best_chain)
    accept_rate = 1.0

    for it in range(max_iters):
        if _remaining() < 5.0:
            break

        # Choose move type
        r = rng.random()
        if r < 0.3:
            new_chain, _ = _try_reinsert_fragments(best_chain, sequence, rng)
        elif r < 0.7:
            new_chain = _reversal_move(best_chain, rng)
        else:
            new_chain = _kink_move(best_chain, rng)

        if new_chain is None:
            continue

        e = _energy_fast(sequence, new_chain)
        if e < best_energy:
            best_chain = new_chain
            best_energy = e
            accept_rate = min(1.0, accept_rate * 1.05)
        elif rng.random() < accept_rate * 0.1:
            # Occasionally accept worse to escape local minima
            best_chain = new_chain

    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using multiple restarts and local search with time guard."""
    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 0  # start at 0 (unfolded)
    n = len(sequence)

    # Number of restarts based on time available
    num_restarts = 400
    for r in range(num_restarts):
        if _remaining() < 3.0:
            break

        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue

        e = _energy_fast(sequence, chain)
        if e < best_energy:
            best_energy = e
            best = chain

        # Apply local search on the best candidate periodically
        if r % 5 == 0 and best is not None and _remaining() > 10.0:
            improved, imp_e = _local_search(sequence, best, rng, max_iters=150)
            if imp_e < best_energy:
                best_energy = imp_e
                best = improved

    # Final local search on the overall best if time permits
    if best is not None and _remaining() > 5.0:
        improved, imp_e = _local_search(sequence, best, rng, max_iters=300)
        if imp_e < best_energy:
            best_energy = imp_e
            best = improved

    if best is None:
        # Fallback: straight line
        best = [(i, 0) for i in range(n)]

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
