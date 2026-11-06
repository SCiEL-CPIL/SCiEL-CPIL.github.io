"""HP lattice protein folding: fragment re-insertion + reversal moves with fast energy."""
# EVOLVE-BLOCK-START
import time
import random

_T0 = time.time()
_DEADLINE = 195.0  # 210s wall-clock limit, 15s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _energy_fast(seq, chain):
    """Fast energy using a position->index map and precomputed H indices."""
    where = {p: i for i, p in enumerate(chain)}
    h_set = set(i for i, c in enumerate(seq) if c == "H")
    contacts = 0
    for i in h_set:
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = where.get((x + dx, y + dy))
            if j is not None and j > i + 1 and j in h_set:
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
    h_set = set(i for i, c in enumerate(sequence) if c == "H")
    for i in range(1, n):
        options = []
        x, y = chain[-1]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            site = (x + dx, y + dy)
            if site in occupied:
                continue
            gain = 0
            if i in h_set:
                sx, sy = site
                for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q = (sx + adx, sy + ady)
                    if q in occupied:
                        idx = occupied[q]
                        if idx < i - 1 and idx in h_set:
                            gain += 1
            free = 0
            sx, sy = site
            for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (sx + adx, sy + ady) not in occupied:
                    free += 1
            # Weight H-H contacts much more heavily
            score = 2.0 * gain + 0.1 * free + rng.random()
            options.append((score, site))
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
    Uses BFS with heuristic ordering for reliability.
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
    rest_occupied = set(rest)

    # Check if right_neighbor is at Manhattan distance L from left_neighbor
    lx, ly = left_neighbor
    rx, ry = right_neighbor
    manhattan = abs(rx - lx) + abs(ry - ly)
    if manhattan > L or (L - manhattan) % 2 != 0:
        return None, 0

    # BFS to find a path of exactly L steps from left_neighbor to right_neighbor
    # avoiding rest_occupied
    # State: (current_pos, path_so_far)
    # We want exactly L steps, so we do BFS layer by layer

    from collections import deque

    # BFS: find any self-avoiding path of length L from left to right
    # For small L (<=5), this is feasible
    def _find_path_bfs(start, end, length, occupied_set):
        """Find a self-avoiding path of exactly `length` steps from start to end."""
        if length == 0:
            return [start] if start == end else None

        # BFS with path tracking
        # visited: set of (position, step_count) to avoid re-exploring
        # But we need self-avoidance, so we track the full path
        # For small lengths, use DFS with pruning

        def _dfs(current, remaining, path, occupied):
            if remaining == 0:
                if current == end:
                    return list(path)
                return None

            x, y = current
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                site = (x + dx, y + dy)
                if site in occupied:
                    continue
                # Heuristic pruning: can we still reach end in remaining steps?
                dist_to_end = abs(end[0] - site[0]) + abs(end[1] - site[1])
                if dist_to_end > remaining or (remaining - dist_to_end) % 2 != 0:
                    continue
                path.append(site)
                occupied.add(site)
                result = _dfs(site, remaining - 1, path, occupied)
                if result is not None:
                    return result
                path.pop()
                occupied.discard(site)
            return None

        return _dfs(start, length, [start], {start} | occupied_set)

    path = _find_path_bfs(left_neighbor, right_neighbor, L, rest_occupied)
    if path is None:
        return None, 0

    # path[0] is left_neighbor, path[-1] should be right_neighbor
    if path[-1] != right_neighbor:
        return None, 0

    # The new fragment positions are path[1:] (excluding the start which is already in chain)
    # Build new chain: chain[:a] + path[1:] + chain[b:]
    # But wait: chain[a-1] is left_neighbor, and path[0] = left_neighbor
    # So the new chain is: chain[:a] + path[1:] + chain[b:]
    # chain[:a] ends with chain[a-1] = left_neighbor
    # path[1:] starts with the next position after left_neighbor
    # path[-1] = right_neighbor = chain[b]
    # chain[b:] starts with chain[b] = right_neighbor

    # Actually, we need to be careful. The fragment occupies positions a, a+1, ..., b-1
    # The new fragment should be path[1:L+1] which has L elements
    # path has L+1 elements: path[0]=left_neighbor, path[1..L] are the L fragment positions

    new_frag = path[1:]  # L elements
    if len(new_frag) != L:
        return None, 0

    new_chain = chain[:a] + new_frag + chain[b:]

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
    """Local search with fragment re-insertion, reversals, and kinks. Adaptive iteration count."""
    best_chain = list(chain)
    best_energy = _energy_fast(sequence, best_chain)
    n = len(best_chain)
    accept_rate = 1.0
    no_improve_count = 0
    max_no_improve = 30

    for it in range(max_iters):
        if _remaining() < 5.0:
            break

        # Adaptive: if stuck, reset and try harder
        if no_improve_count >= max_no_improve:
            # Perturb: do a random reversal to escape
            perturbed = _reversal_move(best_chain, rng, max_len=min(10, n - 2))
            if perturbed is not None:
                best_chain = perturbed
                best_energy = _energy_fast(sequence, best_chain)
            no_improve_count = 0
            max_no_improve = min(60, max_no_improve * 1.5)

        # Choose move type - weight based on what's working
        r = rng.random()
        if r < 0.25:
            new_chain, _ = _try_reinsert_fragments(best_chain, sequence, rng)
        elif r < 0.65:
            new_chain = _reversal_move(best_chain, rng)
        else:
            new_chain = _kink_move(best_chain, rng)

        if new_chain is None:
            no_improve_count += 1
            continue

        e = _energy_fast(sequence, new_chain)
        if e < best_energy:
            best_chain = new_chain
            best_energy = e
            no_improve_count = 0
            accept_rate = min(1.0, accept_rate * 1.02)
        elif rng.random() < accept_rate * 0.05:
            # Occasionally accept worse to escape local minima
            best_chain = new_chain
            no_improve_count += 1
        else:
            no_improve_count += 1

    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using multiple restarts and local search with time guard."""
    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 0  # start at 0 (unfolded)
    n = len(sequence)

    # Number of restarts based on time available
    num_restarts = 500
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
        # More frequent local search, especially when making progress
        if r % 3 == 0 and best is not None and _remaining() > 8.0:
            iters = 200 if _remaining() > 30 else 100
            improved, imp_e = _local_search(sequence, best, rng, max_iters=iters)
            if imp_e < best_energy:
                best_energy = imp_e
                best = improved

    # Final polish: high-iteration local search on the best
    if best is not None and _remaining() > 5.0:
        # Use remaining time for intensive polish
        iters = 500 if _remaining() > 20 else 200
        improved, imp_e = _local_search(sequence, best, rng, max_iters=iters)
        if imp_e < best_energy:
            best_energy = imp_e
            best = improved

    # Second polish if time allows
    if best is not None and _remaining() > 3.0:
        improved, imp_e = _local_search(sequence, best, rng, max_iters=200)
        if imp_e < best_energy:
            best_energy = imp_e
            best = improved

    if best is None:
        # Fallback: straight line
        best = [(i, 0) for i in range(n)]

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
