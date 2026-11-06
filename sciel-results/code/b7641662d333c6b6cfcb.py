"""HP lattice protein folding: beam search + simulated annealing with pivot/reversal/reinsert moves."""
# EVOLVE-BLOCK-START
import time
import random
import math

_T0 = time.time()
_DEADLINE = 198.0  # 210s wall-clock limit, 12s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _energy_fast(seq, chain):
    """Fast energy: -count of H-H contacts on non-adjacent lattice neighbours."""
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


def _grow_beam(sequence, dimension, rng, beam_width=12):
    """
    Beam search growth: keep top beam_width partial chains at each step.
    Score = negative H-H contacts so far + small free-space bonus.
    Returns a list of complete chains (all beam survivors at the end).
    """
    n = len(sequence)
    if n == 0:
        return [[]]
    start = (0, 0)
    h_set = set(i for i, c in enumerate(sequence) if c == "H")

    # beam entries: (chain, occupied_set, score)
    beam = [([start], {start}, 0.0)]

    for i in range(1, n):
        candidates = []
        for chain, occupied, score in beam:
            x, y = chain[-1]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                site = (x + dx, y + dy)
                if site in occupied:
                    continue
                new_score = score
                if i in h_set:
                    sx, sy = site
                    for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        if (sx + adx, sy + ady) in occupied:
                            idx = chain.index((sx + adx, sy + ady))
                            if idx < i - 1 and idx in h_set:
                                new_score -= 1  # more negative = better
                # small free-space bonus to avoid trapping
                free = 0
                sx, sy = site
                for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if (sx + adx, sy + ady) not in occupied:
                        free += 1
                new_score += 0.05 * (4 - free)  # prefer more open sites

                new_chain = chain + [site]
                new_occ = occupied | {site}
                candidates.append((new_chain, new_occ, new_score))

        if not candidates:
            return [beam[0][0]]  # return best we have

        # Sort by score (lower = better) and keep top beam_width
        candidates.sort(key=lambda c: c[2])
        # Deduplicate by chain positions
        seen_chains = set()
        beam = []
        for ch, occ, sc in candidates:
            key = tuple(ch)
            if key not in seen_chains:
                seen_chains.add(key)
                beam.append((ch, occ, sc))
            if len(beam) >= beam_width:
                break

    return [ch for ch, occ, sc in beam]


def _pivot_move(chain, rng):
    """
    Pivot move: pick a cut point i (1..n-2), rotate the suffix chain[i:]
    by 90 or 180 degrees around chain[i-1].
    """
    n = len(chain)
    if n < 4:
        return None
    i = rng.randint(1, n - 2)
    px, py = chain[i - 1]
    cx, cy = chain[i]

    # Direction from p to c
    dx = cx - px
    dy = cy - py

    # 90-degree rotations: (dx,dy) -> (-dy,dx) or (dy,-dx)
    # 180: (dx,dy) -> (-dx,-dy)
    rotations = [(-dy, dx), (dy, -dx), (-dx, -dy)]
    rng.shuffle(rotations)

    for ndx, ndy in rotations:
        # New direction for chain[i]
        new_ci = (px + ndx, py + ndy)
        if new_ci in chain and new_ci != chain[i]:
            continue

        # Build new suffix: rotate all relative positions
        # Original suffix relative to chain[i]: offsets
        new_chain = chain[:i] + [new_ci]
        # For j > i, new position = new_ci + (old[j] - old[i])
        valid = True
        for j in range(i + 1, n):
            ox, oy = chain[j]
            nx2 = new_ci[0] + (ox - cx)
            ny2 = new_ci[1] + (oy - cy)
            new_chain.append((nx2, ny2))

        if _is_valid_chain(new_chain):
            return new_chain
    return None


def _reversal_move(chain, rng, max_len=None):
    """Random reversal move."""
    n = len(chain)
    if n < 4:
        return None
    if max_len is None:
        max_len = min(10, n - 2)
    L = rng.randint(2, max_len)
    a = rng.randint(0, n - L - 1)
    b = a + L
    new_chain = chain[:a] + chain[a:b + 1][::-1] + chain[b + 1:]
    if _is_valid_chain(new_chain):
        return new_chain
    return None


def _kink_move(chain, rng):
    """Kink move: move a single interior residue to an adjacent free site."""
    n = len(chain)
    if n < 3:
        return None
    i = rng.randint(1, n - 2)
    x, y = chain[i]
    occ = set(chain)
    options = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        site = (x + dx, y + dy)
        if site in occ and site != chain[i]:
            continue
        px, py = chain[i - 1]
        if abs(site[0] - px) + abs(site[1] - py) != 1:
            continue
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


def _reinsert_fragment(chain, sequence, rng, max_frag_len=6):
    """
    Cut out a fragment and try to regrow it via DFS path finding.
    """
    n = len(chain)
    if n < 6:
        return None

    L = rng.randint(2, min(max_frag_len, n - 3))
    a = rng.randint(1, n - L - 1)
    b = a + L

    left_neighbor = chain[a - 1]
    right_neighbor = chain[b]

    rest = chain[:a] + chain[b:]
    rest_occupied = set(rest)

    lx, ly = left_neighbor
    rx, ry = right_neighbor
    manhattan = abs(rx - lx) + abs(ry - ly)
    if manhattan > L or (L - manhattan) % 2 != 0:
        return None

    # DFS to find self-avoiding path of exactly L steps
    def _dfs(current, remaining, path, occupied):
        if remaining == 0:
            if current == right_neighbor:
                return list(path)
            return None
        x, y = current
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            site = (x + dx, y + dy)
            if site in occupied:
                continue
            dist_to_end = abs(right_neighbor[0] - site[0]) + abs(right_neighbor[1] - site[1])
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

    path = _dfs(left_neighbor, L, [left_neighbor], {left_neighbor} | rest_occupied)
    if path is None or path[-1] != right_neighbor:
        return None

    new_frag = path[1:]
    if len(new_frag) != L:
        return None

    new_chain = chain[:a] + new_frag + chain[b:]
    if _is_valid_chain(new_chain):
        return new_chain
    return None


def _simulated_annealing(sequence, chain, rng, t_start=1.0, t_end=0.01, max_iters=3000):
    """
    Simulated annealing with pivot, reversal, kink, and reinsert moves.
    Returns (best_chain, best_energy).
    """
    best_chain = list(chain)
    best_energy = _energy_fast(sequence, best_chain)
    cur_chain = list(chain)
    cur_energy = best_energy
    n = len(cur_chain)

    # Adaptive move weights
    move_weights = [1.0, 1.0, 1.0, 0.5]  # pivot, reversal, kink, reinsert
    move_success = [0, 0, 0, 0]
    move_attempts = [0, 0, 0, 0]

    total_steps = max(100, max_iters)
    for it in range(total_steps):
        if _remaining() < 5.0:
            break

        # Geometric cooling
        if total_steps > 1:
            frac = it / total_steps
            T = t_start * (t_end / t_start) ** frac
        else:
            T = t_start

        # Choose move type
        r = rng.random()
        cumulative = [0]
        for w in move_weights:
            cumulative.append(cumulative[-1] + w)
        total_w = cumulative[-1]
        r *= total_w
        if r < cumulative[1]:
            move_type = 0  # pivot
        elif r < cumulative[2]:
            move_type = 1  # reversal
        elif r < cumulative[3]:
            move_type = 2  # kink
        else:
            move_type = 3  # reinsert

        move_attempts[move_type] += 1

        if move_type == 0:
            new_chain = _pivot_move(cur_chain, rng)
        elif move_type == 1:
            new_chain = _reversal_move(cur_chain, rng)
        elif move_type == 2:
            new_chain = _kink_move(cur_chain, rng)
        else:
            new_chain = _reinsert_fragment(cur_chain, sequence, rng)

        if new_chain is None:
            continue

        new_energy = _energy_fast(sequence, new_chain)
        delta = new_energy - cur_energy

        if delta <= 0:
            cur_chain = new_chain
            cur_energy = new_energy
            move_success[move_type] += 1
            if cur_energy < best_energy:
                best_chain = list(cur_chain)
                best_energy = cur_energy
        else:
            # Accept with probability exp(-delta/T)
            if T > 0 and rng.random() < math.exp(-delta / T):
                cur_chain = new_chain
                cur_energy = new_energy
                move_success[move_type] += 1

        # Adapt move weights every 200 iterations
        if it % 200 == 199:
            for k in range(4):
                if move_attempts[k] > 0:
                    rate = move_success[k] / move_attempts[k]
                    move_weights[k] = max(0.1, rate * 2.0)
                move_attempts[k] = 0
                move_success[k] = 0

    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using beam search + multiple SA runs with time guard."""
    rng = random.Random(777 + len(sequence) * 13)
    n = len(sequence)

    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]

    # Generate diverse initial chains via beam search
    initial_chains = _grow_beam(sequence, dimension, rng, beam_width=15)

    best_chain = None
    best_energy = 0

    # Evaluate all initial chains
    for ch in initial_chains:
        e = _energy_fast(sequence, ch)
        if e < best_energy:
            best_energy = e
            best_chain = list(ch)

    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy_fast(sequence, best_chain)

    # Run multiple SA passes from different starting points
    # Allocate time budget across SA runs
    sa_runs = []
    # Start with the best, then a few others
    sa_runs.append(best_chain)
    for ch in initial_chains[:5]:
        if ch != best_chain:
            sa_runs.append(ch)
        if len(sa_runs) >= 6:
            break

    # If we have very little time, just do one SA run
    if _remaining() < 20.0:
        if best_chain is not None and _remaining() > 5.0:
            improved, imp_e = _simulated_annealing(
                sequence, best_chain, rng,
                t_start=0.8, t_end=0.02, max_iters=1500
            )
            if imp_e < best_energy:
                best_energy = imp_e
                best_chain = improved
        return [list(p) for p in best_chain]

    # Distribute time among SA runs
    num_sa = len(sa_runs)
    time_per_sa = (_remaining() - 10.0) / max(1, num_sa)

    for idx, start_chain in enumerate(sa_runs):
        if _remaining() < 8.0:
            break

        # Scale iterations by available time
        iters = int(time_per_sa * 80)  # rough: ~80 iters per second
        iters = max(500, min(5000, iters))

        # Later runs can use slightly lower T for refinement
        t_start = 1.0 if idx == 0 else 0.6
        t_end = 0.01

        improved, imp_e = _simulated_annealing(
            sequence, start_chain, rng,
            t_start=t_start, t_end=t_end, max_iters=iters
        )
        if imp_e < best_energy:
            best_energy = imp_e
            best_chain = improved

    # Final polish on the best
    if best_chain is not None and _remaining() > 8.0:
        iters = int((_remaining() - 5.0) * 60)
        iters = max(500, min(3000, iters))
        improved, imp_e = _simulated_annealing(
            sequence, best_chain, rng,
            t_start=0.3, t_end=0.005, max_iters=iters
        )
        if imp_e < best_energy:
            best_energy = imp_e
            best_chain = improved

    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
