"""HP lattice protein folding: contact-graph guided local search with fast collision detection."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 205.0

_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    n = len(chain)
    if n < 2:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y = p
            for dx, dy in _DELTAS:
                q = (x + dx, y + dy)
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def _backtrack_fold(sequence, time_limit, rng):
    """
    Backtracking search with greedy initialization and local refinement.
    """
    n = len(sequence)
    if n < 2:
        return [(0, 0)] * n

    is_h = [c == 'H' for c in sequence]

    best_chain = None
    best_energy = 0

    def _greedy_grow(seq, rng, hh_weight=1.0, free_weight=0.1, noise=0.3):
        nn = len(seq)
        if nn < 2:
            return [(0, 0)] * nn
        start = (0, 0)
        chain = [start]
        occupied = {start: 0}

        for i in range(1, nn):
            options = []
            for site in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                x, y = chain[-1]
                cand = (x + site[0], y + site[1])
                if cand in occupied:
                    continue
                gain = 0
                if seq[i] == 'H':
                    cx, cy = cand
                    for dx, dy in _DELTAS:
                        q = (cx + dx, cy + dy)
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and seq[j] == 'H':
                            gain += 1
                free = 0
                cx, cy = cand
                for dx, dy in _DELTAS:
                    if (cx + dx, cy + dy) not in occupied:
                        free += 1
                score = hh_weight * gain + free_weight * free + noise * rng.random()
                options.append((score, cand))
            if not options:
                return None
            best_score, best_site = max(options)
            occupied[best_site] = i
            chain.append(best_site)

        return chain

    # Try multiple greedy starts with different parameters
    initial_chains = []
    configs = [
        (2.0, 0.0, 0.5),
        (1.0, 0.5, 0.5),
        (0.5, 1.0, 0.5),
        (1.5, 0.2, 0.3),
        (1.0, 0.3, 1.0),
        (3.0, 0.0, 0.2),
        (1.0, 0.0, 0.8),
        (2.5, 0.1, 0.4),
        (1.2, 0.6, 0.6),
    ]
    growth_deadline = time.time() + min(3.0, time_limit * 0.15)
    for hh_w, free_w, noise in configs:
        if time.time() > growth_deadline:
            break
        for attempt in range(4):
            if time.time() > growth_deadline:
                break
            chain = _greedy_grow(sequence, rng, hh_w, free_w, noise)
            if chain is not None:
                initial_chains.append(chain)

    if not initial_chains:
        initial_chains.append([(i, 0) for i in range(n)])

    # Evaluate initial chains
    scored = []
    for chain in initial_chains:
        e = _energy(sequence, chain)
        scored.append((e, chain))
    scored.sort(key=lambda x: x[0])
    candidates = [s[1] for s in scored[:min(8, len(scored))]]

    def _local_refine(chain, time_limit, rng):
        """Iterated local search with fast collision detection."""
        nn = len(chain)
        current_energy = _energy(sequence, chain)
        best_energy = current_energy
        best_chain = chain[:]
        T = 1.2
        cooling = 0.99996
        min_t = 0.005

        deadline = time.time() + time_limit
        no_improve = 0
        max_no_improve = 4000
        stagnation_count = 0

        h_indices = [i for i in range(nn) if sequence[i] == 'H']

        def _try_reversal(state_chain, i, j):
            """Try reversal from i to j with fast set-based collision check."""
            if i >= j:
                return None
            if i > 0:
                px, py = state_chain[i - 1]
                ex, ey = state_chain[j]
                if abs(px - ex) + abs(py - ey) != 1:
                    return None
            if j < nn - 1:
                sx, sy = state_chain[i]
                nx, ny = state_chain[j + 1]
                if abs(sx - nx) + abs(sy - ny) != 1:
                    return None
            new_chain = state_chain[:i] + state_chain[i:j+1][::-1] + state_chain[j+1:]
            if len(set(new_chain)) != nn:
                return None
            return new_chain

        def _try_pivot(state_chain, i, angle):
            """Try pivot at index i with fast set-based collision check."""
            if i <= 0 or i >= nn - 1:
                return None
            px, py = state_chain[i]
            tail = state_chain[i+1:]
            rotated = []
            for tx, ty in tail:
                rx, ry = tx - px, ty - py
                for _ in range(angle % 4):
                    rx, ry = -ry, rx
                rotated.append((px + rx, py + ry))
            new_chain = state_chain[:i+1] + rotated
            if len(set(new_chain)) != nn:
                return None
            return new_chain

        def _generate_move():
            """Generate a move with bias toward H-rich regions."""
            move_type = rng.random()
            if move_type < 0.55:
                # Reversal
                if h_indices and rng.random() < 0.4:
                    hi = rng.choice(h_indices)
                    span = rng.randint(2, min(25, nn - 1))
                    i = max(0, hi - span // 2)
                    j = min(nn - 1, i + span)
                    if j <= i:
                        i = rng.randint(0, nn - 2)
                        j = rng.randint(i + 1, nn - 1)
                else:
                    i = rng.randint(0, nn - 2)
                    j = rng.randint(i + 1, nn - 1)
                return ('rev', i, j)
            else:
                # Pivot
                if h_indices and rng.random() < 0.4:
                    hi = rng.choice(h_indices)
                    i = max(1, min(nn - 2, hi + rng.randint(-3, 3)))
                else:
                    i = rng.randint(1, nn - 2)
                angle = rng.choice([1, 2, 3])
                return ('pivot', i, angle)

        while time.time() < deadline:
            if _remaining_time() < 2.0:
                break

            move = _generate_move()
            if move[0] == 'rev':
                new_chain = _try_reversal(chain, move[1], move[2])
            else:
                new_chain = _try_pivot(chain, move[1], move[2])

            if new_chain is None:
                no_improve += 1
                continue

            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy

            accept = False
            if delta <= 0:
                accept = True
            elif T > min_t:
                exp_arg = -delta / T
                if exp_arg > -50:
                    accept = rng.random() < math.exp(exp_arg)

            if accept:
                chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
                    no_improve = 0
                    stagnation_count = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

            if no_improve > max_no_improve:
                if _remaining_time() < 3.0:
                    break
                chain = best_chain[:]
                current_energy = best_energy
                for _ in range(rng.randint(3, 8)):
                    move = _generate_move()
                    if move[0] == 'rev':
                        nc = _try_reversal(chain, move[1], move[2])
                    else:
                        nc = _try_pivot(chain, move[1], move[2])
                    if nc is not None:
                        chain = nc
                        current_energy = _energy(sequence, chain)
                        break
                no_improve = 0
                T = max(T * 1.4, 0.5)
                stagnation_count += 1
                if stagnation_count > 6:
                    break

            T *= cooling
            if T < min_t:
                break

        return best_chain, best_energy

    # Allocate time to local refinement
    total_time = _remaining_time() - 3.0
    if total_time < 1.0:
        return candidates[0] if candidates else [(i, 0) for i in range(n)]

    scored_candidates = sorted(
        [(c, _energy(sequence, c)) for c in candidates],
        key=lambda x: x[1]
    )

    n_cands = len(scored_candidates)
    if n_cands == 1:
        allocations = [total_time]
    elif n_cands == 2:
        allocations = [total_time * 0.6, total_time * 0.4]
    else:
        # Give more time to better candidates
        weights = []
        for k in range(n_cands):
            w = 1.0 / (k + 1)
            weights.append(w)
        s = sum(weights)
        allocations = [w * total_time / s for w in weights]

    for idx, (chain, _) in enumerate(scored_candidates):
        if _remaining_time() < 2.0:
            break
        alloc = min(allocations[idx], _remaining_time() - 1.5)
        if alloc < 0.3:
            continue
        refined_chain, refined_energy = _local_refine(chain, alloc, rng)
        if best_chain is None or refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain

    if best_chain is None:
        best_chain = candidates[0] if candidates else [(i, 0) for i in range(n)]

    return best_chain


def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    """
    rng = random.Random(42)
    n = len(sequence)

    if n < 2:
        return [[0, 0]] * n

    time_limit = max(0.5, _remaining_time() - 5.0)

    best_chain = _backtrack_fold(sequence, time_limit, rng)

    if best_chain is None or len(best_chain) != n:
        best_chain = [(i, 0) for i in range(n)]

    # Validate: ensure self-avoiding and unit steps
    seen = set()
    valid = True
    for i, (x, y) in enumerate(best_chain):
        if (x, y) in seen:
            valid = False
            break
        seen.add((x, y))
        if i > 0:
            px, py = best_chain[i-1]
            if abs(px - x) + abs(py - y) != 1:
                valid = False
                break

    if not valid:
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
