"""HP lattice folding: optimized beam search growth + multi-move local search."""
# EVOLVE-BLOCK-START
import random
import time

_T0 = time.time()
_DEADLINE = 200.0


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _compute_energy(sequence, chain):
    """Compute energy from scratch using position->index map."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = where.get((x + dx, y + dy))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _beam_search(sequence, beam_width):
    """
    Beam search growth with incremental energy tracking.
    Returns list of (energy, chain) for complete conformations.
    """
    n = len(sequence)
    if n <= 1:
        return [(0, [(0, 0)])]

    # Beam entry: (energy, chain, pos_to_idx)
    start = (0, 0)
    beam = [(0, [start], {start: 0})]

    for i in range(1, n):
        new_beam = []
        seq_i = sequence[i]
        for energy, chain, pos_to_idx in beam:
            last = chain[-1]
            lx, ly = last
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                site = (lx + dx, ly + dy)
                if site in pos_to_idx:
                    continue
                new_chain = chain + [site]
                new_pos_to_idx = dict(pos_to_idx)
                new_pos_to_idx[site] = i
                delta_contacts = 0
                if seq_i == 'H':
                    sx, sy = site
                    for dx2, dy2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        j = new_pos_to_idx.get((sx + dx2, sy + dy2), -1)
                        if 0 <= j <= i - 2 and sequence[j] == 'H':
                            delta_contacts += 1
                new_energy = energy - delta_contacts
                new_beam.append((new_energy, new_chain, new_pos_to_idx))

        if not new_beam:
            return []

        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]

    if not beam:
        return []

    results = []
    for energy, chain, pos_to_idx in beam:
        exact_energy = _compute_energy(sequence, chain)
        results.append((exact_energy, chain))
    return results


def _local_search(sequence, chain, rng, time_budget):
    """
    Local search with kink moves, loop moves, and simulated annealing.
    """
    n = len(chain)
    if n < 3:
        return chain

    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    current_chain = list(chain)
    current_energy = best_energy

    start_time = time.time()
    T = 3.0
    T_min = 0.05
    cooling = 0.9997

    while T > T_min and (time.time() - start_time) < time_budget and _remaining() > 2.0:
        move_type = rng.random()
        if move_type < 0.55:
            # Kink move: move one residue
            i = rng.randint(1, n - 2)
            old_pos = current_chain[i]
            occupied = set(current_chain)
            candidates = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (old_pos[0] + dx, old_pos[1] + dy)
                if q == current_chain[i - 1] or q == current_chain[i + 1]:
                    continue
                if q not in occupied:
                    candidates.append(q)
            if not candidates:
                T *= cooling
                continue
            new_pos = rng.choice(candidates)
            # Verify self-avoidance after move
            test_chain = current_chain[:]
            test_chain[i] = new_pos
            if len(set(test_chain)) != n:
                T *= cooling
                continue
            # Compute energy delta
            pos_to_idx = {p: idx for idx, p in enumerate(current_chain)}
            delta = 0
            if sequence[i] == 'H':
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    j = pos_to_idx.get((old_pos[0] + dx, old_pos[1] + dy), -1)
                    if j >= 0 and j not in (i - 1, i, i + 1) and sequence[j] == 'H':
                        delta += 1  # losing contact
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    j = pos_to_idx.get((new_pos[0] + dx, new_pos[1] + dy), -1)
                    if j >= 0 and j not in (i - 1, i, i + 1) and sequence[j] == 'H':
                        delta -= 1  # gaining contact
            else:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    j = pos_to_idx.get((old_pos[0] + dx, old_pos[1] + dy), -1)
                    if j >= 0 and j not in (i - 1, i, i + 1) and sequence[j] == 'H':
                        delta += 1
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -4)):
                    j = pos_to_idx.get((new_pos[0] + dx, new_pos[1] + dy), -1)
                    if j >= 0 and j not in (i - 1, i, i + 1) and sequence[j] == 'H':
                        delta -= 1
            delta_e = -delta
            if delta_e <= 0 or rng.random() < (2.718281828459045 ** (-delta_e / T)):
                current_chain[i] = new_pos
                current_energy += delta_e
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)

        elif move_type < 0.80:
            # Reversal move: reverse a segment
            if n < 4:
                T *= cooling
                continue
            i = rng.randint(1, n - 3)
            length = rng.randint(2, min(n - i - 1, 10))
            j = i + length - 1
            if j >= n - 1:
                T *= cooling
                continue
            # Try reversing chain[i:j+1]
            segment = current_chain[i:j + 1]
            reversed_segment = segment[::-1]
            # Check if reversal is valid (no self-avoidance violations)
            test_chain = current_chain[:]
            test_chain[i:j + 1] = reversed_segment
            if len(set(test_chain)) != n:
                T *= cooling
                continue
            # Check unit steps at boundaries
            if i > 0:
                p1 = test_chain[i - 1]
                p2 = test_chain[i]
                if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) != 1:
                    T *= cooling
                    continue
            if j < n - 1:
                p1 = test_chain[j]
                p2 = test_chain[j + 1]
                if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) != 1:
                    T *= cooling
                    continue
            new_energy = _compute_energy(sequence, test_chain)
            delta_e = new_energy - current_energy
            if delta_e <= 0 or rng.random() < (2.718281828459045 ** (-delta_e / T)):
                current_chain = test_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
        else:
            # Fragment re-insertion: pick a short fragment, remove it, try re-inserting elsewhere
            if n < 6:
                T *= cooling
                continue
            frag_len = rng.randint(2, min(5, n // 3))
            i = rng.randint(1, n - frag_len - 1)
            fragment = current_chain[i:i + frag_len]
            # Remove fragment
            remaining = current_chain[:i] + current_chain[i + frag_len:]
            if len(set(remaining)) != len(remaining):
                T *= cooling
                continue
            # Try inserting fragment at various positions
            best_insert = None
            best_insert_energy = current_energy
            for pos in range(1, len(remaining) - frag_len + 1):
                # Check if insertion creates valid chain
                test_chain = remaining[:pos] + fragment + remaining[pos:]
                if len(set(test_chain)) != n:
                    continue
                # Check unit steps
                valid = True
                for k in range(n - 1):
                    if abs(test_chain[k][0] - test_chain[k + 1][0]) + abs(test_chain[k][1] - test_chain[k + 1][1]) != 1:
                        valid = False
                        break
                if not valid:
                    continue
                e = _compute_energy(sequence, test_chain)
                if e < best_insert_energy:
                    best_insert_energy = e
                    best_insert = test_chain
            if best_insert is not None:
                delta_e = best_insert_energy - current_energy
                if delta_e <= 0 or rng.random() < (2.718281828459045 ** (-delta_e / T)):
                    current_chain = best_insert
                    current_energy = best_insert_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)

        T *= cooling

    return best_chain


def fold(sequence, dimension):
    """Best conformation using beam search + multi-move local search."""
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)

    if n <= 1:
        return [[0]]

    best = None
    best_energy = 0

    # Phase 1: Beam search with varying widths
    beam_configs = [
        (8, 1),
        (15, 1),
        (25, 1),
        (15, 2),
        (10, 3),
    ]

    solutions = []
    for beam_width, num_runs in beam_configs:
        if _remaining() < 5.0:
            break
        for run in range(num_runs):
            if _remaining() < 3.0:
                break
            results = _beam_search(sequence, beam_width)
            for energy, chain in results:
                solutions.append((energy, chain))
                if energy < best_energy:
                    best_energy = energy
                    best = chain

    if best is None:
        best = [(i, 0) for i in range(n)]
        best_energy = _compute_energy(sequence, best)

    # Deduplicate and sort solutions
    seen = set()
    unique_solutions = []
    for energy, chain in solutions:
        key = tuple(chain)
        if key not in seen:
            seen.add(key)
            unique_solutions.append((energy, chain))
    unique_solutions.sort(key=lambda x: x[0])

    # Phase 2: Local search on top solutions
    top_k = min(8, len(unique_solutions))
    for k in range(top_k):
        if _remaining() < 2.0:
            break
        _, chain = unique_solutions[k]
        ls_time = min(5.0, max(0.5, _remaining() * 0.25 / max(1, top_k - k)))
        improved = _local_search(sequence, chain, rng, ls_time)
        imp_energy = _compute_energy(sequence, improved)
        if imp_energy < best_energy:
            best_energy = imp_energy
            best = improved

    # Final local search on best
    if _remaining() > 1.0:
        ls_time = min(4.0, _remaining() * 0.4)
        improved = _local_search(sequence, best, rng, ls_time)
        imp_energy = _compute_energy(sequence, improved)
        if imp_energy < best_energy:
            best_energy = imp_energy
            best = improved

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
