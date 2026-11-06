"""HP lattice protein folding: beam search with deadline guard."""
import time
import heapq

_DEADLINE = 330.0
_MARGIN = 10.0  # safety margin before the hard kill

# Precomputed unit step offsets for 3D cubic lattice
_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]


def _key(p):
    return (p[0], p[1], p[2])


def _energy(chain, seq, where):
    """Compute H-H contact energy (negative). where maps position -> index."""
    contacts = 0
    n = len(chain)
    for i in range(n):
        if seq[i] != 'H':
            continue
        p = chain[i]
        for dx, dy, dz in _OFFSETS:
            q = (p[0] + dx, p[1] + dy, p[2] + dz)
            j = where.get(q)
            if j is not None and j > i + 1 and seq[j] == 'H':
                contacts += 1
    return -contacts


def _beam_fold(seq, dimension, beam_width, time_limit):
    """
    Beam search for HP chain folding.
    Returns (best_chain, best_energy) or (None, None).
    """
    n = len(seq)
    if n == 0:
        return None, None
    if n == 1:
        return [(0, 0, 0)], 0

    is_h = [c == 'H' for c in seq]
    start = (0, 0, 0)

    # Each beam entry: (score, chain_tuple, occupied_set, where_dict, energy)
    # score = energy + potential (lower is better, we use min-heap)
    # Actually we want max score; use negative for min-heap.

    start_chain = (start,)
    start_occ = {start}
    start_where = {start: 0}
    start_energy = 0

    # Potential: for each H in chain, count H-neighbours among occupied
    # (this is the same as current energy for a full chain, but for partial
    # chains we also add a bonus for H residues that have H neighbours
    # which could form contacts later)
    # Simplified: score = energy (current contacts) + 0.5 * (number of H-H adjacencies
    # that are NOT chain-adjacent but ARE lattice-adjacent) -- actually energy already
    # counts those. Let's use: score = energy + small bonus for compactness.
    # Better: score = energy (more negative = better). Use -score for min-heap.

    # For partial chains, energy only counts completed contacts.
    # Add potential: for each H residue, count how many H neighbours it has
    # in occupied set (excluding chain-adjacent). This equals current energy contribution.
    # To guide search, add a bonus for H residues that are "exposed" (have free H-neighbour
    # slots) -- but that's complex. Keep it simple: just use energy + a small
    # compactness bonus based on bounding box volume.

    def _score(energy, chain):
        # Compactness bonus: smaller bounding box = better (more negative score)
        if len(chain) < 2:
            return energy
        xs = [p[0] for p in chain]
        ys = [p[1] for p in chain]
        zs = [p[2] for p in chain]
        vol = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1) * (max(zs) - min(zs) + 1)
        # Scale: for n residues, ideal vol ~ n^(2/3) or so
        # Penalty proportional to vol / n^2 to encourage compactness
        return energy - 0.01 * vol / max(len(chain), 1)

    # Initialize beam
    beam = [(-_score(0, start_chain), start_chain, start_occ, start_where, 0)]
    # beam entries: (neg_score, chain_tuple, occupied, where, energy)

    best_chain = None
    best_energy = 0

    for step in range(1, n):
        new_beam = []
        deadline_check = time.time()
        if deadline_check > time_limit - _MARGIN:
            break

        for neg_sc, chain, occ, where, energy in beam:
            last = chain[-1]
            for dx, dy, dz in _OFFSETS:
                np_ = (last[0] + dx, last[1] + dy, last[2] + dz)
                if np_ in occ:
                    continue

                new_occ = occ | {np_}
                new_where = dict(where)
                new_where[np_] = step
                new_chain = chain + (np_,)

                # Compute new energy incrementally
                new_energy = energy
                if is_h[step]:
                    for edx, edy, edz in _OFFSETS:
                        q = (np_[0] + edx, np_[1] + edy, np_[2] + edz)
                        j = where.get(q)
                        if j is not None and j < step - 1 and is_h[j]:
                            new_energy -= 1
                        elif j is not None and j == step - 1:
                            # chain-adjacent, not a contact
                            pass

                sc = _score(new_energy, new_chain)
                new_beam.append((-sc, new_chain, new_occ, new_where, new_energy))

            # Periodic deadline check
            if time.time() > time_limit - _MARGIN:
                break

        if not new_beam:
            break

        # Sort and keep top beam_width
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]

        # Track best complete chain (only when step == n-1)
        if step == n - 1:
            for neg_sc, chain, occ, where, energy in beam:
                if best_chain is None or energy < best_energy:
                    best_chain = chain
                    best_energy = energy

    # If we didn't reach the end, try to complete the best partial chain
    if best_chain is None and beam:
        # Take the best partial chain and try to extend greedily
        for neg_sc, chain, occ, where, energy in beam[:5]:
            if len(chain) >= n:
                if best_chain is None or energy < best_energy:
                    best_chain = chain
                    best_energy = energy
            else:
                # Greedy extension
                try:
                    ext_chain, ext_energy = _greedy_complete(chain, occ, where, energy, seq, is_h, n)
                    if ext_chain is not None:
                        if best_chain is None or ext_energy < best_energy:
                            best_chain = ext_chain
                            best_energy = ext_energy
                except:
                    pass

    if best_chain is None:
        # Fallback: straight line
        best_chain = tuple((i, 0, 0) for i in range(n))
        best_energy = _energy(list(best_chain), seq, {p: i for i, p in enumerate(best_chain)})

    return list(best_chain), best_energy


def _greedy_complete(chain, occ, where, energy, seq, is_h, n):
    """Greedily extend a partial chain to full length."""
    chain = list(chain)
    occ = set(occ)
    where = dict(where)
    energy = energy

    for step in range(len(chain), n):
        last = chain[-1]
        best_site = None
        best_gain = -999
        for dx, dy, dz in _OFFSETS:
            np_ = (last[0] + dx, last[1] + dy, last[2] + dz)
            if np_ in occ:
                continue
            gain = 0
            if is_h[step]:
                for edx, edy, edz in _OFFSETS:
                    q = (np_[0] + edx, np_[1] + edy, np_[2] + edz)
                    j = where.get(q)
                    if j is not None and j < step - 1 and is_h[j]:
                        gain += 1
            if gain > best_gain:
                best_gain = gain
                best_site = np_
        if best_site is None:
            return None, None
        chain.append(best_site)
        occ.add(best_site)
        where[best_site] = step
        energy += -best_gain  # gain is number of new contacts, energy decreases

    return chain, energy


def fold(sequence, dimension):
    """
    Fold an HP sequence on the simple cubic lattice using beam search.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]

    t0 = time.time()
    total_budget = _DEADLINE - _MARGIN  # ~320s
    time_per_seq = total_budget / 26  # approximate, but we don't know how many calls

    # Adaptive beam width based on sequence length and available time
    # Shorter sequences: can afford wider beams
    if n <= 50:
        base_beam = 100
    elif n <= 100:
        base_beam = 80
    else:
        base_beam = 60

    # We'll do multiple passes with increasing beam width if time allows
    best_chain = None
    best_energy = 0

    # First pass: moderate beam
    chain, energy = _beam_fold(sequence, dimension, base_beam, t0 + total_budget * 0.4)
    if chain is not None:
        best_chain = chain
        best_energy = energy

    # Check remaining time
    elapsed = time.time() - t0
    remaining = total_budget - elapsed

    if remaining > 5.0:
        # Second pass: wider beam
        wider = int(base_beam * 1.5)
        chain2, energy2 = _beam_fold(sequence, dimension, wider, t0 + total_budget * 0.75)
        if chain2 is not None and energy2 < best_energy:
            best_chain = chain2
            best_energy = energy2

    elapsed = time.time() - t0
    remaining = total_budget - elapsed

    if remaining > 5.0:
        # Third pass: even wider
        widest = int(base_beam * 2.0)
        chain3, energy3 = _beam_fold(sequence, dimension, widest, t0 + total_budget)
        if chain3 is not None and energy3 < best_energy:
            best_chain = chain3
            best_energy = energy3

    if best_chain is None:
        # Fallback: straight line
        best_chain = [[i] + [0] * (dimension - 1) for i in range(n)]

    # Ensure output format: list of lists of integers
    result = []
    for p in best_chain:
        if isinstance(p, tuple):
            result.append(list(p))
        elif isinstance(p, list):
            result.append(list(p))
        else:
            result.append([p] + [0] * (dimension - 1))

    return result
