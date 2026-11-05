import time
import random
import math

_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

_START_TIME = time.time()
_DEADLINE = _START_TIME + 298.0

def _time_left():
    return _DEADLINE - time.time()

def _energy_full(sequence, chain):
    """Compute energy: -number of H-H contacts on neighbouring non-consecutive sites."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] != 'H':
            continue
        for dx, dy, dz in _OFFSETS:
            q = (p[0] + dx, p[1] + dy, p[2] + dz)
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts

def _beam_search(sequence, dimension, beam_width=300):
    """
    Beam search for HP protein folding.
    Maintains a beam of the best partial conformations at each step.
    Score combines energy (lower better) and a free-space bonus so the
    chain does not trap itself; a little noise adds diversity.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]

    start = (0, 0, 0)
    beam = [(0.0, [start], {start}, {start: 0}, 0)]

    for i in range(1, n):
        if _time_left() < 8.0:
            break

        candidates = []
        seq_i = sequence[i]

        for score, chain, occ, pos_idx, energy in beam:
            last = chain[-1]
            lx, ly, lz = last

            for dx, dy, dz in _OFFSETS:
                site = (lx + dx, ly + dy, lz + dz)
                if site in occ:
                    continue

                # incremental energy gain from new H-H contacts
                gain = 0
                if seq_i == 'H':
                    for dx2, dy2, dz2 in _OFFSETS:
                        q = (site[0] + dx2, site[1] + dy2, site[2] + dz2)
                        j = pos_idx.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1

                # free neighbours of the new site (room to grow)
                free = 0
                for dx2, dy2, dz2 in _OFFSETS:
                    q = (site[0] + dx2, site[1] + dy2, site[2] + dz2)
                    if q not in occ:
                        free += 1

                new_energy = energy - gain
                new_score = new_energy - 0.05 * free + random.random() * 0.01

                new_chain = chain + [site]
                new_occ = set(occ)
                new_occ.add(site)
                new_pos = dict(pos_idx)
                new_pos[site] = i

                candidates.append((new_score, new_chain, new_occ, new_pos, new_energy))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0])
        beam = candidates[:beam_width]

    if not beam:
        return None
    best = min(beam, key=lambda x: x[0])
    return list(best[1])

def _rot(p, axis, angle):
    """Rotate point p about the given axis by 90/180/270 degrees (angle 1/2/3)."""
    x, y, z = p
    if axis == 0:  # x-axis
        if angle == 1:
            return (x, -z, y)
        if angle == 2:
            return (x, -y, -z)
        return (x, z, -y)
    if axis == 1:  # y-axis
        if angle == 1:
            return (z, y, -x)
        if angle == 2:
            return (-x, y, -z)
        return (-z, y, x)
    # z-axis
    if angle == 1:
        return (y, -x, z)
    if angle == 2:
        return (-x, -y, z)
    return (-y, x, z)

def _local_search(sequence, dimension, chain, rng, max_iterations=5000):
    """
    Local search combining segment-reversal moves and pivot-rotation moves,
    both using O(1) incremental energy deltas. Simulated-annealing acceptance.
    """
    n = len(chain)
    if n < 5:
        return chain

    chain = [tuple(p) for p in chain]
    occ = set(chain)
    pos_idx = {p: i for i, p in enumerate(chain)}

    current_energy = _energy_full(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy

    T = 1.5
    T_min = 0.004
    alpha = 0.9997

    for it in range(max_iterations):
        if _time_left() < 4.0:
            break

        if rng.random() < 0.5:
            # ---- segment reversal ----
            max_seg = min(n - 2, 14)
            if max_seg < 2:
                continue
            r = rng.random()
            if r < 0.5:
                seg_len = rng.randint(2, 4)
            elif r < 0.8:
                seg_len = rng.randint(4, 8)
            else:
                seg_len = rng.randint(8, max_seg)
            seg_len = min(seg_len, max_seg)
            if seg_len < 2:
                continue
            s = rng.randint(1, n - seg_len - 1)
            e = s + seg_len - 1

            # reverse the segment in-place and track moved positions
            moved = []
            for k in range(seg_len // 2):
                a = s + k
                b = e - k
                pa, pb = chain[a], chain[b]
                moved.append((a, pa))
                moved.append((b, pb))
                chain[a], chain[b] = pb, pa
                occ.discard(pa)
                occ.discard(pb)
                occ.add(pa)
                occ.add(pb)
                pos_idx[pa] = a
                pos_idx[pb] = b

            # check boundary unit-steps + self-avoidance of moved positions
            valid = True
            # boundary at s-1 <-> s (now holds old chain[e])
            if s > 0:
                p1 = chain[s - 1]
                p2 = chain[s]
                if abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2]) != 1:
                    valid = False
            # boundary at e <-> e+1 (now holds old chain[s])
            if e < n - 1:
                p1 = chain[e]
                p2 = chain[e + 1]
                if abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2]) != 1:
                    valid = False
            # self-avoidance: no two moved positions equal
            seen = set()
            for a, p in moved:
                if p in seen:
                    valid = False
                    break
                seen.add(p)

            if valid:
                delta = 0
                for a, _ in moved:
                    if sequence[a] == 'H':
                        p = chain[a]
                        for dx, dy, dz in _OFFSETS:
                            q = (p[0]+dx, p[1]+dy, p[2]+dz)
                            j = pos_idx.get(q)
                            if j is not None and abs(a - j) > 1 and sequence[j] == 'H':
                                delta -= 1
                new_energy = current_energy + delta
                if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = chain[:]
                else:
                    # revert
                    for a, p in moved:
                        chain[a] = p
                        pos_idx[p] = a
            else:
                # revert
                for a, p in moved:
                    chain[a] = p
                    pos_idx[p] = a
        else:
            # ---- pivot rotation ----
            pivot = rng.randint(1, n - 2)
            axis = rng.randint(0, 2)
            angle = rng.choice([1, 2, 3])

            subchain = chain[pivot+1:]
            if not subchain:
                continue

            new_subchain = [_rot(p, axis, angle) for p in subchain]

            # Check pivot connection validity
            pivot_pos = chain[pivot]
            first_new = new_subchain[0]
            if abs(pivot_pos[0] - first_new[0]) + abs(pivot_pos[1] - first_new[1]) + abs(pivot_pos[2] - first_new[2]) != 1:
                continue

            # Check self-avoidance: new positions must not conflict with existing (non-subchain) positions
            old_positions = set(subchain)
            new_positions = set(new_subchain)

            conflict = False
            for p in new_positions:
                if p in occ and p not in old_positions:
                    conflict = True
                    break

            if conflict:
                continue

            # Incremental energy calculation
            affected_indices = set(range(pivot+1, n))
            for idx in range(pivot+1, n):
                x, y, z = new_subchain[idx - pivot - 1]
                for dx, dy, dz in _OFFSETS:
                    nb = (x+dx, y+dy, z+dz)
                    if nb in pos_idx:
                        j = pos_idx[nb]
                        if j <= pivot:
                            affected_indices.add(j)

            new_occ = set(chain[:pivot+1]) | new_positions
            new_pos_idx = dict(pos_idx)
            for i, p in enumerate(new_subchain):
                new_pos_idx[p] = pivot + 1 + i

            old_contacts = 0
            for i in affected_indices:
                if sequence[i] != 'H':
                    continue
                x, y, z = chain[i]
                for dx, dy, dz in _OFFSETS:
                    nb = (x+dx, y+dy, z+dz)
                    if nb in occ:
                        j = pos_idx[nb]
                        if abs(i-j) > 1 and sequence[j] == 'H' and j > i:
                            old_contacts += 1

            new_contacts = 0
            for i in affected_indices:
                if sequence[i] != 'H':
                    continue
                if i > pivot:
                    x, y, z = new_subchain[i - pivot - 1]
                else:
                    x, y, z = chain[i]
                for dx, dy, dz in _OFFSETS:
                    nb = (x+dx, y+dy, z+dz)
                    if nb in new_occ:
                        j = new_pos_idx[nb]
                        if abs(i-j) > 1 and sequence[j] == 'H' and j > i:
                            new_contacts += 1

            delta_e = -(new_contacts - old_contacts)
            new_energy = current_energy + delta_e

            # Acceptance criterion
            if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-9)):
                # Apply move
                for p in old_positions:
                    occ.remove(p)
                    del pos_idx[p]

                for i, p in enumerate(new_subchain):
                    occ.add(p)
                    pos_idx[p] = pivot + 1 + i

                chain[pivot+1:] = new_subchain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
            else:
                pass

        T = max(T * alpha, T_min)

    return best_chain

def _quantum_tunneling_search(chain, sequence, dimension, time_budget=None):
    """
    Quantum tunneling inspired search: allow moves that temporarily increase energy
    to escape local minima, with a simulated annealing-like acceptance criterion.
    Uses incremental energy updates for speed.
    """
    n = len(chain)
    if n < 5:
        return chain

    chain = [tuple(p) for p in chain]

    occ = set(chain)
    pos_to_idx = {p: i for i, p in enumerate(chain)}

    current_energy = _energy_full(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy

    # Rotation functions
    def rotate_point(p, axis, angle):
        return _rot(p, axis, angle)

    it = 0
    max_iters = 50000
    no_improve_count = 0
    patience = 4000

    # Simulated annealing parameters
    T_start = 5.0
    T_end = 0.1
    cooling_rate = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start

    while it < max_iters:
        if _time_left() < 4.0:
            break
        if time_budget is not None and time.time() > time_budget:
            break

        it += 1
        T *= cooling_rate

        # Pick a random pivot point (not at ends)
        pivot = random.randint(1, n - 2)

        # Pick rotation axis and angle
        axis = random.randint(0, 2)
        angle = random.choice([1, 2, 3])

        # Extract subchain
        subchain = chain[pivot+1:]
        if not subchain:
            continue

        # Apply rotation to subchain
        new_subchain = [rotate_point(p, axis, angle) for p in subchain]

        # Check pivot connection validity
        pivot_pos = chain[pivot]
        first_new = new_subchain[0]
        if abs(pivot_pos[0] - first_new[0]) + abs(pivot_pos[1] - first_new[1]) + abs(pivot_pos[2] - first_new[2]) != 1:
            continue

        # Check self-avoidance: new positions must not conflict with existing (non-subchain) positions
        old_positions = set(subchain)
        new_positions = set(new_subchain)

        conflict = False
        for p in new_positions:
            if p in occ and p not in old_positions:
                conflict = True
                break

        if conflict:
            continue

        # Incremental energy calculation
        affected_indices = set(range(pivot+1, n))
        for idx in range(pivot+1, n):
            x, y, z = new_subchain[idx - pivot - 1]
            for dx, dy, dz in _OFFSETS:
                nb = (x+dx, y+dy, z+dz)
                if nb in pos_to_idx:
                    j = pos_to_idx[nb]
                    if j <= pivot:
                        affected_indices.add(j)

        new_occ = set(chain[:pivot+1]) | new_positions
        new_pos_idx = dict(pos_to_idx)
        for i, p in enumerate(new_subchain):
            new_pos_idx[p] = pivot + 1 + i

        old_contacts = 0
        for i in affected_indices:
            if sequence[i] != 'H':
                continue
            x, y, z = chain[i]
            for dx, dy, dz in _OFFSETS:
                nb = (x+dx, y+dy, z+dz)
                if nb in occ:
                    j = pos_to_idx[nb]
                    if abs(i-j) > 1 and sequence[j] == 'H' and j > i:
                        old_contacts += 1

        new_contacts = 0
        for i in affected_indices:
            if sequence[i] != 'H':
                continue
            if i > pivot:
                x, y, z = new_subchain[i - pivot - 1]
            else:
                x, y, z = chain[i]
            for dx, dy, dz in _OFFSETS:
                nb = (x+dx, y+dy, z+dz)
                if nb in new_occ:
                    j = new_pos_idx[nb]
                    if abs(i-j) > 1 and sequence[j] == 'H' and j > i:
                        new_contacts += 1

        delta_e = -(new_contacts - old_contacts)
        new_energy = current_energy + delta_e

        # Acceptance criterion
        if delta_e < 0 or random.random() < pow(2.718281828, -delta_e / T):
            # Apply move
            for p in old_positions:
                occ.remove(p)
                del pos_to_idx[p]

            for i, p in enumerate(new_subchain):
                occ.add(p)
                pos_to_idx[p] = pivot + 1 + i

            chain[pivot+1:] = new_subchain
            current_energy = new_energy
            no_improve_count = 0
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = chain[:]
        else:
            no_improve_count += 1

        # Early stopping if no improvement for a while
        if no_improve_count > patience:
            break

    return best_chain

def _polish_search(chain, sequence, dimension, time_budget=None):
    """
    Final high-precision polish pass with strict improvement-only acceptance
    and short-range moves for fine-tuning.
    """
    n = len(chain)
    if n < 5:
        return chain

    chain = [tuple(p) for p in chain]

    occ = set(chain)
    pos_to_idx = {p: i for i, p in enumerate(chain)}

    current_energy = _energy_full(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy

    it = 0
    max_iters = 20000
    no_improve_count = 0
    patience = 2000

    while it < max_iters:
        if _time_left() < 2.0:
            break
        if time_budget is not None and time.time() > time_budget:
            break

        it += 1

        # Prefer pivots closer to the end for short-range moves
        # This targets local refinements
        pivot = random.randint(max(1, n - 20), n - 2)

        # Pick rotation axis and angle
        axis = random.randint(0, 2)
        angle = random.choice([1, 2, 3])

        # Extract subchain
        subchain = chain[pivot+1:]
        if not subchain:
            continue

        # Apply rotation to subchain
        new_subchain = [_rot(p, axis, angle) for p in subchain]

        # Check pivot connection validity
        pivot_pos = chain[pivot]
        first_new = new_subchain[0]
        if abs(pivot_pos[0] - first_new[0]) + abs(pivot_pos[1] - first_new[1]) + abs(pivot_pos[2] - first_new[2]) != 1:
            continue

        # Check self-avoidance
        old_positions = set(subchain)
        new_positions = set(new_subchain)

        conflict = False
        for p in new_positions:
            if p in occ and p not in old_positions:
                conflict = True
                break

        if conflict:
            continue

        # Compute new energy
        new_energy = _energy_full(chain[:pivot+1] + new_subchain, sequence)

        if new_energy < current_energy:
            # Apply move
            for p in old_positions:
                occ.remove(p)
                del pos_to_idx[p]

            for i, p in enumerate(new_subchain):
                occ.add(p)
                pos_to_idx[p] = pivot + 1 + i

            chain[pivot+1:] = new_subchain
            current_energy = new_energy
            no_improve_count = 0
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = chain[:]
        else:
            no_improve_count += 1

        if no_improve_count > patience:
            break

    return best_chain

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []

    rng = random.Random(hash(sequence) & 0xFFFFFFFF)

    # Allocate time: 35% beam search, 25% local search, 25% quantum tunneling, 15% polish
    total_budget = 298.0
    beam_time = total_budget * 0.35
    local_search_time = total_budget * 0.25
    quantum_time = total_budget * 0.25
    polish_time = total_budget * 0.15

    beam_deadline = time.time() + beam_time
    local_search_deadline = time.time() + beam_time + local_search_time
    quantum_deadline = time.time() + beam_time + local_search_time + quantum_time
    polish_deadline = time.time() + total_budget

    # Override deadline for beam search phase
    global _DEADLINE
    old_deadline = _DEADLINE
    _DEADLINE = beam_deadline

    chain = _beam_search(sequence, dimension, beam_width=300)

    if chain is None:
        # Fallback: simple greedy walk
        chain = [(0, 0, 0)]
        occ = {chain[0]}
        for i in range(1, n):
            last = chain[-1]
            found = False
            for dx, dy, dz in _OFFSETS:
                np = (last[0]+dx, last[1]+dy, last[2]+dz)
                if np not in occ:
                    chain.append(np)
                    occ.add(np)
                    found = True
                    break
            if not found:
                break
        if len(chain) < n:
            return [list(p) for p in chain]

    # Local search phase
    _DEADLINE = local_search_deadline
    refined_chain = _local_search(sequence, dimension, chain, rng, max_iterations=5000)

    # Quantum tunneling phase
    _DEADLINE = quantum_deadline
    qt_chain = _quantum_tunneling_search(refined_chain, sequence, dimension, time_budget=quantum_deadline)

    # Polish phase - high-precision final refinement
    _DEADLINE = polish_deadline
    final_chain = _polish_search(qt_chain, sequence, dimension, time_budget=polish_deadline)

    # Restore deadline
    _DEADLINE = old_deadline

    return [list(p) for p in final_chain]
