"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import time
import random
import math

# Precompute neighbor offsets for 3D cubic lattice
_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0


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


def _beam_search(sequence, dimension, beam_width=200, seed=42):
    """
    Beam search for HP protein folding with adaptive scoring.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]

    rng = random.Random(seed)
    start = (0, 0, 0)
    beam = [(0.0, [start], {start}, {start: 0}, 0)]

    for i in range(1, n):
        if _time_left() < 10.0:
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
                
                # Adaptive free-space penalty
                if n < 25:
                    free_penalty = 0.02
                elif n < 50:
                    free_penalty = 0.04
                elif n < 100:
                    free_penalty = 0.06
                else:
                    free_penalty = 0.08

                # Add slight bias toward compactness for longer chains
                compactness_bonus = 0.0 if i < 10 else (0.005 * i if n > 50 else 0.0)
                
                new_score = new_energy - free_penalty * free + compactness_bonus + rng.random() * 0.03

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


def _local_search(sequence, chain, max_iterations=15000, seed=42):
    """
    Local search combining segment-reversal moves and pivot-rotation moves.
    Uses simulated annealing with adaptive temperature.
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

    rng = random.Random(seed)

    # Adaptive temperature schedule
    T = 2.5
    T_min = 0.003
    # Slower cooling for longer searches
    alpha = 0.9997 if max_iterations > 12000 else 0.9994

    # Track recent improvements for adaptive move selection
    recent_improvements = 0

    for it in range(max_iterations):
        if _time_left() < 5.0:
            break

        # Dynamically adjust move probabilities based on recent performance
        if recent_improvements > 10:
            # We're in a good basin, be more conservative
            reversal_prob = 0.60
        elif recent_improvements < 2:
            # Struggling, be more exploratory
            reversal_prob = 0.45
        else:
            reversal_prob = 0.52

        if rng.random() < reversal_prob:
            # ---- segment reversal ----
            if n < 20:
                max_seg = min(n - 2, 10)
            elif n < 50:
                max_seg = min(n - 2, 18)
            elif n < 100:
                max_seg = min(n - 2, 25)
            else:
                max_seg = min(n - 2, 30)
            
            if max_seg < 2:
                continue
            
            # Weighted random segment length - favor shorter segments
            r = rng.random()
            if r < 0.4:
                seg_len = rng.randint(2, min(3, max_seg))
            elif r < 0.75:
                seg_len = rng.randint(3, min(8, max_seg))
            else:
                seg_len = rng.randint(min(8, max_seg), max_seg)
            seg_len = min(seg_len, max_seg)
            if seg_len < 2:
                continue
            
            s = rng.randint(1, n - seg_len - 1)
            e = s + seg_len - 1

            orig_positions = [(s + k, chain[s + k]) for k in range(seg_len)]
            
            for k in range(seg_len // 2):
                a = s + k
                b = e - k
                pa, pb = chain[a], chain[b]
                chain[a], chain[b] = pb, pa
                pos_idx[pa] = a
                pos_idx[pb] = b

            valid = True
            if s > 0:
                p1 = chain[s - 1]
                p2 = chain[s]
                if abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2]) != 1:
                    valid = False
            if e < n - 1:
                p1 = chain[e]
                p2 = chain[e + 1]
                if abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2]) != 1:
                    valid = False
            
            if valid:
                seen = set()
                for k in range(seg_len):
                    p = chain[s + k]
                    if p in seen:
                        valid = False
                        break
                    seen.add(p)

            if valid:
                delta = 0
                for k in range(seg_len):
                    a = s + k
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
                        recent_improvements += 1
                    else:
                        recent_improvements = max(0, recent_improvements - 1)
                else:
                    for k in range(seg_len):
                        a, p = orig_positions[k]
                        chain[a] = p
                        pos_idx[p] = a
                    recent_improvements = max(0, recent_improvements - 1)
            else:
                for k in range(seg_len):
                    a, p = orig_positions[k]
                    chain[a] = p
                    pos_idx[p] = a
                recent_improvements = max(0, recent_improvements - 1)
        else:
            # ---- pivot rotation of subchain [pivot+1:] ----
            pivot = rng.randint(1, max(2, n // 2))
            axis = rng.randint(0, 2)
            angle = rng.choice([1, 2, 3])

            sub = chain[pivot+1:]
            if not sub:
                continue
            new_sub = [_rot(p, axis, angle) for p in sub]

            old_set = set(sub)
            new_set = set(new_sub)
            conflict = False
            for p in new_set:
                if p in occ and p not in old_set:
                    conflict = True
                    break
            pp = chain[pivot]
            f = new_sub[0]
            if abs(pp[0]-f[0]) + abs(pp[1]-f[1]) + abs(pp[2]-f[2]) != 1:
                conflict = True

            if conflict:
                continue

            for p in old_set:
                occ.discard(p)
                pos_idx.pop(p, None)
            for k, p in enumerate(new_sub):
                idx = pivot + 1 + k
                occ.add(p)
                pos_idx[p] = idx
                chain[idx] = p

            delta = 0
            for k, p in enumerate(new_sub):
                idx = pivot + 1 + k
                if sequence[idx] == 'H':
                    for dx, dy, dz in _OFFSETS:
                        q = (p[0]+dx, p[1]+dy, p[2]+dz)
                        j = pos_idx.get(q)
                        if j is not None and abs(idx - j) > 1 and sequence[j] == 'H':
                            delta -= 1
            new_energy = current_energy + delta
            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
                    recent_improvements += 1
                else:
                    recent_improvements = max(0, recent_improvements - 1)
            else:
                for k, p in enumerate(sub):
                    idx = pivot + 1 + k
                    occ.discard(new_sub[k])
                    occ.add(p)
                    pos_idx[new_sub[k]] = idx
                    del pos_idx[new_sub[k]]
                    pos_idx[p] = idx
                    chain[idx] = p
                recent_improvements = max(0, recent_improvements - 1)

        T = max(T * alpha, T_min)

    return best_chain


def _basin_jump(sequence, chain, max_jumps=20, seed=42):
    """
    Basin jumping: large perturbations followed by local search recovery.
    """
    n = len(chain)
    if n < 10:
        return chain

    best_chain = list(chain)
    best_energy = _energy_full(sequence, chain)
    rng = random.Random(seed)

    for jump_idx in range(max_jumps):
        if _time_left() < 15.0:
            break

        cand = [tuple(p) for p in chain]
        occ = set(cand)

        # Large random perturbation
        if n > 15 and rng.random() < 0.5:
            max_seg = min(n - 2, max(10, n // 4))
            seg_len = rng.randint(max(4, n // 6), max_seg)
            s = rng.randint(1, n - seg_len - 1)
            e = s + seg_len - 1
            cand[s:e+1] = cand[s:e+1][::-1]
            if len(set(cand)) == n:
                valid = True
                if s > 0:
                    p1 = cand[s-1]
                    p2 = cand[s]
                    if abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2]) != 1:
                        valid = False
                if e < n - 1:
                    p1 = cand[e]
                    p2 = cand[e+1]
                    if abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2]) != 1:
                        valid = False
                if not valid:
                    cand[s:e+1] = cand[s:e+1][::-1]

        # Additional pivot rotations for larger jumps
        num_pivots = rng.randint(1, 3) if n > 20 else 1
        for _ in range(num_pivots):
            pivot = rng.randint(1, n - 2)
            axis = rng.randint(0, 2)
            angle = rng.choice([1, 2, 3])
            sub = cand[pivot+1:]
            new_sub = [_rot(p, axis, angle) for p in sub]
            old_set = set(sub)
            new_set = set(new_sub)
            ok = all((p not in occ or p in old_set) for p in new_set)
            pp = cand[pivot]
            f = new_sub[0]
            if ok and abs(pp[0]-f[0]) + abs(pp[1]-f[1]) + abs(pp[2]-f[2]) == 1:
                cand[pivot+1:] = new_sub

        # Short local search
        improved = _local_search(sequence, cand, max_iterations=800, seed=seed + jump_idx)
        improved_energy = _energy_full(sequence, improved)
        if improved_energy < best_energy:
            best_chain = improved
            best_energy = improved_energy

    return best_chain


def _refined_local_search(sequence, chain, max_iterations=25000, seed=12345):
    """
    High-precision final polish with tighter tolerance and more iterations.
    Uses a slower cooling schedule for better convergence.
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

    rng = random.Random(seed)

    # Very slow cooling for final polish
    T = 1.0
    T_min = 0.001
    alpha = 0.9998

    for it in range(max_iterations):
        if _time_left() < 5.0:
            break

        # Focus on small moves in final polish
        if rng.random() < 0.7:
            # Small segment reversal (length 2-6)
            if n < 15:
                seg_len = rng.randint(2, min(4, n - 2))
            else:
                seg_len = rng.randint(2, min(6, n - 2))
            if seg_len < 2 or n - seg_len - 1 < 1:
                continue
            
            s = rng.randint(1, n - seg_len - 1)
            e = s + seg_len - 1

            orig_positions = [(s + k, chain[s + k]) for k in range(seg_len)]
            
            for k in range(seg_len // 2):
                a = s + k
                b = e - k
                pa, pb = chain[a], chain[b]
                chain[a], chain[b] = pb, pa
                pos_idx[pa] = a
                pos_idx[pb] = b

            valid = True
            if s > 0:
                p1 = chain[s - 1]
                p2 = chain[s]
                if abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2]) != 1:
                    valid = False
            if e < n - 1:
                p1 = chain[e]
                p2 = chain[e + 1]
                if abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2]) != 1:
                    valid = False
            
            if valid:
                seen = set()
                for k in range(seg_len):
                    p = chain[s + k]
                    if p in seen:
                        valid = False
                        break
                    seen.add(p)

            if valid:
                delta = 0
                for k in range(seg_len):
                    a = s + k
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
                    for k in range(seg_len):
                        a, p = orig_positions[k]
                        chain[a] = p
                        pos_idx[p] = a
            else:
                for k in range(seg_len):
                    a, p = orig_positions[k]
                    chain[a] = p
                    pos_idx[p] = a
        else:
            # Small pivot rotation
            pivot = rng.randint(1, min(5, n - 2))
            axis = rng.randint(0, 2)
            angle = rng.choice([1, 2, 3])

            sub = chain[pivot+1:]
            if not sub:
                continue
            new_sub = [_rot(p, axis, angle) for p in sub]

            old_set = set(sub)
            new_set = set(new_sub)
            conflict = False
            for p in new_set:
                if p in occ and p not in old_set:
                    conflict = True
                    break
            pp = chain[pivot]
            f = new_sub[0]
            if abs(pp[0]-f[0]) + abs(pp[1]-f[1]) + abs(pp[2]-f[2]) != 1:
                conflict = True

            if conflict:
                continue

            for p in old_set:
                occ.discard(p)
                pos_idx.pop(p, None)
            for k, p in enumerate(new_sub):
                idx = pivot + 1 + k
                occ.add(p)
                pos_idx[p] = idx
                chain[idx] = p

            delta = 0
            for k, p in enumerate(new_sub):
                idx = pivot + 1 + k
                if sequence[idx] == 'H':
                    for dx, dy, dz in _OFFSETS:
                        q = (p[0]+dx, p[1]+dy, p[2]+dz)
                        j = pos_idx.get(q)
                        if j is not None and abs(idx - j) > 1 and sequence[j] == 'H':
                            delta -= 1
            new_energy = current_energy + delta
            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
            else:
                for k, p in enumerate(sub):
                    idx = pivot + 1 + k
                    occ.discard(new_sub[k])
                    occ.add(p)
                    pos_idx[new_sub[k]] = idx
                    del pos_idx[new_sub[k]]
                    pos_idx[p] = idx
                    chain[idx] = p

        T = max(T * alpha, T_min)

    return best_chain


def fold(sequence, dimension):
    """
    Fold a protein sequence on the cubic lattice.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]

    best_chain = None
    best_energy = 0

    # Time budget allocation
    seq_start = time.time()
    if n < 20:
        time_budget = 10.0
    elif n < 50:
        time_budget = 30.0
    elif n < 80:
        time_budget = 55.0
    elif n < 120:
        time_budget = 75.0
    else:
        time_budget = 95.0
    time_budget = min(time_budget, _time_left() - 15.0)

    # Phase 1: Multiple beam searches with different seeds and widths for diversity
    beam_configs = [
        (120, 111),
        (150, 222),
        (180, 333),
        (200, 444),
        (250, 555),
        (300, 666),
        (160, 777),
        (220, 888),
        (140, 999),
        (280, 101),
    ]

    beam_time_alloc = time_budget * 0.35
    per_beam = beam_time_alloc / len(beam_configs)

    for beam_width, seed in beam_configs:
        if time.time() - seq_start > time_budget * 0.40:
            break
        if _time_left() < 25.0:
            break

        chain = _beam_search(sequence, dimension, beam_width=beam_width, seed=seed)
        if chain is None or len(chain) != n:
            continue

        energy = _energy_full(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy

    # Phase 2: Local search refinement with multiple restarts from best
    if best_chain is not None and _time_left() > 35.0:
        ls_configs = [
            (10000, 11),
            (12000, 22),
            (14000, 33),
            (16000, 44),
            (12000, 55),
            (14000, 66),
            (15000, 77),
        ]
        
        ls_time_alloc = time_budget * 0.35
        for max_iter, seed in ls_configs:
            if _time_left() < 20.0:
                break
            improved = _local_search(sequence, best_chain, max_iterations=max_iter, seed=seed)
            improved_energy = _energy_full(sequence, improved)
            if improved_energy < best_energy:
                best_chain = improved
                best_energy = improved_energy

    # Phase 3: Basin jumping with more diverse perturbations
    if best_chain is not None and _time_left() > 40.0:
        jumps = 12 if n < 50 else (20 if n < 100 else 28)
        tunneled = _basin_jump(sequence, best_chain, max_jumps=jumps, seed=7777)
        tunneled_energy = _energy_full(sequence, tunneled)
        if tunneled_energy < best_energy:
            best_chain = tunneled
            best_energy = tunneled_energy

    # Phase 4: High-precision final polish
    if best_chain is not None and _time_left() > 30.0:
        polish_iters = 20000 if n < 50 else (25000 if n < 100 else 30000)
        polished = _refined_local_search(sequence, best_chain, max_iterations=polish_iters, seed=9999)
        polished_energy = _energy_full(sequence, polished)
        if polished_energy < best_energy:
            best_chain = polished
            best_energy = polished_energy

    # Phase 5: Ultra-fine polish if time remains
    if best_chain is not None and _time_left() > 20.0:
        fine_iters = 15000 if n < 50 else (20000 if n < 100 else 25000)
        fine_polished = _refined_local_search(sequence, best_chain, max_iterations=fine_iters, seed=8888)
        fine_energy = _energy_full(sequence, fine_polished)
        if fine_energy < best_energy:
            best_chain = fine_polished
            best_energy = fine_energy

    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = _energy_full(sequence, best_chain)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
