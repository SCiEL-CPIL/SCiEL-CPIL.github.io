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
_DEADLINE = _START_TIME + 285.0


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


def _beam_search(sequence, dimension, beam_width=150):
    """
    Beam search for HP protein folding with adaptive free-space penalty.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]

    start = (0, 0, 0)
    # State: (score, chain, occ_set, pos_dict, energy)
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
                
                # Adaptive penalty: penalize free space less as the chain grows,
                # but penalize it more if the chain is short to encourage compaction early.
                if n < 20:
                    free_penalty = 0.08
                elif n < 50:
                    free_penalty = 0.05
                else:
                    free_penalty = 0.03

                new_score = new_energy - free_penalty * free + random.random() * 0.02

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


def _local_search(sequence, chain, max_iterations=20000, polish_mode=False):
    """
    Local search combining segment-reversal moves, pivot-rotation moves, and single-step moves.
    Uses simulated annealing. In polish_mode, uses lower temperature and more single-step moves.
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

    rng = random.Random()

    # Adaptive temperature
    if polish_mode:
        T = 0.5
        T_min = 0.001
        alpha = 0.9999
    else:
        T = 2.0
        T_min = 0.005
        alpha = 0.9997 if max_iterations > 15000 else 0.9995

    no_improve_count = 0
    max_no_improve = 5000

    for it in range(max_iterations):
        if _time_left() < 5.0:
            break

        # Restart if stuck
        if no_improve_count > max_no_improve:
            chain = best_chain[:]
            occ = set(chain)
            pos_idx = {p: i for i, p in enumerate(chain)}
            current_energy = best_energy
            T = max(T * 2, 1.0)
            no_improve_count = 0

        move_type = rng.random()
        if polish_mode:
            # In polish mode, favor small moves
            if move_type < 0.4:
                move = 'single'
            elif move_type < 0.8:
                move = 'rev'
            else:
                move = 'pivot'
        else:
            if move_type < 0.55:
                move = 'rev'
            elif move_type < 0.95:
                move = 'pivot'
            else:
                move = 'single'

        accepted = False
        if move == 'rev':
            # ---- segment reversal ----
            if n < 20:
                max_seg = min(n - 2, 8)
            elif n < 50:
                max_seg = min(n - 2, 16)
            else:
                max_seg = min(n - 2, 24)
            
            if max_seg < 2:
                continue
            
            r = rng.random()
            if r < 0.5:
                seg_len = rng.randint(2, min(3, max_seg))
            elif r < 0.85:
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
                    accepted = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = chain[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    for k in range(seg_len):
                        a, p = orig_positions[k]
                        chain[a] = p
                        pos_idx[p] = a
                        no_improve_count += 1
            else:
                for k in range(seg_len):
                    a, p = orig_positions[k]
                    chain[a] = p
                    pos_idx[p] = a
                no_improve_count += 1

        elif move == 'pivot':
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
                no_improve_count += 1
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
                accepted = True
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                for k, p in enumerate(sub):
                    idx = pivot + 1 + k
                    occ.discard(new_sub[k])
                    occ.add(p)
                    pos_idx.pop(new_sub[k], None)
                    pos_idx[p] = idx
                    chain[idx] = p
                no_improve_count += 1

        else:
            # ---- single step move ----
            # Choose a random internal index
            idx = rng.randint(1, n - 2)
            p_old = chain[idx]
            # Try to move it to a neighbor
            # Find a valid neighbor
            valid_neighbors = []
            for dx, dy, dz in _OFFSETS:
                p_new = (p_old[0]+dx, p_old[1]+dy, p_old[2]+dz)
                # Must not be occupied (except by neighbors in chain, which are handled by validity check)
                # Actually, simpler: just try all 6, check if valid
                valid_neighbors.append(p_new)
            
            rng.shuffle(valid_neighbors)
            moved = False
            for p_new in valid_neighbors:
                # Temporarily apply
                chain[idx] = p_new
                # Check self-avoidance: p_new must not be in occ (except p_old which we are moving from)
                # But wait, p_old is still in occ. We need to check if p_new is in occ AND p_new != p_old
                # And we need to check if the move breaks the chain continuity (which it will if p_new is not adjacent to idx-1 and idx+1)
                
                # Check if p_new is occupied by a non-adjacent residue
                if p_new in occ and p_new != p_old:
                    # Check if it's a neighbor in the chain
                    # If idx-1 is at p_new or idx+1 is at p_new, it's a valid move (just swapping positions effectively? No, chain order matters)
                    # Actually, if p_new is occupied by idx-1 or idx+1, the chain would have a 0-length step or overlap.
                    # So p_new must NOT be in occ, OR if it is, it must be p_old (which we are leaving)
                    # Since p_old is in occ, and p_new != p_old, p_new must not be in occ.
                    continue
                
                # Check continuity
                prev_p = chain[idx-1]
                next_p = chain[idx+1]
                if abs(prev_p[0]-p_new[0]) + abs(prev_p[1]-p_new[1]) + abs(prev_p[2]-p_new[2]) != 1:
                    continue
                if abs(next_p[0]-p_new[0]) + abs(next_p[1]-p_new[1]) + abs(next_p[2]-p_new[2]) != 1:
                    continue
                
                # Valid move!
                # Update structures
                occ.discard(p_old)
                occ.add(p_new)
                pos_idx.pop(p_old, None)
                pos_idx[p_new] = idx
                
                # Calculate delta
                delta = 0
                if sequence[idx] == 'H':
                    for dx, dy, dz in _OFFSETS:
                        q = (p_new[0]+dx, p_new[1]+dy, p_new[2]+dz)
                        j = pos_idx.get(q)
                        if j is not None and abs(idx - j) > 1 and sequence[j] == 'H':
                            delta -= 1
                    # Remove old contacts
                    for dx, dy, dz in _OFFSETS:
                        q = (p_old[0]+dx, p_old[1]+dy, p_old[2]+dz)
                        j = pos_idx.get(q)
                        if j is not None and abs(idx - j) > 1 and sequence[j] == 'H':
                            delta += 1
                
                new_energy = current_energy + delta
                if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
                    current_energy = new_energy
                    accepted = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = chain[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    moved = True
                    break
                else:
                    # Revert
                    occ.discard(p_new)
                    occ.add(p_old)
                    pos_idx.pop(p_new, None)
                    pos_idx[p_old] = idx
                    chain[idx] = p_old
                    no_improve_count += 1
            
            if not moved:
                no_improve_count += 1

        T = max(T * alpha, T_min)

    return best_chain


def _basin_jump(sequence, chain, max_jumps=20):
    """
    Basin jumping: large perturbations followed by local search recovery.
    """
    n = len(chain)
    if n < 10:
        return chain

    best_chain = list(chain)
    best_energy = _energy_full(sequence, chain)
    rng = random.Random()

    for _ in range(max_jumps):
        if _time_left() < 15.0:
            break

        cand = [tuple(p) for p in chain]

        # Large random perturbation
        if n > 15 and rng.random() < 0.5:
            max_seg = min(n - 2, max(8, n // 4))
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
                    pivot = rng.randint(1, n - 2)
                    axis = rng.randint(0, 2)
                    angle = rng.choice([1, 2, 3])
                    sub = cand[pivot+1:]
                    new_sub = [_rot(p, axis, angle) for p in sub]
                    old_set = set(sub)
                    occ = set(cand)
                    new_set = set(new_sub)
                    ok = all((p not in occ or p in old_set) for p in new_set)
                    pp = cand[pivot]
                    f = new_sub[0]
                    if ok and abs(pp[0]-f[0]) + abs(pp[1]-f[1]) + abs(pp[2]-f[2]) == 1:
                        cand[pivot+1:] = new_sub

        if n > 15:
            for _ in range(rng.randint(1, 3)):
                pivot = rng.randint(1, n - 2)
                axis = rng.randint(0, 2)
                angle = rng.choice([1, 2, 3])
                sub = cand[pivot+1:]
                new_sub = [_rot(p, axis, angle) for p in sub]
                old_set = set(sub)
                occ = set(cand)
                new_set = set(new_sub)
                ok = all((p not in occ or p in old_set) for p in new_set)
                pp = cand[pivot]
                f = new_sub[0]
                if ok and abs(pp[0]-f[0]) + abs(pp[1]-f[1]) + abs(pp[2]-f[2]) == 1:
                    cand[pivot+1:] = new_sub

        improved = _local_search(sequence, cand, max_iterations=1000)
        improved_energy = _energy_full(sequence, improved)
        if improved_energy < best_energy:
            best_chain = improved
            best_energy = improved_energy

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

    seq_start = time.time()
    if n < 20:
        time_budget = 8.0
    elif n < 50:
        time_budget = 25.0
    elif n < 80:
        time_budget = 50.0
    else:
        time_budget = 90.0
    time_budget = min(time_budget, _time_left() - 10.0)

    # Phase 1: Beam Search
    beam_configs = [
        (150, 12345),
        (200, 54321),
        (250, 98765),
        (120, 11111),
        (300, 22222),
    ]

    for beam_width, seed in beam_configs:
        if time.time() - seq_start > time_budget * 0.30:
            break
        if _time_left() < 25.0:
            break

        random.seed(seed)
        chain = _beam_search(sequence, dimension, beam_width=beam_width)
        if chain is None or len(chain) != n:
            continue

        energy = _energy_full(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy

    # Phase 2: Local Search
    if best_chain is not None and _time_left() > 30.0:
        ls_configs = [
            (8000, 42, False),
            (10000, 123, False),
            (12000, 456, False),
            (15000, 789, False),
        ]
        
        for max_iter, seed, polish in ls_configs:
            if _time_left() < 20.0:
                break
            random.seed(seed)
            improved = _local_search(sequence, best_chain, max_iterations=max_iter, polish_mode=polish)
            improved_energy = _energy_full(sequence, improved)
            if improved_energy < best_energy:
                best_chain = improved
                best_energy = improved_energy

    # Phase 3: Basin Jumping
    if best_chain is not None and _time_left() > 35.0:
        jumps = 15 if n < 50 else 25
        tunneled = _basin_jump(sequence, best_chain, max_jumps=jumps)
        tunneled_energy = _energy_full(sequence, tunneled)
        if tunneled_energy < best_energy:
            best_chain = tunneled
            best_energy = tunneled_energy

    # Phase 4: High-Precision Polish
    if best_chain is not None and _time_left() > 25.0:
        polish_iters = 20000 if n < 50 else 30000
        polished = _local_search(sequence, best_chain, max_iterations=polish_iters, polish_mode=True)
        polished_energy = _energy_full(sequence, polished)
        if polished_energy < best_energy:
            best_chain = polished
            best_energy = polished_energy

    # Phase 5: Final Ultra-Fine Polish
    if best_chain is not None and _time_left() > 15.0:
        fine_iters = 15000 if n < 50 else 25000
        fine_polished = _local_search(sequence, best_chain, max_iterations=fine_iters, polish_mode=True)
        fine_energy = _energy_full(sequence, fine_polished)
        if fine_energy < best_energy:
            best_chain = fine_polished
            best_energy = fine_energy

    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
