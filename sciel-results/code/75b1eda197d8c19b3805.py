"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _neighbours_set(point):
    """Return set of neighbors for faster membership checks."""
    s = set()
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            s.add(tuple(q))
    return s


def _are_adjacent(a, b):
    """Check if two lattice points are adjacent (unit distance)."""
    return sum(abs(a[i] - b[i]) for i in range(len(a))) == 1


def _grow_compact(sequence, dimension, rng):
    """Growth with compactness bias: prefer sites that minimize bounding box volume."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            # Compute H-H contact gain
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            # Compactness: distance from centroid
            cx = sum(p[0] for p in chain) / len(chain)
            cy = sum(p[1] for p in chain) / len(chain)
            cz = sum(p[2] for p in chain) / len(chain)
            dist = abs(site[0] - cx) + abs(site[1] - cy) + abs(site[2] - cz)
            # Free space
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Score: prioritize H contacts, then compactness, then free space
            score = gain * 3.0 - dist * 0.5 + free * 0.1 + rng.random() * 0.3
            options.append((score, site))
        if not options:
            return None
        options.sort(reverse=True)
        k = min(3, len(options))
        site = options[rng.randint(0, k - 1)][1]
        occupied[site] = i
        chain.append(site)
    return chain


def _grow_greedy(sequence, dimension, rng):
    """Standard greedy growth."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 2.0 + 0.1 * free + rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        options.sort(reverse=True)
        k = min(3, len(options))
        site = options[rng.randint(0, k - 1)][1]
        occupied[site] = i
        chain.append(site)
    return chain


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _compute_2opt_delta(sequence, chain, occupied, i, j):
    """
    Compute energy delta for reversing segment [i..j] (both inclusive).
    Returns (delta, valid) where valid indicates the move is structurally valid.
    """
    n = len(chain)
    if i <= 0 or j >= n - 1 or j - i < 1:
        return 0, False

    # Check connectivity after reversal
    # Before: chain[i-1] -- chain[i] ... chain[j] -- chain[j+1]
    # After:  chain[i-1] -- chain[j] ... chain[i] -- chain[j+1]
    if not _are_adjacent(chain[i - 1], chain[j]):
        return 0, False
    if j + 1 < n and not _are_adjacent(chain[i], chain[j + 1]):
        return 0, False

    # The segment sites remain the same, just reordered.
    # Self-avoidance is preserved since same set of sites.
    # But we need to check that no two non-consecutive residues in the new
    # ordering become adjacent (they weren't before). Actually, since the
    # sites are the same and only the order changes, the adjacency between
    # non-consecutive chain positions doesn't change. The only adjacencies
    # that matter are between consecutive chain positions, which we already
    # checked above. So self-avoidance is automatically preserved.

    # Compute energy delta
    # Remove segment from occupied temporarily for delta computation
    old_segment = list(chain[i:j + 1])
    new_segment = list(reversed(old_segment))

    # Build temp occupied without the segment
    temp_occ = dict(occupied)
    for idx in range(i, j + 1):
        temp_occ[chain[idx]] = -1

    delta = 0
    for k in range(i, j + 1):
        old_site = old_segment[k - i]
        new_site = new_segment[k - i]
        if old_site == new_site:
            continue
        if sequence[k] == 'H':
            # Lost contacts at old_site
            for q in _neighbours(old_site):
                jj = temp_occ.get(q)
                if jj is not None and jj >= 0 and jj != k:
                    if sequence[jj] == 'H':
                        delta -= 1
            # Gained contacts at new_site
            for q in _neighbours(new_site):
                jj = temp_occ.get(q)
                if jj is not None and jj >= 0 and jj != k:
                    if sequence[jj] == 'H':
                        delta += 1

    return delta, True


def _multi_replica_sa(sequence, initial_chains, rng, deadline):
    """
    Multi-replica simulated annealing with 2-opt moves.
    Each replica runs independently; we track the global best.
    """
    n = len(sequence)
    if n < 3:
        return initial_chains[0], _energy(sequence, initial_chains[0])

    num_replicas = min(4, len(initial_chains))
    if num_replicas == 0:
        return initial_chains[0], _energy(sequence, initial_chains[0])

    # Global best
    global_best = None
    global_best_e = 0
    for ch in initial_chains[:num_replicas]:
        e = _energy(sequence, ch)
        if global_best is None or e < global_best_e:
            global_best = list(ch)
            global_best_e = e

    # Each replica state
    replicas = []
    for idx in range(num_replicas):
        ch = list(initial_chains[idx])
        e = _energy(sequence, ch)
        occ = {p: i for i, p in enumerate(ch)}
        replicas.append({
            'chain': ch,
            'energy': e,
            'occupied': occ,
            'best_chain': list(ch),
            'best_energy': e,
            'temp': 2.0 + 0.5 * idx,  # Different starting temps
            'no_improve': 0,
        })

    cooling_rate = 0.997
    min_temp = 0.005
    steps_per_check = max(1, n // 5)

    while time.time() < deadline:
        any_improved = False
        for rep in replicas:
            if time.time() >= deadline:
                break
            chain = rep['chain']
            occ = rep['occupied']
            temp = rep['temp']

            # Do several 2-opt moves per replica iteration
            for _ in range(steps_per_check):
                if time.time() >= deadline:
                    break
                # Random segment: pick random start and length
                # Bias toward shorter segments (more likely valid)
                seg_len = rng.randint(2, min(n // 2, 20))
                i = rng.randint(1, n - seg_len - 1)
                j = i + seg_len - 1

                delta, valid = _compute_2opt_delta(sequence, chain, occ, i, j)
                if not valid:
                    continue

                # Accept/reject
                accept = False
                if delta <= 0:
                    accept = True
                else:
                    if temp > 0.001:
                        accept = rng.random() < math.exp(-delta / temp)

                if accept:
                    # Apply reversal
                    old_segment = chain[i:j + 1]
                    new_segment = list(reversed(old_segment))
                    for k in range(i, j + 1):
                        old_site = old_segment[k - i]
                        new_site = new_segment[k - i]
                        if old_site != new_site:
                            occ[old_site] = -1
                            occ[new_site] = k
                        chain[k] = new_site

                    rep['energy'] += delta
                    if rep['energy'] < rep['best_energy']:
                        rep['best_energy'] = rep['energy']
                        rep['best_chain'] = list(chain)
                        rep['no_improve'] = 0
                        if rep['energy'] < global_best_e:
                            global_best_e = rep['energy']
                            global_best = list(chain)
                            any_improved = True
                    else:
                        rep['no_improve'] += 1
                else:
                    rep['no_improve'] += 1

            # Cool down
            rep['temp'] *= cooling_rate

            # Reheat if stuck
            if rep['no_improve'] > 200:
                rep['chain'] = list(rep['best_chain'])
                rep['energy'] = rep['best_energy']
                rep['occupied'] = {p: i for i, p in enumerate(rep['chain'])}
                rep['no_improve'] = 0
                rep['temp'] = max(rep['temp'] * 3, 1.0)

        # If no replica improved in this full pass, check if we should continue
        if not any_improved and all(r['temp'] < min_temp for r in replicas):
            # All replicas have cooled down; do one more pass with reheated temps
            for rep in replicas:
                if rep['temp'] < min_temp:
                    rep['temp'] = 1.0
            # If still no improvement after reheat, break
            if not any_improved:
                break

    return global_best, global_best_e


def fold(sequence, dimension):
    """Best conformation via compact growth + multi-replica 2-opt SA."""
    start_time = time.time()
    # 240s total for 26 sequences, so ~9.2s each. Use 8s per sequence for safety.
    total_deadline = start_time + 8.0

    rng = random.Random(12345 + len(sequence))
    initial_chains = []

    # Phase 1: Generate diverse initial chains with compactness bias
    num_grow = 6
    for i in range(num_grow):
        if time.time() > total_deadline - 3.0:
            break
        if i % 2 == 0:
            chain = _grow_compact(sequence, dimension, rng)
        else:
            chain = _grow_greedy(sequence, dimension, rng)
        if chain is not None:
            initial_chains.append(chain)

    if not initial_chains:
        # Fallback: straight line
        initial_chains.append([tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))])

    # Phase 2: Multi-replica SA with 2-opt moves
    sa_deadline = total_deadline - 0.5
    best_chain, best_energy = _multi_replica_sa(sequence, initial_chains, rng, sa_deadline)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
