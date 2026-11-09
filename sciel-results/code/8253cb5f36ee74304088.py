"""HP lattice protein folding: diverse growth + constraint-satisfaction backtracking + simulated annealing + polish."""
# EVOLVE-BLOCK-START
import random
import time
from math import exp

_START_TIME = time.time()
_DEADLINE = 195.0  # Safety margin before 210s kill


def _remaining_time():
    return max(0.0, _DEADLINE - (time.time() - _START_TIME))


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute HP energy: -number of H-H contacts (non-adjacent in chain)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _energy_delta(sequence, chain, idx, old_site, new_site, where):
    """Compute energy delta for moving residue idx from old_site to new_site."""
    delta = 0
    if sequence[idx] == 'H':
        for q in _neighbours(old_site):
            j = where.get(q)
            if j is not None and j != idx:
                if abs(j - idx) > 1 and sequence[j] == 'H':
                    delta += 1
    if sequence[idx] == 'H':
        for q in _neighbours(new_site):
            j = where.get(q)
            if j is not None and j != idx:
                if abs(j - idx) > 1 and sequence[j] == 'H':
                    delta -= 1
    return delta


def _grow(sequence, dimension, rng, bias=1.0):
    """One randomized greedy chain growth."""
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
            score = gain * bias + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _backtrack_fold(sequence, max_time=8.0, seed=42, node_budget=200000):
    """
    Constraint-satisfaction backtracking with contact-density heuristic.
    Places residues left-to-right; for H residues, prefers sites with many
    existing H neighbours (potential contacts). Uses randomized DFS with a
    node budget and time guard. Returns a chain or None.
    """
    n = len(sequence)
    if n < 3:
        return [(0, 0), (1, 0)] if n == 2 else [(0, 0)]

    rng = random.Random(seed)
    start = time.time()
    nodes = 0

    # Precompute for each position the list of H positions after it (for lookahead)
    # We'll use a simple heuristic: for an H at position i, count H residues in
    # positions i+2..n-1 that could plausibly be neighbours.

    def dfs(pos, chain, occupied):
        nonlocal nodes
        if pos >= n:
            return list(chain)
        if nodes > node_budget:
            return None
        if time.time() - start > max_time:
            return None

        last = chain[-1]
        residue = sequence[pos]

        # Candidate sites: neighbours of last, not occupied
        candidates = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Score the site
            if residue == 'H':
                # Count existing H neighbours (potential contacts)
                h_neighbours = 0
                for q in _neighbours(site):
                    if q in occupied:
                        idx = occupied[q]
                        if idx < pos - 1 and sequence[idx] == 'H':
                            h_neighbours += 1
                # Also count how many H residues remain that could fill neighbours
                # (simple: number of H in sequence[pos+2:])
                score = h_neighbours * 3.0
            else:
                # P residue: prefer sites with more free space (avoid trapping)
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                score = 0.1 * free
            # Add small random tiebreaker
            score += rng.random() * 0.5
            candidates.append((score, site))

        if not candidates:
            return None

        # Sort by score descending, then shuffle ties for diversity
        candidates.sort(key=lambda x: -x[0])
        # For diversity, occasionally pick a random candidate instead of best
        if rng.random() < 0.2:
            site = rng.choice(candidates)[1]
        else:
            site = candidates[0][1]

        chain.append(site)
        occupied[site] = pos
        nodes += 1

        result = dfs(pos + 1, chain, occupied)
        chain.pop()
        del occupied[site]

        if result is not None:
            return result
        return None

    # Try multiple seeds for diversity
    for attempt in range(5):
        if time.time() - start > max_time:
            break
        result = dfs(0, [(0, 0)], {(0, 0): 0})
        if result is not None:
            return result
    return None


def _local_move(sequence, chain, where, current_energy, T, rng):
    """Attempt a single-residue move (pivot)."""
    n = len(chain)
    if n <= 2:
        return chain, where, current_energy, False

    idx = rng.randint(1, n - 2)
    old_site = chain[idx]
    prev_site = chain[idx - 1]
    next_site = chain[idx + 1]

    candidates = []
    for site in _neighbours(old_site):
        if site == old_site:
            continue
        if abs(site[0] - prev_site[0]) + abs(site[1] - prev_site[1]) != 1:
            continue
        if abs(site[0] - next_site[0]) + abs(site[1] - next_site[1]) != 1:
            continue
        if site in where:
            occ_idx = where[site]
            if occ_idx != idx:
                continue
        candidates.append(site)

    if not candidates:
        return chain, where, current_energy, False

    new_site = rng.choice(candidates)
    if new_site == old_site:
        return chain, where, current_energy, False

    delta_e = _energy_delta(sequence, chain, idx, old_site, new_site, where)

    if delta_e <= 0 or rng.random() < exp(-delta_e / T):
        del where[old_site]
        where[new_site] = idx
        chain[idx] = new_site
        return chain, where, current_energy + delta_e, True
    else:
        return chain, where, current_energy, False


def _reverse_move(sequence, chain, where, current_energy, T, rng):
    """Attempt a segment reversal."""
    n = len(chain)
    if n < 4:
        return chain, where, current_energy, False

    i = rng.randint(0, n - 3)
    j = rng.randint(i + 2, n - 1)

    if i > 0:
        if abs(chain[i - 1][0] - chain[j][0]) + abs(chain[i - 1][1] - chain[j][1]) != 1:
            return chain, where, current_energy, False
    if j < n - 1:
        if abs(chain[i][0] - chain[j + 1][0]) + abs(chain[i][1] - chain[j + 1][1]) != 1:
            return chain, where, current_energy, False

    old_segment = list(chain[i:j + 1])
    new_segment = old_segment[::-1]

    new_where = dict(where)
    for pt in old_segment:
        del new_where[pt]

    for pt in new_segment:
        if pt in new_where:
            return chain, where, current_energy, False

    chain[i:j + 1] = new_segment
    for k in range(i, j + 1):
        where[chain[k]] = k

    new_energy = _energy(sequence, chain)
    delta_e = new_energy - current_energy

    if delta_e <= 0 or rng.random() < exp(-delta_e / T):
        return chain, where, new_energy, True
    else:
        chain[i:j + 1] = old_segment
        for k in range(i, j + 1):
            where[chain[k]] = k
        return chain, where, current_energy, False


def _rotate_move(sequence, chain, where, current_energy, T, rng):
    """Attempt a 90-degree rotation of a segment."""
    n = len(chain)
    if n < 4:
        return chain, where, current_energy, False

    i = rng.randint(0, n - 3)
    j = rng.randint(i + 2, n - 1)

    pivot = chain[i]
    seg = chain[i:j + 1]

    rotated_seg = [pivot]
    for pt in seg[1:]:
        dx = pt[0] - pivot[0]
        dy = pt[1] - pivot[1]
        new_pt = (pivot[0] + dy, pivot[1] - dx)
        rotated_seg.append(new_pt)

    new_where = dict(where)
    for pt in seg[1:]:
        if pt in new_where:
            del new_where[pt]

    for pt in rotated_seg[1:]:
        if pt in new_where:
            return chain, where, current_energy, False

    if j < n - 1:
        if abs(rotated_seg[-1][0] - chain[j + 1][0]) + abs(rotated_seg[-1][1] - chain[j + 1][1]) != 1:
            return chain, where, current_energy, False

    chain[i:j + 1] = rotated_seg
    for k in range(i, j + 1):
        where[chain[k]] = k

    new_energy = _energy(sequence, chain)
    delta_e = new_energy - current_energy

    if delta_e <= 0 or rng.random() < exp(-delta_e / T):
        return chain, where, new_energy, True
    else:
        chain[i:j + 1] = seg
        for k in range(i, j + 1):
            where[chain[k]] = k
        return chain, where, current_energy, False


def _anneal(sequence, dimension, chain, max_time=10.0, seed=12345):
    """Simulated annealing with local moves, reversals, and rotations."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    T = 3.0
    T_min = 0.005
    cooling = 0.999
    rng = random.Random(seed)
    no_improve_count = 0
    max_no_improve = 5000

    start = time.time()
    while T > T_min and (time.time() - start) < max_time:
        r = rng.random()
        if r < 0.45:
            chain, where, current_energy, accepted = _local_move(sequence, chain, where, current_energy, T, rng)
        elif r < 0.75:
            chain, where, current_energy, accepted = _reverse_move(sequence, chain, where, current_energy, T, rng)
        else:
            chain, where, current_energy, accepted = _rotate_move(sequence, chain, where, current_energy, T, rng)

        if accepted:
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        if no_improve_count > max_no_improve:
            T *= 0.95
            no_improve_count = 0

        T *= cooling

    return best_chain, best_energy


def _polish(sequence, dimension, chain, max_time=5.0, seed=99999):
    """High-precision polish pass: only accept moves that improve energy."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    rng = random.Random(seed)
    no_improve_count = 0
    max_no_improve = 3000

    start = time.time()
    while (time.time() - start) < max_time:
        r = rng.random()
        if r < 0.6:
            idx = rng.randint(1, n - 2)
            old_site = chain[idx]
            prev_site = chain[idx - 1]
            next_site = chain[idx + 1]

            candidates = []
            for site in _neighbours(old_site):
                if site == old_site:
                    continue
                if abs(site[0] - prev_site[0]) + abs(site[1] - prev_site[1]) != 1:
                    continue
                if abs(site[0] - next_site[0]) + abs(site[1] - next_site[1]) != 1:
                    continue
                if site in where:
                    occ_idx = where[site]
                    if occ_idx != idx:
                        continue
                candidates.append(site)

            if candidates:
                new_site = rng.choice(candidates)
                if new_site != old_site:
                    delta_e = _energy_delta(sequence, chain, idx, old_site, new_site, where)
                    if delta_e <= 0 or (delta_e == 1 and rng.random() < 0.05):
                        del where[old_site]
                        where[new_site] = idx
                        chain[idx] = new_site
                        current_energy += delta_e
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(chain)
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        elif r < 0.85:
            if n >= 4:
                i = rng.randint(0, n - 3)
                j = rng.randint(i + 2, min(i + 6, n - 1))

                if i > 0:
                    if abs(chain[i - 1][0] - chain[j][0]) + abs(chain[i - 1][1] - chain[j][1]) != 1:
                        no_improve_count += 1
                        continue
                if j < n - 1:
                    if abs(chain[i][0] - chain[j + 1][0]) + abs(chain[i][1] - chain[j + 1][1]) != 1:
                        no_improve_count += 1
                        continue

                old_segment = list(chain[i:j + 1])
                new_segment = old_segment[::-1]

                new_where = dict(where)
                for pt in old_segment:
                    del new_where[pt]

                valid = True
                for pt in new_segment:
                    if pt in new_where:
                        valid = False
                        break

                if valid:
                    chain[i:j + 1] = new_segment
                    for k in range(i, j + 1):
                        where[chain[k]] = k

                    new_energy = _energy(sequence, chain)
                    delta_e = new_energy - current_energy

                    if delta_e <= 0:
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(chain)
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                    else:
                        chain[i:j + 1] = old_segment
                        for k in range(i, j + 1):
                            where[chain[k]] = k
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        else:
            if n >= 4:
                i = rng.randint(0, n - 3)
                j = rng.randint(i + 2, min(i + 5, n - 1))

                pivot = chain[i]
                seg = list(chain[i:j + 1])

                rotated_seg = [pivot]
                for pt in seg[1:]:
                    dx = pt[0] - pivot[0]
                    dy = pt[1] - pivot[1]
                    new_pt = (pivot[0] + dy, pivot[1] - dx)
                    rotated_seg.append(new_pt)

                new_where = dict(where)
                for pt in seg[1:]:
                    if pt in new_where:
                        del new_where[pt]

                valid = True
                for pt in rotated_seg[1:]:
                    if pt in new_where:
                        valid = False
                        break

                if valid and j < n - 1:
                    if abs(rotated_seg[-1][0] - chain[j + 1][0]) + abs(rotated_seg[-1][1] - chain[j + 1][1]) != 1:
                        no_improve_count += 1
                        continue

                    chain[i:j + 1] = rotated_seg
                    for k in range(i, j + 1):
                        where[chain[k]] = k

                    new_energy = _energy(sequence, chain)
                    delta_e = new_energy - current_energy

                    if delta_e <= 0:
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(chain)
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                    else:
                        chain[i:j + 1] = seg
                        for k in range(i, j + 1):
                            where[chain[k]] = k
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1

        if no_improve_count > max_no_improve:
            break

    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold a sequence using growth, backtracking, annealing, and polish."""
    n = len(sequence)
    rng = random.Random(12345 + n)
    best, best_energy = None, 1

    # Phase 0: Constraint-satisfaction backtracking (new)
    # Only run if sequence length is manageable and we have time
    if n <= 60 and _remaining_time() > 30:
        bt_time = min(6.0, _remaining_time() * 0.15)
        bt_chain = _backtrack_fold(sequence, max_time=bt_time, seed=42 + n)
        if bt_chain is not None:
            initial_chains = [bt_chain]
        else:
            initial_chains = []
    else:
        initial_chains = []

    # Phase 1: Generate diverse initial conformations via greedy growth
    growth_budget = 400
    biases = [1.0, 1.5, 2.0, 0.5, 0.8, 1.2, 1.8, 0.6, 1.4, 0.9]
    for trial in range(growth_budget):
        if _remaining_time() < 20:
            break
        bias = biases[trial % len(biases)]
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is not None:
            initial_chains.append(chain)

    if not initial_chains:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in best]

    # Phase 2: Anneal each initial conformation
    time_per_anneal = max(0.3, _remaining_time() / max(1, len(initial_chains)) * 0.8)
    initial_chains.sort(key=lambda c: _energy(sequence, c))

    for idx, chain in enumerate(initial_chains):
        if _remaining_time() < 10:
            break
        if idx < 3:
            t = time_per_anneal * 2.0
        elif idx < 10:
            t = time_per_anneal * 1.5
        else:
            t = time_per_anneal
        t = min(t, _remaining_time() - 5)
        if t <= 0:
            break
        annealed_chain, annealed_energy = _anneal(sequence, dimension, chain, max_time=t, seed=12345 + idx)
        if annealed_energy < best_energy:
            best, best_energy = annealed_chain, annealed_energy

    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    # Phase 3: Final polish
    if _remaining_time() > 5 and best is not None:
        polish_time = min(8.0, _remaining_time() - 3)
        best, best_energy = _polish(sequence, dimension, best, max_time=polish_time, seed=99999)

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
