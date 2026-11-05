import time
import random
import math
from collections import deque

# Precompute neighbor offsets for 3D cubic lattice
_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # Safety margin before 330s kill


def _time_left():
    return _DEADLINE - time.time()


def _energy_correct(sequence, chain):
    """Compute energy: -number of H-H contacts on neighboring non-consecutive sites."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] != 'H':
            continue
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            q = (p[0] + dx, p[1] + dy, p[2] + dz)
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _beam_search(sequence, dimension, beam_width=200):
    """
    Beam search for HP protein folding.
    Maintains a beam of the best partial conformations at each step.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]

    beam = [(0, ((0, 0, 0),), {(0, 0, 0): 0})]

    for i in range(1, n):
        if _time_left() < 10.0:
            break

        new_beam = []
        seen = set()

        for energy_so_far, chain_tuple, occupied in beam:
            last = chain_tuple[-1]
            seq_i = sequence[i]

            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                site = (last[0] + dx, last[1] + dy, last[2] + dz)
                if site in occupied:
                    continue

                gain = 0
                if seq_i == 'H':
                    for dx2, dy2, dz2 in _NEIGHBOR_OFFSETS:
                        q = (site[0] + dx2, site[1] + dy2, site[2] + dz2)
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1

                new_energy = energy_so_far - gain

                # Use last 4 positions for symmetry breaking deduplication
                key = (site, chain_tuple[-4:] if len(chain_tuple) >= 4 else chain_tuple)
                if key in seen:
                    continue
                seen.add(key)

                new_chain = chain_tuple + (site,)
                new_occupied = dict(occupied)
                new_occupied[site] = i

                new_beam.append((new_energy, new_chain, new_occupied))

        if not new_beam:
            break

        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]

    if not beam:
        return None

    best_energy, best_chain_tuple, _ = beam[0]
    return list(best_chain_tuple)


def _fast_local_search(sequence, dimension, chain, rng, max_iterations=50000):
    """
    Optimized local search using incremental energy updates for segment reversals.
    """
    n = len(chain)
    if n < 5:
        return chain

    current_chain = [tuple(p) for p in chain]
    current_energy = _energy_correct(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    T = 2.0
    T_min = 0.005
    alpha = 0.9999

    for iteration in range(max_iterations):
        if _time_left() < 5.0:
            break

        # Choose segment length
        r = rng.random()
        if r < 0.6:
            seg_len = rng.randint(2, 5)
        elif r < 0.9:
            seg_len = rng.randint(5, 10)
        else:
            seg_len = rng.randint(10, min(20, n - 2))

        seg_len = min(seg_len, n - 2)
        if seg_len < 2:
            continue

        start_idx = rng.randint(1, n - seg_len - 1)
        end_idx = start_idx + seg_len - 1

        # 1. Check Connectivity (O(1))
        p_left = current_chain[start_idx - 1]
        p_right = current_chain[end_idx + 1]
        p_new_start = current_chain[end_idx]
        p_new_end = current_chain[start_idx]

        if sum(abs(a - b) for a, b in zip(p_left, p_new_start)) != 1:
            continue
        if sum(abs(a - b) for a, b in zip(p_new_end, p_right)) != 1:
            continue

        # 2. Calculate Energy Delta
        pos_to_idx = {p: i for i, p in enumerate(current_chain)}

        old_delta = 0
        new_delta = 0

        for k in range(start_idx, end_idx + 1):
            pos_k = current_chain[k]
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                neighbor_pos = (pos_k[0] + dx, pos_k[1] + dy, pos_k[2] + dz)
                j = pos_to_idx.get(neighbor_pos)
                if j is None:
                    continue

                if start_idx <= j <= end_idx:
                    if j <= k:
                        continue
                    if sequence[k] == 'H' and sequence[j] == 'H' and abs(k - j) > 1:
                        old_delta += 1
                    new_k = start_idx + end_idx - k
                    new_j = start_idx + end_idx - j
                    if sequence[new_k] == 'H' and sequence[new_j] == 'H' and abs(new_k - new_j) > 1:
                        new_delta += 1
                else:
                    if sequence[k] == 'H' and sequence[j] == 'H' and abs(k - j) > 1:
                        old_delta += 1
                    new_k = start_idx + end_idx - k
                    if sequence[new_k] == 'H' and sequence[j] == 'H' and abs(new_k - j) > 1:
                        new_delta += 1

        delta_e = new_delta - old_delta

        # 3. Acceptance
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            # Perform reversal
            current_chain[start_idx:end_idx + 1] = current_chain[start_idx:end_idx + 1][::-1]
            current_energy += delta_e
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)

        T = max(T_min, T * alpha)

    return best_chain


def _greedy_growth(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts."""
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}

    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            site = (last[0] + dx, last[1] + dy, last[2] + dz)
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for dx2, dy2, dz2 in _NEIGHBOR_OFFSETS:
                    q = (site[0] + dx2, site[1] + dy2, site[2] + dz2)
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            score = gain + rng.random() * 0.5
            options.append((score, site))

        if not options:
            return None

        options.sort(reverse=True)
        site = options[0][1]
        occupied[site] = i
        chain.append(site)

    return chain


def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}

    for i in range(1, len(sequence)):
        options = []
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            site = (chain[-1][0] + dx, chain[-1][1] + dy, chain[-1][2] + dz)
            if site not in occupied:
                options.append(site)
        if not options:
            return None
        site = rng.choice(options)
        occupied[site] = i
        chain.append(site)

    return chain


def fold(sequence, dimension):
    """
    Best conformation using Multi-Start Simulated Annealing with Tail Regeneration moves.
    """
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))

    # Global deadline
    start_time = time.time()
    deadline = start_time + 310.0

    best_chain = None
    best_energy = 0

    # Phase 1: Generate initial population
    init_chains = []

    # Try Beam Search first for a strong initial guess
    if n <= 60:
        bc = _beam_search(sequence, dimension, beam_width=100)
        if bc:
            init_chains.append(bc)

    # Greedy
    for _ in range(10):
        c = _greedy_growth(sequence, dimension, rng)
        if c:
            init_chains.append(c)

    # Random
    for _ in range(5):
        c = _random_walk(sequence, dimension, rng)
        if c:
            init_chains.append(c)

    if not init_chains:
        return [list([i] + [0] * (dimension - 1)) for i in range(n)]

    # Phase 2: Optimize each chain
    num_chains = len(init_chains)
    # Allocate time: 80% to local search, 20% to beam search if not used or for re-optimization?
    # Let's just distribute local search time.
    time_budget = deadline - time.time()
    time_per_chain = time_budget / num_chains

    for c in init_chains:
        if time.time() > deadline:
            break

        # Run local search
        optimized = _fast_local_search(sequence, dimension, c, rng)
        if optimized:
            e = _energy_correct(sequence, optimized)
            if e < best_energy:
                best_energy = e
                best_chain = optimized

    if best_chain is None:
        best_chain = init_chains[0]

    return [list(p) for p in best_chain]
