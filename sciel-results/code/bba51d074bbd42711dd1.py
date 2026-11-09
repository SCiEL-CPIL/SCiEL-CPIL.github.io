"""Rewritten program: multi-restart tail-regrowth SA with incremental energy,
better initialization, and a global deadline guard."""

import random
import time
import math
import sys

# Global deadline: 330 s wall-clock budget for all sequences.
_GLOBAL_DEADLINE = 330.0
_start_wall = time.time()


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - (time.time() - _start_wall))


def _energy(sequence, chain):
    """Calculate total energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    h_positions = [i for i, c in enumerate(sequence) if c == 'H']
    contacts = 0
    for i in h_positions:
        p = chain[i]
        x, y, z = p
        for q in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
            j = where.get(q)
            if j is not None and j > i + 1:
                if sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _grow_greedy(sequence, rng, bias=1.0):
    """
    Generate a self-avoiding walk with a greedy bias towards H-H contacts and space.
    """
    n = len(sequence)
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    h_set = set(i for i, c in enumerate(sequence) if c == 'H')

    for i in range(1, n):
        last = chain[-1]
        options = []
        x, y, z = last
        for site in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
            if site in occupied:
                continue

            score = 0.0
            if i in h_set:
                sx, sy, sz = site
                for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and j in h_set:
                            score += 3.0

            sx, sy, sz = site
            empty_neighbors = 0
            for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                if q not in occupied and q != last:
                    empty_neighbors += 1
            score += empty_neighbors * 0.8

            score += rng.random() * (1.0 - bias) * 2.0
            options.append((score, site))

        if not options:
            return None

        options.sort(reverse=True)
        if bias > 0.7:
            site = options[0][1]
        else:
            top_k = min(4, len(options))
            site = options[rng.randint(0, top_k-1)][1]

        chain.append(site)
        occupied[site] = i

    return chain


def _tail_regrowth(sequence, chain, occupied, k, rng, temperature=1.0):
    """
    Regrow the tail from index k to end.
    Returns new_chain, new_occupied, or (None, None) if stuck.
    """
    n = len(chain)
    if k >= n:
        return chain, occupied

    new_occupied = dict(occupied)
    for i in range(k, n):
        del new_occupied[chain[i]]

    new_chain = chain[:k]
    h_set = set(i for i, c in enumerate(sequence) if c == 'H')

    for i in range(k, n):
        last = new_chain[-1]
        options = []
        x, y, z = last
        for site in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
            if site in new_occupied:
                continue

            score = 0.0
            if i in h_set:
                sx, sy, sz = site
                for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                    if q in new_occupied:
                        j = new_occupied[q]
                        if j < i - 1 and j in h_set:
                            score += 3.0

            sx, sy, sz = site
            empty_neighbors = 0
            for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                if q not in new_occupied and q != last:
                    empty_neighbors += 1
            score += empty_neighbors * 0.8

            score += rng.random() * temperature
            options.append((score, site))

        if not options:
            return None, None

        if temperature < 0.05:
            options.sort(reverse=True)
            site = options[0][1]
        else:
            scores = [s for s, _ in options]
            max_score = max(scores)
            exp_scores = [math.exp((s - max_score) / temperature) for s in scores]
            total = sum(exp_scores)
            r = rng.random() * total
            cumulative = 0.0
            site = options[0][1]
            for idx, (_, s) in enumerate(options):
                cumulative += exp_scores[idx]
                if r <= cumulative:
                    site = s
                    break

        new_chain.append(site)
        new_occupied[site] = i

    return new_chain, new_occupied


def _sa_fold(sequence, rng, time_limit):
    """
    Simulated Annealing with variable-length tail regrowth.
    """
    n = len(sequence)
    if n < 2:
        return [(i, 0, 0) for i in range(n)]

    # Multiple greedy initializations with different biases
    best_init = None
    best_init_energy = float('inf')
    biases = [0.3, 0.6, 0.9, 1.0]
    for b in biases:
        init_chain = _grow_greedy(sequence, rng, bias=b)
        if init_chain is None:
            continue
        e = _energy(sequence, init_chain)
        if e < best_init_energy:
            best_init_energy = e
            best_init = init_chain

    if best_init is None:
        best_init = [(i, 0, 0) for i in range(n)]
        best_init_energy = _energy(sequence, best_init)

    current_chain = best_init
    current_occupied = {p: i for i, p in enumerate(current_chain)}
    current_energy = best_init_energy

    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    T = 3.0
    T_min = 0.01
    cooling_rate = 0.997

    stagnation_counter = 0
    max_stagnation = 800

    while T > T_min:
        if time.time() - start_time > time_limit:
            break

        r = rng.random()
        if r < 0.5:
            tail_len = rng.randint(1, 12)
        elif r < 0.85:
            tail_len = rng.randint(12, min(35, n-1))
        else:
            tail_len = rng.randint(35, max(35, n//2))

        tail_len = max(1, min(n - 1, tail_len))
        k = n - tail_len

        new_chain, new_occupied = _tail_regrowth(sequence, current_chain, current_occupied, k, rng, T)

        if new_chain is None:
            T *= cooling_rate
            continue

        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy

        if delta_e < 0 or (T > 0 and rng.random() < math.exp(-delta_e / T)):
            current_chain = new_chain
            current_occupied = new_occupied
            current_energy = new_energy

            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max_stagnation:
                T = max(T, 1.5)
                stagnation_counter = 0
        else:
            stagnation_counter += 1

        T *= cooling_rate

    return best_chain


def fold(sequence, dimension):
    """
    Best conformation using multiple SA restarts with a global deadline.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]

    remaining = _remaining_time()
    if remaining < 2.0:
        # Not enough time; just return a greedy fold
        rng = random.Random(hash(sequence) % (2**32))
        chain = _grow_greedy(sequence, rng, bias=1.0)
        if chain is None:
            chain = [(i, 0, 0) for i in range(n)]
        return [list(p) for p in chain]

    # Allocate time for this sequence, leaving margin for other sequences
    # We don't know how many sequences remain, so use a fraction of remaining time
    # but cap it to avoid using all time on one sequence.
    # Use min(remaining * 0.4, 12.0) seconds per sequence as a heuristic.
    seq_time_limit = min(remaining * 0.4, 12.0)

    if seq_time_limit < 1.0:
        rng = random.Random(hash(sequence) % (2**32))
        chain = _grow_greedy(sequence, rng, bias=1.0)
        if chain is None:
            chain = [(i, 0, 0) for i in range(n)]
        return [list(p) for p in chain]

    best_chain = None
    best_energy = float('inf')

    # Number of restarts based on available time
    slice_time = max(0.8, seq_time_limit / 5.0)
    num_restarts = max(1, int(seq_time_limit / slice_time))

    rng = random.Random(hash(sequence) % (2**32))

    for restart in range(num_restarts):
        if _remaining_time() < 3.0:
            break
        sa_chain = _sa_fold(sequence, rng, slice_time)
        if sa_chain is not None:
            e = _energy(sequence, sa_chain)
            if e < best_energy:
                best_energy = e
                best_chain = sa_chain

    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]

    return [list(p) for p in best_chain]
