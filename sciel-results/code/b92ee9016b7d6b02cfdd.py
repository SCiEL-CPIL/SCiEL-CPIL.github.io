"""HP lattice protein folding: simulated annealing with multiple move types."""
# EVOLVE-BLOCK-START
import random
import time
import math
import sys

_START = time.time()
DEADLINE = 195.0  # safety margin before 210s kill


def _remaining():
    return DEADLINE - time.time()


def _neighbours(point):
    d = len(point)
    for axis in range(d):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _initial_chain(sequence, dimension, rng):
    """Greedy self-avoiding walk with H-contact bias."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        best_site = None
        best_score = -1e18
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            score = rng.random()
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j <= i - 2 and sequence[j] == 'H':
                        score += 10.0
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += 0.5 * free
            if score > best_score:
                best_score = score
                best_site = site
        if best_site is None:
            return None
        occupied[best_site] = i
        chain.append(best_site)
    return chain


def _energy_from_chain(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _try_move(chain, move_type, rng):
    """Attempt a move; return new chain or None if invalid."""
    n = len(chain)
    if n < 2:
        return None

    if move_type == 0:  # pivot: rotate a suffix
        if n < 3:
            return None
        k = rng.randint(1, n - 2)  # pivot at position k, rotate suffix k+1..n-1
        anchor = chain[k]
        suffix = chain[k + 1:]
        # try 3 rotations (90, 180, 270 degrees)
        for rot in range(1, 4):
            new_suffix = _rotate_suffix(suffix, anchor, rot)
            if new_suffix is None:
                continue
            # check self-avoidance: new suffix positions must not collide with prefix
            prefix_set = set(chain[:k + 1])
            valid = True
            for p in new_suffix:
                if p in prefix_set:
                    valid = False
                    break
            if valid:
                return chain[:k + 1] + new_suffix
        return None

    elif move_type == 1:  # crankshaft: rotate a segment of length 3 or 4
        if n < 5:
            return None
        seg_len = rng.choice([3, 4])
        if n < seg_len + 2:
            return None
        start = rng.randint(1, n - seg_len - 1)
        end = start + seg_len - 1
        anchor_left = chain[start - 1]
        anchor_right = chain[end + 1]
        segment = chain[start:end + 1]
        # rotate segment around anchor_left
        for rot in range(1, 4):
            new_segment = _rotate_around(segment, anchor_left, rot)
            if new_segment is None:
                continue
            # check: new segment must not collide with prefix (except anchor_left) and suffix (except anchor_right)
            prefix_set = set(chain[:start])
            suffix_set = set(chain[end + 1:])
            valid = True
            for idx, p in enumerate(new_segment):
                # first position is adjacent to anchor_left, last adjacent to anchor_right
                if idx == 0:
                    if p in prefix_set:
                        valid = False
                        break
                elif idx == seg_len - 1:
                    if p in suffix_set:
                        valid = False
                        break
                else:
                    if p in prefix_set or p in suffix_set:
                        valid = False
                        break
            if valid:
                # verify connectivity: last of new_segment must be adjacent to anchor_right
                last = new_segment[-1]
                if last[0] != anchor_right[0] or last[1] != anchor_right[1]:
                    # check unit step
                    if abs(last[0] - anchor_right[0]) + abs(last[1] - anchor_right[1]) != 1:
                        valid = False
            if valid:
                return chain[:start] + new_segment + chain[end + 1:]
        return None

    elif move_type == 2:  # single-step move: move one internal residue to adjacent free site
        if n < 3:
            return None
        i = rng.randint(1, n - 2)
        old = chain[i]
        for new in _neighbours(old):
            if new == chain[i - 1] or new == chain[i + 1]:
                continue
            # check not occupied by any other residue
            occupied = set(chain)
            occupied.discard(old)
            if new in occupied:
                continue
            # check that moving i to new keeps i-1 to i and i to i+1 as unit steps
            if abs(new[0] - chain[i - 1][0]) + abs(new[1] - chain[i - 1][1]) != 1:
                continue
            if abs(new[0] - chain[i + 1][0]) + abs(new[1] - chain[i + 1][1]) != 1:
                continue
            new_chain = chain[:]
            new_chain[i] = new
            return new_chain
        return None

    return None


def _rotate_suffix(suffix, anchor, rot):
    """Rotate suffix around anchor by rot*90 degrees. Return new suffix or None."""
    # anchor is the point before the suffix
    # rotate each point in suffix around anchor
    new_suffix = []
    for p in suffix:
        dx = p[0] - anchor[0]
        dy = p[1] - anchor[1]
        for _ in range(rot):
            dx, dy = -dy, dx
        new_suffix.append((anchor[0] + dx, anchor[1] + dy))
    # verify unit steps within new_suffix
    for i in range(1, len(new_suffix)):
        if abs(new_suffix[i][0] - new_suffix[i - 1][0]) + abs(new_suffix[i][1] - new_suffix[i - 1][1]) != 1:
            return None
    # verify first of new_suffix is adjacent to anchor
    if abs(new_suffix[0][0] - anchor[0]) + abs(new_suffix[0][1] - anchor[1]) != 1:
        return None
    return new_suffix


def _rotate_around(segment, anchor, rot):
    """Rotate segment around anchor. Return new segment or None."""
    new_segment = []
    for p in segment:
        dx = p[0] - anchor[0]
        dy = p[1] - anchor[1]
        for _ in range(rot):
            dx, dy = -dy, dx
        new_segment.append((anchor[0] + dx, anchor[1] + dy))
    # verify unit steps
    for i in range(1, len(new_segment)):
        if abs(new_segment[i][0] - new_segment[i - 1][0]) + abs(new_segment[i][1] - new_segment[i - 1][1]) != 1:
            return None
    # verify first is adjacent to anchor
    if abs(new_segment[0][0] - anchor[0]) + abs(new_segment[0][1] - anchor[1]) != 1:
        return None
    return new_segment


def _anneal(sequence, dimension, rng, time_budget):
    """Simulated annealing with multiple move types."""
    chain = _initial_chain(sequence, dimension, rng)
    if chain is None:
        # fallback: straight line
        chain = [(i, 0) for i in range(len(sequence))]
    
    current_energy = _energy_from_chain(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    T = 5.0  # initial temperature
    T_min = 0.1
    # adaptive cooling
    no_improve = 0
    patience = 500
    
    t0 = time.time()
    step = 0
    while time.time() - t0 < time_budget:
        step += 1
        move_type = rng.choice([0, 0, 1, 1, 2])  # bias toward pivot and crankshaft
        new_chain = _try_move(chain, move_type, rng)
        if new_chain is None:
            continue
        
        new_energy = _energy_from_chain(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        # adaptive cooling
        if no_improve > patience:
            T *= 0.8
            no_improve = 0
        
        if T < T_min:
            T = T_min
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation via simulated annealing."""
    rng = random.Random(12345 + len(sequence))
    
    # time budget per sequence: distribute total time
    total_budget = DEADLINE - (time.time() - _START)
    # reserve some time for overhead
    per_seq_budget = max(1.0, total_budget / 10.0 - 1.0)
    
    # run multiple SA restarts if time allows
    best_chain = None
    best_energy = 1
    
    t0 = time.time()
    restart = 0
    while time.time() - t0 < per_seq_budget:
        restart += 1
        chain, energy = _anneal(sequence, dimension, rng, per_seq_budget / (restart + 1))
        if chain is not None and energy < best_energy:
            best_chain = chain
            best_energy = energy
        # if we've already used most of the budget, stop
        if time.time() - t0 > per_seq_budget * 0.8:
            break
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(len(sequence))]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
