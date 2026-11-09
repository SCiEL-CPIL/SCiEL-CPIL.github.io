"""HP lattice protein folding: biased greedy growth + simulated annealing (reversal/pivot)."""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = time.time()
DEADLINE = 195.0  # safety margin before 210s wall-clock kill


def _remaining_time():
    return DEADLINE - time.time()


def _neighbours_2d(x, y):
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _initial_chain(sequence, dimension, rng):
    """Diverse biased greedy growth: prefer sites creating new H-H contacts, then free space."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0)]
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    for i in range(1, n):
        cx, cy = chain[-1]
        candidates = []
        for (sx, sy) in _neighbours_2d(cx, cy):
            if (sx, sy) in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for (qx, qy) in _neighbours_2d(sx, sy):
                    j = occupied.get((qx, qy), -1)
                    if j >= 0 and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = 0
            for (qx, qy) in _neighbours_2d(sx, sy):
                if (qx, qy) not in occupied:
                    free += 1
            score = gain * 2.0 + free * 0.1 + rng.random() * 0.5
            candidates.append((score, (sx, sy)))
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0], reverse=True)
        # pick top with some randomness
        if rng.random() < 0.7:
            site = candidates[0][1]
        elif len(candidates) > 1:
            site = candidates[1][1]
        else:
            site = candidates[0][1]
        occupied[site] = i
        chain.append(site)
    return chain


def _compute_energy(sequence, chain):
    """Compute HP energy: -number of non-consecutive H-H contacts."""
    n = len(chain)
    where = {}
    for i, (x, y) in enumerate(chain):
        where[(x, y)] = i
    contacts = 0
    for i, (x, y) in enumerate(chain):
        if sequence[i] == 'H':
            for (qx, qy) in _neighbours_2d(x, y):
                j = where.get((qx, qy), -1)
                if j >= 0 and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _sa_energy_update(sequence, chain, i, j, old_energy):
    """
    Compute energy delta for a reversal move reversing chain[i..j] (inclusive).
    Only contacts involving positions in [i..j] can change.
    Returns (new_energy, valid) where valid indicates no self-overlap.
    """
    n = len(chain)
    if i >= j:
        return old_energy, True
    # Build position map for quick lookup
    where = {}
    for k, (x, y) in enumerate(chain):
        where[(x, y)] = k

    # The reversed segment: new position of k (i<=k<=j) is chain[i+j-k]
    # Check self-avoidance: no two distinct positions in [i..j] map to same lattice point
    # Actually after reversal, position k gets coords of old (i+j-k).
    # Self-avoidance violated if for any a<b in [i..j], chain[a] == chain[b] originally? No.
    # After reversal, the set of lattice points occupied by [i..j] is the same set.
    # Self-avoidance among [i..j] is preserved (same points). But boundary contacts change.
    # Also need to ensure no point in [i..j] collides with outside? The set of points is unchanged,
    # so no new collisions with outside. Reversal always preserves self-avoidance.
    # So valid = True always for reversal.

    # Energy delta: only contacts where at least one endpoint is in [i..j] can change.
    # We recompute energy contributions involving indices in [i..j].
    # Simpler: recompute full energy? That's O(n). For efficiency, do delta.
    # For correctness and simplicity given budget, recompute full energy.
    new_chain = chain[:i] + chain[i:j+1][::-1] + chain[j+1:]
    new_energy = _compute_energy(sequence, new_chain)
    return new_energy, True, new_chain


def _sa_energy_update_pivot(sequence, chain, i, direction, old_energy):
    """
    Pivot: rotate the segment after index i (positions i+1..n-1) by 90 degrees around chain[i].
    direction: 0,1,2,3 for 0,90,180,270 degrees.
    Returns (new_energy, valid, new_chain) or (None, False, None) if self-overlap.
    """
    n = len(chain)
    if i >= n - 1:
        return old_energy, True, chain
    if direction == 0:
        return old_energy, True, chain

    px, py = chain[i]
    # Rotation around (px,py)
    def rotate(x, y, d):
        dx, dy = x - px, y - py
        if d == 1:   # 90 deg CCW: (dx,dy) -> (-dy, dx)
            return (px - dy, py + dx)
        elif d == 2: # 180: (dx,dy) -> (-dx,-dy)
            return (px - dx, py - dy)
        elif d == 3: # 270 CCW: (dx,dy) -> (dy, -dx)
            return (px + dy, py - dx)
        return (x, y)

    new_chain = list(chain[:i+1])
    for k in range(i+1, n):
        x, y = chain[k]
        new_chain.append(rotate(x, y, direction))

    # Check self-avoidance
    seen = set()
    for k, (x, y) in enumerate(new_chain):
        if (x, y) in seen:
            return None, False, None
        seen.add((x, y))

    new_energy = _compute_energy(sequence, new_chain)
    return new_energy, True, new_chain


def _anneal(sequence, dimension, chain, time_budget, rng):
    """Simulated annealing with reversal and pivot moves."""
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    # Cooling schedule
    t_start = max(time_budget, 0.5)
    t_end = 0.01
    alpha = (t_end / t_start) ** 0.5  # geometric cooling factor per step
    # We'll run until time budget is used
    steps_done = 0
    while _remaining_time() > 0.5:
        T = t_start * (t_end / t_start) ** (steps_done / max(1, int(t_start * 200)))
        if T < t_end:
            T = t_end
        steps_done += 1

        if rng.random() < 0.5:
            # Reversal move
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            # Ensure segment length reasonable
            if j - i < 2:
                continue
            new_energy, valid, new_chain = _sa_energy_update(sequence, chain, i, j, current_energy)
            if not valid:
                continue
            delta = new_energy - current_energy
            if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
                chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
        else:
            # Pivot move
            i = rng.randint(0, n - 2)
            direction = rng.randint(1, 3)
            new_energy, valid, new_chain = _sa_energy_update_pivot(sequence, chain, i, direction, current_energy)
            if not valid:
                continue
            delta = new_energy - current_energy
            if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
                chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation via multiple biased greedy growths followed by simulated annealing."""
    n = len(sequence)
    if n <= 1:
        return [[0, 0]] if dimension == 2 else [[0] * dimension]

    rng = random.Random(42 + len(sequence))
    best_chain = None
    best_energy = 0

    # Generate diverse initial conformations
    num_growths = 30
    initial_chains = []
    for _ in range(num_growths):
        chain = _initial_chain(sequence, dimension, rng)
        if chain is not None:
            initial_chains.append(chain)

    if not initial_chains:
        # Fallback: straight line
        chain = [(i, 0) for i in range(n)]
        initial_chains = [chain]

    # Evaluate all initial chains
    for chain in initial_chains:
        e = _compute_energy(sequence, chain)
        if best_chain is None or e < best_energy:
            best_chain = chain
            best_energy = e

    # Simulated annealing on the best initial chain, and a few others if time permits
    # Budget split: use remaining time for SA
    sa_budget = _remaining_time() * 0.85
    if sa_budget > 0.3:
        # SA on best
        improved_chain, improved_energy = _anneal(sequence, dimension, best_chain, sa_budget * 0.6, rng)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy

        # If time remains, SA on second-best initial
        if _remaining_time() > 1.0 and len(initial_chains) > 1:
            # Find second best
            energies = [( _compute_energy(sequence, c), i) for i, c in enumerate(initial_chains)]
            energies.sort()
            second_idx = None
            for e, idx in energies:
                if idx != initial_chains.index(best_chain) if best_chain in initial_chains else True:
                    if initial_chains[idx] != best_chain:
                        second_idx = idx
                        break
            if second_idx is not None:
                remaining = _remaining_time() * 0.4
                if remaining > 0.3:
                    improved_chain2, improved_energy2 = _anneal(sequence, dimension, initial_chains[second_idx], remaining, rng)
                    if improved_energy2 < best_energy:
                        best_chain = improved_chain2
                        best_energy = improved_energy2

    # Final local polish: short SA with very low temperature
    if _remaining_time() > 0.5:
        polish_budget = _remaining_time() * 0.5
        polished_chain, polished_energy = _anneal(sequence, dimension, best_chain, polish_budget, rng)
        if polished_energy < best_energy:
            best_chain = polished_chain
            best_energy = polished_energy

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
