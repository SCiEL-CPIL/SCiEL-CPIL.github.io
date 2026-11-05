"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

# Precompute neighbor offsets for 3D cubic lattice
_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # safety margin before 330s kill


def _time_left():
    return _DEADLINE - time.time()


def _grow_greedy(sequence, dimension, rng):
    """
    Randomized greedy chain growth with backtracking-lite.
    Prefers sites that create H-H contacts, then has free space.
    Uses a stack-based approach for potential backtracking.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]

    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}

    for i in range(1, n):
        # Get available neighbor sites
        last = chain[-1]
        options = []
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            site = (last[0] + dx, last[1] + dy, last[2] + dz)
            if site in occupied:
                continue

            # Compute gain: number of new H-H contacts if we place here
            gain = 0
            if sequence[i] == 'H':
                for dx2, dy2, dz2 in _NEIGHBOR_OFFSETS:
                    q = (site[0] + dx2, site[1] + dy2, site[2] + dz2)
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1

            # Free space heuristic
            free = 0
            for dx2, dy2, dz2 in _NEIGHBOR_OFFSETS:
                q = (site[0] + dx2, site[1] + dy2, site[2] + dz2)
                if q not in occupied:
                    free += 1

            score = gain * 10.0 + free * 0.1 + rng.random() * 2.0
            options.append((score, site))

        if not options:
            return None

        # Select best option
        options.sort(key=lambda x: -x[0])
        # With some probability, pick a suboptimal option to add diversity
        if len(options) > 1 and rng.random() < 0.3:
            idx = min(1, len(options) - 1)
            site = options[idx][1]
        else:
            site = options[0][1]

        chain.append(site)
        occupied[site] = i

    return chain


def _local_search(sequence, dimension, chain, rng, max_iterations=500):
    """
    Simple local search: try to improve energy by moving a single residue
    to a nearby unoccupied site, or by reversing a segment.
    Returns improved chain or original if no improvement found.
    """
    n = len(sequence)
    if n < 4:
        return chain

    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)

    for iteration in range(max_iterations):
        if _time_left() < 5.0:
            break

        # Try moving a random internal residue
        i = rng.randint(1, n - 2)
        old_pos = best_chain[i]

        # Find an unoccupied neighbor of old_pos
        candidates = []
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            new_pos = (old_pos[0] + dx, old_pos[1] + dy, old_pos[2] + dz)
            # Check if new_pos is not occupied by another residue
            if new_pos in best_chain:
                continue
            # Ensure the move doesn't break connectivity
            # Check that previous and next residues can still reach
            prev_pos = best_chain[i - 1]
            next_pos = best_chain[i + 1]
            # Distance check: new_pos must be adjacent to at least one of prev/next
            # Actually, for a single move, we need new_pos adjacent to prev_pos OR next_pos
            # But since we're moving residue i, the chain connectivity requires:
            # dist(prev_pos, new_pos) == 1 AND dist(new_pos, next_pos) == 1
            # This is very restrictive. Let's use segment reversal instead.
            pass

        # Instead, try segment reversal
        if rng.random() < 0.5 and n >= 5:
            # Pick a segment to reverse
            seg_len = rng.randint(2, min(6, n - 2))
            start_idx = rng.randint(1, n - seg_len - 1)
            end_idx = start_idx + seg_len - 1

            # Reverse the segment
            new_chain = list(best_chain)
            new_chain[start_idx:end_idx + 1] = best_chain[start_idx:end_idx + 1][::-1]

            # Check self-avoidance
            if len(set(new_chain)) != n:
                continue

            # Check unit steps
            valid = True
            for k in range(n - 1):
                diff = sum(abs(a - b) for a, b in zip(new_chain[k], new_chain[k + 1]))
                if diff != 1:
                    valid = False
                    break

            if not valid:
                continue

            new_energy = _energy(sequence, new_chain)
            if new_energy < best_energy:
                best_chain = new_chain
                best_energy = new_energy
        else:
            # Try a small perturbation: pick a residue and try to swap with nearby
            # Actually, let's try a different move: pick two non-adjacent positions
            # and try to "slide" one along the chain
            i = rng.randint(1, n - 2)
            j = rng.randint(1, n - 2)
            if abs(i - j) < 3:
                continue

            # Try swapping positions i and j in the chain (not the coordinates)
            # This is a permutation move - check if it maintains unit steps
            new_chain = list(best_chain)
            new_chain[i], new_chain[j] = best_chain[j], best_chain[i]

            if len(set(new_chain)) != n:
                continue

            valid = True
            for k in range(n - 1):
                diff = sum(abs(a - b) for a, b in zip(new_chain[k], new_chain[k + 1]))
                if diff != 1:
                    valid = False
                    break

            if not valid:
                continue

            new_energy = _energy(sequence, new_chain)
            if new_energy < best_energy:
                best_chain = new_chain
                best_energy = new_energy

    return best_chain


def _energy(sequence, chain):
    """Compute energy: -number of H-H contacts on neighboring non-consecutive sites."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i, p in enumerate(chain):
        if sequence[i] != 'H':
            continue
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            q = (p[0] + dx, p[1] + dy, p[2] + dz)
            j = where.get(q)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts // 1  # Each contact counted twice, so divide by... wait
    # Actually each pair (i,j) is counted when we're at i and see j, and when at j and see i.
    # So we need to divide by 2. But the original code only counts j > i+1, so it counts each once.
    # Let me fix this.


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


def _grow_with_backtrack(sequence, dimension, rng, max_backtracks=50):
    """
    Greedy growth with limited backtracking.
    When stuck, backtrack a few steps and try alternative paths.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]

    # Use a stack for backtracking
    # State: (chain, occupied, next_index, alternatives)
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    backtracks = 0

    for i in range(1, n):
        last = chain[-1]
        options = []
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            site = (last[0] + dx, last[1] + dy, last[2] + dz)
            if site in occupied:
                continue

            gain = 0
            if sequence[i] == 'H':
                for dx2, dy2, dz2 in _NEIGHBOR_OFFSETS:
                    q = (site[0] + dx2, site[1] + dy2, site[2] + dz2)
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1

            free = sum(1 for dx2, dy2, dz2 in _NEIGHBOR_OFFSETS
                       if (site[0] + dx2, site[1] + dy2, site[2] + dz2) not in occupied)

            score = gain * 10.0 + free * 0.1 + rng.random() * 2.0
            options.append((score, site))

        if not options:
            # Stuck - try backtracking
            if backtracks >= max_backtracks or len(chain) <= 1:
                return None

            # Remove last few residues and try again
            remove_count = min(3, len(chain) - 1)
            for _ in range(remove_count):
                last_pos = chain.pop()
                del occupied[last_pos]
            backtracks += 1
            # Adjust i to match new chain length
            i = len(chain) - 1
            continue

        options.sort(key=lambda x: -x[0])
        # Add randomness
        if len(options) > 1 and rng.random() < 0.2:
            idx = min(1, len(options) - 1)
            site = options[idx][1]
        else:
            site = options[0][1]

        chain.append(site)
        occupied[site] = i

    return chain


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

    rng = random.Random(12345 + len(sequence))

    best_chain = None
    best_energy = 0  # Energy is negative or zero

    # Time budget for this sequence
    # Distribute time based on sequence length
    # Longer sequences get more time
    time_budget = min(120.0, max(10.0, n * 0.5))  # seconds
    seq_start = time.time()

    # Phase 1: Multiple greedy growths
    num_growths = min(2000, max(100, int(time_budget * 20)))

    for attempt in range(num_growths):
        if time.time() - seq_start > time_budget:
            break
        if _time_left() < 10.0:
            break

        # Alternate between different growth strategies
        if attempt % 3 == 0:
            chain = _grow_greedy(sequence, dimension, rng)
        elif attempt % 3 == 1:
            chain = _grow_with_backtrack(sequence, dimension, rng, max_backtracks=20)
        else:
            # Slightly different greedy with more randomness
            chain = _grow_greedy(sequence, dimension, rng)

        if chain is None:
            continue

        energy = _energy_correct(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy

    # Phase 2: Local search on the best chains found
    if best_chain is not None and _time_left() > 15.0:
        # Run local search on the best chain
        improved = _local_search(sequence, dimension, best_chain, rng, max_iterations=200)
        improved_energy = _energy_correct(sequence, improved)
        if improved_energy < best_energy:
            best_chain = improved
            best_energy = improved_energy

    # Fallback: straight line if no valid chain found
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = _energy_correct(sequence, best_chain)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
