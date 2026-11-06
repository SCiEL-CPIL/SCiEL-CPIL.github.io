"""HP lattice protein folding on the cubic lattice.
Strategy: Multi-restart randomized growth + simulated annealing with kink/pivot/reversal moves.
"""
import random
import time

# EVOLVE-BLOCK-START

# Global deadline guard
_TOTAL_DEADLINE = 330.0  # seconds wall-clock
_start_time = None


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _time_remaining():
    return _TOTAL_DEADLINE - _get_elapsed() - 5.0  # 5s safety margin


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy: -number of H-H contacts (non-adjacent along chain)."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            score = gain + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options, key=lambda x: x[0])[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _kink_move(chain, rng):
    """Try a kink move: pick two non-adjacent positions and reverse the subchain."""
    n = len(chain)
    if n < 4:
        return None
    i = rng.randint(0, n - 4)
    j = rng.randint(i + 2, n - 1)
    # Check that the reversed segment doesn't collide
    segment = chain[i + 1:j + 1]
    new_segment = segment[::-1]
    # Check self-avoidance for the new positions
    occupied = set(chain)
    # Remove old segment positions
    for p in segment:
        occupied.discard(p)
    # Check new positions
    for p in new_segment:
        if p in occupied:
            return None
    # Also need to check that the connection at i and j is valid
    # chain[i] connects to new_segment[0] (which was segment[-1] = chain[j])
    # new_segment[-1] (which was segment[0] = chain[i+1]) connects to chain[j+1]
    if abs(new_segment[0][0] - chain[i][0]) + abs(new_segment[0][1] - chain[i][1]) + abs(new_segment[0][2] - chain[i][2]) != 1:
        return None
    if j + 1 < n:
        if abs(new_segment[-1][0] - chain[j + 1][0]) + abs(new_segment[-1][1] - chain[j + 1][1]) + abs(new_segment[-1][2] - chain[j + 1][2]) != 1:
            return None
    new_chain = chain[:i + 1] + new_segment + chain[j + 1:]
    return new_chain


def _pivot_move(chain, rng):
    """Try a pivot move: rotate a subchain by 90 or 180 degrees."""
    n = len(chain)
    if n < 4:
        return None
    i = rng.randint(0, n - 3)
    j = rng.randint(i + 2, n - 1)
    segment = chain[i + 1:j + 1]
    if len(segment) < 2:
        return None
    # Try different rotations
    for rot in range(4):
        # Rotate segment around chain[i]
        rotated = []
        for p in segment:
            # Translate so chain[i] is origin
            dx = p[0] - chain[i][0]
            dy = p[1] - chain[i][1]
            dz = p[2] - chain[i][2]
            # Apply rotation
            for _ in range(rot % 4):
                dx, dy = -dy, dx  # 90 deg rotation in xy plane
            new_p = (chain[i][0] + dx, chain[i][1] + dy, chain[i][2] + dz)
            rotated.append(new_p)
        # Check self-avoidance
        occupied = set(chain)
        for p in segment:
            occupied.discard(p)
        valid = True
        for p in rotated:
            if p in occupied:
                valid = False
                break
        if not valid:
            continue
        # Check connectivity at boundaries
        if abs(rotated[0][0] - chain[i][0]) + abs(rotated[0][1] - chain[i][1]) + abs(rotated[0][2] - chain[i][2]) != 1:
            continue
        if j + 1 < n:
            if abs(rotated[-1][0] - chain[j + 1][0]) + abs(rotated[-1][1] - chain[j + 1][1]) + abs(rotated[-1][2] - chain[j + 1][2]) != 1:
                continue
        new_chain = chain[:i + 1] + rotated + chain[j + 1:]
        return new_chain
    return None


def _simulated_annealing(sequence, chain, rng, time_budget):
    """Simulated annealing with kink and pivot moves."""
    n = len(chain)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Temperature schedule
    T = max(1.0, abs(current_energy) * 0.5)
    T_min = 0.01
    cooling_rate = 0.995
    
    start_t = time.time()
    while time.time() - start_t < time_budget and T > T_min:
        move_type = rng.random()
        if move_type < 0.5:
            new_chain = _kink_move(chain, rng)
        else:
            new_chain = _pivot_move(chain, rng)
        
        if new_chain is None:
            T *= cooling_rate
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < pow(2.718281828, -delta_e / T):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        T *= cooling_rate
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold a sequence using multi-restart growth + simulated annealing."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    # Allocate time based on sequence length
    if n <= 48:
        num_restarts = 30
        sa_budget_per_restart = 2.0
    elif n <= 64:
        num_restarts = 20
        sa_budget_per_restart = 3.0
    elif n <= 96:
        num_restarts = 12
        sa_budget_per_restart = 4.0
    else:
        num_restarts = 8
        sa_budget_per_restart = 5.0
    
    best_chain = None
    best_energy = 1
    
    for restart in range(num_restarts):
        if _time_remaining() < 10.0:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
        
        # Simulated annealing to improve
        sa_budget = min(sa_budget_per_restart, _time_remaining() / 2.0)
        if sa_budget > 0.5:
            improved_chain, improved_energy = _simulated_annealing(sequence, chain, rng, sa_budget)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved_chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
