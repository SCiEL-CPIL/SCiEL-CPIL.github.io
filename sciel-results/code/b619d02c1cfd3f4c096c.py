# EVOLVE-BLOCK-START
import random
import time
import sys

# Global deadline guard
_DEADLINE = 330.0  # seconds wall-clock limit with safety margin
_start_time = None


def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute the HP contact energy (negative number of H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _pivot_move(chain, n, rng, max_tail_len=8):
    """
    Perform a pivot move: rotate a random suffix of the chain.
    Returns a new chain if valid, None otherwise.
    Limit tail length to keep it fast.
    """
    if n <= 2:
        return None
    
    # Choose pivot point (index from 1 to n-2)
    # Bias towards shorter tails for speed
    max_pivot = min(n - 2, max_tail_len)
    pivot = rng.randint(1, max_pivot)
    
    prefix = chain[:pivot + 1]
    suffix = chain[pivot + 1:]
    
    # The vector from pivot to next
    v = tuple(suffix[0][d] - prefix[-1][d] for d in range(len(prefix[-1])))
    
    # Generate all possible rotations of this vector in 3D
    # For a unit vector, there are 5 other unit vectors at distance 1
    candidates = []
    for axis in range(3):
        for step in (-1, 1):
            new_v = list(v)
            new_v[axis] = step
            # Check it's a unit step and different from original
            if new_v != list(v):
                candidates.append(tuple(new_v))
    
    if not candidates:
        return None
    
    # Randomly pick a rotation
    new_first = tuple(prefix[-1][d] + candidates[rng.randint(0, len(candidates) - 1)][d] 
                      for d in range(len(prefix[-1])))
    
    # Apply the rotation to the entire suffix
    # The rotation is: new_pos = old_pos - old_first + new_first
    dx = new_first[0] - suffix[0][0]
    dy = new_first[1] - suffix[0][1]
    dz = new_first[2] - suffix[0][2]
    
    new_suffix = [(suffix[i][0] + dx, suffix[i][1] + dy, suffix[i][2] + dz) for i in range(len(suffix))]
    
    new_chain = prefix + new_suffix
    
    # Check self-avoidance
    occupied = set(new_chain)
    if len(occupied) != len(new_chain):
        return None
    
    return new_chain


def _tail_regen_move(chain, sequence, n, rng, regen_len=4):
    """
    Tail regeneration: remove the last few residues and regrow them greedily.
    Short length for speed.
    """
    if n <= regen_len + 2:
        return None
    
    # Keep a random prefix length
    keep = n - rng.randint(regen_len, min(regen_len + 2, n - 2))
    if keep < 2:
        return None
    
    prefix = chain[:keep]
    occupied = set(prefix)
    
    # Regrow the tail
    new_chain = list(prefix)
    for i in range(keep, n):
        last = new_chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Score: prefer H-H contacts
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        # Check if it's an H residue in the prefix
                        # Optimization: We don't have index map here, so we just check if q is in occupied.
                        # We need to know if the residue at q is H.
                        # Since we don't have the index map, we can't easily check sequence[idx].
                        # However, for a short tail regen, we can just accept any contact as a "good" move 
                        # or we can pass the index map. 
                        # Let's just use a heuristic: any contact is good, free space is good.
                        gain += 1 
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        
        if not options:
            return None
        
        site = max(options)[1]
        occupied.add(site)
        new_chain.append(site)
    
    return new_chain


def _simulated_annealing(sequence, dimension, initial_chain, max_iters, rng, deadline):
    """
    Run simulated annealing starting from initial_chain.
    Returns the best chain found.
    """
    n = len(sequence)
    current = list(initial_chain)
    current_energy = _energy(sequence, current)
    
    best = list(current)
    best_energy = current_energy
    
    # Initial and final temperatures
    T_start = 1.0
    T_end = 0.01
    cooling_rate = (T_end / T_start) ** (1.0 / max(1, max_iters))
    
    T = T_start
    
    # Check time every 1000 iterations
    for it in range(max_iters):
        if it % 1000 == 0 and time.time() > deadline:
            break
        
        # Choose move type
        # Use shorter moves for speed
        if rng.random() < 0.2 and n > 10:
            new_chain = _tail_regen_move(current, sequence, n, rng, regen_len=4)
        else:
            new_chain = _pivot_move(current, n, rng, max_tail_len=8)
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Metropolis acceptance
        if delta < 0 or rng.random() < pow(2.718281828, -delta / max(T, 1e-10)):
            current = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best = list(current)
        
        T *= cooling_rate
    
    return best, best_energy


def _greedy_grow(sequence, dimension, rng):
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
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def fold(sequence, dimension):
    """
    Find a good conformation for the given HP sequence.
    Uses multi-start simulated annealing with tail regeneration moves.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: 10 seconds
    deadline = _start_time + _DEADLINE - 10.0
    
    n = len(sequence)
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = 10 ** 9
    
    # Phase 1: Generate diverse starting points via greedy growth
    starts = []
    # Reduce number of starts to allow more time per SA
    num_starts = min(20, max(5, 200 // n))
    for _ in range(num_starts):
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is not None:
            starts.append(chain)
    
    if not starts:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in best_chain]
    
    # Phase 2: Run simulated annealing from each start
    # Allocate time evenly
    time_per_sa = (deadline - time.time()) / len(starts) if len(starts) > 0 else 0
    
    for idx, start_chain in enumerate(starts):
        if time.time() > deadline:
            break
        
        elapsed = time.time() - _start_time
        remaining = deadline - time.time()
        
        # Conservative iteration estimate: 1000 iters/sec
        max_iters = int(remaining * 1000)
        max_iters = max(100, min(max_iters, 20000)) # Cap iterations
        
        sa_chain, sa_energy = _simulated_annealing(
            sequence, dimension, start_chain, max_iters, rng, deadline
        )
        
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_chain = sa_chain
    
    if best_chain is None:
        best_chain = starts[0]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
