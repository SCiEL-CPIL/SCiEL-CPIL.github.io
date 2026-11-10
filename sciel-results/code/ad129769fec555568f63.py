"""HP lattice protein folding using simulated annealing with chain-reversal moves."""
import random
import math
import time


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute the energy of a conformation. Energy = -number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _is_valid_walk(chain):
    """Check if the chain is a valid self-avoiding walk with unit steps."""
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            prev = chain[i - 1]
            dist = sum(abs(a - b) for a, b in zip(p, prev))
            if dist != 1:
                return False
    return True


def _initial_spiral(sequence):
    """Generate a compact initial conformation using a spiral-like growth."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0)]
    
    # Start with a small spiral pattern
    chain = [(0, 0)]
    occupied = {(0, 0)}
    direction = [(1, 0)]  # right
    pos = (0, 0)
    
    # Simple spiral: right 1, down 1, left 2, up 2, right 3, down 3, etc.
    steps = 1
    dir_idx = 0
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
    for i in range(1, n):
        placed = False
        for _ in range(4):
            d = dirs[dir_idx % 4]
            new_pos = (pos[0] + d[0], pos[1] + d[1])
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied.add(new_pos)
                pos = new_pos
                placed = True
                break
            dir_idx += 1
        
        if not placed:
            # Try to find any free neighbour
            for q in _neighbours(pos):
                if q not in occupied:
                    chain.append(q)
                    occupied.add(q)
                    pos = q
                    placed = True
                    break
        
        if not placed:
            # Fallback: straight line
            for j in range(i, n):
                chain.append((j, 0))
            break
    
    return chain


def _reverse_segment(chain, i, j):
    """Reverse the segment of the chain from index i to j (inclusive)."""
    if i >= j:
        return chain
    return chain[:i] + list(reversed(chain[i:j+1])) + chain[j+1:]


def fold(sequence, dimension):
    """Find a low-energy conformation using simulated annealing."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension for _ in range(n)]
    
    start_time = time.time()
    deadline = start_time + 180  # Safety margin before 210s kill
    
    # Generate initial conformation
    rng = random.Random(42 + len(sequence))
    chain = _initial_spiral(sequence)
    
    # Ensure the chain has exactly n residues
    while len(chain) < n:
        last = chain[-1]
        for q in _neighbours(last):
            if q not in set(chain):
                chain.append(q)
                break
        else:
            chain.append((last[0], last[1] + len(chain)))
    
    chain = chain[:n]
    
    # Compute initial energy
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Simulated annealing parameters
    T = 1.0
    cooling = 0.9995
    min_T = 0.01
    
    # Track time per iteration to estimate budget
    max_iters = 500000
    
    for it in range(max_iters):
        # Check deadline
        if time.time() > deadline - 5:
            break
        
        # Choose a random segment to reverse
        if n < 3:
            break
        
        i = rng.randint(0, n - 2)
        j = rng.randint(i + 1, n - 1)
        
        # Ensure the reversal doesn't break connectivity
        # Check if chain[i-1] is adjacent to chain[j] and chain[i] is adjacent to chain[j+1]
        if i > 0:
            prev = chain[i - 1]
            new_next = chain[j]
            if sum(abs(a - b) for a, b in zip(prev, new_next)) != 1:
                continue
        
        if j < n - 1:
            nxt = chain[j + 1]
            new_prev = chain[i]
            if sum(abs(a - b) for a, b in zip(nxt, new_prev)) != 1:
                continue
        
        # Perform the reversal
        new_chain = _reverse_segment(chain, i, j)
        
        # Check self-avoidance
        if not _is_valid_walk(new_chain):
            continue
        
        # Compute energy change
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        # Cool down
        T *= cooling
        if T < min_T:
            T = min_T
    
    # Verify the best chain is valid
    if not _is_valid_walk(best_chain):
        # Fallback to a simple straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
