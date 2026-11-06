"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 195.0  # Safety margin for 210s kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


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


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
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


def _pivot_move(chain, rng):
    """Perform a single pivot move: rotate a subchain by 90, 180, or 270 degrees around a pivot point."""
    if len(chain) < 4:
        return None
    
    # Choose a random pivot index (not the first or last)
    pivot_idx = rng.randint(1, len(chain) - 2)
    pivot = chain[pivot_idx]
    
    # Get the segment to rotate (from pivot_idx to end)
    segment = chain[pivot_idx:]
    
    # Try different rotations
    for angle in [90, 180, 270, -90]:
        # Rotate segment around pivot
        rotated = []
        for p in segment:
            dx, dy = p[0] - pivot[0], p[1] - pivot[1]
            if angle == 90:
                ndx, ndy = -dy, dx
            elif angle == -90:
                ndx, ndy = dy, -dx
            elif angle == 180:
                ndx, ndy = -dx, -dy
            else:
                ndx, ndy = dx, dy
            rotated.append((pivot[0] + ndx, pivot[1] + ndy))
        
        # Check if the rotated segment is self-avoiding and doesn't collide with the fixed part
        new_chain = chain[:pivot_idx] + rotated
        new_occupied = set(new_chain)
        if len(new_occupied) == len(new_chain):
            # Check unit steps
            valid = True
            for i in range(1, len(new_chain)):
                dx = abs(new_chain[i][0] - new_chain[i-1][0])
                dy = abs(new_chain[i][1] - new_chain[i-1][1])
                if dx + dy != 1:
                    valid = False
                    break
            if valid:
                return new_chain
    
    return None


def _local_search(sequence, chain, max_iterations=1000, time_limit=5.0):
    """Simulated annealing local search to improve a conformation."""
    if chain is None:
        return None
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    temperature = 2.0
    cooling = 0.995
    
    rng = random.Random(hash(tuple(current_chain)) % (2**32))
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 10:
            break
        
        # Try a pivot move
        new_chain = _pivot_move(current_chain, rng)
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < pow(2.718281828, -delta / max(temperature, 0.1)):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        temperature *= cooling
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using multiple strategies: greedy growth + local search."""
    seq_len = len(sequence)
    
    # Adjust strategy based on sequence length
    if seq_len <= 30:
        num_grow = 300
        local_search_iters = 2000
        local_search_time = 8.0
    elif seq_len <= 50:
        num_grow = 200
        local_search_iters = 1500
        local_search_time = 6.0
    elif seq_len <= 70:
        num_grow = 150
        local_search_iters = 1000
        local_search_time = 4.0
    else:
        num_grow = 100
        local_search_iters = 800
        local_search_time = 3.0
    
    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0
    
    # First pass: pure greedy growth
    for i in range(num_grow):
        if _time_left() < 20:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain
    
    # Second pass: local search on best chains
    if best_chain is not None and _time_left() > 30:
        # Try local search on the best chain
        improved = _local_search(sequence, best_chain, max_iterations=local_search_iters, 
                                  time_limit=local_search_time)
        if improved is not None:
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved
        
        # Try local search on a few other good chains
        for i in range(min(20, num_grow // 10)):
            if _time_left() < 25:
                break
            
            chain = _grow(sequence, dimension, rng)
            if chain is None:
                continue
            
            value = _energy(sequence, chain)
            if value < best_energy + 3:  # Only improve chains close to best
                improved = _local_search(sequence, chain, max_iterations=local_search_iters // 2,
                                          time_limit=local_search_time / 2)
                if improved is not None:
                    improved_energy = _energy(sequence, improved)
                    if improved_energy < best_energy:
                        best_energy = improved_energy
                        best_chain = improved
    
    # Final fallback: straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(seq_len)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
