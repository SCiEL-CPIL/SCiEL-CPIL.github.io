"""Rewritten HP lattice protein folding using Iterated Local Search with Hill Climbing."""
import random
import math
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute the negative H-H contact energy."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _initial_spiral(sequence, dimension):
    """Create a compact initial conformation using a spiral-like pattern."""
    n = len(sequence)
    if n <= 1:
        return [(0,) * dimension]
    
    # Start at origin, try to make a compact spiral
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    direction = (1, 0)  # start going right
    
    for i in range(1, n):
        placed = False
        # Try turning right, straight, left
        for turn in (1, 0, -1):
            if turn == 1:
                new_dir = (direction[1], -direction[0])
            elif turn == -1:
                new_dir = (-direction[1], direction[0])
            else:
                new_dir = direction
            
            new_pos = (chain[-1][0] + new_dir[0], chain[-1][1] + new_dir[1])
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied[new_pos] = i
                direction = new_dir
                placed = True
                break
        
        if not placed:
            # Fallback: try all 4 directions
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_pos = (chain[-1][0] + dx, chain[-1][1] + dy)
                if new_pos not in occupied:
                    chain.append(new_pos)
                    occupied[new_pos] = i
                    direction = (dx, dy)
                    placed = True
                    break
        
        if not placed:
            # Give up on spiral, use a straight line from here
            for j in range(i, n):
                chain.append((chain[-1][0], chain[-1][1] + 1))
            break
    
    return chain


def _try_single_step(chain, occupied, i, rng):
    """Try to move residue i to a valid neighbor. Returns (new_chain, new_occupied) or None."""
    n = len(chain)
    if i == 0 or i == n - 1:
        return None
    
    old_pos = chain[i]
    candidates = []
    for q in _neighbours(old_pos):
        # Check if q is free or is the current position
        if q in occupied and occupied[q] != i:
            continue
        # Check connectivity
        if abs(q[0] - chain[i-1][0]) + abs(q[1] - chain[i-1][1]) != 1:
            continue
        if abs(q[0] - chain[i+1][0]) + abs(q[1] - chain[i+1][1]) != 1:
            continue
        if q != old_pos:
            candidates.append(q)
    
    if not candidates:
        return None
    
    new_pos = rng.choice(candidates)
    
    # Perform move
    new_chain = list(chain)
    new_chain[i] = new_pos
    new_occupied = dict(occupied)
    del new_occupied[old_pos]
    new_occupied[new_pos] = i
    
    return new_chain, new_occupied


def _try_reversal(chain, occupied, i, j, rng):
    """Try to reverse segment i..j. Returns (new_chain, new_occupied) or None."""
    n = len(chain)
    if i >= j:
        return None
    
    # Check connectivity
    if i > 0:
        prev = chain[i-1]
        new_first = chain[j]
        if abs(prev[0] - new_first[0]) + abs(prev[1] - new_first[1]) != 1:
            return None
    if j < n - 1:
        next_pos = chain[j+1]
        new_last = chain[i]
        if abs(next_pos[0] - new_last[0]) + abs(next_pos[1] - new_last[1]) != 1:
            return None
    
    # Self-avoidance is preserved by reversal of a valid segment?
    # Yes, the set of positions is the same.
    
    new_chain = list(chain)
    new_chain[i:j+1] = chain[i:j+1][::-1]
    new_occupied = dict(occupied)
    for k in range(i, j+1):
        new_occupied[new_chain[k]] = k
        
    return new_chain, new_occupied


def _local_search(sequence, chain, rng, max_iter=1000):
    """Hill Climbing local search."""
    n = len(sequence)
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    for _ in range(max_iter):
        improved = False
        # Try a few random moves
        for _ in range(20):
            move_type = rng.random()
            new_chain = None
            new_occupied = None
            
            if move_type < 0.5:
                # Single step
                i = rng.randint(1, n - 2)
                res = _try_single_step(chain, occupied, i, rng)
                if res:
                    new_chain, new_occupied = res
            else:
                # Reversal
                i = rng.randint(0, n - 2)
                j = rng.randint(i + 1, n - 1)
                res = _try_reversal(chain, occupied, i, j, rng)
                if res:
                    new_chain, new_occupied = res
            
            if new_chain is None:
                continue
                
            new_energy = _compute_energy(sequence, new_chain)
            
            if new_energy < current_energy:
                chain = new_chain
                occupied = new_occupied
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                improved = True
                break # Restart move selection with new state
        
        if not improved:
            break
            
    return best_chain, best_energy


def _perturb(chain, rng, num_moves=10):
    """Apply random moves to perturb the chain."""
    n = len(chain)
    occupied = {p: i for i, p in enumerate(chain)}
    for _ in range(num_moves):
        move_type = rng.random()
        if move_type < 0.5:
            i = rng.randint(1, n - 2)
            res = _try_single_step(chain, occupied, i, rng)
            if res:
                chain, occupied = res
        else:
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            res = _try_reversal(chain, occupied, i, j, rng)
            if res:
                chain, occupied = res
    return chain


def fold(sequence, dimension):
    """Best conformation using Iterated Local Search."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    start_time = time.time()
    time_limit = 110.0 # Leave some buffer
    
    best_chain = None
    best_energy = 0
    
    # Number of ILS iterations depends on sequence length
    # Shorter sequences can afford more iterations
    if n <= 20:
        max_ils_iter = 50
    elif n <= 50:
        max_ils_iter = 30
    else:
        max_ils_iter = 20
        
    rng = random.Random(42)
    
    for it in range(max_ils_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize or use previous best
        if it == 0:
            chain = _initial_spiral(sequence, dimension)
        else:
            # Perturb the best found so far
            chain = _perturb(best_chain, rng, num_moves=5 + n//10)
        
        # Local Search
        chain, energy = _local_search(sequence, chain, rng, max_iter=500)
        
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        best_chain = _initial_spiral(sequence, dimension)
        
    return [list(p) for p in best_chain]
