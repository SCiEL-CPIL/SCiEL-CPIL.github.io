"""Rewritten HP lattice protein folding using simulated annealing with chain-reversal moves."""
import random
import math


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


def _simulate_annealing(sequence, dimension, rng, num_iterations=5000):
    """Simulated annealing with chain-reversal and single-step moves."""
    n = len(sequence)
    if n <= 1:
        return _initial_spiral(sequence, dimension)
    
    # Start from a good initial conformation
    chain = _initial_spiral(sequence, dimension)
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = current_energy
    
    # Simulated annealing parameters
    T_start = 1.0
    T_end = 0.01
    cooling_rate = (T_end / T_start) ** (1.0 / num_iterations)
    
    for it in range(num_iterations):
        T = T_start * (cooling_rate ** it)
        
        # Choose move type: 60% chain reversal, 40% single step
        move_type = rng.random()
        
        if move_type < 0.6 and n > 4:
            # Chain reversal move: pick two indices i < j, reverse chain[i..j]
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            
            # Check if reversal is valid: the new endpoints must connect
            # Old: ... chain[i-1] - chain[i] ... chain[j] - chain[j+1] ...
            # New: ... chain[i-1] - chain[j] ... chain[i] - chain[j+1] ...
            # Need: chain[j] adjacent to chain[i-1] (if i > 0)
            # Need: chain[i] adjacent to chain[j+1] (if j < n-1)
            valid = True
            
            if i > 0:
                prev = chain[i - 1]
                new_first = chain[j]
                if abs(prev[0] - new_first[0]) + abs(prev[1] - new_first[1]) != 1:
                    valid = False
            
            if j < n - 1:
                next_pos = chain[j + 1]
                new_last = chain[i]
                if abs(next_pos[0] - new_last[0]) + abs(next_pos[1] - new_last[1]) != 1:
                    valid = False
            
            if not valid:
                continue
            
            # Check self-avoidance for the reversed segment
            # The segment chain[i..j] is reversed, so positions chain[i+1]..chain[j-1] 
            # must not collide with the rest
            # Actually, since we're just reversing the order, the set of positions is the same,
            # so self-avoidance is automatically maintained as long as the connections work.
            # The only concern is that the new connections don't create conflicts, which we checked.
            
            # Perform the reversal
            old_segment = chain[i:j + 1]
            chain[i:j + 1] = old_segment[::-1]
            
            # Update occupied
            for k in range(i, j + 1):
                occupied[chain[k]] = k
            
            new_energy = _compute_energy(sequence, chain)
            
            if new_energy < current_energy or rng.random() < math.exp((new_energy - current_energy) / T):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
            else:
                # Revert
                chain[i:j + 1] = old_segment
                for k in range(i, j + 1):
                    occupied[chain[k]] = k
        
        else:
            # Single step move: pick a residue (not first or last) and try to move it
            i = rng.randint(1, n - 2)
            old_pos = chain[i]
            
            # Try all 4 neighbors
            candidates = []
            for q in _neighbours(old_pos):
                if q not in occupied or occupied[q] == i:
                    # Check if q is adjacent to both chain[i-1] and chain[i+1]
                    if abs(q[0] - chain[i-1][0]) + abs(q[1] - chain[i-1][1]) == 1 and \
                       abs(q[0] - chain[i+1][0]) + abs(q[1] - chain[i+1][1]) == 1:
                        candidates.append(q)
            
            if not candidates:
                continue
            
            new_pos = rng.choice(candidates)
            if new_pos == old_pos:
                continue
            
            # Perform the move
            chain[i] = new_pos
            del occupied[old_pos]
            occupied[new_pos] = i
            
            new_energy = _compute_energy(sequence, chain)
            
            if new_energy < current_energy or rng.random() < math.exp((new_energy - current_energy) / T):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
            else:
                # Revert
                chain[i] = old_pos
                del occupied[new_pos]
                occupied[old_pos] = i
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using simulated annealing with multiple restarts."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    best_chain = None
    best_energy = 0
    
    # Multiple restarts with different seeds
    num_restarts = 3
    
    for restart in range(num_restarts):
        rng = random.Random(12345 + len(sequence) * 100 + restart)
        
        # Adjust iterations based on sequence length
        if n <= 30:
            num_iter = 8000
        elif n <= 60:
            num_iter = 6000
        else:
            num_iter = 4000
        
        chain = _simulate_annealing(sequence, dimension, rng, num_iter)
        energy = _compute_energy(sequence, chain)
        
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
