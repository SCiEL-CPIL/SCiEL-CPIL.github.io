"""Rewritten HP lattice protein folding using Simulated Annealing with 1-moves."""
import random
import time
import math

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _compute_energy(sequence, chain):
    """Compute the energy of a conformation. Energy = -number of H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _is_valid_walk(chain):
    """Check if the chain is self-avoiding and has unit steps."""
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            prev = chain[i-1]
            dist = sum(abs(a-b) for a, b in zip(p, prev))
            if dist != 1:
                return False
    return True

def _initial_spiral(sequence, dimension, seed):
    """Generate a compact spiral-like initial conformation."""
    n = len(sequence)
    rng = random.Random(seed)
    chain = [(0, 0)]
    occupied = {(0, 0)}
    # Directions: Right, Down, Left, Up
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dir_idx = 0
    
    x, y = 0, 0
    for i in range(1, n):
        moved = False
        # Try to move in the current direction, then turn if blocked
        for _ in range(4):
            dx, dy = directions[dir_idx % 4]
            nx, ny = x + dx, y + dy
            if (nx, ny) not in occupied:
                x, y = nx, ny
                chain.append((x, y))
                occupied.add((x, y))
                moved = True
                break
            dir_idx += 1
        if not moved:
            # Fallback: random walk
            while i < n:
                found = False
                for _ in range(4):
                    dx, dy = directions[rng.randint(0,3)]
                    nx, ny = x + dx, y + dy
                    if (nx, ny) not in occupied:
                        x, y = nx, ny
                        chain.append((x, y))
                        occupied.add((x, y))
                        i += 1
                        found = True
                        break
                if not found:
                    # Stuck, return line
                    return [(i, 0) for i in range(n)]
            break
    return chain

def fold(sequence, dimension):
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    start_time = time.time()
    deadline = start_time + 175.0 # 175s to be safe
    
    best_chain = None
    best_energy = 0 # Start with 0 (no contacts) as upper bound for "best" (min energy)
    
    # Use a fixed seed for reproducibility of the "best" found, but allow multiple restarts
    base_seed = 12345 + n
    
    for restart in range(5): # 5 restarts
        if time.time() > deadline - 5.0:
            break
            
        rng = random.Random(base_seed + restart)
        
        # Initialize
        chain = _initial_spiral(sequence, dimension, base_seed + restart)
        if not _is_valid_walk(chain):
            chain = [(i, 0) for i in range(n)]
        
        occupied = set(chain)
        current_energy = _compute_energy(sequence, chain)
        
        if best_chain is None or current_energy < best_energy:
            best_energy = current_energy
            best_chain = chain[:]
            
        # Simulated Annealing
        T = 1.0
        T_min = 0.01
        alpha = 0.999
        
        while time.time() < deadline - 5.0:
            index = rng.randint(0, n - 1)
            p = chain[index]
            
            possible_moves = []
            for q in _neighbours(p):
                if q in occupied:
                    continue
                valid = True
                if index > 0:
                    prev_p = chain[index-1]
                    if sum(abs(a-b) for a,b in zip(prev_p, q)) != 1:
                        valid = False
                if index < n - 1:
                    next_p = chain[index+1]
                    if sum(abs(a-b) for a,b in zip(next_p, q)) != 1:
                        valid = False
                if valid:
                    possible_moves.append(q)
            
            if possible_moves:
                new_p = rng.choice(possible_moves)
                
                # Apply move
                occupied.remove(p)
                chain[index] = new_p
                occupied.add(new_p)
                
                new_energy = _compute_energy(sequence, chain)
                delta = new_energy - current_energy
                
                if delta < 0 or (T > T_min and rng.random() < math.exp(-delta / T)):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = chain[:]
                else:
                    # Revert
                    occupied.remove(new_p)
                    chain[index] = p
                    occupied.add(p)
            
            T *= alpha
            if T < T_min:
                break
                
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
