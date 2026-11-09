"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = time.time()
_DEADLINE = 205.0  # Safety margin before 210s kill


def _remaining_time():
    return max(0.0, _DEADLINE - (time.time() - _START_TIME))


def _neighbours(point):
    # Optimized for 2D
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth."""
    start = (0, 0)
    chain = [start]
    occupied = {start}
    
    # Directions: Up, Down, Left, Right
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    for i in range(1, len(sequence)):
        last_x, last_y = chain[-1]
        options = []
        for dx, dy in dirs:
            nx, ny = last_x + dx, last_y + dy
            if (nx, ny) in occupied:
                continue
            options.append((nx, ny))
        
        if not options:
            return None
        
        next_pos = rng.choice(options)
        chain.append(next_pos)
        occupied.add(next_pos)
        
    return chain


def _calc_energy(sequence, chain):
    """Calculate energy of a chain. O(N) with set lookups."""
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y = p
            for q in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                j = pos_to_idx.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def fold(sequence, dimension):
    """Best conformation using Simulated Annealing."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension for _ in range(n)]
    
    # 1. Generate initial valid chain
    rng = random.Random(12345 + n)
    chain = None
    for _ in range(50):
        chain = _grow(sequence, dimension, rng)
        if chain is not None:
            break
    
    if chain is None:
        chain = [(i, 0) for i in range(n)]
        
    if n < 3:
        return [list(p) for p in chain]

    # 2. Initialize SA structures
    occupied = set(chain)
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    current_energy = _calc_energy(sequence, chain)
    
    best_energy = current_energy
    best_chain = list(chain)
    
    # SA Parameters
    T_start = 2.0
    T_end = 0.01
    
    # Allocate time per sequence
    # 10 sequences. Total budget ~200s.
    # 18s per sequence = 180s total. Safe.
    time_limit = 18.0 
    start_time = time.time()
    
    def get_contacts_for_residue(i, p):
        if sequence[i] != 'H':
            return 0
        count = 0
        x, y = p
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        for q in neighbors:
            j = pos_to_idx.get(q)
            if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                count += 1
        return count
    
    while (time.time() - start_time) < time_limit:
        i = rng.randint(1, n - 2)
        p_old = chain[i]
        
        x, y = p_old
        candidates = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        
        valid_moves = []
        for p_new in candidates:
            if p_new not in occupied:
                valid_moves.append(p_new)
        
        if not valid_moves:
            continue
            
        p_new = rng.choice(valid_moves)
        
        c_old = get_contacts_for_residue(i, p_old)
        c_new = get_contacts_for_residue(i, p_new)
        
        delta_e = -(c_new - c_old)
        
        if delta_e < 0:
            accept = True
        else:
            elapsed = time.time() - start_time
            progress = elapsed / time_limit
            T = T_start * (T_end / T_start) ** progress
            if T > 0:
                prob = math.exp(-delta_e / T)
                accept = (rng.random() < prob)
            else:
                accept = False
        
        if accept:
            occupied.remove(p_old)
            occupied.add(p_new)
            
            del pos_to_idx[p_old]
            pos_to_idx[p_new] = i
            
            chain[i] = p_new
            
            current_energy += delta_e
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
