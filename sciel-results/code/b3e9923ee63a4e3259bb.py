import random
import time
import math

# EVOLVE-BLOCK-START

def _calc_energy(sequence, chain):
    """Calculate the energy of the conformation. Energy is -number of H-H contacts."""
    n = len(chain)
    if n < 2:
        return 0
    
    # Map position to index for O(1) lookup
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
        
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            x, y = chain[i]
            # Check 4 neighbors
            for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                j = pos_to_idx.get((nx, ny), -1)
                # Ensure it's not the adjacent residue in the chain (i-1 or i+1)
                if j != -1 and abs(j - i) > 1:
                    if sequence[j] == 'H':
                        # Avoid double counting: only count if j > i
                        if j > i:
                            contacts += 1
    return -contacts

def _make_initial_chain_safe(length):
    """Generate a compact initial chain, falling back to a line if random fails."""
    # Try random walk
    for _ in range(50):
        chain = [(0, 0)]
        occupied = {(0, 0)}
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        success = True
        for _ in range(1, length):
            random.shuffle(directions)
            last_x, last_y = chain[-1]
            moved = False
            for dx, dy in directions:
                new_pos = (last_x + dx, last_y + dy)
                if new_pos not in occupied:
                    chain.append(new_pos)
                    occupied.add(new_pos)
                    moved = True
                    break
            if not moved:
                success = False
                break
        if success and len(chain) == length:
            return chain
            
    # Fallback: Straight line
    return [(i, 0) for i in range(length)]

def fold(sequence, dimension):
    if dimension != 2:
        raise ValueError("Only 2D supported")
        
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
        
    # Time management
    if not hasattr(fold, '_start_time'):
        fold._start_time = time.time()
        
    elapsed = time.time() - fold._start_time
    total_time_limit = 115.0 # Leave 5s margin for overhead
    
    if elapsed > total_time_limit:
        # Time's up. Return a valid chain quickly.
        chain = _make_initial_chain_safe(n)
        return [list(p) for p in chain]
        
    # Allocate time for this sequence
    # If we have a lot of time left, use more. If little, use less.
    # Assume 10 sequences.
    # If this is the first, we have ~115s.
    # If this is the 10th, we have ~11s.
    # We don't know the index.
    # Strategy: Use a fixed chunk of time, e.g., 10s, but capped by remaining time.
    local_deadline = time.time() + min(10.0, total_time_limit - elapsed)
    
    # 1. Initialize
    chain = _make_initial_chain_safe(n)
    current_energy = _calc_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # 2. Simulated Annealing
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.999
    
    rng = random.Random(12345 + n)
    
    last_reheat = time.time()
    
    while time.time() < local_deadline:
        # Reheat every 2 seconds to escape local minima
        if time.time() - last_reheat > 2.0:
            T = 5.0
            last_reheat = time.time()
            
        # Generate Move
        if rng.random() < 0.5:
            # Pivot Move
            if n < 3:
                continue
            pivot_idx = rng.randint(1, n - 2)
            pivot_point = chain[pivot_idx - 1]
            
            orig_seg = chain[pivot_idx:]
            
            valid_rotations = []
            # Pre-compute fixed set
            fixed_set = set(chain[:pivot_idx])
            
            for rot in range(1, 4):
                new_seg = []
                for p in orig_seg:
                    dx = p[0] - pivot_point[0]
                    dy = p[1] - pivot_point[1]
                    
                    if rot == 1:
                        ndx, ndy = -dy, dx
                    elif rot == 2:
                        ndx, ndy = -dx, -dy
                    else:
                        ndx, ndy = dy, -dx
                        
                    new_p = (pivot_point[0] + ndx, pivot_point[1] + ndy)
                    new_seg.append(new_p)
                
                # Check collisions
                collision = False
                for p in new_seg:
                    if p in fixed_set:
                        collision = True
                        break
                        
                if not collision:
                    valid_rotations.append(new_seg)
                    
            if valid_rotations:
                new_seg = valid_rotations[rng.randint(0, len(valid_rotations)-1)]
                new_chain = chain[:pivot_idx] + new_seg
                new_energy = _calc_energy(sequence, new_chain)
                
                delta_e = new_energy - current_energy
                if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                    chain = new_chain
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                    
        else:
            # End Move
            if n < 2:
                continue
            last_x, last_y = chain[-1]
            neighbors = [(last_x+1, last_y), (last_x-1, last_y), (last_x, last_y+1), (last_x, last_y-1)]
            rng.shuffle(neighbors)
            
            moved = False
            # Optimize: check collision against chain[:-1]
            # Using a set is faster for large N, but for N=100, list check is okay.
            # Let's use a set for the first N-1 elements.
            # But creating a set every time is O(N).
            # We can maintain a set of occupied sites.
            # However, updating the set on every move is also O(1) but we need to remove/add.
            # Given N=100, O(N) check is fine.
            
            occupied_except_last = set(chain[:-1])
            for new_p in neighbors:
                if new_p not in occupied_except_last:
                    new_chain = chain[:-1] + [new_p]
                    new_energy = _calc_energy(sequence, new_chain)
                    
                    delta_e = new_energy - current_energy
                    if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                        chain = new_chain
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(chain)
                        moved = True
                        break
            
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
