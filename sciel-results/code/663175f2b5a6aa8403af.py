"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math

def _neighbours(point):
    """Generate 4-neighbors of a point on a 2D lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _calculate_energy(sequence, chain):
    """
    Calculate the energy of the conformation.
    Energy = - (number of H-H contacts)
    Contacts are non-bonded pairs of H residues that are adjacent on the lattice.
    """
    if not chain:
        return 0
    
    # Map position to index for O(1) lookup
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    contacts = 0
    # We only need to check neighbors in one direction to avoid double counting,
    # or just check all and divide by 2, or check specific directions.
    # Let's check Right and Down to count each pair once.
    for i, p in enumerate(chain):
        if sequence[i] != 'H':
            continue
        x, y = p
        # Check Right
        if (x + 1, y) in pos_to_idx:
            j = pos_to_idx[(x + 1, y)]
            if j != i + 1 and j != i - 1: # Not bonded
                if sequence[j] == 'H':
                    contacts += 1
        # Check Down
        if (x, y + 1) in pos_to_idx:
            j = pos_to_idx[(x, y + 1)]
            if j != i + 1 and j != i - 1: # Not bonded
                if sequence[j] == 'H':
                    contacts += 1
                    
    return -contacts

def _init_spiral(length):
    """
    Generate a spiral self-avoiding walk.
    This provides a compact, valid starting structure.
    """
    if length == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    x, y = 0, 0
    # Directions: Right, Down, Left, Up
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dir_idx = 0
    step_count = 0
    max_steps = 0
    
    while len(chain) < length:
        # Move in current direction
        dx, dy = dirs[dir_idx]
        x += dx
        y += dy
        
        # Check if we are about to hit the chain (simple check: if we've been here)
        # For a standard spiral, we don't hit the chain until we complete a layer.
        # However, a simple spiral might not be valid for all lengths if not careful.
        # Let's use a safer "random compact" init if spiral fails, but spiral usually works.
        # Actually, a simple square spiral is always self-avoiding.
        
        chain.append((x, y))
        step_count += 1
        
        # Logic for turning:
        # In a standard spiral, we turn after 1, 1, 2, 2, 3, 3... steps
        # Let's implement a standard spiral generation.
        
    # The above while loop is a bit messy for exact length. Let's generate a full spiral and truncate.
    # But truncating a spiral might leave "dangling" ends that are fine, as long as it's a path.
    # A path is just a sequence of points. Truncating a valid path yields a valid path.
    
    # Let's regenerate properly:
    chain = [(0, 0)]
    x, y = 0, 0
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dir_idx = 0
    steps_in_dir = 0
    current_steps = 0
    steps_needed = 1
    
    while len(chain) < length:
        dx, dy = dirs[dir_idx]
        x += dx
        y += dy
        chain.append((x, y))
        steps_in_dir += 1
        
        if steps_in_dir == steps_needed:
            steps_in_dir = 0
            dir_idx = (dir_idx + 1) % 4
            # Increase step count every 2 turns
            if dir_idx % 2 == 0:
                steps_needed += 1
                
    return chain

def _simulated_annealing(sequence, dimension, initial_chain, rng, max_iters=2000, T_start=1.0, T_end=0.01):
    """
    Perform Simulated Annealing to optimize the chain.
    """
    if not initial_chain:
        return None, 0
        
    chain = list(initial_chain)
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _calculate_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    N = len(chain)
    
    # Pre-calculate neighbors for faster access? 
    # For N<=100, recalculating energy is O(N). 
    # Updating energy incrementally is complex. Let's just recalculate.
    # 2000 iterations * 100 residues * 4 neighbors = 800k ops. Very fast.
    
    for i in range(max_iters):
        # Cooling schedule
        T = T_start * (T_end / T_start) ** (i / max_iters)
        
        # Pick a random move
        move_type = rng.random()
        
        new_chain = None
        new_energy = None
        
        if move_type < 0.7:
            # Single residue move
            # Pick a random internal residue (not ends, to keep path connected easily? 
            # Actually, moving ends is fine too, but internal is more interesting)
            # Let's pick any index 1 to N-2
            idx = rng.randint(1, N - 2)
            x, y = chain[idx]
            
            # Try 4 neighbors
            neighbors = list(_neighbours((x, y)))
            rng.shuffle(neighbors)
            
            for nx, ny in neighbors:
                # Check if neighbor is empty
                if (nx, ny) in occupied:
                    continue
                # Check if moving here creates a bond with a non-adjacent residue?
                # The move is valid if the new position is empty.
                # We must ensure the new position doesn't collide with chain[idx-1] or chain[idx+1] 
                # which are already handled by 'occupied' check? 
                # Wait, 'occupied' contains all points. 
                # If (nx, ny) is in occupied, it's invalid.
                # If it's not, it's valid.
                
                # Perform move
                old_pos = (x, y)
                new_pos = (nx, ny)
                
                # Update occupied
                del occupied[old_pos]
                occupied[new_pos] = idx
                
                # Update chain
                chain[idx] = new_pos
                
                # Calculate new energy
                new_energy = _calculate_energy(sequence, chain)
                
                # Acceptance
                delta_E = new_energy - current_energy
                if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
                    # Accept
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                    new_chain = chain # Keep the move
                    break
                else:
                    # Reject, revert
                    chain[idx] = old_pos
                    del occupied[new_pos]
                    occupied[old_pos] = idx
                    new_chain = None
                    break
        
        else:
            # Chain reversal
            # Pick two indices i < j
            i = rng.randint(0, N - 1)
            j = rng.randint(0, N - 1)
            if i > j:
                i, j = j, i
            
            # Reverse segment i..j
            # Check if the reversed segment is self-avoiding?
            # Reversal of a self-avoiding walk is always self-avoiding internally.
            # We just need to check the connections at the boundaries.
            # The internal bonds are reversed, so they are still valid bonds.
            # The bonds (i-1, i) and (j, j+1) change to (i-1, j) and (i, j+1).
            # We must check if chain[i-1] is adjacent to chain[j] (if i>0)
            # and if chain[i] is adjacent to chain[j+1] (if j<N-1)
            
            valid = True
            if i > 0:
                p1 = chain[i-1]
                p2 = chain[j]
                if abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) != 1:
                    valid = False
            
            if valid and j < N - 1:
                p1 = chain[i]
                p2 = chain[j+1]
                if abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) != 1:
                    valid = False
            
            if not valid:
                continue
                
            # Perform reversal
            segment = chain[i:j+1]
            segment.reverse()
            for k in range(i, j+1):
                chain[k] = segment[k-i]
            
            # Update occupied (positions don't change, just indices)
            # Wait, occupied maps position -> index.
            # The set of positions is the same. So occupied keys are same.
            # But values (indices) change for the reversed segment.
            for k in range(i, j+1):
                pos = chain[k]
                occupied[pos] = k
            
            new_energy = _calculate_energy(sequence, chain)
            
            delta_E = new_energy - current_energy
            if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
            else:
                # Revert
                segment.reverse()
                for k in range(i, j+1):
                    chain[k] = segment[k-i]
                for k in range(i, j+1):
                    pos = chain[k]
                    occupied[pos] = k

    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    """
    N = len(sequence)
    if N == 0:
        return []
    if N == 1:
        return [[0, 0]]
    
    rng = random.Random(42)
    
    # Multi-start strategy
    best_chain = None
    best_energy = 0 # Energy is negative, so 0 is worse than any valid fold
    
    # Strategy 1: Spiral initialization
    spiral_chain = _init_spiral(N)
    chain, energy = _simulated_annealing(sequence, dimension, spiral_chain, rng, max_iters=1000)
    if chain and energy < best_energy:
        best_chain = chain
        best_energy = energy
        
    # Strategy 2: Random compact initialization (Random Walk that doesn't self-intersect)
    # This is harder to generate. Let's try a "greedy" random walk.
    # Or just run SA again from a slightly perturbed spiral.
    
    # Let's try 5 more starts with different seeds/initializations
    for run in range(5):
        # Perturb the spiral
        chain = list(spiral_chain)
        # Randomly swap some internal segments? 
        # Or just start SA from the spiral again with different T?
        # Better: Generate a different initial SAW.
        
        # Simple Random SAW:
        start = (0, 0)
        cur = start
        path = [start]
        occupied = {start: 0}
        dirs = [(1,0), (-1,0), (0,1), (0,-1)]
        
        for i in range(1, N):
            moves = []
            for dx, dy in dirs:
                nx, ny = cur[0] + dx, cur[1] + dy
                if (nx, ny) not in occupied:
                    moves.append((nx, ny))
            if not moves:
                break
            nxt = rng.choice(moves)
            path.append(nxt)
            occupied[nxt] = i
            cur = nxt
            
        if len(path) == N:
             chain, energy = _simulated_annealing(sequence, dimension, path, rng, max_iters=1000)
             if chain and energy < best_energy:
                 best_chain = chain
                 best_energy = energy
                 
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(N)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
