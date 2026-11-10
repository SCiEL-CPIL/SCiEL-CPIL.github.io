"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math
import numpy as np

# Grid offsets for 4-neighbors (up, down, left, right)
DX = np.array([0, 0, 1, -1])
DY = np.array([1, -1, 0, 0])

def _compute_energy(sequence, chain):
    """Compute the energy of a given conformation."""
    n = len(chain)
    # Map position to index for quick lookup
    # Using a dictionary is fast enough for n <= 100
    pos_to_idx = {tuple(p): i for i, p in enumerate(chain)}
    
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            x, y = chain[i]
            for dx, dy in zip(DX, DY):
                nx, ny = x + dx, y + dy
                j = pos_to_idx.get((nx, ny))
                if j is not None:
                    # Check if they are not adjacent in the chain
                    if abs(i - j) > 1:
                        if sequence[j] == 'H':
                            contacts += 1
    return -contacts

def _delta_energy(sequence, chain, i, new_pos, pos_to_idx):
    """
    Calculate the change in energy if residue i moves to new_pos.
    Assumes new_pos is empty and valid (not occupied by j where |i-j|<=1).
    """
    if sequence[i] != 'H':
        return 0
    
    delta = 0
    old_pos = chain[i]
    
    # Remove old contacts
    x, y = old_pos
    for dx, dy in zip(DX, DY):
        nx, ny = x + dx, y + dy
        j = pos_to_idx.get((nx, ny))
        if j is not None and abs(i - j) > 1:
            if sequence[j] == 'H':
                delta += 1 # Losing a contact increases energy (less negative)
                
    # Add new contacts
    nx, ny = new_pos
    for dx, dy in zip(DX, DY):
        nnx, nny = nx + dx, ny + dy
        j = pos_to_idx.get((nnx, nny))
        if j is not None and abs(i - j) > 1:
            if sequence[j] == 'H':
                delta -= 1 # Gaining a contact decreases energy (more negative)
                
    return delta

def _simulate_annealing(sequence, initial_chain, max_iters, start_temp, end_temp, rng):
    """
    Perform Simulated Annealing starting from initial_chain.
    Returns (best_chain, best_energy).
    """
    n = len(initial_chain)
    if n < 4:
        return list(initial_chain), _compute_energy(sequence, initial_chain)
    
    # Initialize state
    chain = list(initial_chain)
    pos_to_idx = {tuple(p): i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = current_energy
    
    # Cooling schedule
    # Using geometric cooling
    cooling_factor = (end_temp / start_temp) ** (1.0 / max(1, max_iters))
    
    # Precompute H positions for faster checking? 
    # sequence is a string, so sequence[i] is O(1).
    
    no_improve_counter = 0
    max_no_improve = max(1000, n * 10)
    
    for it in range(max_iters):
        temp = start_temp * (cooling_factor ** it)
        if temp < end_temp:
            temp = end_temp
            
        # Choose a random residue to move (exclude endpoints to keep chain connected easily? 
        # Actually, moving endpoints is fine if we just check validity, but 
        # standard SA on SAWs usually moves internal residues or uses pivot moves.
        # Moving a single residue (1-opt) requires the new position to be empty 
        # and not adjacent to the immediate neighbors of i in the chain.
        
        i = rng.randint(1, n - 2) # Internal residues only for 1-opt simplicity
        
        x, y = chain[i]
        # Try all 4 neighbors
        # Shuffle directions for randomness
        dirs = [0, 1, 2, 3]
        rng.shuffle(dirs)
        
        moved = False
        for d in dirs:
            nx, ny = x + DX[d], y + DY[d]
            
            # Check if new position is occupied
            if (nx, ny) in pos_to_idx:
                continue
                
            # Check if new position is adjacent to chain neighbors of i
            # Neighbors of i in chain are i-1 and i+1
            prev_pos = chain[i-1]
            next_pos = chain[i+1]
            
            if (nx == prev_pos[0] and ny == prev_pos[1]) or \
               (nx == next_pos[0] and ny == next_pos[1]):
                continue
            
            # Valid move
            # Calculate delta energy
            delta = _delta_energy(sequence, chain, i, (nx, ny), pos_to_idx)
            new_energy = current_energy + delta
            
            # Acceptance criterion
            accept = False
            if delta <= 0:
                accept = True
            else:
                # Metropolis criterion
                if temp > 0:
                    prob = math.exp(-delta / temp)
                    if rng.random() < prob:
                        accept = True
            
            if accept:
                # Perform move
                # Update pos_to_idx
                del pos_to_idx[(x, y)]
                pos_to_idx[(nx, ny)] = i
                chain[i] = (nx, ny)
                current_energy = new_energy
                moved = True
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
                    
                break # Only one move per iteration
        
        # If no move was accepted, just count no improve
        if not moved:
            no_improve_counter += 1
        
        # Early termination if stuck for too long
        if no_improve_counter > max_no_improve:
            break
            
    return best_chain, best_energy

def _local_search(sequence, chain, max_iters=10000, rng=None):
    """
    Strict local search (hill climbing) to polish the solution.
    Only accepts moves that strictly improve energy.
    """
    if rng is None:
        rng = random.Random(42)
        
    n = len(chain)
    if n < 4:
        return list(chain), _compute_energy(sequence, chain)
        
    current_chain = list(chain)
    pos_to_idx = {tuple(p): i for i, p in enumerate(current_chain)}
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    for _ in range(max_iters):
        improved = False
        # Shuffle indices to avoid bias
        indices = list(range(1, n-1))
        rng.shuffle(indices)
        
        for i in indices:
            x, y = current_chain[i]
            dirs = [0, 1, 2, 3]
            rng.shuffle(dirs)
            
            for d in dirs:
                nx, ny = x + DX[d], y + DY[d]
                if (nx, ny) in pos_to_idx:
                    continue
                prev_pos = current_chain[i-1]
                next_pos = current_chain[i+1]
                if (nx == prev_pos[0] and ny == prev_pos[1]) or \
                   (nx == next_pos[0] and ny == next_pos[1]):
                    continue
                    
                delta = _delta_energy(sequence, current_chain, i, (nx, ny), pos_to_idx)
                if delta < 0:
                    # Accept
                    del pos_to_idx[(x, y)]
                    pos_to_idx[(nx, ny)] = i
                    current_chain[i] = (nx, ny)
                    current_energy += delta
                    improved = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    break # Move to next residue
                    
            if improved:
                # Continue scanning other residues in this pass
                pass
            else:
                # No improvement for this residue
                pass
                
        if not improved:
            break
            
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    """
    start_time = time.time()
    deadline = start_time + 170.0 # Safety margin for 180s/210s limit
    
    n = len(sequence)
    if n <= 2:
        return [[0,0], [1,0]] if n==2 else [[0,0]]
        
    rng = random.Random(42)
    
    best_chain = None
    best_energy = float('inf')
    
    # Strategy: Multiple restarts of SA with different initializations
    # Initialization: Random walk that tries to stay compact
    
    def generate_initial_walk():
        """Generate a random self-avoiding walk with a slight bias towards center."""
        # Start at 0,0
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        for i in range(1, n):
            # Get neighbors of last point
            x, y = chain[-1]
            neighbors = []
            for dx, dy in zip(DX, DY):
                nx, ny = x + dx, y + dy
                if (nx, ny) not in occupied:
                    # Bias: prefer directions that keep x,y small (compactness)
                    # Simple heuristic: prefer moves that reduce max(x,y) or keep within box
                    score = abs(nx) + abs(ny) + rng.random()
                    neighbors.append((score, (nx, ny)))
            
            if not neighbors:
                return None # Failed to grow
            
            # Pick the best (lowest score) or random from top few
            neighbors.sort(key=lambda x: x[0])
            # Pick from top 3 to maintain diversity
            k = min(3, len(neighbors))
            chosen = rng.choice(neighbors[:k])
            pos = chosen[1]
            chain.append(pos)
            occupied[pos] = i
            
        return chain

    # Number of restarts depends on n
    if n <= 20:
        num_restarts = 500
        sa_iters = 5000
    elif n <= 50:
        num_restarts = 200
        sa_iters = 2000
    else:
        num_restarts = 100
        sa_iters = 1000
        
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        init_chain = generate_initial_walk()
        if init_chain is None:
            continue
            
        # Run SA
        # Temperature depends on n
        start_temp = 2.0
        end_temp = 0.1
        
        sa_chain, sa_energy = _simulate_annealing(
            sequence, init_chain, sa_iters, start_temp, end_temp, rng
        )
        
        # Polish with local search
        ls_chain, ls_energy = _local_search(
            sequence, sa_chain, max_iters=min(2000, n*20), rng=rng
        )
        
        if ls_energy < best_energy:
            best_energy = ls_energy
            best_chain = ls_chain
            
        # If we found a very good solution, we can be more aggressive or stop early
        # For HP, energy is -contacts. Max contacts ~ n/2.
        # If best_energy is very low, we might break early, but let's keep searching for robustness.

    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
