import time
import random

def _neighbours(point):
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def _local_search(chain, sequence, max_iterations=5000, deadline=None):
    """Local search using random single-residue moves."""
    n = len(chain)
    current_chain = [tuple(p) for p in chain]
    
    # Build pos_to_idx
    pos_to_idx = {p: i for i, p in enumerate(current_chain)}
    
    # Compute initial energy
    current_energy = 0
    for i, p in enumerate(current_chain):
        if sequence[i] != 'H':
            continue
        for nb in _neighbours(p):
            if nb in pos_to_idx:
                j = pos_to_idx[nb]
                if abs(i - j) > 1 and sequence[j] == 'H':
                    current_energy -= 1
    current_energy //= 2  # Each pair counted twice
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    for _ in range(max_iterations):
        if deadline and time.time() > deadline:
            break
        
        # Pick a random interior residue (not endpoints)
        i = random.randint(1, n - 2)
        old_pos = current_chain[i]
        
        # Try moving to a free neighbor
        candidates = []
        for nb in _neighbours(old_pos):
            if nb in pos_to_idx:
                continue
            # Check if moving here would create self-intersection with adjacent residues
            if nb == current_chain[i-1] or nb == current_chain[i+1]:
                continue
            candidates.append(nb)
        
        if not candidates:
            continue
        
        new_pos = random.choice(candidates)
        
        # Calculate energy change
        # Remove old position contributions
        old_delta = 0
        if sequence[i] == 'H':
            for nb in _neighbours(old_pos):
                if nb in pos_to_idx:
                    j = pos_to_idx[nb]
                    if j != i and abs(i - j) > 1 and sequence[j] == 'H':
                        old_delta -= 1
        
        # Calculate new position contributions
        new_delta = 0
        if sequence[i] == 'H':
            for nb in _neighbours(new_pos):
                if nb in pos_to_idx:
                    j = pos_to_idx[nb]
                    if j != i and abs(i - j) > 1 and sequence[j] == 'H':
                        new_delta -= 1
        
        # Also check interactions with i-1 and i+1 (they might be H)
        # When residue i moves, it can form/break contacts with i-1 and i+1
        # But those are excluded by abs(i-j) > 1, so no change there
        
        delta_e = (new_delta - old_delta) // 2  # Each pair counted twice
        
        if delta_e <= 0 or random.random() < 0.1:  # Accept if better or with 10% probability
            # Perform the move
            del pos_to_idx[old_pos]
            current_chain[i] = new_pos
            pos_to_idx[new_pos] = i
            current_energy += delta_e
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold the sequence using Beam Search on the 3D cubic lattice.
    Returns a list of lists of integers representing the coordinates.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    if n == 2:
        return [[0, 0, 0], [1, 0, 0]]
    
    start_time = time.time()
    deadline = start_time + 290
    
    # Symmetry breaking:
    # R0 at (0,0,0), R1 at (1,0,0)
    initial_chain = [(0, 0, 0), (1, 0, 0)]
    initial_occupied = {(0, 0, 0), (1, 0, 0)}
    initial_pos_to_idx = {(0, 0, 0): 0, (1, 0, 0): 1}
    initial_energy = 0
    
    beam = [(initial_chain, initial_occupied, initial_pos_to_idx, initial_energy)]
    
    # Beam width: adjust based on chain length
    if n <= 50:
        K = 200
    elif n <= 80:
        K = 120
    else:
        K = 80
    
    for i in range(2, n):
        new_beam = []
        residue = sequence[i]
        
        for state in beam:
            chain, occupied, pos_to_idx, energy = state
            last_pos = chain[-1]
            
            for neighbor in _neighbours(last_pos):
                if neighbor in occupied:
                    continue
                
                # Calculate new energy
                new_energy = energy
                if residue == 'H':
                    for nb in _neighbours(neighbor):
                        if nb in occupied:
                            j = pos_to_idx[nb]
                            # Exclude the immediately preceding residue in the chain
                            if j != i - 1 and sequence[j] == 'H':
                                new_energy -= 1
                
                # Calculate free space for heuristic
                free_space = 0
                for nb in _neighbours(neighbor):
                    if nb not in occupied:
                        free_space += 1
                
                # Score: Energy is primary (minimize), free space is secondary (maximize)
                score = new_energy - 0.05 * free_space
                
                # Create new state
                new_chain = chain + [neighbor]
                new_occupied = occupied | {neighbor}
                new_pos_to_idx = pos_to_idx | {neighbor: i}
                
                # Store score in the tuple for sorting
                new_beam.append((score, new_chain, new_occupied, new_pos_to_idx, new_energy))
        
        if not new_beam:
            break
            
        # Sort by score
        new_beam.sort(key=lambda x: x[0])
        
        # Keep top K
        beam = []
        for item in new_beam[:K]:
            _, chain, occupied, pos_to_idx, energy = item
            beam.append((chain, occupied, pos_to_idx, energy))
            
        if not beam:
            break
            
        # Time check
        if time.time() > deadline:
            break

    if not beam:
        # Fallback: straight line
        return [[i, 0, 0] for i in range(n)]
    
    # Take top few candidates and apply local search
    top_k = min(5, len(beam))
    best_chain = None
    best_energy = float('inf')
    
    # Allocate remaining time for local search
    elapsed = time.time() - start_time
    remaining = deadline - start_time - elapsed
    if remaining < 10:
        # Not enough time for local search, return best from beam
        best_chain = beam[0][0]
    else:
        # Split remaining time among top candidates
        time_per_candidate = max(1, remaining / top_k - 5)  # Leave 5s buffer
        
        for idx in range(top_k):
            candidate_chain = beam[idx][0]
            candidate_energy = beam[idx][3]
            
            # Apply local search
            ls_chain, ls_energy = _local_search(candidate_chain, sequence, max_iterations=10000, deadline=start_time + time_per_candidate)
            
            if ls_energy < best_energy:
                best_energy = ls_energy
                best_chain = ls_chain
            
            if time.time() > deadline - 5:
                break
    
    if best_chain is None:
        best_chain = beam[0][0]
    
    return [list(p) for p in best_chain]
