import time
import random
import math

def _neighbours(point):
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)

def _calculate_energy(sequence, chain):
    occupied = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i in range(len(sequence)):
        if sequence[i] == 'H':
            p = chain[i]
            for q in _neighbours(p):
                j = occupied.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _try_translate_tail(chain, occupied, k, direction, sequence):
    """
    Try to translate the tail chain[k+1:] by 'direction'.
    Returns (new_chain, new_occupied, new_energy) or (None, None, None) if collision.
    """
    n = len(chain)
    # Check if new positions collide with head [0..k]
    # We don't need to check tail against itself because it's a rigid translation.
    # We only need to check tail_new against head.
    
    head_occupied = {chain[i] for i in range(k + 1)}
    
    # Calculate new positions
    new_tail = []
    for i in range(k + 1, n):
        old_pos = chain[i]
        new_pos = (old_pos[0] + direction[0], old_pos[1] + direction[1], old_pos[2] + direction[2])
        if new_pos in head_occupied:
            return None, None, None
        new_tail.append(new_pos)
        
    # Also check if new_tail positions collide with each other?
    # No, rigid translation preserves relative distances.
    
    # Construct new chain
    new_chain = chain[:k+1] + new_tail
    new_occupied = {p: i for i, p in enumerate(new_chain)}
    
    # Calculate energy change? 
    # For simplicity, recalculate energy. It's O(N^2) or O(N*6). 
    # With N=136, O(N) is fine.
    new_energy = _calculate_energy(sequence, new_chain)
    
    return new_chain, new_occupied, new_energy

def fold(sequence, dimension):
    start_time = time.time()
    deadline = start_time + 280.0
    
    n = len(sequence)
    if n == 0:
        return []
    
    best_chain = [(i, 0, 0) for i in range(n)]
    best_energy = _calculate_energy(sequence, best_chain)
    
    rng = random.Random(12345)
    
    directions = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    
    while time.time() < deadline:
        # 1. Generate initial chain
        chain = [(0, 0, 0)]
        occupied = {chain[0]: 0}
        success = True
        
        for i in range(1, n):
            last_pos = chain[-1]
            candidates = []
            for q in _neighbours(last_pos):
                if q not in occupied:
                    gain = 0
                    if sequence[i] == 'H':
                        for r in _neighbours(q):
                            if r in occupied and occupied[r] < i - 1 and sequence[occupied[r]] == 'H':
                                gain += 1
                    score = gain + rng.random() * 0.5
                    candidates.append((score, q))
            
            if not candidates:
                success = False
                break
            
            if rng.random() < 0.7:
                best_score, best_pos = max(candidates)
            else:
                best_score, best_pos = rng.choice(candidates)
            
            chain.append(best_pos)
            occupied[best_pos] = i
        
        if not success:
            continue
            
        current_chain = chain
        current_occupied = occupied
        current_energy = _calculate_energy(sequence, current_chain)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = current_chain[:]
            
        # 2. Simulated Annealing
        T = 1.0
        T_min = 0.05
        cooling_rate = 0.995
        steps = 500
        
        for step in range(steps):
            if time.time() > deadline:
                break
                
            # Choose move type
            move_type = rng.random()
            
            if move_type < 0.6:
                # Tail Translation
                # Pick a random split point k between 1 and n-2
                k = rng.randint(1, n - 2)
                direction = rng.choice(directions)
                
                new_chain, new_occupied, new_energy = _try_translate_tail(
                    current_chain, current_occupied, k, direction, sequence
                )
                
                if new_chain is not None:
                    delta = new_energy - current_energy
                    if delta < 0 or rng.random() < math.exp(-delta / T):
                        current_chain = new_chain
                        current_occupied = new_occupied
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = current_chain[:]
            else:
                # Tail Regrowth
                K = rng.randint(3, 10)
                start_idx = n - K
                if start_idx <= 0:
                    continue
                
                # Backup
                old_tail = current_chain[start_idx:]
                old_occupied = current_occupied.copy()
                
                # Remove
                for pos in old_tail:
                    del current_occupied[pos]
                del current_chain[start_idx:]
                
                # Regrow
                success = True
                for i in range(start_idx, n):
                    last_pos = current_chain[-1]
                    candidates = []
                    for q in _neighbours(last_pos):
                        if q not in current_occupied:
                            gain = 0
                            if sequence[i] == 'H':
                                for r in _neighbours(q):
                                    if r in current_occupied and current_occupied[r] < i - 1 and sequence[current_occupied[r]] == 'H':
                                        gain += 1
                            score = gain + rng.random()
                            candidates.append((score, q))
                            
                    if not candidates:
                        success = False
                        break
                        
                    best_score, best_pos = max(candidates)
                    current_chain.append(best_pos)
                    current_occupied[best_pos] = i
                    
                if success:
                    new_energy = _calculate_energy(sequence, current_chain)
                    delta = new_energy - current_energy
                    if delta < 0 or rng.random() < math.exp(-delta / T):
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = current_chain[:]
                    else:
                        # Revert
                        current_chain = current_chain[:start_idx] + old_tail
                        current_occupied = old_occupied
                else:
                    # Revert
                    current_chain = current_chain[:start_idx] + old_tail
                    current_occupied = old_occupied
            
            T *= cooling_rate
            if T < T_min:
                T = T_min

    return [list(p) for p in best_chain]
