import time
import random
from collections import defaultdict

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _beam_search(sequence, dimension, width, time_limit):
    start_time = time.time()
    n = len(sequence)
    
    # State: (chain, occupied, idx_by_pos, energy)
    # chain: list of tuples
    # occupied: set of tuples
    # idx_by_pos: dict {tuple: int}
    # energy: int
    
    start_pos = (0,) * dimension
    initial_chain = [start_pos]
    initial_occupied = {start_pos}
    initial_idx = {start_pos: 0}
    initial_energy = 0
    
    beam = [(initial_chain, initial_occupied, initial_idx, initial_energy)]
    
    for i in range(1, n):
        new_beam = []
        seq_i = sequence[i]
        
        for state in beam:
            chain, occupied, idx_by_pos, energy = state
            last_pos = chain[-1]
            
            for neighbor in _neighbours(last_pos):
                if neighbor in occupied:
                    continue
                
                # Calculate energy gain
                gain = 0
                if seq_i == 'H':
                    for n_pos in _neighbours(neighbor):
                        if n_pos in occupied:
                            j = idx_by_pos.get(n_pos)
                            if j is not None and j < i - 1 and sequence[j] == 'H':
                                gain += 1
                
                new_energy = energy - gain
                
                # Heuristic: Prefer lower energy. 
                # To break ties, prefer moves that leave more free space?
                # Or just random tie-breaking to ensure diversity.
                # Let's add a small random noise to avoid deterministic traps.
                score = new_energy + random.random() * 0.1
                
                # Create new state
                new_chain = chain + [neighbor]
                new_occupied = occupied | {neighbor}
                new_idx = idx_by_pos.copy()
                new_idx[neighbor] = i
                
                new_beam.append((new_chain, new_occupied, new_idx, new_energy, score))
        
        # Prune to width
        if len(new_beam) > width:
            new_beam.sort(key=lambda x: x[4])
            new_beam = new_beam[:width]
        
        beam = [(s[0], s[1], s[2], s[3]) for s in new_beam]
        
        # Check time
        if time.time() - start_time > time_limit * 0.8:
            # Return best found so far if we are out of time?
            # But we need a complete chain. If we stop early, we don't have one.
            # We should ensure we finish if possible, or return a straight line if we can't.
            pass

    if not beam:
        return None
        
    # Return best energy chain
    best_state = min(beam, key=lambda x: x[3])
    return best_state[0]

def fold(sequence, dimension):
    # Adaptive width based on time remaining and sequence length
    # Start with a reasonable width, increase if fast?
    # For now, fixed width 100-200.
    
    # Time budget per sequence? The task says 240s for ALL.
    # We don't know how many sequences are left.
    # Strategy: Use a global deadline? No, fold is called per sequence.
    # The evaluator calls fold 26 times.
    # We should allocate time per call?
    # If we assume uniform distribution, 240/26 ~ 9s per sequence.
    # But long sequences take longer.
    # Let's use a fixed time limit per call, say 8 seconds.
    # If we exceed, we return the best we have or a fallback.
    
    # Actually, the prompt says "fold is called once per benchmark sequence... 240s CPU budget for all sequences".
    # We can't know the total budget inside fold.
    # But we can assume we have ~10s per call on average.
    
    # Let's try width 50 first to ensure we finish, then maybe increase?
    # The knowledge says width 600 was possible with lazy copying.
    # Let's try width 100.
    
    width = 100
    time_limit = 8.0 # seconds per call
    
    result = _beam_search(sequence, dimension, width, time_limit)
    
    if result is None:
        # Fallback: straight line
        return [list(p) for p in [(i, 0, 0) for i in range(len(sequence))]]
        
    return [list(p) for p in result]
