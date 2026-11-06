"""HP lattice protein folding on the cubic lattice.
Strategy: Beam Search with dynamic width.
"""
# EVOLVE-BLOCK-START
import time

_TOTAL_DEADLINE = 325.0  # seconds wall-clock (evaluator kills at 330s)
_start_time = None


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _time_remaining():
    return _TOTAL_DEADLINE - _get_elapsed() - 5.0  # 5s safety margin


def _beam_search(sequence, dimension):
    """
    Beam search for HP lattice protein folding.
    Returns the best chain found.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    # Adjust beam width based on sequence length to prevent timeout
    if n > 100:
        beam_width = 20
    elif n > 50:
        beam_width = 30
    else:
        beam_width = 50
        
    start_pos = (0, 0, 0)
    initial_chain = [start_pos]
    initial_occupied = {start_pos}
    initial_pos_to_idx = {start_pos: 0}
    
    beam = [(0, initial_chain, initial_occupied, initial_pos_to_idx)]
    
    for step in range(1, n):
        if _time_remaining() < 0.5:
            best_state = min(beam, key=lambda x: x[0])
            return list(best_state[1])
            
        new_beam = []
        current_residue = sequence[step]
        
        for energy, chain, occupied, pos_to_idx in beam:
            last_pos = chain[-1]
            prev_pos = chain[-2] if step > 1 else None
            
            candidates = [
                (last_pos[0] + 1, last_pos[1], last_pos[2]),
                (last_pos[0] - 1, last_pos[1], last_pos[2]),
                (last_pos[0], last_pos[1] + 1, last_pos[2]),
                (last_pos[0], last_pos[1] - 1, last_pos[2]),
                (last_pos[0], last_pos[1], last_pos[2] + 1),
                (last_pos[0], last_pos[1], last_pos[2] - 1)
            ]
            
            for next_pos in candidates:
                if next_pos in occupied:
                    continue
                
                delta_e = 0
                if current_residue == 'H':
                    nbs = [
                        (next_pos[0] + 1, next_pos[1], next_pos[2]),
                        (next_pos[0] - 1, next_pos[1], next_pos[2]),
                        (next_pos[0], next_pos[1] + 1, next_pos[2]),
                        (next_pos[0], next_pos[1] - 1, next_pos[2]),
                        (next_pos[0], next_pos[1], next_pos[2] + 1),
                        (next_pos[0], next_pos[1], next_pos[2] - 1)
                    ]
                    
                    for nb in nbs:
                        if nb == prev_pos:
                            continue
                        idx = pos_to_idx.get(nb)
                        if idx is not None:
                            if sequence[idx] == 'H':
                                delta_e += 1
                
                new_energy = energy - delta_e
                
                new_chain = chain + [next_pos]
                new_occupied = occupied | {next_pos}
                new_pos_to_idx = pos_to_idx.copy()
                new_pos_to_idx[next_pos] = step
                
                new_beam.append((new_energy, new_chain, new_occupied, new_pos_to_idx))
        
        if not new_beam:
            best_state = min(beam, key=lambda x: x[0])
            return list(best_state[1])
            
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
        
        if step % 10 == 0 and _time_remaining() < 5.0:
            best_state = min(beam, key=lambda x: x[0])
            return list(best_state[1])

    best_state = min(beam, key=lambda x: x[0])
    return list(best_state[1])


def fold(sequence, dimension):
    """
    Fold the sequence using Beam Search.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    chain = _beam_search(sequence, dimension)
    
    return [list(p) for p in chain]
# EVOLVE-BLOCK-END
