"""HP lattice protein folding on the cubic lattice.
Strategy: Beam Search with incremental energy and adaptive width.
"""
# EVOLVE-BLOCK-START
import time
import random

_TOTAL_DEADLINE = 325.0
_start_time = None


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _time_remaining():
    return _TOTAL_DEADLINE - _get_elapsed() - 5.0


def _beam_search(sequence, dimension, beam_width=None):
    """
    Beam search for HP lattice protein folding.
    Returns the best chain found.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    if beam_width is None:
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


def _local_search(chain, sequence, dimension, max_steps=5000):
    """
    Perform local search using kink moves and simple flips.
    """
    n = len(chain)
    if n < 4:
        return chain
    
    occupied = set(chain)
    pos_to_idx = {pos: i for i, pos in enumerate(chain)}
    
    def compute_energy(chain):
        energy = 0
        occupied = set(chain)
        for i in range(n):
            if sequence[i] == 'H':
                pos = chain[i]
                nbs = [
                    (pos[0] + 1, pos[1], pos[2]),
                    (pos[0] - 1, pos[1], pos[2]),
                    (pos[0], pos[1] + 1, pos[2]),
                    (pos[0], pos[1] - 1, pos[2]),
                    (pos[0], pos[1], pos[2] + 1),
                    (pos[0], pos[1], pos[2] - 1)
                ]
                for nb in nbs:
                    if nb in occupied:
                        # Check if nb is not adjacent in chain
                        j = pos_to_idx.get(nb)
                        if j is not None and abs(i - j) > 1:
                            if sequence[j] == 'H':
                                energy -= 1
        return energy // 2  # Divide by 2 because each contact counted twice

    current_energy = compute_energy(chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    for step in range(max_steps):
        if _time_remaining() < 2.0:
            break
        
        # Try kink moves: move residue i to a new position
        # Find a residue that can be moved
        i = random.randint(1, n - 2)
        pos = chain[i]
        nbs = [
            (pos[0] + 1, pos[1], pos[2]),
            (pos[0] - 1, pos[1], pos[2]),
            (pos[0], pos[1] + 1, pos[2]),
            (pos[0], pos[1] - 1, pos[2]),
            (pos[0], pos[1], pos[2] + 1),
            (pos[0], pos[1], pos[2] - 1)
        ]
        
        random.shuffle(nbs)
        moved = False
        for new_pos in nbs:
            # Check if new_pos is valid (not occupied, except by i itself)
            if new_pos == pos:
                continue
            if new_pos in occupied and new_pos != pos:
                continue
            # Check if move is valid (not adjacent to i-1 or i+1 in a way that breaks self-avoidance)
            # Actually, we need to ensure the new position doesn't collide with other parts of the chain
            # and that the move is a valid lattice step from the previous position
            # For simplicity, let's just try to move and check self-avoidance
            test_chain = list(chain)
            test_chain[i] = new_pos
            
            # Check self-avoidance
            test_occupied = set(test_chain)
            if len(test_occupied) != n:
                continue
            
            # Check if the move creates a valid walk (unit steps)
            valid = True
            for k in range(1, n):
                p1 = test_chain[k - 1]
                p2 = test_chain[k]
                dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])
                if dist != 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Compute new energy
            new_energy = compute_energy(test_chain)
            if new_energy < best_energy:
                best_chain = test_chain
                best_energy = new_energy
                chain = test_chain
                occupied = set(chain)
                pos_to_idx = {pos: i for i, pos in enumerate(chain)}
                current_energy = new_energy
                moved = True
                break
            elif new_energy == current_energy:
                # Accept neutral moves with some probability
                if random.random() < 0.5:
                    chain = test_chain
                    occupied = set(chain)
                    pos_to_idx = {pos: i for i, pos in enumerate(chain)}
                    current_energy = new_energy
                    moved = True
                    break
        
        if not moved:
            continue
    
    return best_chain


def fold(sequence, dimension):
    """
    Fold the sequence using Beam Search followed by local search.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    # Adaptive beam width
    if n > 100:
        beam_width = 20
    elif n > 50:
        beam_width = 30
    else:
        beam_width = 50
    
    chain = _beam_search(sequence, dimension, beam_width=beam_width)
    
    # Local search if time permits
    if _time_remaining() > 10.0 and n <= 64:
        chain = _local_search(chain, sequence, dimension, max_steps=2000)
    elif _time_remaining() > 5.0 and n <= 48:
        chain = _local_search(chain, sequence, dimension, max_steps=1000)
    
    return [list(p) for p in chain]
# EVOLVE-BLOCK-END
