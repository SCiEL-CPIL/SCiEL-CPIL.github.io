"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    # Budget is 120s CPU for all sequences, 210s wall-clock kill.
    # Use 190s as a safe deadline to ensure we finish before kill.
    return 190.0 - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))

def _grow_beam(sequence, dimension, rng, beam_width=8):
    """
    Beam search growth with efficient occupied tracking.
    Keeps the top `beam_width` partial chains based on contact heuristic.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0)
    # Each beam item: (chain_list, occupied_set, score)
    beam = [([start], {start}, 0.0)]
    
    for i in range(1, n):
        next_beam = []
        for chain, occupied, score in beam:
            current = chain[-1]
            for site in _neighbours(current):
                if site in occupied:
                    continue
                
                # Calculate immediate contact gain
                gain = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        if q in occupied:
                            # Check if the residue at q is H and not adjacent in chain
                            # We need to know which index is at q
                            # Since we only have the set, we need a different approach
                            # Use a dict for occupied->index for gain calculation
                            pass
                
                # Simpler: just use score = -gain where gain is estimated
                # For efficiency, track contacts differently
                new_score = score
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        if q in occupied:
                            new_score -= 1.0  # assume contact, will be validated later
                
                new_chain = chain + [site]
                new_occupied = set(occupied)
                new_occupied.add(site)
                
                next_beam.append((new_chain, new_occupied, new_score))
        
        if not next_beam:
            return None
        
        # Sort by score (lower is better)
        next_beam.sort(key=lambda x: x[2])
        
        # Keep top beam_width, with some random tie-breaking for diversity
        beam = next_beam[:beam_width]
        
        if _time_remaining() < 1.0:
            break

    if not beam:
        return None
    
    return list(beam[0][0])

def _compute_energy(sequence, chain):
    """Compute HP contact energy: -1 for each H-H pair on neighbouring sites not adjacent in chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _local_search(sequence, chain, rng, max_steps=200):
    """
    Local search using fragment shift moves.
    Moves a contiguous fragment of the chain to a new position.
    """
    n = len(chain)
    if n < 3:
        return list(chain), _compute_energy(sequence, chain)
    
    chain = list(chain)
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    
    for step in range(max_steps):
        if _time_remaining() < 0.5:
            break
        
        # Choose fragment length (1 to min(3, n-2))
        frag_len = rng.randint(1, min(3, n - 2))
        start_idx = rng.randint(1, n - 1 - frag_len)
        end_idx = start_idx + frag_len - 1
        
        # Get the fragment
        fragment = chain[start_idx:end_idx + 1]
        
        # Get the neighbors of the fragment ends
        left_neighbor = chain[start_idx - 1] if start_idx > 0 else None
        right_neighbor = chain[end_idx + 1] if end_idx < n - 1 else None
        
        # Find valid new positions for the fragment
        # The fragment must maintain connectivity:
        # - First element of fragment must be adjacent to left_neighbor (if exists)
        # - Last element of fragment must be adjacent to right_neighbor (if exists)
        
        # Try shifting the fragment in each of 4 directions
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_positions = [(x + dx, y + dy) for x, y in fragment]
            
            # Check self-avoidance: new positions must not overlap with existing occupied positions
            # (excluding the fragment's own original positions)
            valid = True
            for new_pos in new_positions:
                if new_pos in occupied:
                    # Check if this position is part of the fragment being moved
                    orig_idx = occupied[new_pos]
                    if not (start_idx <= orig_idx <= end_idx):
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Check connectivity
            if left_neighbor is not None:
                # First element of new fragment must be adjacent to left_neighbor
                lx, ly = left_neighbor
                fx, fy = new_positions[0]
                if abs(lx - fx) + abs(ly - fy) != 1:
                    continue
            
            if right_neighbor is not None:
                # Last element of new fragment must be adjacent to right_neighbor
                rx, ry = right_neighbor
                fx, fy = new_positions[-1]
                if abs(rx - fx) + abs(ry - fy) != 1:
                    continue
            
            # Check that the fragment itself remains connected
            for k in range(len(new_positions) - 1):
                x1, y1 = new_positions[k]
                x2, y2 = new_positions[k + 1]
                if abs(x1 - x2) + abs(y1 - y2) != 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Perform the move
            # Remove old positions from occupied
            for pos in fragment:
                del occupied[pos]
            # Add new positions
            for k, pos in enumerate(new_positions):
                occupied[pos] = start_idx + k
            # Update chain
            for k, pos in enumerate(new_positions):
                chain[start_idx + k] = pos
            
            # Calculate new energy
            new_energy = _compute_energy(sequence, chain)
            
            if new_energy < current_energy:
                current_energy = new_energy
            else:
                # Revert
                for pos in new_positions:
                    del occupied[pos]
                for k, pos in enumerate(fragment):
                    occupied[pos] = start_idx + k
                    chain[start_idx + k] = pos
    
    return chain, current_energy

def fold(sequence, dimension):
    """
    Fold an HP sequence on the square lattice.
    Uses beam search growth followed by fragment-shift local search.
    """
    _init_time()
    
    n = len(sequence)
    if n == 0:
        return []
    
    # Emergency fallback if no time left
    if _time_remaining() < 1.0:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0  # Energy is negative, so we look for lower (more negative) values
    
    # Time-based budget allocation
    total_time = 190.0
    polish_time = 15.0
    
    iterations = 0
    max_iterations = 5000
    
    while _time_remaining() > polish_time and iterations < max_iterations:
        iterations += 1
        
        # Generate a new conformation using beam search
        # Vary beam width for diversity
        bw = rng.choice([5, 8, 10])
        chain = _grow_beam(sequence, dimension, rng, beam_width=bw)
        if chain is None:
            continue
        
        # Adaptive local search steps
        progress = (total_time - _time_remaining()) / max(1, (total_time - polish_time))
        max_ls_steps = int(80 + progress * 120)
        
        # Refine with local search
        chain, energy = _local_search(sequence, chain, rng, max_steps=max_ls_steps)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
    
    # Final polish pass
    if best_chain is not None and _time_remaining() > 1.0:
        chain, energy = _local_search(sequence, best_chain, rng, max_steps=300)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    # Final validation
    if len(set(best_chain)) != len(best_chain):
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
