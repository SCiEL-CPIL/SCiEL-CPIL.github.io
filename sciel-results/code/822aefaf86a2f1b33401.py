"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _beam_fold(sequence, dimension, beam_width=500):
    """Beam search over partial conformations.
    
    Each beam entry is a tuple: (neg_energy, chain, occupied_dict)
    where occupied_dict maps position -> residue index.
    """
    n = len(sequence)
    start = (0,) * dimension
    
    # Initial state
    initial_chain = [start]
    initial_occupied = {start: 0}
    
    # Beam: list of (neg_energy, chain, occupied)
    beam = [(0, initial_chain, initial_occupied)]
    
    for i in range(1, n):
        new_beam = []
        
        for neg_energy, chain, occupied in beam:
            last_pos = chain[-1]
            
            for site in _neighbours(last_pos):
                if site in occupied:
                    continue
                
                # Compute energy gain
                gain = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        if q in occupied:
                            j = occupied[q]
                            if j < i - 1 and sequence[j] == 'H':
                                gain += 1
                
                # New neg_energy (more negative = better)
                new_neg_energy = neg_energy - gain
                
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                
                new_beam.append((new_neg_energy, new_chain, new_occupied))
        
        if not new_beam:
            break
        
        # Keep top beam_width states
        new_beam.sort(key=lambda x: x[0], reverse=True)
        beam = new_beam[:beam_width]
    
    # Find best complete chain
    best_chain = None
    best_neg_energy = float('-inf')
    
    for neg_energy, chain, occupied in beam:
        if len(chain) == n:
            if neg_energy > best_neg_energy:
                best_neg_energy = neg_energy
                best_chain = chain
    
    return best_chain


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _greedy_fold(sequence, dimension):
    """Fallback greedy growth."""
    rng = random.Random(42 + len(sequence))
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied.get(q, -1) < i - 1 and sequence[occupied.get(q, -1)] == 'H')
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        
        if not options:
            return None
        
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    
    return chain


def fold(sequence, dimension):
    """Fold a protein sequence using beam search with time management."""
    start_time = time.time()
    deadline = start_time + 320  # Leave 10s safety margin for 330s limit
    
    n = len(sequence)
    
    # Adaptive beam width based on sequence length
    if n <= 50:
        beam_width = 500
    elif n <= 100:
        beam_width = 300
    else:
        beam_width = 150
    
    # Run beam search
    best_chain = _beam_fold(sequence, dimension, beam_width=beam_width)
    
    # If beam search didn't find a complete chain, try greedy fallback
    if best_chain is None or len(best_chain) != n:
        best_chain = _greedy_fold(sequence, dimension)
    
    # If still no valid chain, use straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
