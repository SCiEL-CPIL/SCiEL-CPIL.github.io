"""
HP lattice protein folding on the cubic lattice.
Strategy: Beam search with incremental contact tracking + local optimization.
Deadline guard for 330s wall-clock limit.
"""
import time
import random
import math
import heapq
from collections import defaultdict

# EVOLVE-BLOCK-START

_DEADLINE = 325.0  # Leave 5s margin for the 330s limit
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _neighbours(point):
    """Yield the 6 neighbours of a 3D lattice point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _compute_energy(sequence, chain):
    """Compute H-H contact energy (negative count)."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _beam_search(sequence, dimension, beam_width, rng, time_budget):
    """Beam search for folding. Returns list of (chain, energy) for top candidates."""
    n = len(sequence)
    is_h = [c == 'H' for c in sequence]
    
    start = (0,) * dimension
    start_occ = {start: 0}
    start_chain = [start]
    start_contacts = 0
    
    # Each beam entry: (negative_contacts, chain, occ_dict, contacts)
    # We sort by negative_contacts (i.e., most contacts first)
    beam = [(0, start_chain, start_occ, 0)]
    
    for i in range(1, n):
        if _time_left() < 2.0:
            break
        
        new_beam = []
        for neg_score, chain, occ, contacts in beam:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occ:
                    continue
                new_occ = dict(occ)
                new_occ[site] = i
                new_contacts = contacts
                if is_h[i]:
                    for q in _neighbours(site):
                        j = new_occ.get(q)
                        if j is not None and j < i - 1 and is_h[j]:
                            new_contacts += 1
                new_chain = chain + [site]
                new_neg = -new_contacts
                new_beam.append((new_neg, new_chain, new_occ, new_contacts))
        
        if not new_beam:
            break
        
        # Sort by score (most negative = best)
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
    
    results = []
    for neg_score, chain, occ, contacts in beam:
        energy = -contacts
        results.append((chain, energy))
    return results

def _pivot_move(chain, i, axis, sign, dimension):
    """
    Perform a pivot move at index i.
    Rotates the tail (i+1 to end) around chain[i] by 90 degrees around 'axis' with 'sign'.
    Returns new_chain if valid, else None.
    """
    n = len(chain)
    if i <= 1 or i >= n - 1:
        return None
    
    head = chain[:i+1]
    tail = chain[i+1:]
    pivot = chain[i]
    
    # Pre-compute head occupied set for fast collision check
    head_set = set(head)
    
    new_tail = []
    seen_in_tail = set()
    
    # Transformation logic for 90 degree rotation
    # We are rotating the vector (pos - pivot) around 'axis'
    # Standard rotation matrices for 90 deg:
    # Axis 0 (x): y' = -z, z' = y (for sign=1) or y' = z, z' = -y (for sign=-1)
    # Axis 1 (y): x' = z, z' = -x (for sign=1) or x' = -z, z' = x (for sign=-1)
    # Axis 2 (z): x' = -y, y' = x (for sign=1) or x' = y, y' = -x (for sign=-1)
    
    for pos in tail:
        vec = [pos[k] - pivot[k] for k in range(dimension)]
        
        if axis == 0:
            if sign == 1:
                new_vec = [vec[0], -vec[2], vec[1]]
            else:
                new_vec = [vec[0], vec[2], -vec[1]]
        elif axis == 1:
            if sign == 1:
                new_vec = [vec[2], vec[1], -vec[0]]
            else:
                new_vec = [-vec[2], vec[1], vec[0]]
        else: # axis == 2
            if sign == 1:
                new_vec = [-vec[1], vec[0], vec[2]]
            else:
                new_vec = [vec[1], -vec[0], vec[2]]
        
        new_pos = tuple(pivot[k] + new_vec[k] for k in range(dimension))
        
        if new_pos in seen_in_tail or new_pos in head_set:
            return None
        
        seen_in_tail.add(new_pos)
        new_tail.append(new_pos)
        
    return head + new_tail

def _local_optimize_sa(sequence, chain, time_budget):
    """
    Simulated Annealing optimization.
    """
    n = len(chain)
    dimension = len(chain[0])
    rng = random.Random(12345 + n)
    
    current_chain = list(chain)
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # SA Parameters
    # Tune temperature based on chain length
    T = 1.0 
    alpha = 0.995 # Cooling rate
    min_T = 0.001
    
    # Number of steps is determined by time
    while _time_left() > 1.0 and T > min_T:
        # Pick a random pivot point
        # Bias towards middle? Or uniform? Uniform is safer for global search.
        i = rng.randint(2, n - 2)
        axis = rng.randint(0, dimension - 1)
        sign = rng.choice([-1, 1])
        
        new_chain = _pivot_move(current_chain, i, axis, sign, dimension)
        
        if new_chain is None:
            # If pivot fails, try a kink move? 
            # For now, just cool down slightly faster or skip
            T *= alpha
            continue
            
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or (rng.random() < math.exp(-delta / T)):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        T *= alpha
        
        # Check time more frequently in inner loop
        if _time_left() < 1.0:
            break
            
    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold a sequence on the cubic lattice. Returns list of lattice points."""
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    # Choose beam width based on chain length
    if n <= 30:
        beam_width = 2000
    elif n <= 50:
        beam_width = 1000
    elif n <= 80:
        beam_width = 500
    else:
        beam_width = 200
    
    # Run beam search
    results = _beam_search(sequence, dimension, beam_width, rng, _time_left())
    
    if not results:
        # Fallback: straight line
        return [[i] + [0] * (dimension - 1) for i in range(n)]
    
    # Sort by energy
    results.sort(key=lambda x: x[1])
    
    # Take the best candidate from beam search
    # We could also take top 2 and run SA on both if time permits, 
    # but for simplicity and speed, we take the best one.
    best_chain = results[0][0]
    
    # Run Simulated Annealing
    # Allocate remaining time to SA
    # We already checked time in beam search, so we have some left.
    best_chain, best_energy = _local_optimize_sa(sequence, best_chain, _time_left())
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
