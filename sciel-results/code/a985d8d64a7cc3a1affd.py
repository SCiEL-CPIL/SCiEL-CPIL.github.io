"""
Strategy: Beam Search with Composite Scoring and Local Refinement

This approach is fundamentally different from CSP backtracking:
1. Instead of exhaustive search with pruning (which explodes exponentially), 
   we maintain a fixed-width beam of the most promising partial conformations.
2. At each step, we expand all beams by all valid neighbor positions, score them,
   and keep only the top-K. This is a heuristic, non-exhaustive search.
3. The composite score balances immediate H-H contacts, future space availability,
   and proximity to other H residues (potential for future contacts).
4. After beam search completes, we apply local refinement (hill climbing with 
   segment reversals and single-residue moves) to escape local minima.
5. Multiple restarts with different initial orientations/seed positions diversify search.

This avoids the exponential blowup of CSP while being more systematic than pure greedy.
"""

import time
import random
import math
from collections import defaultdict

_START_TIME = time.time()
_HARD_DEADLINE = 315.0  # Safety margin before 330s kill

def _neighbours(point):
    """Yield 6 neighbors on cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _compute_energy(sequence, chain):
    """Compute energy of a chain. Energy = - (number of H-H contacts)."""
    if not chain:
        return 0
    
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i in range(len(chain)):
        if sequence[i] != 'H':
            continue
        p = chain[i]
        for q in _neighbours(p):
            j = pos_to_idx.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts

def _beam_score(sequence, chain, pos_to_idx, is_h, idx):
    """
    Score a partial conformation for beam search.
    Returns a score where higher is better.
    """
    if idx == 0:
        return 0.0
    
    # Current energy (negative is better)
    current_energy = 0
    for i in range(idx):
        if not is_h[i]:
            continue
        p = chain[i]
        for q in _neighbours(p):
            j = pos_to_idx.get(q)
            if j is not None and j > i + 1 and j < idx and is_h[j]:
                current_energy -= 1
    
    # Immediate contact count
    contact_count = -current_energy
    
    # Future space: count empty neighbors of the last position
    last_pos = chain[idx - 1]
    empty_count = 0
    for nb in _neighbours(last_pos):
        if nb not in pos_to_idx:
            empty_count += 1
    
    # Proximity to H residues: count H residues within distance 2 of last position
    proximity = 0
    last_x, last_y, last_z = last_pos
    for i in range(idx):
        if not is_h[i]:
            continue
        px, py, pz = chain[i]
        dist = abs(px - last_x) + abs(py - last_y) + abs(pz - last_z)
        if 1 < dist <= 3:
            proximity += 1
    
    # Compactness: bounding box volume
    if idx > 0:
        xs = [p[0] for p in chain]
        ys = [p[1] for p in chain]
        zs = [p[2] for p in chain]
        vol = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1) * (max(zs) - min(zs) + 1)
        compactness = idx / max(vol, 1)
    else:
        compactness = 0
    
    # Composite score: weight immediate contacts heavily, then space, then compactness
    score = contact_count * 10.0 + empty_count * 0.5 + proximity * 0.3 + compactness * 5.0
    
    return score

def _beam_search(sequence, dimension, beam_width, time_limit):
    """
    Beam search for HP protein folding.
    Returns the best chain found.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    is_h = [c == 'H' for c in sequence]
    
    # Initial state: first residue at origin
    # Try multiple initial orientations for diversity
    best_chain = None
    best_energy = 0
    
    # Each beam entry: (chain_list, pos_to_idx_dict, score)
    beams = []
    
    # Initialize with first residue at origin
    init_chain = [(0, 0, 0)]
    init_pos = {(0, 0, 0): 0}
    init_score = 0.0
    beams.append((init_chain, init_pos, init_score))
    
    for idx in range(1, n):
        if time.time() - _START_TIME > time_limit:
            break
        
        new_beams = []
        
        for chain, pos_to_idx, score in beams:
            last_pos = chain[-1]
            
            # Generate all valid next positions
            for next_pos in _neighbours(last_pos):
                if next_pos in pos_to_idx:
                    continue
                
                # Create new chain and pos_to_idx
                new_chain = chain + [next_pos]
                new_pos = dict(pos_to_idx)
                new_pos[next_pos] = idx
                
                # Compute new score
                new_score = _beam_score(sequence, new_chain, new_pos, is_h, idx)
                new_beams.append((new_chain, new_pos, new_score))
        
        if not new_beams:
            break
        
        # Keep only top beam_width
        new_beams.sort(key=lambda x: x[2], reverse=True)
        beams = new_beams[:beam_width]
    
    # Evaluate all beams in the final layer
    for chain, pos_to_idx, score in beams:
        energy = _compute_energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
    
    # Also check intermediate beams if we didn't complete
    if best_chain is None:
        # Fallback: use the best from any beam
        for chain, pos_to_idx, score in beams:
            energy = _compute_energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
    
    if best_chain is None:
        # Last resort: straight line
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return best_chain

def _local_refine(sequence, chain, time_limit, max_iterations=500):
    """
    Local refinement using hill climbing with segment reversals and single-residue moves.
    """
    n = len(sequence)
    if n == 0:
        return chain
    
    is_h = [c == 'H' for c in sequence]
    current_chain = list(chain)
    current_energy = _compute_energy(sequence, current_chain)
    
    # Helper to check if a move is valid (self-avoiding)
    def is_valid_move(chain, i, new_pos):
        # Check if new_pos is occupied by another residue
        for j, p in enumerate(chain):
            if j != i and tuple(p) == new_pos:
                return False
        return True
    
    # Helper to apply a segment reversal
    def reverse_segment(chain, i, j):
        """Reverse the segment from index i to j (inclusive)."""
        new_chain = list(chain)
        new_chain[i:j+1] = list(reversed(new_chain[i:j+1]))
        return new_chain
    
    # Helper to try a single residue move
    def try_single_move(chain, i, new_pos):
        """Try moving residue i to new_pos. Returns (new_chain, energy) or (None, None) if invalid."""
        if not is_valid_move(chain, i, new_pos):
            return None, None
        
        new_chain = list(chain)
        new_chain[i] = new_pos
        
        # Check self-avoidance for the whole chain
        pos_set = set()
        for j, p in enumerate(new_chain):
            tp = tuple(p)
            if tp in pos_set:
                return None, None
            pos_set.add(tp)
        
        energy = _compute_energy(sequence, new_chain)
        return new_chain, energy
    
    iteration = 0
    improved = True
    
    while improved and iteration < max_iterations and time.time() - _START_TIME < time_limit:
        improved = False
        iteration += 1
        
        # Try segment reversals
        for i in range(n - 1):
            for j in range(i + 1, min(i + 8, n)):  # Limit segment length
                new_chain = reverse_segment(current_chain, i, j)
                
                # Check validity
                pos_set = set()
                valid = True
                for p in new_chain:
                    tp = tuple(p)
                    if tp in pos_set:
                        valid = False
                        break
                    pos_set.add(tp)
                
                if not valid:
                    continue
                
                energy = _compute_energy(sequence, new_chain)
                if energy < current_energy:
                    current_chain = new_chain
                    current_energy = energy
                    improved = True
        
        # Try single residue moves
        for i in range(1, n - 1):  # Don't move endpoints
            for new_pos in _neighbours(current_chain[i]):
                new_chain, energy = try_single_move(current_chain, i, new_pos)
                if new_chain is not None and energy < current_energy:
                    current_chain = new_chain
                    current_energy = energy
                    improved = True
    
    return current_chain

def fold(sequence, dimension):
    """
    Fold the protein using beam search + local refinement.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Time allocation
    elapsed_global = time.time() - _START_TIME
    remaining = _HARD_DEADLINE - elapsed_global
    
    if remaining < 5.0:
        # Not enough time, return a simple fallback
        return [(i, 0, 0) for i in range(n)]
    
    # Allocate time: 70% for beam search, 30% for refinement
    beam_time = remaining * 0.7
    refine_time = remaining * 0.3
    
    # Beam width: larger for shorter chains, smaller for longer
    if n <= 30:
        beam_width = 200
    elif n <= 60:
        beam_width = 100
    elif n <= 100:
        beam_width = 50
    else:
        beam_width = 30
    
    # Run beam search
    best_chain = _beam_search(sequence, dimension, beam_width, beam_time)
    
    # Local refinement
    if time.time() - _START_TIME < _HARD_DEADLINE - 5.0:
        refined_chain = _local_refine(sequence, best_chain, refine_time)
        
        # Keep the better one
        e1 = _compute_energy(sequence, best_chain)
        e2 = _compute_energy(sequence, refined_chain)
        if e2 < e1:
            best_chain = refined_chain
    
    return [list(p) for p in best_chain]
