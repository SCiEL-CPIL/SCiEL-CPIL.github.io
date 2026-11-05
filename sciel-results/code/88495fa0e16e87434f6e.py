"""
HP Lattice Protein Folding using Beam Search + Simulated Annealing with Pivot Moves.

Strategy:
1. Use beam search to explore multiple partial conformations in parallel,
   keeping the top K most promising partial folds at each step.
2. Refine using Simulated Annealing with Pivot Moves.
3. Multi-start approach to escape local minima.
4. Final polish pass with aggressive local search.
"""
# EVOLVE-BLOCK-START
import random
import time
import math
import sys
from collections import defaultdict

_DEADLINE = None
_BUDGET = 330.0
_SAFETY = 0.92

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _BUDGET * _SAFETY
    return _DEADLINE

def _is_deadline_exceeded():
    return time.time() >= _get_deadline()

def _neighbours(p):
    x, y, z = p
    return ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1))

def _compute_energy(seq, chain, pos_map, h_indices):
    """Compute energy: -number of H-H non-bonded contacts."""
    contacts = 0
    for i in h_indices:
        p = chain[i]
        for q in _neighbours(p):
            j = pos_map.get(q)
            if j is not None and j != i - 1 and j != i + 1 and j > i:
                if seq[j] == 'H':
                    contacts += 1
    return -contacts

def _beam_search(seq, dim, rng, beam_width=20, max_time=2.0):
    """Beam search for HP chain folding.
    
    Returns list of (energy, chain, pos_map) for the best beam_width complete folds.
    """
    n = len(seq)
    if n == 0:
        return []
    if n == 1:
        return [(0, [(0,0,0)], {(0,0,0): 0})]
    
    h_set = set(i for i, s in enumerate(seq) if s == 'H')
    t_start = time.time()
    
    # Each beam entry: (partial_chain, occupied_dict, current_energy)
    # Start with first residue at origin
    beam = [([(0,0,0)], {(0,0,0): 0}, 0)]
    
    for step in range(1, n):
        if _is_deadline_exceeded() or (time.time() - t_start) > max_time:
            break
        if not beam:
            break
            
        new_beam = []
        seen = set()
        
        for chain, occupied, energy in beam:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                # Check for duplicate partial chains
                key = (tuple(chain), site)
                if key in seen:
                    continue
                seen.add(key)
                
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = step
                
                # Compute incremental energy
                delta_e = 0
                if step in h_set:
                    for q in _neighbours(site):
                        j = new_occupied.get(q)
                        if j is not None and j < step - 1 and seq[j] == 'H':
                            delta_e -= 1
                
                new_energy = energy + delta_e
                new_beam.append((new_chain, new_occupied, new_energy))
        
        if not new_beam:
            break
        
        # Sort by energy and keep top beam_width
        new_beam.sort(key=lambda x: x[2])
        beam = new_beam[:beam_width]
        
        # Early termination if we have a complete fold with good energy
        if step == n - 1:
            break
    
    # If beam search didn't complete, try to finish the best partial
    if step < n - 1 and beam:
        # Complete the best partial using greedy growth
        results = []
        for chain, occupied, energy in beam[:5]:
            completed = _complete_greedy(seq, chain, occupied, h_set, rng)
            if completed:
                pos_map = {p: i for i, p in enumerate(completed)}
                e = _compute_energy(seq, completed, pos_map, h_set)
                results.append((e, completed, pos_map))
        if results:
            results.sort(key=lambda x: x[0])
            return results[:beam_width]
        return []
    
    results = []
    for chain, occupied, energy in beam:
        if len(chain) == n:
            pos_map = {p: i for i, p in enumerate(chain)}
            e = _compute_energy(seq, chain, pos_map, h_set)
            results.append((e, chain, pos_map))
    
    results.sort(key=lambda x: x[0])
    return results[:beam_width]

def _complete_greedy(seq, chain, occupied, h_set, rng):
    """Complete a partial chain using greedy growth."""
    n = len(seq)
    if len(chain) == n:
        return chain
    
    h_indices = [i for i in range(len(chain), n) if i in h_set]
    h_set_remaining = set(h_indices)
    
    for i in range(len(chain), n):
        last = chain[-1]
        candidates = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            score = 0.0
            if i in h_set_remaining:
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and seq[j] == 'H':
                        score += 3.0
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += free * 0.05
            score += rng.random() * 0.5
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: -x[0])
        if rng.random() < 0.7:
            site = candidates[0][1]
        else:
            k = min(3, len(candidates))
            site = candidates[rng.randint(0, k-1)][1]
        
        chain.append(site)
        occupied[site] = i
    
    return chain

def _apply_pivot(chain, pos_map, pivot_idx, rotation_matrix, seq):
    """
    Attempt a pivot move.
    pivot_idx: The index of the first residue in the tail to be rotated.
    rotation_matrix: A 3x3 rotation matrix (list of lists).
    Returns (success, new_chain, new_pos_map)
    """
    n = len(chain)
    if pivot_idx <= 0 or pivot_idx >= n:
        return False, None, None
    
    center = chain[pivot_idx - 1]
    
    v0 = (chain[pivot_idx][0] - center[0], 
          chain[pivot_idx][1] - center[1], 
          chain[pivot_idx][2] - center[2])
    
    new_v0 = (
        rotation_matrix[0][0]*v0[0] + rotation_matrix[0][1]*v0[1] + rotation_matrix[0][2]*v0[2],
        rotation_matrix[1][0]*v0[0] + rotation_matrix[1][1]*v0[1] + rotation_matrix[1][2]*v0[2],
        rotation_matrix[2][0]*v0[0] + rotation_matrix[2][1]*v0[1] + rotation_matrix[2][2]*v0[2]
    )
    
    new_pos0 = (center[0] + new_v0[0], center[1] + new_v0[1], center[2] + new_v0[2])
    
    if new_pos0 in pos_map:
        if pos_map[new_pos0] < pivot_idx:
            return False, None, None
    
    new_tail = [None] * (n - pivot_idx)
    
    for k in range(pivot_idx, n):
        vk = (chain[k][0] - center[0], 
              chain[k][1] - center[1], 
              chain[k][2] - center[2])
        new_vk = (
            rotation_matrix[0][0]*vk[0] + rotation_matrix[0][1]*vk[1] + rotation_matrix[0][2]*vk[2],
            rotation_matrix[1][0]*vk[0] + rotation_matrix[1][1]*vk[1] + rotation_matrix[1][2]*vk[2],
            rotation_matrix[2][0]*vk[0] + rotation_matrix[2][1]*vk[1] + rotation_matrix[2][2]*vk[2]
        )
        new_pos = (center[0] + new_vk[0], center[1] + new_vk[1], center[2] + new_vk[2])
        new_tail[k - pivot_idx] = new_pos
        
    seen_new_positions = set()
    for k in range(pivot_idx, n):
        np = new_tail[k - pivot_idx]
        if np in seen_new_positions:
            return False, None, None
        seen_new_positions.add(np)
        
        if np in pos_map:
            if pos_map[np] < pivot_idx:
                return False, None, None
    
    new_chain = list(chain)
    new_pos_map = dict(pos_map)
    
    for k in range(pivot_idx, n):
        del new_pos_map[chain[k]]
        
    for k in range(pivot_idx, n):
        new_pos_map[new_tail[k - pivot_idx]] = k
        new_chain[k] = new_tail[k - pivot_idx]
        
    return True, new_chain, new_pos_map

def _local_search(seq, chain, dim, rng, time_limit, aggressive=False):
    """Simulated Annealing with Pivot Moves."""
    n = len(seq)
    if n <= 2:
        return chain
    
    h_indices = [i for i, s in enumerate(seq) if s == 'H']
    pos_map = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(seq, chain, pos_map, h_indices)
    best_energy = current_energy
    best_chain = list(chain)
    
    if aggressive:
        start_temp = max(2.0, -current_energy * 0.8)
        end_temp = 0.005
        cooling_rate = 6.0
    else:
        start_temp = max(1.0, -current_energy * 0.5)
        end_temp = 0.01
        cooling_rate = 5.0
    
    t_start = time.time()
    t_end = t_start + time_limit
    
    rot_90_x = [[1,0,0],[0,0,-1],[0,1,0]]
    rot_90_y = [[0,0,1],[0,1,0],[-1,0,0]]
    rot_90_z = [[0,-1,0],[1,0,0],[0,0,1]]
    rot_180_x = [[1,0,0],[0,-1,0],[0,0,-1]]
    rot_180_y = [[-1,0,0],[0,1,0],[0,0,-1]]
    rot_180_z = [[-1,0,0],[0,-1,0],[0,0,1]]
    
    rotations = [rot_90_x, rot_90_y, rot_90_z, rot_180_x, rot_180_y, rot_180_z]
    
    no_improve = 0
    max_no_improve = 5000 if aggressive else 10000
    
    while time.time() < t_end and not _is_deadline_exceeded():
        elapsed = time.time() - t_start
        progress = min(1.0, elapsed / time_limit)
        temp = start_temp * math.exp(-cooling_rate * progress) + end_temp
        
        pivot_idx = rng.randint(1, n - 1)
        rot = rotations[rng.randint(0, len(rotations)-1)]
        
        success, new_chain, new_pos_map = _apply_pivot(chain, pos_map, pivot_idx, rot, seq)
        
        if not success:
            no_improve += 1
            if no_improve > max_no_improve:
                break
            continue
        
        new_energy = _compute_energy(seq, new_chain, new_pos_map, h_indices)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(temp, 1e-10)):
            chain = new_chain
            pos_map = new_pos_map
            current_energy = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
            
    return best_chain

def fold(sequence, dimension):
    """Fold HP chain on cubic lattice. Returns list of lattice points."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    rng = random.Random(42 + n)
    h_indices = [i for i, s in enumerate(sequence) if s == 'H']
    
    best_chain = None
    best_energy = float('inf')
    
    # Allocate time based on chain length
    if n <= 48:
        total_time = 3.0
    elif n <= 80:
        total_time = 4.5
    elif n <= 120:
        total_time = 6.0
    else:
        total_time = 8.0
        
    max_total = (_get_deadline() - time.time()) * 0.95
    total_time = min(total_time, max_total)
    
    t_start = time.time()
    
    # Phase 1: Beam search to get good initial solutions
    beam_width = 30 if n <= 60 else 15
    beam_time = min(total_time * 0.3, 1.5)
    beam_results = _beam_search(sequence, dimension, rng, beam_width=beam_width, max_time=beam_time)
    
    for e, chain, pos_map in beam_results:
        if e < best_energy:
            best_energy = e
            best_chain = list(chain)
    
    # Phase 2: SA refinement on beam search results
    remaining = total_time - (time.time() - t_start)
    if remaining > 0.5:
        for e, chain, pos_map in beam_results[:5]:
            if _is_deadline_exceeded():
                break
            sa_time = min(remaining * 0.2, 2.0 if n <= 48 else 3.0)
            refined = _local_search(sequence, chain, dimension, rng, sa_time)
            pos_map = {p: i for i, p in enumerate(refined)}
            new_e = _compute_energy(sequence, refined, pos_map, h_indices)
            if new_e < best_energy:
                best_energy = new_e
                best_chain = list(refined)
            remaining = total_time - (time.time() - t_start)
            if remaining < 0.5:
                break
    
    # Phase 3: Multi-start with greedy growth + SA
    remaining = total_time - (time.time() - t_start)
    num_restarts = 0
    while remaining > 1.0 and not _is_deadline_exceeded():
        num_restarts += 1
        bias = 0.5 + rng.random() * 1.5
        chain = _initial_grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue
        
        pos_map = {p: i for i, p in enumerate(chain)}
        e = _compute_energy(sequence, chain, pos_map, h_indices)
        
        if e < best_energy:
            best_energy = e
            best_chain = list(chain)
        
        remaining = total_time - (time.time() - t_start)
        if remaining < 0.5:
            break
        
        sa_time = min(remaining * 0.3, 2.0 if n <= 48 else 3.0)
        chain = _local_search(sequence, chain, dimension, rng, sa_time)
        
        pos_map = {p: i for i, p in enumerate(chain)}
        e = _compute_energy(sequence, chain, pos_map, h_indices)
        
        if e < best_energy:
            best_energy = e
            best_chain = list(chain)
            
        remaining = total_time - (time.time() - t_start)
    
    # Phase 4: Final aggressive polish
    if best_chain is not None:
        remaining = total_time - (time.time() - t_start)
        if remaining > 1.0 and not _is_deadline_exceeded():
            polish_time = min(remaining * 0.5, 3.0)
            polished = _local_search(sequence, best_chain, dimension, rng, polish_time, aggressive=True)
            pos_map = {p: i for i, p in enumerate(polished)}
            e = _compute_energy(sequence, polished, pos_map, h_indices)
            if e < best_energy:
                best_energy = e
                best_chain = list(polished)
    
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]

def _initial_grow(seq, dim, rng, bias=1.0):
    """Randomized greedy growth with H-contact bias."""
    n = len(seq)
    if n == 0:
        return []
    if n == 1:
        return [(0,0,0)]
    
    chain = [(0,0,0)]
    occupied = {(0,0,0): 0}
    h_set = {i for i, s in enumerate(seq) if s == 'H'}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            score = 0.0
            if i in h_set:
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and seq[j] == 'H':
                        score += 3.0
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += free * 0.05
            score += rng.random() * bias
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: -x[0])
        if rng.random() < 0.7:
            site = candidates[0][1]
        else:
            k = min(3, len(candidates))
            site = candidates[rng.randint(0, k-1)][1]
        
        chain.append(site)
        occupied[site] = i
    
    return chain
# EVOLVE-BLOCK-END
