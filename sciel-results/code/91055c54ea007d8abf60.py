"""HP lattice protein folding using CSP with backjumping and heuristic search."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from collections import defaultdict

# Global deadline tracking
_START_TIME = time.time()
_TOTAL_TIME_BUDGET = 185.0  # Leave 5s margin before 190s limit


def _remaining_time():
    return _TOTAL_TIME_BUDGET - (time.time() - _START_TIME)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute energy: -number of H-H non-adjacent contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _heuristic_score(position, sequence, chain, occupied, i):
    """Score a candidate position for residue i.
    Prioritize positions with high potential for future H-H contacts.
    """
    score = 0.0
    if sequence[i] == 'H':
        # Count existing H neighbors (potential immediate contacts)
        for q in _neighbours(position):
            if q in occupied:
                j = occupied[q]
                if j < i - 1 and sequence[j] == 'H':
                    score += 10.0
        # Count free neighbors (potential future contacts)
        free_neighbors = 0
        for q in _neighbours(position):
            if q not in occupied:
                free_neighbors += 1
        score += free_neighbors * 2.0
    else:
        # For P residues, prefer positions with more free space
        free_neighbors = 0
        for q in _neighbours(position):
            if q not in occupied:
                free_neighbors += 1
        score += free_neighbors * 0.5
    return score


def _backtrack_csp(sequence, dimension, rng, max_depth=None):
    """CSP-based backtracking search for protein folding.
    
    Places residues one at a time, using heuristic to guide placement.
    Returns best chain found within time budget.
    """
    n = len(sequence)
    if max_depth is None:
        max_depth = n
    
    best_chain = None
    best_energy = 0  # Start at 0 (unfolded)
    
    def backtrack(i, chain, occupied):
        nonlocal best_chain, best_energy
        
        if _remaining_time() < 2.0:
            return
        
        if i >= max_depth:
            # Evaluate partial chain
            if len(chain) > 0:
                # Complete the chain with a simple greedy extension
                complete_chain = _complete_chain(chain, occupied, sequence, i, rng)
                if complete_chain is not None:
                    energy = _compute_energy(sequence, complete_chain)
                    if energy < best_energy:
                        best_energy = energy
                        best_chain = complete_chain[:]
            return
        
        # Get candidate positions
        if i == 0:
            candidates = [(0, 0)]
        else:
            last = chain[-1]
            candidates = []
            for pos in _neighbours(last):
                if pos not in occupied:
                    candidates.append(pos)
        
        if not candidates:
            return
        
        # Score and sort candidates
        scored = []
        for pos in candidates:
            s = _heuristic_score(pos, sequence, chain, occupied, i)
            scored.append((s, pos))
        scored.sort(key=lambda x: -x[0])
        
        # Try top candidates (limit to top 3 for speed)
        for s, pos in scored[:3]:
            if _remaining_time() < 2.0:
                return
            
            chain.append(pos)
            occupied[pos] = i
            
            backtrack(i + 1, chain, occupied)
            
            chain.pop()
            del occupied[pos]
    
    # Start backtracking
    backtrack(0, [], {})
    
    return best_chain, best_energy


def _complete_chain(chain, occupied, sequence, start_i, rng):
    """Complete a partial chain using greedy growth."""
    n = len(sequence)
    if start_i >= n:
        return chain[:]
    
    for i in range(start_i, n):
        last = chain[-1]
        candidates = []
        for pos in _neighbours(last):
            if pos not in occupied:
                # Score this position
                score = 0.0
                if sequence[i] == 'H':
                    for q in _neighbours(pos):
                        if q in occupied:
                            j = occupied[q]
                            if j < i - 1 and sequence[j] == 'H':
                                score += 5.0
                # Add small random factor for diversity
                score += rng.random() * 0.1
                candidates.append((score, pos))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: -x[0])
        best_pos = candidates[0][1]
        chain.append(best_pos)
        occupied[best_pos] = i
    
    return chain


def _greedy_fold(sequence, dimension, rng):
    """Fast greedy folding for initial solution."""
    n = len(sequence)
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for pos in _neighbours(last):
            if pos not in occupied:
                score = 0.0
                if sequence[i] == 'H':
                    for q in _neighbours(pos):
                        if q in occupied:
                            j = occupied[q]
                            if j < i - 1 and sequence[j] == 'H':
                                score += 1.0
                free = sum(1 for q in _neighbours(pos) if q not in occupied)
                score += 0.1 * free + rng.random()
                candidates.append((score, pos))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: -x[0])
        best_pos = candidates[0][1]
        chain.append(best_pos)
        occupied[best_pos] = i
    
    return chain


def _local_search(sequence, dimension, chain, rng, time_limit=5.0):
    """Improve a chain using local search (move, flip, rotate)."""
    n = len(sequence)
    current_energy = _compute_energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    start_time = time.time()
    while time.time() - start_time < time_limit:
        # Try a random perturbation
        move_type = rng.choice(['move', 'flip', 'rotate'])
        
        if move_type == 'move':
            # Move a single residue to a nearby free position
            if n <= 2:
                break
            i = rng.randint(1, n - 2)  # Don't move ends
            old_pos = chain[i]
            candidates = [pos for pos in _neighbours(old_pos) if pos not in set(chain)]
            if not candidates:
                continue
            new_pos = rng.choice(candidates)
            
            # Check if move is valid (no collisions)
            test_chain = chain[:]
            test_chain[i] = new_pos
            if len(set(test_chain)) != n:
                continue
            
            # Check connectivity
            valid = True
            for j in range(1, n):
                dx = abs(test_chain[j][0] - test_chain[j-1][0])
                dy = abs(test_chain[j][1] - test_chain[j-1][1])
                if dx + dy != 1:
                    valid = False
                    break
            if not valid:
                continue
            
            test_energy = _compute_energy(sequence, test_chain)
            if test_energy < best_energy:
                best_energy = test_energy
                best_chain = test_chain[:]
        
        elif move_type == 'flip':
            # Flip a segment
            if n <= 3:
                break
            i = rng.randint(1, n - 2)
            length = rng.randint(1, min(5, n - i - 1))
            
            # Try flipping the segment by rotating it
            test_chain = chain[:]
            segment = test_chain[i:i+length+1]
            # Rotate segment 180 degrees
            for k in range(len(segment)):
                segment[k] = (-segment[k][0], -segment[k][1])
            test_chain[i:i+length+1] = segment
            
            # Check validity
            if len(set(test_chain)) != n:
                continue
            valid = True
            for j in range(1, n):
                dx = abs(test_chain[j][0] - test_chain[j-1][0])
                dy = abs(test_chain[j][1] - test_chain[j-1][1])
                if dx + dy != 1:
                    valid = False
                    break
            if not valid:
                continue
            
            test_energy = _compute_energy(sequence, test_chain)
            if test_energy < best_energy:
                best_energy = test_energy
                best_chain = test_chain[:]
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold a protein sequence into a low-energy conformation.
    
    Args:
        sequence: String of 'H' and 'P' residues
        dimension: 2 for square lattice
    
    Returns:
        List of [x, y] integer lattice points
    """
    n = len(sequence)
    rng = random.Random(42 + len(sequence))
    
    # Get initial solution via greedy
    initial_chain = _greedy_fold(sequence, dimension, rng)
    if initial_chain is None:
        # Fallback: straight line
        return [[i, 0] for i in range(n)]
    
    best_chain = initial_chain
    best_energy = _compute_energy(sequence, best_chain)
    
    # Try CSP-based search if time allows
    if _remaining_time() > 10.0 and n <= 50:
        csp_chain, csp_energy = _backtrack_csp(sequence, dimension, rng, max_depth=min(n, 15))
        if csp_chain is not None and csp_energy < best_energy:
            best_chain = csp_chain
            best_energy = csp_energy
    
    # Local search to improve
    if _remaining_time() > 5.0:
        time_limit = min(_remaining_time() - 3.0, 10.0)
        improved_chain, improved_energy = _local_search(sequence, dimension, best_chain, rng, time_limit)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
