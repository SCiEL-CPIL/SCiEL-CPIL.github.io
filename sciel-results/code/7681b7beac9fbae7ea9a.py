import random
import time
import sys

# Global deadline
_DEADLINE = None
_START_TIME = None

def _init_deadline():
    global _DEADLINE, _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    if _DEADLINE is None:
        _DEADLINE = _START_TIME + 290  # 290s safety margin before 300s kill

def _time_left():
    _init_deadline()
    return _DEADLINE - time.time()


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute energy of a chain configuration."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _greedy_seed(sequence, dimension, rng):
    """Generate a good initial seed using greedy growth."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        best_site = None
        best_score = -999999
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            # Score: prefer H-H contacts, then free space
            score = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            score += 10
            # Free space bonus
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += 0.1 * free
            score += rng.random() * 0.5
            
            if score > best_score:
                best_score = score
                best_site = site
        
        if best_site is None:
            return None
        
        occupied[best_site] = i
        chain.append(best_site)
    
    return chain


def _sa_fold(sequence, dimension, rng, max_iters):
    """Simulated annealing for a single restart."""
    # Generate initial seed
    chain = _greedy_seed(sequence, dimension, rng)
    if chain is None:
        return None, 0
    
    n = len(sequence)
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = current_energy
    
    # SA parameters
    T_start = 2.0
    T_end = 0.05
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))
    T = T_start
    
    for it in range(max_iters):
        if _time_left() < 2.0:
            break
        
        # Choose a move type
        move_type = rng.random()
        
        if move_type < 0.7:
            # End Move: Move the last residue
            idx = n - 1
            old_pos = chain[idx]
            candidates = []
            for new_pos in _neighbours(old_pos):
                if new_pos == chain[idx-1]:
                    continue
                if new_pos in occupied and occupied[new_pos] != idx:
                    continue
                # Check self-avoidance with other residues (already handled by occupied check mostly, 
                # but occupied contains all residues. If new_pos is in occupied and not idx, it's blocked.
                # If new_pos is NOT in occupied, it's free.
                # So the check above is sufficient for self-avoidance if occupied is accurate.
                candidates.append(new_pos)
            
            if not candidates:
                continue
            
            new_pos = rng.choice(candidates)
            
            # Compute energy change
            delta = 0
            if sequence[idx] == 'H':
                # Remove contacts from old position
                for q in _neighbours(old_pos):
                    if q in occupied:
                        j = occupied[q]
                        if j != idx and j < idx - 1 and sequence[j] == 'H':
                            delta += 1  # removing a contact increases energy
                # Add contacts at new position
                for q in _neighbours(new_pos):
                    if q in occupied:
                        j = occupied[q]
                        if j != idx and j < idx - 1 and sequence[j] == 'H':
                            delta -= 1  # adding a contact decreases energy
            
            new_energy = current_energy + delta
            
            # Metropolis criterion
            if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
                # Accept move
                del occupied[old_pos]
                chain[idx] = new_pos
                occupied[new_pos] = idx
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
        
        else:
            # Pivot Move: Rotate a segment of length 2 around a pivot
            if n < 4:
                continue
            
            # Choose pivot index i (1 to n-3) so segment i+1, i+2 exists
            pivot_idx = rng.randint(1, n - 3)
            
            # Segment to move: indices pivot_idx+1, pivot_idx+2
            seg_start = pivot_idx + 1
            seg_end = pivot_idx + 2
            
            p_pivot = chain[pivot_idx]
            p_seg0 = chain[seg_start]
            p_seg1 = chain[seg_end]
            
            # Remove segment from occupied
            del occupied[p_seg0]
            del occupied[p_seg1]
            
            # Find valid new positions for the segment
            candidates = []
            for new_p0 in _neighbours(p_pivot):
                # new_p0 cannot be the pivot's other neighbor (chain[pivot_idx-1])
                if new_p0 == chain[pivot_idx - 1]:
                    continue
                # new_p0 must be free (not in occupied)
                if new_p0 in occupied:
                    continue
                
                for new_p1 in _neighbours(new_p0):
                    # new_p1 must be free
                    if new_p1 in occupied:
                        continue
                    # new_p1 cannot be new_p0
                    if new_p1 == new_p0:
                        continue
                    # Check if this is the same position as before (skip no-ops)
                    if new_p0 == p_seg0 and new_p1 == p_seg1:
                        continue
                    
                    # Check collision with the rest of the chain
                    # occupied currently contains all residues EXCEPT seg0 and seg1
                    # So if new_p0 or new_p1 are in occupied, it's a collision.
                    # We already checked `if new_p0 in occupied` and `if new_p1 in occupied`.
                    # So if we pass those checks, they are free.
                    
                    candidates.append((new_p0, new_p1))
            
            # Restore segment to occupied if we don't accept a move
            # But we need to try a candidate first.
            if candidates:
                new_p0, new_p1 = rng.choice(candidates)
                
                # Compute energy delta
                delta = 0
                # Residue at seg_start (H or P)
                if sequence[seg_start] == 'H':
                    # Remove contacts from old p_seg0
                    for q in _neighbours(p_seg0):
                        if q in occupied:
                            j = occupied[q]
                            if j != seg_start and j != seg_end and j < seg_start - 1 and sequence[j] == 'H':
                                delta += 1
                    # Add contacts at new_p0
                    for q in _neighbours(new_p0):
                        if q in occupied:
                            j = occupied[q]
                            if j != seg_start and j != seg_end and j < seg_start - 1 and sequence[j] == 'H':
                                delta -= 1
                        elif q == new_p1 and sequence[seg_end] == 'H':
                            # Contact with the other part of the segment
                            # This contact is internal to the moved segment, so it doesn't change 
                            # relative to the rest of the chain, but we need to be careful.
                            # Actually, energy is defined by H-H contacts on the lattice.
                            # The contact between seg_start and seg_end exists in both old and new config 
                            # IF they are neighbors. 
                            # In old config: p_seg0 and p_seg1 are neighbors.
                            # In new config: new_p0 and new_p1 are neighbors.
                            # So this internal contact is constant. We don't need to count it in delta.
                            pass
                
                # Residue at seg_end (H or P)
                if sequence[seg_end] == 'H':
                    # Remove contacts from old p_seg1
                    for q in _neighbours(p_seg1):
                        if q in occupied:
                            j = occupied[q]
                            if j != seg_start and j != seg_end and j < seg_start - 1 and sequence[j] == 'H':
                                delta += 1
                    # Add contacts at new_p1
                    for q in _neighbours(new_p1):
                        if q in occupied:
                            j = occupied[q]
                            if j != seg_start and j != seg_end and j < seg_start - 1 and sequence[j] == 'H':
                                delta -= 1
                
                new_energy = current_energy + delta
                
                if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
                    # Accept move
                    chain[seg_start] = new_p0
                    chain[seg_end] = new_p1
                    occupied[new_p0] = seg_start
                    occupied[new_p1] = seg_end
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                else:
                    # Reject move, restore occupied (already restored by not deleting? 
                    # No, we deleted them. We must put them back.)
                    occupied[p_seg0] = seg_start
                    occupied[p_seg1] = seg_end
            else:
                # No candidates, restore occupied
                occupied[p_seg0] = seg_start
                occupied[p_seg1] = seg_end
        
        T *= cooling
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold a protein sequence using simulated annealing with multiple restarts."""
    _init_deadline()
    
    n = len(sequence)
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = 0  # Initialize to 0 (unfolded) or 1? Energy is negative. 0 is better than -100.
    
    # Determine number of restarts based on time budget
    # Each restart with SA takes roughly O(n * iters) time
    # Let's do multiple restarts with varying SA iteration counts
    
    # First, get a good greedy seed
    greedy = _greedy_seed(sequence, dimension, rng)
    if greedy is not None:
        e = _compute_energy(sequence, greedy)
        best_chain = list(greedy)
        best_energy = e
    
    # Now do SA restarts
    while _time_left() > 5.0:
        # Determine iterations based on remaining time and sequence length
        time_per_restart = _time_left() * 0.4
        iters = int(time_per_restart * 1e6 / max(1, n))
        iters = max(100, min(iters, 50000))
        
        chain, energy = _sa_fold(sequence, dimension, rng, iters)
        
        if chain is not None and energy < best_energy:
            best_chain = list(chain)
            best_energy = energy
        
        # Check if we've found a very good solution
        if best_energy <= -n // 2:
            break
    
    if best_chain is None:
        # Fallback: straight line (might be invalid if N > space, but N is small)
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
