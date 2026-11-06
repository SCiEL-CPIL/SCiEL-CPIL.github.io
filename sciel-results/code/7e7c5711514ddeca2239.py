"""HP lattice protein folding using multi-strategy growth with local search refinement."""
# EVOLVE-BLOCK-START
import random
import time
import math


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute energy as negative number of H-H contacts (non-adjacent in chain)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _contact_priority(sequence):
    """Build priority for H-H pairs based on sequence distance (closer = higher priority)."""
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    priority = {}
    for i, a in enumerate(h_indices):
        for j in range(i + 1, len(h_indices)):
            b = h_indices[j]
            dist = abs(a - b)
            # Closer in sequence gets higher priority
            if dist < 20:
                priority[(a, b)] = 1.0 / dist
    return priority


def _grow_contact_guided(sequence, dimension, rng, priority):
    """Growth that prioritizes satisfying high-priority H-H contacts."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            gain = 0
            if sequence[i] == 'H':
                # Count new H-H contacts with previously placed H residues
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        # Bonus for high-priority pairs
                        pair_key = (min(j, i), max(j, i))
                        gain += 1.0 + priority.get(pair_key, 0.0)
            
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain + 0.05 * free + rng.random() * 0.3
            options.append((score, site))
        
        if not options:
            return None
        
        site = max(options, key=lambda x: x[0])[1]
        occupied[site] = i
        chain.append(site)
    
    return chain


def _grow_random(sequence, dimension, rng):
    """Simple random growth."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                options.append((free + rng.random(), site))
        
        if not options:
            return None
        
        site = max(options, key=lambda x: x[0])[1]
        occupied[site] = i
        chain.append(site)
    
    return chain


def _grow_spiral(sequence, dimension, rng):
    """Spiral-like growth for compact conformations."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dir_idx = 0
    steps_in_dir = 0
    max_steps = 1
    
    for i in range(1, len(sequence)):
        placed = False
        for attempt in range(4):
            dx, dy = directions[dir_idx % 4]
            site = (chain[-1][0] + dx, chain[-1][1] + dy)
            
            if site not in occupied:
                occupied[site] = i
                chain.append(site)
                placed = True
                steps_in_dir += 1
                if steps_in_dir >= max_steps:
                    dir_idx += 1
                    steps_in_dir = 0
                    if dir_idx % 2 == 0:
                        max_steps += 1
                break
        
        if not placed:
            # Try all directions
            for d in directions:
                site = (chain[-1][0] + d[0], chain[-1][1] + d[1])
                if site not in occupied:
                    occupied[site] = i
                    chain.append(site)
                    placed = True
                    break
        
        if not placed:
            return None
    
    return chain


def _local_search(sequence, chain, rng, max_iters=500):
    """Improve conformation using pivot moves and k-reversals."""
    if chain is None:
        return None
    
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    n = len(sequence)
    
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    for _ in range(max_iters):
        # Try a random move
        move_type = rng.random()
        
        if move_type < 0.6:
            # Pivot move: rotate a segment
            if n < 4:
                continue
            pivot = rng.randint(1, n - 3)
            # Try rotating the segment from pivot+1 to some endpoint
            seg_len = min(rng.randint(1, n - pivot - 1), 4)
            if seg_len < 2:
                continue
            
            # Get the segment to rotate
            segment = best_chain[pivot + 1:pivot + 1 + seg_len]
            if len(segment) < 2:
                continue
            
            # Try 3 rotations (90, 180, 270 degrees)
            for rot in range(1, 4):
                # Rotate segment around pivot point
                cx, cy = best_chain[pivot]
                new_segment = []
                for (x, y) in segment:
                    rx, ry = x - cx, y - cy
                    if rot == 1:
                        new_segment.append((cx + (-ry), cy + rx))
                    elif rot == 2:
                        new_segment.append((cx - rx, cy - ry))
                    else:
                        new_segment.append((cx + ry, cy - rx))
                
                # Check if new positions are valid
                valid = True
                for idx, pos in enumerate(new_segment):
                    target_idx = pivot + 1 + idx
                    if pos in occupied and occupied[pos] != target_idx:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Compute energy change
                old_energies = {}
                for idx, pos in enumerate(segment):
                    target_idx = pivot + 1 + idx
                    if sequence[target_idx] == 'H':
                        for q in _neighbours(pos):
                            j = occupied.get(q)
                            if j is not None and j > target_idx + 1 and sequence[j] == 'H':
                                old_energies[(target_idx, j)] = 1
                
                # Apply move
                for idx, pos in enumerate(new_segment):
                    target_idx = pivot + 1 + idx
                    occupied.pop(best_chain[target_idx])
                    best_chain[target_idx] = pos
                    occupied[pos] = target_idx
                
                new_energy = _compute_energy(sequence, best_chain)
                
                if new_energy <= best_energy:
                    best_energy = new_energy
                    break
                else:
                    # Revert
                    for idx, pos in enumerate(segment):
                        target_idx = pivot + 1 + idx
                        occupied.pop(best_chain[target_idx])
                        best_chain[target_idx] = pos
                        occupied[pos] = target_idx
        
        else:
            # k-reversal: reverse a segment of the chain
            if n < 4:
                continue
            k = rng.randint(2, min(6, n - 2))
            start_idx = rng.randint(1, n - k - 1)
            end_idx = start_idx + k
            
            # Check if reversal is valid (no self-intersection)
            segment = best_chain[start_idx:end_idx + 1]
            reversed_segment = list(reversed(segment))
            
            valid = True
            for idx, pos in enumerate(reversed_segment):
                target_idx = start_idx + idx
                if pos in occupied and occupied[pos] != target_idx:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Apply reversal
            for idx, pos in enumerate(reversed_segment):
                target_idx = start_idx + idx
                occupied.pop(best_chain[target_idx])
                best_chain[target_idx] = pos
                occupied[pos] = target_idx
            
            new_energy = _compute_energy(sequence, best_chain)
            
            if new_energy <= best_energy:
                best_energy = new_energy
            else:
                # Revert
                for idx, pos in enumerate(segment):
                    target_idx = start_idx + idx
                    occupied.pop(best_chain[target_idx])
                    best_chain[target_idx] = pos
                    occupied[pos] = target_idx
    
    return best_chain


def fold(sequence, dimension):
    """Fold sequence into low-energy conformation using multi-strategy approach."""
    start_time = time.time()
    deadline = start_time + 180  # Safety margin before 210s kill
    
    n = len(sequence)
    priority = _contact_priority(sequence)
    
    best_chain = None
    best_energy = 0
    
    # Generate initial conformations using different strategies
    initial_chains = []
    
    rng1 = random.Random(42 + n)
    chain1 = _grow_contact_guided(sequence, dimension, rng1, priority)
    if chain1:
        initial_chains.append(chain1)
    
    rng2 = random.Random(123 + n)
    for _ in range(10):
        chain = _grow_random(sequence, dimension, rng2)
        if chain:
            initial_chains.append(chain)
    
    rng3 = random.Random(777 + n)
    chain3 = _grow_spiral(sequence, dimension, rng3)
    if chain3:
        initial_chains.append(chain3)
    
    # If no chains generated, use a straight line
    if not initial_chains:
        initial_chains.append([tuple([i] + [0] * (dimension - 1)) for i in range(n)])
    
    # Evaluate and keep top candidates
    candidates = []
    for chain in initial_chains:
        energy = _compute_energy(sequence, chain)
        candidates.append((energy, chain))
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    # Sort by energy and keep top 5 for local search
    candidates.sort(key=lambda x: x[0])
    top_candidates = [c[1] for c in candidates[:5]]
    
    # Apply local search to each candidate
    for chain in top_candidates:
        if time.time() > deadline - 5:
            break
        
        rng = random.Random(int(time.time() * 1000) % 100000)
        improved = _local_search(sequence, chain, rng, max_iters=200)
        
        if improved:
            energy = _compute_energy(sequence, improved)
            if energy < best_energy:
                best_energy = energy
                best_chain = improved
    
    # Final fallback
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
