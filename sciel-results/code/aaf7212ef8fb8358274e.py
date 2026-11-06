"""HP lattice protein folding on the square lattice using advanced local search."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline guard
START_TIME = time.time()
TIME_LIMIT = 200.0  # Safety margin for 210s limit

def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def _neighbours(point):
    """Yield the 4 neighbors of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _energy(sequence, chain):
    """Calculate the energy of a conformation.
    Energy = - (number of non-consecutive H-H contacts)
    """
    if not chain:
        return 0
    
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
        
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                    if i < j:
                        contacts += 1
    return -contacts

def _grow_greedy(sequence, rng, bias=0.5):
    """Generate an initial conformation using greedy growth with H-bias."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    h_indices = {i for i, res in enumerate(sequence) if res == 'H'}
    
    for i in range(1, len(sequence)):
        last_pos = chain[-1]
        options = []
        
        for site in _neighbours(last_pos):
            if site in occupied:
                continue
            
            score = 0.0
            # Bias towards H-H contacts
            if i in h_indices:
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j in h_indices and abs(i - j) > 1:
                            score += 10.0
            
            # Small random noise
            score += rng.random() * bias
            
            options.append((score, site))
        
        if not options:
            return None
            
        # Pick the best option
        best_site = max(options, key=lambda x: x[0])[1]
        occupied[best_site] = i
        chain.append(best_site)
        
    return chain

def _apply_slither(chain, rng, occupied):
    """Attempt a slither move: move the last residue to a free neighbor."""
    n = len(chain)
    if n < 2:
        return None
    
    last_idx = n - 1
    last_pos = chain[last_idx]
    
    # Remove last pos from occupied
    occupied.pop(last_pos)
    
    # Try moving to a neighbor
    candidates = []
    for site in _neighbours(last_pos):
        if site not in occupied:
            # Check if it connects to the rest of the chain (it does, by definition of slither)
            # But we must ensure it doesn't collide with anything else
            candidates.append(site)
    
    if not candidates:
        # Put it back
        occupied[last_pos] = last_idx
        return None
        
    new_pos = rng.choice(candidates)
    occupied[new_pos] = last_idx
    chain[last_idx] = new_pos
    return chain

def _apply_pivot(chain, rng, occupied):
    """Attempt a pivot move: rotate a segment around a pivot point."""
    n = len(chain)
    if n < 4:
        return None
    
    # Choose a pivot index (1 to n-2)
    pivot_idx = rng.randint(1, n - 2)
    pivot_pos = chain[pivot_idx]
    
    # Choose a segment length to rotate (from pivot+1 to some end)
    # Let's rotate the segment from pivot_idx+1 to a random end j
    max_j = n - 1
    min_j = pivot_idx + 1
    if min_j >= max_j:
        return None
        
    j = rng.randint(min_j, max_j)
    
    # The segment is chain[pivot_idx+1 : j+1]
    # We need to rotate this segment around pivot_pos
    # This is complex to implement directly with just positions.
    # A simpler "Pivot" in SAW literature often means rotating the entire tail 
    # or a sub-chain such that it stays connected.
    
    # Let's try a simpler move: "Crystallization" or "Reversal" is often more effective.
    # Let's implement a proper Reversal move which is robust.
    return None # Placeholder, we will use Reversal instead

def _apply_reversal(chain, rng, occupied):
    """Attempt a reversal move: reverse a segment of the chain."""
    n = len(chain)
    if n < 4:
        return None
    
    # Choose two indices i < j
    i = rng.randint(0, n - 3)
    j = rng.randint(i + 2, n - 1)
    
    # The segment to reverse is chain[i+1 : j+1]
    # We keep chain[0..i] and chain[j+1..n-1] fixed.
    # The new chain will be chain[0..i] + reversed(chain[i+1..j]) + chain[j+1..n-1]
    
    # Check validity of the new connections:
    # 1. chain[i] to chain[j] (new first of reversed segment)
    # 2. chain[i+1] to chain[j+1] (new last of reversed segment, if j+1 < n)
    
    # Actually, the standard reversal in SAW is:
    # Original: ... A - B - C - D ...
    # Reverse B-C: ... A - C - B - D ...
    # We need to check if A-C is a valid step and C-B is a valid step (it is, same as B-C)
    # And if B-D is a valid step (it is, same as C-D? No, B is now connected to D).
    # Wait, the connections are:
    # Old: A-B, B-C, C-D
    # New: A-C, C-B, B-D
    # We need to check:
    # 1. A and C are adjacent.
    # 2. B and D are adjacent.
    # 3. No self-intersection in the new segment.
    
    a = chain[i]
    b = chain[i+1]
    c = chain[j]
    d = chain[j+1] if j+1 < n else None
    
    # Check adjacency A-C
    if abs(a[0] - c[0]) + abs(a[1] - c[1]) != 1:
        return None
        
    # Check adjacency B-D
    if d is not None:
        if abs(b[0] - d[0]) + abs(b[1] - d[1]) != 1:
            return None
    else:
        # If D doesn't exist, B becomes the new end. No adjacency check needed for D.
        pass
        
    # Now check self-avoidance for the reversed segment.
    # The reversed segment is chain[j], chain[j-1], ..., chain[i+1]
    # We need to ensure none of these positions collide with the fixed parts.
    
    # Remove the segment from occupied
    segment_indices = list(range(i+1, j+1))
    segment_positions = [chain[k] for k in segment_indices]
    
    for k in segment_indices:
        del occupied[chain[k]]
        
    # Check if any position in the reversed segment collides with the rest
    # The rest is chain[0..i] and chain[j+1..n-1]
    # We already checked the boundary connections.
    # We just need to check if any position in the reversed segment is in the remaining occupied set.
    
    for pos in segment_positions:
        if pos in occupied:
            # Collision! Restore and fail
            for k in segment_indices:
                occupied[chain[k]] = k
            return None
            
    # No collision. Apply the move.
    # Update chain
    new_segment = segment_positions[::-1]
    for k, pos in zip(segment_indices, new_segment):
        chain[k] = pos
        occupied[pos] = k
        
    return chain

def _local_search(sequence, chain, rng, time_budget):
    """Perform local search using Reversal moves with Simulated Annealing."""
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    deadline = time.time() + time_budget
    temp = 5.0
    min_temp = 0.1
    
    while time.time() < deadline and temp > min_temp:
        # Try a reversal move
        new_chain = _apply_reversal(current_chain, rng, occupied)
        
        if new_chain is not None:
            new_energy = _energy(sequence, new_chain)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / temp):
                current_chain = new_chain
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
        else:
            # If move failed, maybe try a slither?
            # For now, just cool down.
            pass
            
        temp *= 0.995
        
    return best_chain

def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    """
    if _time_left() < 1.0:
        return [[i, 0] for i in range(len(sequence))]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = float('inf')
    
    # 1. Generate initial conformations
    # We will generate a few diverse initial chains
    num_initial = 5
    initial_chains = []
    
    for _ in range(num_initial):
        c = _grow_greedy(sequence, rng)
        if c:
            initial_chains.append(c)
            
    if not initial_chains:
        # Fallback
        return [[i, 0] for i in range(len(sequence))]
        
    # Sort by energy
    initial_chains.sort(key=lambda c: _energy(sequence, c))
    
    # 2. Optimize the top few
    # Allocate remaining time to the best initial chain
    # Let's optimize the top 2 to balance diversity and depth.
    
    num_to_optimize = 2
    for c in initial_chains[:num_to_optimize]:
        if _time_left() < 5.0:
            break
            
        alloc = _time_left() / 2.0
        if alloc < 1.0:
            break
            
        optimized = _local_search(sequence, c, rng, alloc)
        if optimized:
            e = _energy(sequence, optimized)
            if e < best_energy:
                best_energy = e
                best_chain = optimized
                
    if best_chain is None:
        best_chain = initial_chains[0]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
