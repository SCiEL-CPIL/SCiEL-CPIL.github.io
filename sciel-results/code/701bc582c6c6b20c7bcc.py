"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = time.time()
_DEADLINE = 195.0  # Safety margin before 210s kill

def _remaining_time():
    return max(0.0, _DEADLINE - (time.time() - _START_TIME))

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute HP contact energy: -number of H-H non-bonded contacts."""
    if not chain:
        return 0
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        p = chain[i]
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts

def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk for initialization."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            options.append(site)
        if not options:
            return None
        site = rng.choice(options)
        occupied[site] = i
        chain.append(site)
    return chain

def _fragment_energy(sequence, chain, start, end):
    """
    Calculate the energy contribution of a fragment (start:end) 
    considering internal contacts and contacts with the rest of the chain.
    Note: For efficiency in local search, we usually just recompute global energy
    or use incremental updates. Here we keep it simple for correctness first.
    """
    # This is a placeholder for potential incremental optimization, 
    # but for the fragment approach, we often just check validity and then 
    # compute global energy or use a fast contact map.
    pass

def _rotate_fragment(sequence, dimension, chain, start, end, rng):
    """
    Attempt to rotate a fragment (start:end) around the bond (start-1, start).
    The fragment is treated as a rigid body. We try to place the rest of the fragment
    such that it remains self-avoiding and doesn't collide with the rest of the chain.
    
    Strategy: 
    1. Extract fragment points.
    2. Try all possible rigid rotations/translations of the fragment relative to the anchor.
    3. Check for collisions with the rest of the chain.
    4. If valid, return the new chain.
    
    For 2D, rigid motions are limited. We can try rotating the entire fragment 
    around the anchor point (start-1) by 90 degrees? No, that breaks the bond geometry 
    unless we allow the anchor to move? 
    Actually, standard fragment moves in lattice proteins often involve "pivoting" 
    a sub-chain. 
    Let's use a simpler, robust move: **Pivot Move**.
    Pick a pivot index k. Rotate the sub-chain k..N-1 around the bond (k-1, k).
    This is a valid local move that maintains connectivity and unit steps.
    """
    if start <= 0 or end >= len(chain):
        return None
    
    pivot_idx = start
    anchor = chain[pivot_idx - 1]
    # The segment to move is chain[pivot_idx:]
    # We need to rotate this segment around 'anchor' such that the first step 
    # (anchor -> chain[pivot_idx]) is preserved in length but can change direction?
    # No, a pivot move usually rotates the tail around the bond (pivot-1, pivot).
    # The new position of chain[pivot_idx] must be a neighbor of anchor.
    # The rest of the tail must follow via a rigid rotation or a new random walk?
    # Rigid rotation of the whole tail is hard if it's long and collides.
    
    # Let's use a **Sub-chain Re-walk** strategy which is a form of fragment move.
    # Keep prefix 0..start-1 fixed.
    # Keep suffix end..N-1 fixed? No, that's hard to connect.
    
    # Better: **Local Re-walk**.
    # Keep prefix 0..start-1 fixed.
    # Keep suffix end..N-1 fixed.
    # Re-walk the middle part start..end-1.
    # This allows "unfolding" and "refolding" of a section.
    
    prefix = chain[:start]
    suffix = chain[end:]
    
    if not prefix:
        return None
        
    # The new walk must start from prefix[-1] and end at suffix[0] (if suffix exists)
    # If suffix is empty, it just needs to be a valid walk from prefix[-1].
    
    # This is a constraint satisfaction problem for the middle segment.
    # We use a randomized greedy backtracking search for this segment.
    
    start_pos = prefix[-1]
    target_pos = suffix[0] if suffix else None
    length_to_fill = end - start
    
    # Simple DFS with randomization for the segment
    def _search_segment(pos, remaining, occupied_set, seq_slice, idx_in_seq):
        if remaining == 0:
            if target_pos is None:
                return []
            if pos == target_pos:
                return []
            return None # Failed to reach target
        
        options = []
        for site in _neighbours(pos):
            if site in occupied_set:
                continue
            # Heuristic: prefer sites that allow reaching target
            # For now, just random
            options.append(site)
        
        rng.shuffle(options)
        
        for site in options:
            # Check if we can still reach target if we go here (simple distance check)
            if target_pos is not None:
                dist = abs(site[0] - target_pos[0]) + abs(site[1] - target_pos[1])
                if dist > remaining - 1:
                    continue
            
            occupied_set.add(site)
            res = _search_segment(site, remaining - 1, occupied_set, seq_slice, idx_in_seq + 1)
            if res is not None:
                return [site] + res
            occupied_set.remove(site)
        
        return None

    # Build occupied set for prefix and suffix
    occupied_set = set(prefix)
    if suffix:
        occupied_set.update(suffix)
    
    # Note: The bond between prefix[-1] and suffix[0] is NOT occupied by the walk itself 
    # in the sense that the walk doesn't "pass through" the space between them, 
    # but the lattice points are occupied.
    # However, the new segment must not collide with prefix or suffix.
    
    seq_slice = sequence[start:end]
    
    # Optimization: If the segment is short, this is fast. If long, it might be slow.
    # We limit the length of the segment we re-walk to avoid exponential blowup.
    if length_to_fill > 15:
        return None
        
    new_segment = _search_segment(start_pos, length_to_fill, occupied_set, seq_slice, start)
    
    if new_segment is None:
        return None
        
    # Verify the connection to suffix
    if suffix:
        # The last point of new_segment must be adjacent to suffix[0]
        last_new = new_segment[-1]
        first_suf = suffix[0]
        if abs(last_new[0] - first_suf[0]) + abs(last_new[1] - first_suf[1]) != 1:
            return None
            
    new_chain = prefix + new_segment + suffix
    return new_chain

def fold(sequence, dimension):
    """
    Fragment-based local search.
    1. Initialize with a few random walks or greedy walks.
    2. Iteratively pick a random segment and re-walk it to improve energy.
    3. Use simulated annealing acceptance criteria.
    """
    rng = random.Random(12345 + len(sequence))
    
    # Initialize with a few diverse random walks
    best_chain = None
    best_energy = 1000
    
    for _ in range(10):
        c = _random_walk(sequence, dimension, rng)
        if c:
            e = _energy(sequence, c)
            if e < best_energy:
                best_energy = e
                best_chain = c
                
    if best_chain is None:
        # Fallback straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        best_energy = _energy(sequence, best_chain)
        
    current_chain = best_chain
    current_energy = best_energy
    
    # Simulated Annealing / Local Search
    # T decreases over time
    T = 1.0
    cooling = 0.999
    
    n = len(sequence)
    
    while _remaining_time() > 1.0:
        # Pick a random segment to re-walk
        # Segment length: random between 3 and min(15, n//2)
        min_len = 3
        max_len = min(15, n // 2)
        if min_len > max_len:
            break
            
        seg_len = rng.randint(min_len, max_len)
        start = rng.randint(0, n - seg_len - 1)
        end = start + seg_len
        
        # Skip if segment includes the very beginning or very end? 
        # Re-walking from 0 is just a full re-walk, which is expensive.
        # Re-walking to the end is also a full re-walk of the tail.
        # Let's bias towards internal segments for "fragment" behavior.
        if start == 0 or end == n:
            continue
            
        new_chain = _rotate_fragment(sequence, dimension, current_chain, start, end, rng)
        
        if new_chain is None:
            continue
            
        new_energy = _energy(sequence, new_chain)
        
        delta = new_energy - current_energy
        
        # Acceptance criterion
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain
                
        T *= cooling
        
        # Periodic restart if stuck?
        # For now, simple cooling.
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
