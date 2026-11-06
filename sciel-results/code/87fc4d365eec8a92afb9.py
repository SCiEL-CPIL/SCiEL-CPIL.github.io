"""
Fragment-based Monte Carlo search for HP lattice protein folding.
Instead of growing the chain sequentially, we start with a random self-avoiding walk (SAW)
and use segment moves (rotations/translations of sub-chains) to optimize the energy.
This allows for large-scale rearrangements that single-residue moves cannot achieve.
"""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    # Safety margin: 210s limit, leave 15s buffer -> 195s usable
    return 195.0 - (time.time() - _START_TIME)

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _is_adjacent(p, q):
    return sum(abs(a - b) for a, b in zip(p, q)) == 1

def _energy(sequence, chain):
    """Calculate energy: -number of H-H contacts (non-bonded neighbors)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(sequence)
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _generate_random_saw(n, dimension, rng, max_attempts=10000):
    """Generate a random self-avoiding walk of length n."""
    for _ in range(max_attempts):
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        success = True
        for i in range(1, n):
            last = chain[-1]
            # Shuffle neighbors to avoid bias
            neighbors = list(_neighbours(last))
            rng.shuffle(neighbors)
            moved = False
            for nb in neighbors:
                if nb not in occupied:
                    chain.append(nb)
                    occupied[nb] = i
                    moved = True
                    break
            if not moved:
                success = False
                break
        if success:
            return chain
    return None

def _compute_contact_map(sequence, chain):
    """
    Returns a dictionary mapping (i, j) to True if i, j are H-H contacts (i < j, not adjacent).
    Also returns the energy.
    """
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    n = len(sequence)
    is_h = [c == 'H' for c in sequence]
    
    for i in range(n):
        if not is_h[i]:
            continue
        p = chain[i]
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and j > i + 1 and is_h[j]:
                contacts.add((i, j))
                
    return contacts, -len(contacts)

def _segment_move(sequence, chain, rng):
    """
    Attempt a segment move: pick a segment [i+1, j], rotate/translate it.
    Returns (new_chain, delta_energy, moved)
    """
    n = len(chain)
    if n < 4:
        return None, 0, False
    
    # Pick segment length
    # Short segments (3-5) are fast, long segments (5-10) allow larger moves
    # Limit segment length to avoid O(n^2) checks becoming too slow
    max_seg_len = min(10, n - 2)
    seg_len = rng.randint(3, max_seg_len)
    
    # Pick start index i (segment is i+1 to i+seg_len)
    # i can be 0 to n - seg_len - 1
    i = rng.randint(0, n - seg_len - 1)
    
    # The segment indices are i+1 ... i+seg_len
    # The pivot is i. The segment must remain connected to i and i+seg_len+1 (if exists)
    
    # Get current positions
    pivot_pos = chain[i]
    # The segment is chain[i+1 : i+seg_len+1]
    # The tail is chain[i+seg_len+1 : ] (if any)
    
    # Check if tail exists to constrain the move
    has_tail = (i + seg_len + 1 < n)
    tail_pos = chain[i + seg_len + 1] if has_tail else None
    
    # Generate possible new positions for the segment.
    # Strategy: Try rotating the segment around the pivot or translating it.
    # A simpler robust approach: Pick a random valid move for the first residue of the segment,
    # then grow the rest of the segment greedily/randomly to fit.
    
    # Let's try a "pivot" move: Move the first residue of the segment (i+1) to a neighbor of pivot_pos.
    # Then try to place the rest of the segment.
    
    neighbors = list(_neighbours(pivot_pos))
    rng.shuffle(neighbors)
    
    # Occupied set for quick lookup (excluding the segment itself)
    # We need to check against:
    # 1. chain[0..i]
    # 2. chain[i+seg_len+1..n]
    # The segment itself is being moved, so its old positions are "free"
    
    occupied_fixed = set(chain[:i+1])
    if has_tail:
        occupied_fixed.update(chain[i+seg_len+1:])
    
    # Old positions of the segment (to be freed)
    old_seg_positions = [chain[k] for k in range(i+1, i+seg_len+1)]
    
    # We will try to place the new segment.
    # new_seg[0] must be adjacent to pivot_pos.
    # new_seg[k] must be adjacent to new_seg[k-1].
    # No new_seg[k] can be in occupied_fixed.
    # If has_tail, new_seg[-1] must be adjacent to tail_pos.
    
    for first_new_pos in neighbors:
        if first_new_pos in occupied_fixed:
            continue
        
        # Try to build the rest of the segment
        new_seg = [first_new_pos]
        valid = True
        
        for k in range(1, seg_len):
            last_new = new_seg[-1]
            opts = list(_neighbours(last_new))
            rng.shuffle(opts)
            placed = False
            for nb in opts:
                if nb in occupied_fixed:
                    continue
                # Check if it hits the old segment positions? 
                # Actually, if it hits an old segment position that is NOT in new_seg yet,
                # it's a collision with the "ghost" of the old chain. 
                # But wait, the old positions are free. 
                # However, we must ensure self-avoidance of the *new* chain.
                # The new chain is: chain[:i+1] + new_seg + chain[i+seg_len+1:]
                # So new_seg must not intersect chain[:i+1] or chain[i+seg_len+1:].
                # It *can* intersect the old segment positions, provided those positions
                # are not also in chain[:i+1] or chain[i+seg_len+1:] (which they aren't).
                # BUT, we must ensure new_seg is self-avoiding.
                
                # Check if nb is in new_seg
                if nb in new_seg:
                    continue
                
                # Check if nb is in occupied_fixed
                # (Already checked above)
                
                # Special check: if has_tail, the last step must connect to tail_pos
                if k == seg_len - 1 and has_tail:
                    if not _is_adjacent(nb, tail_pos):
                        continue
                        
                new_seg.append(nb)
                placed = True
                break
            
            if not placed:
                valid = False
                break
        
        if not valid:
            continue
            
        # If we got here, we have a valid new_seg
        # Construct new chain
        new_chain = chain[:i+1] + new_seg + (chain[i+seg_len+1:] if has_tail else [])
        
        # Calculate energy delta
        # To be efficient, we can just recalculate energy for the whole chain if n is small,
        # or update contact map. For simplicity and correctness, and given n<=100,
        # recalculating is O(n) or O(n log n) depending on implementation.
        # Let's use a fast energy delta calculation if possible, or just full recalc.
        # Full recalc is safe.
        
        # Optimization: Only check if the move actually changed the chain
        if new_chain == chain:
            continue
            
        return new_chain, 0, True

    return None, 0, False

def _local_search_fragment(sequence, chain, rng, max_iters=1000):
    """
    Perform fragment moves with simulated annealing acceptance.
    """
    n = len(chain)
    current = list(chain)
    current_energy = _energy(sequence, current)
    
    start_temp = 1.0
    end_temp = 0.01
    
    for it in range(max_iters):
        if _time_remaining() < 5:
            break
            
        temp = start_temp * (end_temp / start_temp) ** (it / max(1, max_iters))
        
        new_chain, _, moved = _segment_move(sequence, current, rng)
        
        if not moved or new_chain is None:
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta/temp)
        if delta < 0:
            accept = True
        else:
            if temp > 0:
                prob = math.exp(-delta / temp)
                accept = rng.random() < prob
            else:
                accept = False
                
        if accept:
            current = new_chain
            current_energy = new_energy
            
    return current

def fold(sequence, dimension):
    """
    Fold the sequence using fragment-based Monte Carlo search.
    """
    _init_time()
    rng = random.Random(12345 + len(sequence))
    
    n = len(sequence)
    
    # Determine restarts based on time and length
    # Longer sequences need fewer restarts but more iterations per restart?
    # Actually, fragment moves are more powerful, so fewer restarts might suffice.
    
    if n <= 20:
        num_restarts = 50
        iters_per_restart = 2000
    elif n <= 50:
        num_restarts = 30
        iters_per_restart = 1500
    else:
        num_restarts = 20
        iters_per_restart = 1000
        
    best_chain = None
    best_energy = 0 # 0 is unfolded/neutral, we want negative (lower is better)
    
    # First, try to find a good initial chain via random growth or random SAW
    # Random SAW is often poor in energy. Greedy growth is better.
    # Let's combine: Use greedy growth for initialization, then fragment search.
    
    # Helper: Greedy growth
    def _grow(sequence, dimension, rng):
        start = (0, 0)
        chain, occupied = [start], {start: 0}
        for i in range(1, len(sequence)):
            options = []
            for site in _neighbours(chain[-1]):
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == "H":
                    gain = sum(1 for q in _neighbours(site)
                               if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                options.append((gain + 0.1 * free + rng.random(), site))
            if not options:
                return None
            site = max(options)[1]
            occupied[site] = i
            chain.append(site)
        return chain

    for restart in range(num_restarts):
        if _time_remaining() < 10:
            break
            
        # 1. Initialize with greedy growth
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            # Fallback to random SAW if growth fails (rare)
            chain = _generate_random_saw(n, dimension, rng)
            if chain is None:
                continue
                
        # 2. Refine with fragment-based search
        # Adjust iterations based on remaining time
        remaining_time = _time_remaining()
        time_per_restart = remaining_time / (num_restarts - restart)
        # Estimate iterations: assume 1ms per iter (conservative for n=100)
        # Actually, fragment move is O(n) for energy check if done naively.
        # Let's just use a fixed number and check time inside the loop.
        
        refined_chain = _local_search_fragment(sequence, chain, rng, max_iters=iters_per_restart)
        
        energy = _energy(sequence, refined_chain)
        if best_chain is None or energy < best_energy:
            best_chain = refined_chain
            best_energy = energy
            
    if best_chain is None:
        # Final fallback: straight line (invalid if n > 1 and self-avoiding required? 
        # Straight line is self-avoiding.)
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
