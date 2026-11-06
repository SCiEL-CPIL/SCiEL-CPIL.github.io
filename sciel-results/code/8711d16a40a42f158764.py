"""HP lattice protein folding: multi-start simulated annealing with efficient moves."""
# EVOLVE-BLOCK-START
import random
import time
import math

_T0 = time.time()
_DEADLINE = 195.0  # wall-clock limit with safety margin (210s kill, leave 15s buffer)


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _compute_energy(sequence, chain):
    """Compute energy: -number of H-H non-bonded contacts."""
    n = len(chain)
    if n < 3:
        return 0
    
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
    
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = (x + dx, y + dy)
            j = pos_to_idx.get(nb)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _grow_greedy(sequence, rng):
    """Generate initial conformation using greedy growth with some randomness."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0)}
    
    for i in range(1, n):
        last = chain[-1]
        x, y = last
        options = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = (x + dx, y + dy)
            if nb not in occupied:
                options.append(nb)
        
        if not options:
            return None
        
        # Score each option: prefer moves that create H-H contacts
        if sequence[i] == 'H':
            scored = []
            for nb in options:
                gain = 0
                nx, ny = nb
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    pp = (nx + dx, ny + dy)
                    if pp in occupied:
                        # Find which residue is at pp
                        for j in range(len(chain)):
                            if chain[j] == pp:
                                if sequence[j] == 'H':
                                    gain += 1
                                break
                scored.append((gain + rng.random() * 0.5, nb))
            scored.sort(reverse=True)
            next_pos = scored[0][1]
        else:
            next_pos = rng.choice(options)
        
        chain.append(next_pos)
        occupied.add(next_pos)
    
    return chain


def _kink_move(chain, rng):
    """
    Perform a kink move: select a residue (not at ends), try moving it to an adjacent
    empty position while maintaining connectivity.
    """
    n = len(chain)
    if n < 3:
        return None
    
    idx = rng.randint(1, n - 2)
    x, y = chain[idx]
    
    # Try all 4 directions
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        new_pos = (x + dx, y + dy)
        
        # Check if new position is occupied
        occupied = set(chain)
        occupied.remove(chain[idx])
        if new_pos in occupied:
            continue
        
        # Check connectivity with neighbors
        prev = chain[idx - 1]
        nxt = chain[idx + 1]
        
        # New position must be adjacent to both prev and nxt
        if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) != 1:
            continue
        if abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) != 1:
            continue
        
        new_chain = list(chain)
        new_chain[idx] = new_pos
        return new_chain
    
    return None


def _pivot_move(chain, rng):
    """
    Perform a pivot move: rotate a suffix of the chain around a pivot point.
    """
    n = len(chain)
    if n < 4:
        return None
    
    # Choose pivot index
    pivot = rng.randint(1, n - 2)
    pivot_pos = chain[pivot]
    suffix = chain[pivot + 1:]
    
    # Try rotating the suffix by 90 degrees clockwise or counter-clockwise
    for rot in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
        # rot represents the direction of the first step from pivot
        # We need to rotate the entire suffix
        
        # Get relative positions from pivot
        rel_positions = [(x - pivot_pos[0], y - pivot_pos[1]) for x, y in suffix]
        
        # Apply 90 degree rotation: (x,y) -> (-y,x) for CCW, (y,-x) for CW
        # Let's try both rotations
        for is_cw in [False, True]:
            rotated_rel = []
            for rx, ry in rel_positions:
                if is_cw:
                    new_rx, new_ry = ry, -rx
                else:
                    new_rx, new_ry = -ry, rx
                rotated_rel.append((new_rx, new_ry))
            
            # Check if the first rotated position is adjacent to pivot
            first_rx, first_ry = rotated_rel[0]
            if abs(first_rx) + abs(first_ry) != 1:
                continue
            
            # Build the new chain
            new_chain = chain[:pivot + 1]
            new_occupied = set(chain[:pivot + 1])
            valid = True
            
            for rx, ry in rotated_rel:
                new_pos = (pivot_pos[0] + rx, pivot_pos[1] + ry)
                if new_pos in new_occupied:
                    valid = False
                    break
                new_chain.append(new_pos)
                new_occupied.add(new_pos)
            
            if valid and len(new_chain) == n:
                # Verify self-avoidance
                if len(set(new_chain)) == n:
                    return new_chain
    
    return None


def _segment_relocation(chain, sequence, rng):
    """
    Relocate a segment of the chain to a new position.
    """
    n = len(chain)
    if n < 5:
        return None
    
    # Choose segment length (2-4 residues)
    seg_len = rng.choice([2, 3, 4])
    
    # Choose start index (not at ends)
    if n <= seg_len + 2:
        return None
    start = rng.randint(1, n - seg_len - 1)
    
    segment = chain[start:start + seg_len]
    before = chain[:start]
    after = chain[start + seg_len:]
    
    # The segment must be moved so that:
    # 1. The first residue of the segment is adjacent to the last residue of 'before'
    # 2. The last residue of the segment is adjacent to the first residue of 'after'
    # 3. No collisions with 'before' or 'after'
    
    before_set = set(before)
    after_set = set(after)
    
    # Try all possible positions for the segment
    # The segment's internal geometry must be preserved
    # We can translate the segment to any position where it fits
    
    # Get the relative geometry of the segment
    rel_seg = [(x - segment[0][0], y - segment[0][1]) for x, y in segment]
    
    # Try placing the segment's first residue at various positions
    # that are adjacent to before[-1]
    bx, by = before[-1]
    candidates = []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        candidates.append((bx + dx, by + dy))
    
    for base_pos in candidates:
        # Build the new segment positions
        new_segment = [(base_pos[0] + rx, base_pos[1] + ry) for rx, ry in rel_seg]
        
        # Check that the last residue of the new segment is adjacent to after[0]
        last_new = new_segment[-1]
        ax, ay = after[0]
        if abs(last_new[0] - ax) + abs(last_new[1] - ay) != 1:
            continue
        
        # Check for collisions
        new_segment_set = set(new_segment)
        if new_segment_set & before_set:
            continue
        if new_segment_set & after_set:
            continue
        
        # Build the new chain
        new_chain = before + new_segment + after
        
        # Verify self-avoidance
        if len(set(new_chain)) == n:
            # Verify connectivity
            valid = True
            for i in range(n - 1):
                if abs(new_chain[i][0] - new_chain[i+1][0]) + abs(new_chain[i][1] - new_chain[i+1][1]) != 1:
                    valid = False
                    break
            if valid:
                return new_chain
    
    return None


def _local_search(sequence, chain, rng, time_limit):
    """
    Perform simulated annealing local search.
    """
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    T = 3.0
    cooling = 0.9995
    min_T = 0.1
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        if _remaining() < 5.0:
            break
        
        # Choose a move type
        r = rng.random()
        if r < 0.4:
            new_chain = _kink_move(chain, rng)
        elif r < 0.7:
            new_chain = _pivot_move(chain, rng)
        else:
            new_chain = _segment_relocation(chain, sequence, rng)
        
        if new_chain is None:
            continue
        
        # Verify validity
        if len(set(new_chain)) != n:
            continue
        valid = True
        for i in range(n - 1):
            if abs(new_chain[i][0] - new_chain[i+1][0]) + abs(new_chain[i][1] - new_chain[i+1][1]) != 1:
                valid = False
                break
        if not valid:
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Simulated annealing acceptance
        if delta < 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(new_chain)
        
        T *= cooling
        if T < min_T:
            T = min_T
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold sequence into low-energy conformation using multi-start simulated annealing."""
    if dimension != 2:
        return [[i] + [0] * (dimension - 1) for i in range(len(sequence))]
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0  # Start with unfolded (energy 0)
    
    # Time budget management
    total_time_budget = min(_remaining() - 10.0, 180.0)
    if total_time_budget < 10:
        total_time_budget = 10
    
    # Number of restarts based on sequence length
    if n <= 30:
        num_restarts = 80
    elif n <= 50:
        num_restarts = 50
    elif n <= 70:
        num_restarts = 30
    else:
        num_restarts = 20
    
    # Time per restart
    time_per_restart = total_time_budget / num_restarts
    if time_per_restart < 0.5:
        time_per_restart = 0.5
    
    for restart in range(num_restarts):
        if _remaining() < 15.0:
            break
        
        # Generate initial conformation
        chain = _grow_greedy(sequence, rng)
        if chain is None or len(chain) != n:
            # Try a simple random walk
            chain = [(0, 0)]
            occupied = {(0, 0)}
            valid = True
            for i in range(1, n):
                last = chain[-1]
                x, y = last
                options = []
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nb = (x + dx, y + dy)
                    if nb not in occupied:
                        options.append(nb)
                if not options:
                    valid = False
                    break
                next_pos = rng.choice(options)
                chain.append(next_pos)
                occupied.add(next_pos)
            if not valid:
                continue
        
        # Verify initial conformation
        if len(set(chain)) != n:
            continue
        valid = True
        for i in range(n - 1):
            if abs(chain[i][0] - chain[i+1][0]) + abs(chain[i][1] - chain[i+1][1]) != 1:
                valid = False
                break
        if not valid:
            continue
        
        # Run local search with time limit
        actual_time = min(time_per_restart, _remaining() - 20.0)
        if actual_time < 0.1:
            break
        
        chain, energy = _local_search(sequence, chain, rng, actual_time)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [[x, y] for x, y in best_chain]
# EVOLVE-BLOCK-END
