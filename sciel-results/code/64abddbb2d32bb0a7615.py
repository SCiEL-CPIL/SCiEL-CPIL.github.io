"""HP lattice protein folding: divide-and-conquer with block optimization and assembly."""
import random
import time
import math

# Deadline guard: 240s CPU budget, keep safety margin
_DEADLINE = time.time() + 230.0  # 10s safety margin before 240s kill
_START_TIME = time.time()


def _neighbours(point):
    """Yield 6 neighbours on cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute energy = -H-H contacts (non-adjacent neighbours)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _grow_block(sequence, start_idx, end_idx, dimension, rng, offset=(0, 0, 0)):
    """
    Grow a sub-chain from start_idx to end_idx (exclusive).
    Returns list of positions for residues start_idx..end_idx-1, or None.
    Uses greedy growth preferring H-H contacts and free space.
    """
    start = offset
    chain, occupied = [start], {start: start_idx}
    for i in range(start_idx + 1, end_idx):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H')
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _optimize_block(sequence, start_idx, end_idx, dimension, rng, n_restarts=50):
    """
    Find best conformation for sequence[start_idx:end_idx] via multiple randomized grows.
    Returns (best_chain, best_energy) or (None, 0).
    """
    best, best_energy = None, 0
    for _ in range(n_restarts):
        if time.time() > _DEADLINE:
            break
        chain = _grow_block(sequence, start_idx, end_idx, dimension, rng)
        if chain is None:
            continue
        # Compute energy within this block only
        local_seq = sequence[start_idx:end_idx]
        local_energy = _compute_energy(local_seq, chain)
        if local_energy < best_energy:
            best, best_energy = chain, local_energy
    if best is None:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(end_idx - start_idx)]
        best_energy = _compute_energy(sequence[start_idx:end_idx], best)
    return best, best_energy


def _assemble_blocks(sequence, dimension, block_boundaries, rng):
    """
    Assemble pre-folded blocks into a full chain.
    block_boundaries: list of (start_idx, end_idx, best_chain) for each block.
    We need to connect blocks so consecutive residues are unit steps.
    Strategy: try to place blocks with offsets that minimize clashes and maximize inter-block H-H contacts.
    """
    n_blocks = len(block_boundaries)
    if n_blocks == 1:
        return block_boundaries[0][2], _compute_energy(sequence, block_boundaries[0][2])
    
    # For each block, we have a relative chain. We need to find absolute positions.
    # Block 0 starts at origin. For each subsequent block, we try to connect
    # its first residue to the last residue of the previous block.
    
    # First, get all block chains
    block_chains = [b[2] for b in block_boundaries]
    block_starts = [b[0] for b in block_boundaries]
    block_ends = [b[1] for b in block_boundaries]
    
    # Place block 0 at origin
    placed = [list(block_chains[0])]
    occupied = {tuple(p): i for i, p in enumerate(placed)}
    
    for b in range(1, n_blocks):
        if time.time() > _DEADLINE:
            break
        prev_last = tuple(placed[-1])
        # The first residue of block b must be adjacent to prev_last
        # Try all 6 neighbours for placement
        best_placement = None
        best_score = float('inf')
        
        for site in _neighbours(prev_last):
            if site in occupied:
                continue
            # Try placing this block with its first residue at site
            # The block chain is relative; offset so that first residue goes to site
            offset = tuple(site[k] - block_chains[b][0][k] for k in range(dimension))
            candidate = [tuple(p[k] + offset[k] for k in range(dimension)) for p in block_chains[b]]
            
            # Check for clashes
            valid = True
            for i, p in enumerate(candidate):
                if p in occupied:
                    valid = False
                    break
            if not valid:
                continue
            
            # Score: minimize clashes (already 0), maximize inter-block H-H contacts
            # Count H-H contacts between this block and previously placed residues
            score = 0
            for i, p in enumerate(candidate):
                if sequence[block_starts[b] + i] == 'H':
                    for q in _neighbours(p):
                        j = occupied.get(q)
                        if j is not None and sequence[j] == 'H':
                            score += 1
            # Also count intra-block contacts (already in block energy, but for assembly scoring)
            # We want to maximize total, so minimize negative score
            if score < best_score:
                best_score = score
                best_placement = candidate
        
        if best_placement is None:
            # Fallback: try all 6 neighbours without checking clashes strictly,
            # or use a straight extension
            # Try extending in a direction not occupied
            for site in _neighbours(prev_last):
                offset = tuple(site[k] - block_chains[b][0][k] for k in range(dimension))
                candidate = [tuple(p[k] + offset[k] for k in range(dimension)) for p in block_chains[b]]
                # Just check first few for clashes
                clash = False
                for p in candidate[:min(5, len(candidate))]:
                    if p in occupied:
                        clash = True
                        break
                if not clash:
                    best_placement = candidate
                    break
            if best_placement is None:
                # Last resort: straight line extension
                last = prev_last
                best_placement = []
                for i in range(len(block_chains[b])):
                    if i == 0:
                        best_placement.append(last)
                    else:
                        last = (last[0] + 1, last[1], last[2])
                        best_placement.append(last)
        
        placed.extend(best_placement)
        for i, p in enumerate(best_placement):
            occupied[tuple(p)] = block_starts[b] + i
    
    full_chain = [tuple(p) for p in placed]
    energy = _compute_energy(sequence, full_chain)
    return full_chain, energy


def _divide_and_conquer_fold(sequence, dimension, rng):
    """
    Divide sequence into blocks, optimize each, then assemble.
    """
    n = len(sequence)
    if n <= 20:
        # Small enough: just do direct optimization
        best, best_energy = _optimize_block(sequence, 0, n, dimension, rng, n_restarts=200)
        return best, best_energy
    
    # Determine block size: aim for ~12-16 residues per block
    target_block_size = max(10, min(16, n // 3))
    n_blocks = max(2, n // target_block_size)
    block_size = n // n_blocks
    remainder = n % n_blocks
    
    # Create block boundaries
    boundaries = []
    idx = 0
    for b in range(n_blocks):
        size = block_size + (1 if b < remainder else 0)
        boundaries.append((idx, idx + size))
        idx += size
    
    # Optimize each block independently
    block_results = []
    for start, end in boundaries:
        if time.time() > _DEADLINE:
            break
        chain, energy = _optimize_block(sequence, start, end, dimension, rng, n_restarts=100)
        block_results.append((start, end, chain))
    
    if len(block_results) < n_blocks:
        # Not enough time, do direct fold on what we have
        return _optimize_block(sequence, 0, n, dimension, rng, n_restarts=50)
    
    # Assemble
    full_chain, energy = _assemble_blocks(sequence, dimension, block_results, rng)
    return full_chain, energy


def _local_search(sequence, chain, dimension, rng, max_iter=200):
    """
    Local search: try small moves (rotate a segment) to improve energy.
    Only allow moves that maintain self-avoidance and connectivity.
    """
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    current_chain = [tuple(p) for p in chain]
    
    for _ in range(max_iter):
        if time.time() > _DEADLINE:
            break
        # Pick a random segment to try rotating/moving
        if n < 4:
            break
        seg_start = rng.randint(1, n - 3)
        seg_len = rng.randint(1, min(4, n - seg_start - 1))
        seg_end = seg_start + seg_len
        
        # Try moving the segment by one unit in a random direction
        axis = rng.randint(0, dimension - 1)
        step = rng.choice([-1, 1])
        
        # Check if move is valid
        new_chain = list(current_chain)
        valid = True
        # The segment must move as a unit
        moved = []
        for i in range(seg_start, seg_end):
            new_pos = list(current_chain[i])
            new_pos[axis] += step
            moved.append(tuple(new_pos))
        
        # Check connectivity at boundaries
        if seg_start > 0:
            # First moved residue must be adjacent to residue before segment
            prev = current_chain[seg_start - 1]
            first_moved = moved[0]
            if abs(prev[0] - first_moved[0]) + abs(prev[1] - first_moved[1]) + abs(prev[2] - first_moved[2]) != 1:
                valid = False
        if seg_end < n:
            # Last moved residue must be adjacent to residue after segment
            next_res = current_chain[seg_end]
            last_moved = moved[-1]
            if abs(next_res[0] - last_moved[0]) + abs(next_res[1] - last_moved[1]) + abs(next_res[2] - last_moved[2]) != 1:
                valid = False
        
        if not valid:
            continue
        
        # Check self-avoidance
        new_chain[seg_start:seg_end] = moved
        # Build set of occupied positions
        occ = set()
        for i, p in enumerate(new_chain):
            if p in occ:
                valid = False
                break
            occ.add(p)
        
        if not valid:
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        if new_energy < current_energy:
            current_chain = new_chain
            current_energy = new_energy
    
    return current_chain, current_energy


def fold(sequence, dimension):
    """
    Best conformation using divide-and-conquer with block optimization and assembly,
    followed by local search polish.
    """
    global _DEADLINE, _START_TIME
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 230.0
    
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Divide and conquer
    if n <= 25:
        # Direct optimization with many restarts
        chain, energy = _optimize_block(sequence, 0, n, dimension, rng, n_restarts=300)
        if chain is not None:
            best_chain = chain
            best_energy = energy
    else:
        chain, energy = _divide_and_conquer_fold(sequence, dimension, rng)
        if chain is not None:
            best_chain = chain
            best_energy = energy
    
    # Phase 2: Local search polish on best chain
    if best_chain is not None:
        if time.time() < _DEADLINE - 5:
            polished_chain, polished_energy = _local_search(sequence, best_chain, dimension, rng, max_iter=500)
            if polished_energy < best_energy:
                best_chain = polished_chain
                best_energy = polished_energy
    
    # Phase 3: Additional randomized restarts if time remains
    if best_chain is None or time.time() < _DEADLINE - 10:
        n_extra = 100 if n <= 30 else 50
        for _ in range(n_extra):
            if time.time() > _DEADLINE - 5:
                break
            chain = _grow_block(sequence, 0, n, dimension, rng)
            if chain is None:
                continue
            energy = _compute_energy(sequence, chain)
            if energy < best_energy:
                best_chain = chain
                best_energy = energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
