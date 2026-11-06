"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 200.0  # Safety margin for 210s kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(sequence, chain):
    """Compute the HP contact energy of a conformation efficiently."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[(p[0], p[1])] = i
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            x, y = chain[i]
            # Check 4 neighbours
            j = where.get((x+1, y))
            if j is not None and j > i+1 and sequence[j] == 'H':
                contacts += 1
            j = where.get((x-1, y))
            if j is not None and j > i+1 and sequence[j] == 'H':
                contacts += 1
            j = where.get((x, y+1))
            if j is not None and j > i+1 and sequence[j] == 'H':
                contacts += 1
            j = where.get((x, y-1))
            if j is not None and j > i+1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _energy_delta(sequence, chain, old_pos, new_pos, removed_indices, added_indices):
    """Compute energy change when moving a segment.
    old_pos: dict mapping old positions to indices in the segment
    new_pos: dict mapping new positions to indices in the segment
    removed_indices: set of indices that were in the segment (now moved)
    added_indices: set of indices that are now in the segment positions
    Returns delta_energy (negative is improvement)
    """
    delta = 0
    # For each H residue in the chain, check if its contact status changed
    # We need to check contacts involving the moved residues
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[(p[0], p[1])] = i
    
    # Rebuild where with new positions for the moved segment
    for old_p, idx in old_pos.items():
        where.pop(old_p, None)
    for new_p, idx in new_pos.items():
        where[new_p] = idx
    
    # Count contacts for H residues that might be affected
    affected = set()
    for idx in removed_indices:
        affected.add(idx)
        # Add neighbors that could be affected
        p = chain[idx]
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None:
                affected.add(j)
    
    # Count new contacts
    contacts = 0
    for i in affected:
        if i < 0 or i >= n or sequence[i] != 'H':
            continue
        x, y = chain[i]
        j = where.get((x+1, y))
        if j is not None and j > i+1 and sequence[j] == 'H':
            contacts += 1
        j = where.get((x-1, y))
        if j is not None and j > i+1 and sequence[j] == 'H':
            contacts += 1
        j = where.get((x, y+1))
        if j is not None and j > i+1 and sequence[j] == 'H':
            contacts += 1
        j = where.get((x, y-1))
        if j is not None and j > i+1 and sequence[j] == 'H':
            contacts += 1
    
    return -contacts


def _polish(sequence, chain, max_iterations=5000, time_limit=5.0, rng=None):
    """Deterministic hill-climbing polish: accept only improving moves.
    Uses a more efficient strategy: try random pivots first, then slither."""
    if chain is None:
        return None
    if rng is None:
        rng = random.Random(42)

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    stall_count = 0

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 15:
            break

        n = len(current_chain)
        improved = False

        # Try a batch of random pivot moves - try more diverse pivots
        for _ in range(30):
            pivot_idx = rng.randint(1, n - 2)
            pivot = current_chain[pivot_idx]
            segment_start = pivot_idx + 1
            if segment_start >= n:
                continue

            segment = current_chain[segment_start:]
            fixed_set = set(current_chain[:pivot_idx + 1])

            for angle in (90, -90, 180):
                rotated = []
                valid = True
                for p in segment:
                    dx, dy = p[0] - pivot[0], p[1] - pivot[1]
                    if angle == 90:
                        ndx, ndy = -dy, dx
                    elif angle == -90:
                        ndx, ndy = dy, -dx
                    else:
                        ndx, ndy = -dx, -dy
                    new_p = (pivot[0] + ndx, pivot[1] + ndy)
                    if new_p in fixed_set:
                        valid = False
                        break
                    rotated.append(new_p)

                if not valid:
                    continue

                new_chain = current_chain[:segment_start] + rotated
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - current_energy

                if delta < 0:
                    current_chain = new_chain
                    current_energy = new_energy
                    improved = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    break  # Accept first improving move and break inner loop
            if improved:
                break  # Break outer batch loop if we improved

        if not improved:
            # Try slither move
            last = current_chain[-1]
            prev = current_chain[-2]
            occupied = set(current_chain[:-1])
            for site in _neighbours(prev):
                if site in occupied:
                    continue
                new_chain = current_chain[:-1] + [site]
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - current_energy
                if delta < 0:
                    current_chain = new_chain
                    current_energy = new_energy
                    improved = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    break

        if not improved:
            stall_count += 1
            if stall_count > 80:
                break
        else:
            stall_count = 0

    return best_chain


def _grow(sequence, dimension, rng, bias=1.0, contact_map=None):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space.
    If contact_map is provided, bias growth toward satisfying high-priority H-H pairs."""
    n = len(sequence)
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    # Precompute which residue pairs are both H and close in sequence (high priority)
    h_positions = [i for i, c in enumerate(sequence) if c == 'H']
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            
            # Contact-map bias: if this H residue should be near a previous H, boost score
            contact_bonus = 0
            if sequence[i] == 'H' and contact_map is not None:
                for j in range(max(0, i-15), i):
                    if sequence[j] == 'H':
                        dist_seq = i - j
                        if dist_seq <= 15:  # Only consider close-in-sequence pairs
                            # Check if site is adjacent to residue j's position
                            px, py = chain[j]
                            sx, sy = site
                            if abs(px - sx) + abs(py - sy) == 1:
                                contact_bonus += (16 - dist_seq) / 16.0
            
            # Heuristic: Prefer sites that are "inside" the current bounding box or close to centroid
            cx = sum(p[0] for p in chain) / len(chain)
            cy = sum(p[1] for p in chain) / len(chain)
            dist = math.sqrt((site[0] - cx)**2 + (site[1] - cy)**2)
            
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Strong bias towards gaining contacts, secondary bias towards compactness (lower dist)
            score = bias * gain + 2.0 * contact_bonus - 0.5 * dist + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move_fast(chain, rng):
    """Fast pivot move: rotate a subchain. Validity check is O(segment_length)."""
    n = len(chain)
    if n < 4:
        return None

    pivot_idx = rng.randint(1, n - 2)
    pivot = chain[pivot_idx]
    segment_start = pivot_idx + 1
    if segment_start >= n:
        return None
        
    segment = chain[segment_start:]
    fixed_set = set(chain[:pivot_idx + 1])
    
    for angle in (90, -90, 180):
        rotated = []
        valid = True
        for p in segment:
            dx, dy = p[0] - pivot[0], p[1] - pivot[1]
            if angle == 90:
                ndx, ndy = -dy, dx
            elif angle == -90:
                ndx, ndy = dy, -dx
            else:  # 180
                ndx, ndy = -dx, -dy
            new_p = (pivot[0] + ndx, pivot[1] + ndy)
            if new_p in fixed_set:
                valid = False
                break
            rotated.append(new_p)
        
        if not valid:
            continue
            
        new_chain = chain[:segment_start] + rotated
        return new_chain

    return None


def _slither_move(chain, rng):
    """Try to move the end of the chain."""
    n = len(chain)
    if n < 4:
        return None
    
    last = chain[-1]
    prev = chain[-2]
    occupied = set(chain[:-1])
    
    for site in _neighbours(prev):
        if site in occupied:
            continue
        new_chain = chain[:-1] + [site]
        return new_chain
    return None


def _local_search(sequence, chain, max_iterations=10000, time_limit=8.0, rng=None):
    """Simulated annealing local search to improve a conformation."""
    if chain is None:
        return None

    if rng is None:
        rng = random.Random(hash(tuple(chain)) % (2**32))

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    n = len(current_chain)
    # Adaptive temperature: start high, cool slower, with reheating on stagnation
    temperature = 8.0
    cooling = 0.999
    stall_count = 0
    max_stall = 30

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 10:
            break

        # Choose move: 60% Pivot, 25% Slither, 15% Segment move
        r = rng.random()
        if r < 0.6:
            new_chain = _pivot_move_fast(current_chain, rng)
        elif r < 0.85:
            new_chain = _slither_move(current_chain, rng)
        else:
            # Segment move: rotate a short segment (length 2-5)
            seg_len = rng.randint(2, min(5, n-2))
            pivot_idx = rng.randint(1, n - seg_len - 1)
            pivot = current_chain[pivot_idx]
            segment = current_chain[pivot_idx+1:pivot_idx+1+seg_len]
            fixed_set = set(current_chain[:pivot_idx+1]) | set(current_chain[pivot_idx+1+seg_len:])
            
            valid_move = None
            for angle in (90, -90, 180):
                rotated = []
                valid = True
                for p in segment:
                    dx, dy = p[0] - pivot[0], p[1] - pivot[1]
                    if angle == 90:
                        ndx, ndy = -dy, dx
                    elif angle == -90:
                        ndx, ndy = dy, -dx
                    else:
                        ndx, ndy = -dx, -dy
                    new_p = (pivot[0] + ndx, pivot[1] + ndy)
                    if new_p in fixed_set:
                        valid = False
                        break
                    rotated.append(new_p)
                if valid:
                    valid_move = rotated
                    break
            
            if valid_move is not None:
                new_chain = current_chain[:pivot_idx+1] + valid_move + current_chain[pivot_idx+1+seg_len:]
            else:
                new_chain = _pivot_move_fast(current_chain, rng)

        if new_chain is None:
            continue

        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy

        # Metropolis criterion with adaptive temperature
        if delta < 0:
            current_chain = new_chain
            current_energy = new_energy
            stall_count = 0
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        else:
            # Accept with probability exp(-delta/T)
            if rng.random() < math.exp(-delta / max(temperature, 0.1)):
                current_chain = new_chain
                current_energy = new_energy
                stall_count += 1
            else:
                stall_count += 1
        
        # Cool down
        temperature *= cooling
        
        # Reheat if stuck
        if stall_count > max_stall:
            temperature = max(temperature, 3.0)
            stall_count = 0

    return best_chain


def _spiral_fold(sequence):
    """Create a compact spiral conformation as a deterministic starting point."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    if n == 2:
        return [(0, 0), (1, 0)]
    
    # Build a spiral that stays compact
    chain = [(0, 0)]
    # Directions: right, up, left, down, repeat but expanding
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dir_idx = 0
    step_count = 0
    steps_per_dir = 1
    
    while len(chain) < n:
        dx, dy = directions[dir_idx]
        # Take steps_per_dir steps in this direction
        for _ in range(steps_per_dir):
            if len(chain) >= n:
                break
            last_x, last_y = chain[-1]
            new_x = last_x + dx
            new_y = last_y + dy
            chain.append((new_x, new_y))
        
        dir_idx = (dir_idx + 1) % 4
        if dir_idx == 2:  # After going left, increase steps
            steps_per_dir += 1
    
    return chain[:n]


def _snake_fold(sequence):
    """Create a snake-like conformation."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    # Snake pattern: go right, down, left, down, right, down...
    width = max(1, int(math.sqrt(n)) + 1)
    
    x, y = 0, 0
    chain.append((x, y))
    
    for i in range(1, n):
        if i % (2 * width) < width:
            # Going right
            x = (i % width) if (i // width) % 2 == 0 else width - 1 - (i % width)
            y = i // width
        else:
            # Going left
            x = width - 1 - ((i - width) % width) if (i // width) % 2 == 1 else (i - width) % width
            y = i // width
        
        # Simplified: just use a proper snake
        row = i // width
        col = i % width
        if row % 2 == 0:
            x = col
        else:
            x = width - 1 - col
        y = row
        
        chain.append((x, y))
    
    return chain


def fold(sequence, dimension):
    """Best conformation using multiple strategies: contact-guided greedy growth + local search."""
    seq_len = len(sequence)

    # Adjust strategy based on sequence length
    if seq_len <= 30:
        num_grow = 600
        local_search_time = 10.0
        polish_time = 5.0
    elif seq_len <= 50:
        num_grow = 400
        local_search_time = 8.0
        polish_time = 4.5
    elif seq_len <= 70:
        num_grow = 300
        local_search_time = 7.0
        polish_time = 4.0
    else:
        num_grow = 200
        local_search_time = 6.0
        polish_time = 3.5

    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0

    # Pass 0: Try deterministic structures first
    for init_chain in [_spiral_fold(sequence), _snake_fold(sequence)]:
        if init_chain is not None and len(init_chain) == seq_len:
            # Verify self-avoidance
            if len(set(init_chain)) == seq_len:
                # Verify unit steps
                valid = True
                for i in range(1, seq_len):
                    dx = abs(init_chain[i][0] - init_chain[i-1][0])
                    dy = abs(init_chain[i][1] - init_chain[i-1][1])
                    if dx + dy != 1:
                        valid = False
                        break
                if valid:
                    value = _energy(sequence, init_chain)
                    if value < best_energy:
                        best_energy = value
                        best_chain = init_chain

    # Pass 1: Contact-guided Greedy Growth
    # Use contact map to guide growth toward satisfying high-priority H-H pairs
    for i in range(num_grow):
        if _time_left() < 25:
            break

        # Bias between 0.5 and 3.0
        bias = 0.5 + rng.random() * 2.5
        # Use contact map for half the attempts
        contact_map = i < num_grow // 2
        chain = _grow(sequence, dimension, rng, bias=bias, contact_map=contact_map)
        if chain is None:
            continue

        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain

    # Pass 2: Local Search on the best chain found
    if best_chain is not None and _time_left() > 30:
        improved = _local_search(sequence, best_chain, max_iterations=20000,
                                  time_limit=local_search_time, rng=rng)
        if improved is not None:
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved

        # Pass 3: Deterministic polish
        if _time_left() > 25:
            polished = _polish(sequence, best_chain, max_iterations=10000,
                               time_limit=polish_time, rng=rng)
            if polished is not None:
                polished_energy = _energy(sequence, polished)
                if polished_energy < best_energy:
                    best_energy = polished_energy
                    best_chain = polished

        # Pass 4: Additional local search with different seed if time permits
        if _time_left() > 35:
            rng2 = random.Random(99999 + seq_len)
            improved2 = _local_search(sequence, best_chain, max_iterations=10000,
                                       time_limit=local_search_time * 0.8, rng=rng2)
            if improved2 is not None:
                e2 = _energy(sequence, improved2)
                if e2 < best_energy:
                    best_energy = e2
                    best_chain = improved2

            # Final polish
            if _time_left() > 25:
                final_polish = _polish(sequence, best_chain, max_iterations=6000,
                                       time_limit=polish_time * 0.7, rng=rng2)
                if final_polish is not None:
                    fp_energy = _energy(sequence, final_polish)
                    if fp_energy < best_energy:
                        best_energy = fp_energy
                        best_chain = final_polish

        # Pass 5: Another round of growth + search if time permits
        if _time_left() > 40:
            extra_grow = 60
            for i in range(extra_grow):
                if _time_left() < 30:
                    break
                bias = 0.5 + rng2.random() * 2.5
                # Use contact map for half the extra attempts
                contact_map = i < extra_grow // 2
                chain = _grow(sequence, dimension, rng2, bias=bias, contact_map=contact_map)
                if chain is None:
                    continue
                value = _energy(sequence, chain)
                if value < best_energy:
                    best_energy = value
                    best_chain = chain
                    # Quick polish
                    if _time_left() > 25:
                        polished = _polish(sequence, best_chain, max_iterations=4000,
                                           time_limit=2.5, rng=rng2)
                        if polished is not None:
                            pe = _energy(sequence, polished)
                            if pe < best_energy:
                                best_energy = pe
                                best_chain = polished

    # Final fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(seq_len)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
