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
    """Compute the HP contact energy of a conformation."""
    where = {}
    for i, p in enumerate(chain):
        where[(p[0], p[1])] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y = p
            j = where.get((x+1, y))
            if j is not None and j > i + 1 and sequence[j] == "H":
                contacts += 1
            j = where.get((x-1, y))
            if j is not None and j > i + 1 and sequence[j] == "H":
                contacts += 1
            j = where.get((x, y+1))
            if j is not None and j > i + 1 and sequence[j] == "H":
                contacts += 1
            j = where.get((x, y-1))
            if j is not None and j > i + 1 and sequence[j] == "H":
                contacts += 1
    return -contacts


def _energy_incremental(sequence, chain, where, pivot_idx, old_segment, new_segment):
    """Compute energy change for a pivot move using pre-built position map.
    Only recomputes contacts involving residues in the moved segment or their neighbors.
    """
    n = len(chain)
    seg_start = pivot_idx + 1
    seg_len = len(new_segment)
    
    # Build a set of positions in the moved segment for quick lookup
    new_seg_positions = set(new_segment)
    old_seg_positions = set(old_segment)
    
    # Residues whose contact status might change:
    # 1. Residues in the moved segment (their positions changed)
    # 2. Residues adjacent to any position in the moved segment (old or new)
    
    # Find all H residues in the moved segment
    moved_h = [seg_start + k for k in range(seg_len) if sequence[seg_start + k] == "H"]
    
    # Find all H residues that are neighbors of the old or new segment positions
    neighbor_h = set()
    for pos in old_seg_positions:
        x, y = pos
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            j = where.get((x+dx, y+dy))
            if j is not None and j < seg_start and sequence[j] == "H":
                neighbor_h.add(j)
    for pos in new_seg_positions:
        x, y = pos
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            j = where.get((x+dx, y+dy))
            if j is not None and j < seg_start and sequence[j] == "H":
                neighbor_h.add(j)
    
    # Count contacts before and after for these residues
    # Before: use old_segment positions
    def count_contacts(residue_idx, pos, exclude_indices):
        """Count H-H contacts for residue at pos, excluding chain-adjacent."""
        if sequence[residue_idx] != "H":
            return 0
        x, y = pos
        count = 0
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            j = where.get((x+dx, y+dy))
            if j is not None and j not in exclude_indices and sequence[j] == "H":
                # Avoid double counting: only count if j > residue_idx or we handle it
                count += 1
        return count
    
    # For efficiency, count total contacts involving moved segment + neighbors
    # Before move:
    old_total = 0
    exclude_moved = set(range(seg_start, seg_start + seg_len))
    
    for idx in moved_h:
        pos = old_segment[idx - seg_start]
        old_total += count_contacts(idx, pos, exclude_moved)
    
    for idx in neighbor_h:
        if idx in exclude_moved:
            continue
        pos = chain[idx]
        # Count contacts from this fixed residue to the moved segment
        x, y = pos
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nb = (x+dx, y+dy)
            if nb in old_seg_positions:
                # Find which residue in moved segment is at nb
                k = old_segment.index(nb) if nb in old_segment else -1
                if k >= 0:
                    moved_idx = seg_start + k
                    if abs(moved_idx - idx) > 1 and sequence[moved_idx] == "H":
                        old_total += 1
    
    # After move:
    new_total = 0
    # Update where temporarily for new positions
    new_where = dict(where)
    for k, pos in enumerate(new_segment):
        new_where[pos] = seg_start + k
    # Remove old positions
    for pos in old_segment:
        if pos in new_where and new_where[pos] >= seg_start:
            del new_where[pos]
    
    for idx in moved_h:
        pos = new_segment[idx - seg_start]
        # Count contacts using new_where
        x, y = pos
        count = 0
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            j = new_where.get((x+dx, y+dy))
            if j is not None and j not in exclude_moved and sequence[j] == "H":
                count += 1
        new_total += count
    
    for idx in neighbor_h:
        if idx in exclude_moved:
            continue
        pos = chain[idx]
        x, y = pos
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nb = (x+dx, y+dy)
            if nb in new_seg_positions:
                k = new_segment.index(nb) if nb in new_segment else -1
                if k >= 0:
                    moved_idx = seg_start + k
                    if abs(moved_idx - idx) > 1 and sequence[moved_idx] == "H":
                        new_total += 1
    
    # Energy change = -(new_contacts - old_contacts) = old_total - new_total (since energy = -contacts)
    # But we need to be careful about double counting. Let's just do full recompute for correctness
    # in the critical path, but use this for quick rejection.
    return old_total - new_total


def _delta_energy(sequence, chain, old_pos, new_pos, idx):
    """Compute energy change when moving residue idx from old_pos to new_pos."""
    # Remove old contacts involving idx
    old_contacts = 0
    if sequence[idx] == 'H':
        ox, oy = old_pos
        for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
            nb = (ox+dx, oy+dy)
            # Check if this neighbor is in chain and is H and not adjacent
            # We need to know which residue is at nb
            for j, p in enumerate(chain):
                if p == nb and j != idx-1 and j != idx+1 and sequence[j] == 'H':
                    old_contacts += 1
                    break
    
    # Add new contacts involving idx
    new_contacts = 0
    if sequence[idx] == 'H':
        nx, ny = new_pos
        for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
            nb = (nx+dx, ny+dy)
            for j, p in enumerate(chain):
                if p == nb and j != idx-1 and j != idx+1 and sequence[j] == 'H':
                    new_contacts += 1
                    break
    
    # Also need to account for contacts that were between old_pos neighbors
    # and new_pos neighbors that change
    # This is complex, so for simplicity we'll do full recompute if delta is close to 0
    return -(new_contacts - old_contacts)


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
    n = len(current_chain)

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 15:
            break

        improved = False

        # Try multiple pivot strategies in sequence
        # Strategy 1: Short pivots (high probability of validity)
        for _ in range(15):
            pivot_idx = rng.randint(1, min(n - 2, max(2, n // 4)))
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
                    break
            if improved:
                break

        # Strategy 2: Medium pivots
        if not improved:
            for _ in range(10):
                pivot_idx = rng.randint(1, min(n - 2, max(3, n // 2)))
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
                        break
                if improved:
                    break

        # Strategy 3: Slither moves
        if not improved:
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

        # Strategy 4: Try moving internal residues (insertion/deletion style)
        if not improved and n > 10:
            # Try to move a random internal residue to a nearby empty spot
            for _ in range(5):
                idx = rng.randint(2, n - 3)  # Avoid ends
                old_pos = current_chain[idx]
                # Try moving to a neighbor of old_pos that's not occupied
                for site in _neighbours(old_pos):
                    if site in set(current_chain):
                        continue
                    # Check if this move maintains connectivity
                    # Simple check: just try it
                    new_chain = current_chain[:]
                    new_chain[idx] = site
                    # Verify self-avoidance
                    if len(set(new_chain)) == n:
                        # Verify unit steps
                        valid = True
                        for j in range(1, n):
                            dx = abs(new_chain[j][0] - new_chain[j-1][0])
                            dy = abs(new_chain[j][1] - new_chain[j-1][1])
                            if dx + dy != 1:
                                valid = False
                                break
                        if valid:
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
                if improved:
                    break

        if not improved:
            stall_count += 1
            if stall_count > 30:
                break
        else:
            stall_count = 0

    return best_chain


def _grow(sequence, dimension, rng, bias=1.0):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    
    # Track centroid incrementally for efficiency
    cx = 0.0
    cy = 0.0
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            # Distance to centroid
            dist = math.sqrt((site[0] - cx)**2 + (site[1] - cy)**2)
            
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # Adaptive scoring: stronger contact bias early, more compactness later
            progress = i / max(1, n - 1)
            contact_weight = bias * (1.5 - 0.5 * progress)
            compact_weight = 0.3 + 0.7 * progress
            
            score = contact_weight * gain - compact_weight * dist + 0.1 * free + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
        # Update centroid
        cx = (cx * i + site[0]) / (i + 1)
        cy = (cy * i + site[1]) / (i + 1)
    
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


def _pivot_move_inplace(chain, where, pivot_idx):
    """Apply a pivot move in-place on chain and update where. Returns (new_energy_delta_approx, success).
    Tries 90, -90, 180 degrees. Modifies chain and where in place if successful.
    """
    n = len(chain)
    if n < 4:
        return None, False
    
    pivot = chain[pivot_idx]
    segment_start = pivot_idx + 1
    if segment_start >= n:
        return None, False
        
    old_segment = chain[segment_start:]
    fixed_set = set(chain[:pivot_idx + 1])
    
    for angle in (90, -90, 180):
        rotated = []
        valid = True
        for p in old_segment:
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
        
        # Apply the move in-place
        # Remove old positions from where
        for p in old_segment:
            if p in where:
                del where[p]
        # Add new positions
        for k, p in enumerate(rotated):
            where[p] = segment_start + k
        # Update chain
        chain[segment_start:] = rotated
        
        return True, True

    return None, False


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
    """Simulated annealing local search with incremental energy for speed."""
    if chain is None:
        return None

    if rng is None:
        rng = random.Random(hash(tuple(chain)) % (2**32))

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    # Build position map for incremental updates
    where = {}
    for i, p in enumerate(current_chain):
        where[(p[0], p[1])] = i

    start_time = time.time()
    n = len(current_chain)
    
    # Adaptive temperature: start higher for longer sequences
    temperature = 3.0 + 0.05 * n
    cooling = 0.998  # Slower cooling for more exploration
    min_temp = 0.3

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 10:
            break

        # Choose move with adaptive probabilities
        r = rng.random()
        
        if r < 0.55:
            # Pivot move - try in-place
            pivot_idx = rng.randint(1, n - 2)
            # Save state for rollback
            old_segment = list(current_chain[pivot_idx+1:])
            old_where = dict(where)
            old_chain_tail = list(current_chain[pivot_idx+1:])
            
            # Try all three angles
            pivot = current_chain[pivot_idx]
            segment_start = pivot_idx + 1
            fixed_set = set(current_chain[:pivot_idx + 1])
            success = False
            for angle in (90, -90, 180):
                rotated = []
                valid = True
                for p in old_segment:
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
                
                # Apply move
                for p in old_segment:
                    if p in where:
                        del where[p]
                for k, p in enumerate(rotated):
                    where[p] = segment_start + k
                current_chain[segment_start:] = rotated
                success = True
                break
            
            if not success:
                continue
            
            new_energy = _energy(sequence, current_chain)
            delta = new_energy - current_energy
            
            # Metropolis criterion
            if delta < 0:
                accept = True
            elif temperature > min_temp:
                if delta < 50:
                    accept_prob = math.exp(-delta / temperature)
                else:
                    accept_prob = 0.0
                accept = rng.random() < accept_prob
            else:
                accept = False
            
            if not accept:
                # Rollback
                for p in rotated:
                    if p in where:
                        del where[p]
                for p in old_segment:
                    where[p] = segment_start + (old_segment.index(p))
                # Rebuild where properly for old segment
                for k, p in enumerate(old_segment):
                    where[p] = segment_start + k
                current_chain[segment_start:] = old_segment
                current_energy = new_energy  # No change since we rolled back
            else:
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                    
        elif r < 0.80:
            # Slither move
            last = current_chain[-1]
            prev = current_chain[-2]
            occupied = set(current_chain[:-1])
            
            moved = False
            for site in _neighbours(prev):
                if site in occupied:
                    continue
                
                # Apply slither
                del where[last]
                where[site] = n - 1
                old_last = last
                current_chain[-1] = site
                
                new_energy = _energy(sequence, current_chain)
                delta = new_energy - current_energy
                
                if delta < 0:
                    accept = True
                elif temperature > min_temp:
                    if delta < 50:
                        accept_prob = math.exp(-delta / temperature)
                    else:
                        accept_prob = 0.0
                    accept = rng.random() < accept_prob
                else:
                    accept = False
                
                if accept:
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    moved = True
                    break
                else:
                    # Rollback
                    del where[site]
                    where[old_last] = n - 1
                    current_chain[-1] = old_last
            
            if not moved:
                continue
                
        else:
            # Long pivot for large-scale rearrangement
            if n > 15:
                pivot_idx = rng.randint(1, max(2, n // 3))
            else:
                pivot_idx = rng.randint(1, n - 2)
            
            old_segment = list(current_chain[pivot_idx+1:])
            pivot = current_chain[pivot_idx]
            segment_start = pivot_idx + 1
            fixed_set = set(current_chain[:pivot_idx + 1])
            
            success = False
            for angle in (90, -90, 180):
                rotated = []
                valid = True
                for p in old_segment:
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
                
                # Apply move
                for p in old_segment:
                    if p in where:
                        del where[p]
                for k, p in enumerate(rotated):
                    where[p] = segment_start + k
                current_chain[segment_start:] = rotated
                success = True
                break
            
            if not success:
                continue
            
            new_energy = _energy(sequence, current_chain)
            delta = new_energy - current_energy
            
            if delta < 0:
                accept = True
            elif temperature > min_temp:
                if delta < 50:
                    accept_prob = math.exp(-delta / temperature)
                else:
                    accept_prob = 0.0
                accept = rng.random() < accept_prob
            else:
                accept = False
            
            if not accept:
                # Rollback
                for p in rotated:
                    if p in where:
                        del where[p]
                for k, p in enumerate(old_segment):
                    where[p] = segment_start + k
                current_chain[segment_start:] = old_segment
            else:
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)

        temperature = max(min_temp, temperature * cooling)

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
    """Best conformation using multiple strategies: greedy growth + local search."""
    seq_len = len(sequence)

    # Adjust strategy based on sequence length
    # Allocate more time to local search for better results
    if seq_len <= 30:
        num_grow = 400
        local_search_time = 15.0
        polish_time = 8.0
        extra_search_time = 10.0
    elif seq_len <= 50:
        num_grow = 300
        local_search_time = 13.0
        polish_time = 7.0
        extra_search_time = 8.0
    elif seq_len <= 70:
        num_grow = 220
        local_search_time = 11.0
        polish_time = 6.0
        extra_search_time = 7.0
    else:
        num_grow = 160
        local_search_time = 9.0
        polish_time = 5.0
        extra_search_time = 6.0

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

    # Pass 1: Greedy Growth
    # We use a range of biases to explore different compactness levels
    for i in range(num_grow):
        if _time_left() < 30:
            break

        # Bias between 0.5 and 3.0
        bias = 0.5 + rng.random() * 2.5
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue

        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain

    # Pass 2: Local Search on the best chain found
    if best_chain is not None and _time_left() > 40:
        improved = _local_search(sequence, best_chain, max_iterations=50000,
                                  time_limit=local_search_time, rng=rng)
        if improved is not None:
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved

        # Pass 3: Deterministic polish
        if _time_left() > 35:
            polished = _polish(sequence, best_chain, max_iterations=15000,
                               time_limit=polish_time, rng=rng)
            if polished is not None:
                polished_energy = _energy(sequence, polished)
                if polished_energy < best_energy:
                    best_energy = polished_energy
                    best_chain = polished

        # Pass 4: Additional local search with different seed if time permits
        if _time_left() > 45:
            rng2 = random.Random(99999 + seq_len)
            improved2 = _local_search(sequence, best_chain, max_iterations=30000,
                                       time_limit=extra_search_time, rng=rng2)
            if improved2 is not None:
                e2 = _energy(sequence, improved2)
                if e2 < best_energy:
                    best_energy = e2
                    best_chain = improved2

            # Final polish
            if _time_left() > 35:
                final_polish = _polish(sequence, best_chain, max_iterations=10000,
                                       time_limit=polish_time * 0.8, rng=rng2)
                if final_polish is not None:
                    fp_energy = _energy(sequence, final_polish)
                    if fp_energy < best_energy:
                        best_energy = fp_energy
                        best_chain = final_polish

        # Pass 5: Another round of growth + search if time permits
        if _time_left() > 50:
            extra_grow = 80
            for i in range(extra_grow):
                if _time_left() < 40:
                    break
                bias = 0.5 + rng2.random() * 2.5
                chain = _grow(sequence, dimension, rng2, bias=bias)
                if chain is None:
                    continue
                value = _energy(sequence, chain)
                if value < best_energy:
                    best_energy = value
                    best_chain = chain
                    # Quick local search
                    if _time_left() > 35:
                        quick_ls = _local_search(sequence, best_chain, max_iterations=12000,
                                                 time_limit=3.0, rng=rng2)
                        if quick_ls is not None:
                            qe = _energy(sequence, quick_ls)
                            if qe < best_energy:
                                best_energy = qe
                                best_chain = quick_ls
                    # Quick polish
                    if _time_left() > 30:
                        polished = _polish(sequence, best_chain, max_iterations=5000,
                                           time_limit=3.0, rng=rng2)
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
