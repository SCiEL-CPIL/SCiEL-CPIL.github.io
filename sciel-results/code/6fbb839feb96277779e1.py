"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 175.0  # Safety margin for 210s kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _calculate_energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng, contact_weight=1.0, free_weight=0.1):
    """Randomized greedy chain growth with configurable bias."""
    start = (0,) * dimension
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
            options.append((contact_weight * gain + free_weight * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _energy_delta_hh(sequence, i, old_pos, new_pos, chain, occupied):
    """Compute the change in H-H contacts when residue i moves from old_pos to new_pos.
    Only consider contacts with residues j where |j-i| > 1 (non-adjacent)."""
    delta = 0
    if sequence[i] == "H":
        # Remove contacts at old_pos
        for q in _neighbours(old_pos):
            j = occupied.get(q)
            if j is not None and abs(j - i) > 1 and sequence[j] == "H":
                delta -= 1
        # Add contacts at new_pos
        for q in _neighbours(new_pos):
            j = occupied.get(q)
            if j is not None and abs(j - i) > 1 and sequence[j] == "H":
                delta += 1
    return delta


def _pivot_move(sequence, chain, occupied, rng):
    """Attempt a pivot move: rotate a suffix of the chain around a pivot point.
    Returns (new_chain, new_occupied, energy_delta) or None if move not possible."""
    n = len(chain)
    if n < 5:
        return None
    # Choose pivot index (at least 3 from start to have room)
    pivot_idx = rng.randint(2, n - 3)
    pivot = chain[pivot_idx]
    prev = chain[pivot_idx - 1]
    # The suffix starts at pivot_idx
    suffix = chain[pivot_idx:]
    # Try random rotations of the suffix
    # Represent suffix as list of relative directions from pivot
    # Actually, let's just try moving the first few residues of the suffix to new positions
    
    # Simpler pivot: pick a sub-chain segment and try to relocate it
    # Let's do a simpler version: pick a random residue (not first/last) and try to move it
    # to a random free neighbouring site, but that's just a single-residue move.
    
    # Better: crankshaft move - pick a contiguous segment and rotate it
    seg_len = rng.randint(2, min(5, n - 2))
    seg_start = rng.randint(1, n - seg_len - 1)
    seg_end = seg_start + seg_len - 1
    
    # The segment is chain[seg_start:seg_end+1]
    # We need to find a new position for this segment that:
    # 1. Is self-avoiding
    # 2. Connects properly to chain[seg_start-1] and chain[seg_end+1]
    
    # This is complex. Let's do a simpler pivot: rotate a 3-residue segment
    # Actually, let me implement a proper pivot move:
    # Pick a pivot point and rotate the tail
    
    # Let me do a simpler but effective move: single residue move
    idx = rng.randint(1, n - 2)  # not first or last
    old_pos = chain[idx]
    # Try all free neighbours
    for new_pos in _neighbours(old_pos):
        if new_pos in occupied:
            continue
        # Check self-avoidance: new_pos must not be occupied by any other residue
        # (it's already free, so OK)
        # Compute energy delta
        delta = _energy_delta_hh(sequence, idx, old_pos, new_pos, chain, occupied)
        # Apply
        del occupied[old_pos]
        occupied[new_pos] = idx
        chain[idx] = new_pos
        return (chain, occupied, delta)
    return None


def _crankshaft_move(sequence, chain, occupied, rng):
    """Crankshaft move: rotate a contiguous segment of the chain.
    Returns (new_chain, new_occupied, energy_delta) or None."""
    n = len(chain)
    if n < 5:
        return None
    # Pick segment length and start
    seg_len = rng.randint(2, min(4, n - 2))
    seg_start = rng.randint(1, n - seg_len - 1)
    seg_end = seg_start + seg_len - 1
    
    # The segment connects to chain[seg_start-1] and chain[seg_end+1]
    left_anchor = chain[seg_start - 1]
    right_anchor = chain[seg_end + 1]
    
    # Remove the segment from occupied
    seg_positions = chain[seg_start:seg_end + 1]
    for pos in seg_positions:
        del occupied[pos]
    
    # Try to find a new placement for the segment
    # The new segment must start at a neighbour of left_anchor and end at a neighbour of right_anchor
    # and be self-avoiding
    
    # Get free neighbours of left_anchor
    start_options = [q for q in _neighbours(left_anchor) if q not in occupied and q != right_anchor]
    if not start_options:
        # Revert
        for i, pos in enumerate(seg_positions):
            occupied[pos] = seg_start + i
            chain[seg_start + i] = pos
        return None
    
    # Try to grow a path of length seg_len from a start option to a neighbour of right_anchor
    # Use BFS/DFS
    found = False
    for start_pos in start_options:
        # DFS to find a path of seg_len residues ending at a neighbour of right_anchor
        def dfs(pos, path, visited):
            nonlocal found
            if len(path) == seg_len:
                # Check if last position is adjacent to right_anchor
                for q in _neighbours(pos):
                    if q == right_anchor:
                        return True
                return False
            for q in _neighbours(pos):
                if q in visited or q in occupied:
                    continue
                if q == right_anchor:
                    # Can't step on right_anchor unless it's the last step
                    if len(path) == seg_len - 1:
                        # Check if this would work
                        pass
                    continue
                visited.add(q)
                path.append(q)
                if dfs(q, path, visited):
                    return True
                path.pop()
                visited.remove(q)
            return False
        
        visited = {start_pos}
        path = [start_pos]
        if dfs(start_pos, path, visited):
            # Apply the new segment
            for i, pos in enumerate(path):
                occupied[pos] = seg_start + i
                chain[seg_start + i] = pos
            # Compute energy delta for all moved residues
            delta = 0
            for i in range(seg_start, seg_end + 1):
                if sequence[i] == "H":
                    # Count H-H contacts for residue i in new position
                    for q in _neighbours(chain[i]):
                        j = occupied.get(q)
                        if j is not None and abs(j - i) > 1 and sequence[j] == "H":
                            delta += 1
            # But we need to be careful: we're counting all contacts, not just changes.
            # Let me recompute energy properly.
            found = True
            break
    
    if not found:
        # Revert
        for i, pos in enumerate(seg_positions):
            occupied[pos] = seg_start + i
            chain[seg_start + i] = pos
        return None
    
    # Compute energy delta by comparing before and after
    # We already have the new chain, so let's just compute the delta
    # Actually, it's easier to compute the full energy change
    # Let me recompute: the delta is the change in total H-H contacts
    # We removed seg_positions and added path
    # For each H residue in the segment, count new contacts minus old contacts
    
    delta = 0
    for i in range(seg_start, seg_end + 1):
        if sequence[i] == "H":
            # Old contacts (with seg_positions[i])
            old_pos = seg_positions[i - seg_start]
            for q in _neighbours(old_pos):
                j = occupied.get(q)
                # But occupied now has the new positions... this is tricky
                # Let me just recompute energy difference differently
                pass
    
    # This is getting complicated. Let me just compute the energy delta by
    # counting all H-H contacts before and after, but that's O(n^2).
    # For simplicity, let's just return the move and let the caller compute energy.
    # Actually, let me restructure: return the new chain and let caller compute energy.
    
    return (chain, occupied, None)  # None means recompute energy


def _simulate_annealing(sequence, chain, occupied, rng, time_budget):
    """Run simulated annealing on a starting conformation.
    Returns (best_chain, best_energy)."""
    n = len(chain)
    best_chain = list(chain)
    best_energy = _calculate_energy(sequence, best_chain)
    current_energy = best_energy
    current_chain = list(chain)
    current_occupied = dict(occupied)
    
    # Temperature schedule
    t0 = max(0.5, -best_energy * 0.1)  # Initial temperature
    t_min = 0.01
    # Number of steps based on time budget
    # Estimate: each step is ~1-5 microseconds
    # Let's do a fixed number of steps proportional to sequence length
    num_steps = min(50000, max(1000, n * 500))
    
    start_time = time.time()
    for step in range(num_steps):
        # Check time
        if time.time() - start_time > time_budget:
            break
        
        # Temperature
        T = t0 * (t_min / t0) ** (step / max(1, num_steps - 1))
        
        # Choose move type
        if rng.random() < 0.6:
            # Single residue move
            idx = rng.randint(1, n - 2)
            old_pos = current_chain[idx]
            moved = False
            for new_pos in _neighbours(old_pos):
                if new_pos in current_occupied:
                    continue
                # Compute energy delta
                delta = _energy_delta_hh(sequence, idx, old_pos, new_pos, current_chain, current_occupied)
                new_energy = current_energy + delta
                # Acceptance criterion
                if new_energy < current_energy or rng.random() < math.exp(-(new_energy - current_energy) / T):
                    # Apply move
                    del current_occupied[old_pos]
                    current_occupied[new_pos] = idx
                    current_chain[idx] = new_pos
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(current_chain)
                    moved = True
                    break
            if not moved:
                continue
        else:
            # Crankshaft move
            if n < 5:
                continue
            seg_len = rng.randint(2, min(4, n - 2))
            seg_start = rng.randint(1, n - seg_len - 1)
            seg_end = seg_start + seg_len - 1
            
            left_anchor = current_chain[seg_start - 1]
            right_anchor = current_chain[seg_end + 1]
            
            seg_positions = current_chain[seg_start:seg_end + 1]
            for pos in seg_positions:
                del current_occupied[pos]
            
            start_options = [q for q in _neighbours(left_anchor) if q not in current_occupied and q != right_anchor]
            found = False
            for start_pos in start_options:
                # DFS to find path
                def dfs(pos, path, visited):
                    if len(path) == seg_len:
                        for q in _neighbours(pos):
                            if q == right_anchor:
                                return True
                        return False
                    for q in _neighbours(pos):
                        if q in visited or q in current_occupied:
                            continue
                        if q == right_anchor:
                            continue
                        visited.add(q)
                        path.append(q)
                        if dfs(q, path, visited):
                            return True
                        path.pop()
                        visited.remove(q)
                    return False
                
                visited = {start_pos}
                path = [start_pos]
                if dfs(start_pos, path, visited):
                    for i, pos in enumerate(path):
                        current_occupied[pos] = seg_start + i
                        current_chain[seg_start + i] = pos
                    # Compute energy delta
                    delta = 0
                    for i in range(seg_start, seg_end + 1):
                        if sequence[i] == "H":
                            for q in _neighbours(current_chain[i]):
                                j = current_occupied.get(q)
                                if j is not None and abs(j - i) > 1 and sequence[j] == "H":
                                    delta += 1
                    # But this counts all contacts, not just changes. Need to subtract old.
                    # Let me just recompute energy for the whole chain
                    new_energy = _calculate_energy(sequence, current_chain)
                    if new_energy < current_energy or rng.random() < math.exp(-(new_energy - current_energy) / T):
                        current_energy = new_energy
                        if new_energy < best_energy:
                            best_energy = new_energy
                            best_chain = list(current_chain)
                    else:
                        # Revert
                        for i, pos in enumerate(seg_positions):
                            current_occupied[pos] = seg_start + i
                            current_chain[seg_start + i] = pos
                    found = True
                    break
            
            if not found:
                for i, pos in enumerate(seg_positions):
                    current_occupied[pos] = seg_start + i
                    current_chain[seg_start + i] = pos
    
    return (best_chain, best_energy)


def fold(sequence, dimension):
    """Best conformation using greedy growth + simulated annealing."""
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    
    best_chain = None
    best_energy = 1  # Worst possible (no contacts)
    
    # Phase 1: Generate diverse starting conformations via greedy growth
    # Try different bias settings
    bias_configs = [
        (1.0, 0.1),
        (1.0, 0.3),
        (1.5, 0.1),
        (2.0, 0.05),
        (1.0, 0.5),
    ]
    
    starting_chains = []
    for cw, fw in bias_configs:
        for _ in range(20):
            if _time_left() < 5.0:
                break
            chain = _grow(sequence, dimension, rng, contact_weight=cw, free_weight=fw)
            if chain is not None:
                energy = _calculate_energy(sequence, chain)
                starting_chains.append((energy, chain))
    
    # Sort by energy and keep top few
    starting_chains.sort(key=lambda x: x[0])
    top_chains = starting_chains[:min(10, len(starting_chains))]
    
    if not top_chains:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = 0
        return [list(p) for p in best_chain]
    
    # Phase 2: Simulated annealing on each starting conformation
    # Allocate time budget for each
    remaining_time = _time_left()
    num_sa = len(top_chains)
    time_per_sa = max(0.1, remaining_time * 0.8 / max(1, num_sa))
    
    for energy, chain in top_chains:
        if _time_left() < 2.0:
            break
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
        
        # Run SA
        occupied = {p: i for i, p in enumerate(chain)}
        sa_chain, sa_energy = _simulate_annealing(sequence, chain, occupied, rng, time_per_sa)
        
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_chain = list(sa_chain)
    
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = 0
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
