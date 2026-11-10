"""HP lattice protein folding on cubic lattice - fast local search with pivot moves."""
import random
import time
import math

_GLOBAL_START = time.time()
_HARD_DEADLINE = _GLOBAL_START + 315.0  # Safety margin before 330s kill


def _neighbours(point):
    """Generate 6 neighbours in 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain, occupied=None):
    """Compute energy: -number of H-H contacts (non-adjacent)."""
    n = len(chain)
    if occupied is None:
        occupied = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = occupied.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _delta_energy_move(sequence, chain, i, old_pos, new_pos, occupied):
    """Compute energy change when moving residue i from old_pos to new_pos.
    Returns delta such that new_energy = old_energy - delta (delta > 0 means improvement)."""
    if sequence[i] != 'H':
        return 0
    delta = 0
    # Lost contacts at old_pos
    for q in _neighbours(old_pos):
        j = occupied.get(q)
        if j is not None and j != i and sequence[j] == 'H':
            delta -= 1
    # Gained contacts at new_pos
    for q in _neighbours(new_pos):
        j = occupied.get(q)
        if j is not None and j != i and sequence[j] == 'H':
            delta += 1
    return delta


def _pivot_move(sequence, chain, pivot_idx, direction, occupied, n):
    """
    Perform a pivot move: rotate the segment from pivot_idx to end by 90 degrees.
    Returns (new_chain, new_energy_delta, success) where success is True if move is valid.
    """
    if pivot_idx < 1 or pivot_idx >= n - 1:
        return None, 0, False
    
    # Get the current position of pivot
    pivot_pos = chain[pivot_idx]
    prev_pos = chain[pivot_idx - 1]
    
    # The vector from prev to pivot
    vec = (pivot_pos[0] - prev_pos[0], pivot_pos[1] - prev_pos[1], pivot_pos[2] - prev_pos[2])
    
    # Find the 3 perpendicular directions (90 degree rotations)
    # vec is one of (±1,0,0), (0,±1,0), (0,0,±1)
    axis = 0 if vec[0] != 0 else (1 if vec[1] != 0 else 2)
    sign = 1 if vec[axis] > 0 else -1
    
    # The two other axes
    other_axes = [a for a in range(3) if a != axis]
    
    # Possible new vectors: rotate by ±90 around the axis perpendicular to vec
    # Actually, for a pivot move, we rotate the entire tail segment
    # The new direction for the first bond of the tail can be any of the 3 other directions
    # (excluding the reverse of vec, which would go back)
    
    new_vectors = []
    for oa in other_axes:
        for s in (-1, 1):
            nv = [0, 0, 0]
            nv[oa] = s
            new_vectors.append(tuple(nv))
    
    # Try each new direction
    for new_vec in new_vectors:
        # Skip if it's the same as current or reverse
        if new_vec == vec:
            continue
        if new_vec == (-vec[0], -vec[1], -vec[2]):
            continue
        
        # Build new positions for the tail
        new_tail = []
        last_pos = pivot_pos
        for k in range(pivot_idx + 1, n):
            # Move in the new direction for the first step, then continue with same relative direction
            if k == pivot_idx + 1:
                new_pos = (last_pos[0] + new_vec[0], last_pos[1] + new_vec[1], last_pos[2] + new_vec[2])
            else:
                # For subsequent steps, we need to rotate the entire segment
                # This is complex; let's use a simpler approach
                # Actually, pivot moves rotate the entire tail rigidly
                # Let's compute the rotation properly
                pass
        
        # This approach is getting complex. Let's use a different strategy:
        # For pivot moves, we'll just try to move the pivot residue and see if it works
        # Or use a simpler "kink" move
        
        break
    
    # Simpler approach: try moving a single residue (kink move)
    idx = pivot_idx
    old = chain[idx]
    prev = chain[idx - 1]
    next_pos = chain[idx + 1]
    
    candidates = []
    for site in _neighbours(prev):
        if site == old:
            continue
        if site in occupied and occupied[site] != idx:
            continue
        dist = sum(abs(site[a] - next_pos[a]) for a in range(3))
        if dist == 1:
            candidates.append(site)
    
    if not candidates:
        return None, 0, False
    
    site = candidates[direction % len(candidates)]
    delta = _delta_energy_move(sequence, chain, idx, old, site, occupied)
    delta_e = -delta  # Energy change
    
    # Check self-avoidance (already done via occupied check)
    # Apply the move
    new_chain = list(chain)
    new_chain[idx] = site
    new_occupied = dict(occupied)
    new_occupied[old] = -1
    new_occupied[site] = idx
    
    return new_chain, delta_e, True


def _grow_greedy(sequence, dimension, rng, temperature=0.0, compact_bias=0.0):
    """Greedy growth with optional randomness and compactness bias. Returns chain or None."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    
    for i in range(1, n):
        best_score = -1e10
        best_site = None
        options = []
        
        # Compute centroid for compactness
        if compact_bias > 0 and len(chain) > 1:
            cx = sum(p[0] for p in chain) / len(chain)
            cy = sum(p[1] for p in chain) / len(chain)
            cz = sum(p[2] for p in chain) / len(chain)
        else:
            cx = cy = cz = 0
        
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # Compactness: distance from centroid
            if compact_bias > 0:
                dist = abs(site[0] - cx) + abs(site[1] - cy) + abs(site[2] - cz)
                compact_score = -dist * compact_bias
            else:
                compact_score = 0
            
            score = gain + 0.05 * free + compact_score
            if temperature > 0:
                score += rng.gauss(0, temperature)
            options.append((score, site))
        
        if not options:
            return None
        
        if temperature > 0:
            # Boltzmann selection
            max_score = max(s for s, _ in options)
            weights = [math.exp((s - max_score) / max(temperature, 1e-6)) for s, _ in options]
            total = sum(weights)
            r = rng.random() * total
            cumsum = 0
            best_site = options[0][1]
            for (score, site), w in zip(options, weights):
                cumsum += w
                if r <= cumsum:
                    best_site = site
                    break
        else:
            best_score, best_site = max(options)
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain


def _local_search_fast(sequence, chain, rng, deadline, max_iters=5000):
    """Fast local search with kink moves only (no 2-opt)."""
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    current_chain = list(best_chain)
    current_occupied = {p: i for i, p in enumerate(current_chain)}
    current_energy = best_energy
    
    t0 = 3.0
    t_end = 0.1
    alpha = (t_end / t0) ** (1.0 / max(max_iters, 1))
    T = t0
    
    iters = 0
    while time.time() < deadline and iters < max_iters:
        iters += 1
        T *= alpha
        if T < t_end:
            T = t_end
        
        # Choose a residue to move (kink move)
        idx = rng.randint(1, n - 2)
        old = current_chain[idx]
        prev = current_chain[idx - 1]
        next_pos = current_chain[idx + 1]
        
        candidates = []
        for site in _neighbours(prev):
            if site == old:
                continue
            if site in current_occupied and current_occupied[site] != idx:
                continue
            dist = sum(abs(site[a] - next_pos[a]) for a in range(3))
            if dist == 1:
                candidates.append(site)
        
        if not candidates:
            continue
        
        site = rng.choice(candidates)
        delta = _delta_energy_move(sequence, current_chain, idx, old, site, current_occupied)
        delta_e = -delta
        
        accept = delta_e < 0 or (T > 0 and rng.random() < math.exp(-delta_e / T))
        
        if accept:
            current_occupied[old] = -1
            current_occupied[site] = idx
            current_chain[idx] = site
            current_energy += delta_e
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation via diverse growth + fast local search."""
    n = len(sequence)
    rng = random.Random(42 + len(sequence) * 7 + sum(ord(c) for c in sequence))
    
    # Time budget: adapt dynamically
    per_seq_time = min(10.0, max(2.0, (_HARD_DEADLINE - time.time()) / 26.0))
    deadline = time.time() + per_seq_time
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Generate diverse initial chains
    initial_chains = []
    
    for temp in [0.0, 0.15, 0.3]:
        if time.time() > deadline - 4.0:
            break
        for compact in [0.0, 0.25]:
            chain = _grow_greedy(sequence, dimension, rng, temperature=temp, compact_bias=compact)
            if chain:
                initial_chains.append(chain)
    
    # Evaluate all initial chains
    for chain in initial_chains:
        e = _compute_energy(sequence, chain)
        if best_chain is None or e < best_energy:
            best_chain = chain
            best_energy = e
    
    if best_chain is None:
        # Fallback: simple snake pattern
        best_chain = [(i % 8, (i // 8) % 8, (i // 64) % 8) for i in range(n)]
        if len(set(best_chain)) < n:
            best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
    
    # Phase 2: Fast local search on best initial chain
    if time.time() < deadline - 0.5:
        best_chain, best_energy = _local_search_fast(
            sequence, best_chain, rng, deadline - 0.5, max_iters=10000
        )
    
    # Phase 3: If time remains, try local search on second-best initial chain
    remaining = deadline - time.time()
    if remaining > 1.5 and len(initial_chains) > 1:
        scored = [(_compute_energy(sequence, c), c) for c in initial_chains]
        scored.sort()
        if len(scored) >= 2:
            e2, c2 = scored[1]
            if e2 > best_energy - 3:
                c2, e2 = _local_search_fast(sequence, c2, rng, time.time() + remaining - 0.3, max_iters=5000)
                if e2 < best_energy:
                    best_chain, best_energy = c2, e2
    
    return [list(p) for p in best_chain]
