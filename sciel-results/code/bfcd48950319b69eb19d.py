"""HP lattice protein folding - beam search with local search refinement."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = time.time() + 295  # safety margin within 330s wall-clock


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy: negative count of H-H contacts (non-consecutive neighbours)."""
    n = len(chain)
    if n < 2:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _partial_energy(sequence, chain):
    """
    Compute partial energy for a partial chain (for beam search).
    Only counts contacts where both residues are already placed.
    """
    n = len(chain)
    if n < 2:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _beam_search(sequence, dimension, beam_width, time_budget, rng):
    """
    Beam search: grow the chain step by step, keeping only the top `beam_width`
    partial conformations at each step based on partial energy.
    Returns the best complete conformation found.
    """
    n = len(sequence)
    if n <= 1:
        return [(0,) * dimension]
    
    # Each beam entry: (energy, chain)
    # Start with just the first residue
    beams = [(0, [(0,) * dimension])]
    
    start_time = time.time()
    
    for i in range(1, n):
        # Check time
        if time.time() - start_time > time_budget:
            break
        
        new_beams = []
        for energy, chain in beams:
            last_pt = chain[-1]
            occupied = set(chain)
            
            for site in _neighbours(last_pt):
                if site in occupied:
                    continue
                
                new_chain = chain + [site]
                # Compute the energy change from adding this residue
                # Only need to check contacts involving the new residue
                delta_e = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        # Check if q is in the chain (excluding the last position)
                        # We need to find if q is at position j where j < i-1
                        # and sequence[j] == 'H'
                        # Since we're building incrementally, we can check the chain
                        for j in range(i - 1):
                            if new_chain[j] == q and sequence[j] == "H":
                                delta_e -= 1
                                break
                
                new_energy = energy + delta_e
                new_beams.append((new_energy, new_chain))
        
        if not new_beams:
            # No valid extensions, return best so far (shouldn't happen for valid SAW)
            break
        
        # Sort by energy (lower is better) and keep top beam_width
        new_beams.sort(key=lambda x: x[0])
        beams = new_beams[:beam_width]
    
    # Return the best complete chain
    if beams:
        best_energy, best_chain = beams[0]
        if len(best_chain) == n:
            return best_chain
    
    # Fallback: return a random walk
    return _random_walk(sequence, dimension, rng)


def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n <= 1:
        return [(0,) * dimension]
    
    chain = [(0,) * dimension]
    occupied = {chain[0]: 0}
    
    for i in range(1, n):
        candidates = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                candidates.append(site)
        
        if not candidates:
            return None
        
        site = rng.choice(candidates)
        occupied[site] = i
        chain.append(site)
    
    return chain


def _pivot_move(chain, pivot, rng, dimension):
    """Apply a pivot move: rotate the tail after pivot by 90 degrees."""
    n = len(chain)
    if pivot >= n - 1:
        return None
    
    bond = [chain[pivot + 1][a] - chain[pivot][a] for a in range(dimension)]
    bond_axis = None
    for a in range(dimension):
        if bond[a] != 0:
            bond_axis = a
            break
    if bond_axis is None:
        return None
    
    other_axes = [a for a in range(dimension) if a != bond_axis]
    rot_axis = rng.choice(other_axes)
    
    pivot_pt = chain[pivot]
    new_chain = list(chain[:pivot + 1])
    occupied = set(chain[:pivot + 1])
    
    new_tail = []
    for pt in chain[pivot + 1:]:
        rel = [pt[a] - pivot_pt[a] for a in range(dimension)]
        new_rel = list(rel)
        if rot_axis == 0:  # around x
            new_rel[1], new_rel[2] = -rel[2], rel[1]
        elif rot_axis == 1:  # around y
            new_rel[0], new_rel[2] = rel[2], -rel[0]
        else:  # around z
            new_rel[0], new_rel[1] = -rel[1], rel[0]
        
        new_pt = tuple(pivot_pt[a] + new_rel[a] for a in range(dimension))
        new_tail.append(new_pt)
    
    for pt in new_tail:
        if pt in occupied:
            return None
        occupied.add(pt)
    
    new_chain.extend(new_tail)
    return new_chain


def _local_search(sequence, chain, rng, time_budget, dimension):
    """Monte Carlo local search to improve conformation energy."""
    n = len(chain)
    if n <= 1:
        return chain
    
    current = list(chain)
    current_energy = _energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    start_time = time.time()
    temp = 1.0
    cooling = 0.999
    
    while time.time() - start_time < time_budget:
        move_type = rng.random()
        
        if move_type < 0.8:
            pivot = rng.randint(0, n - 2)
            new_chain = _pivot_move(current, pivot, rng, dimension)
            if new_chain is None:
                continue
        else:
            if n < 3:
                continue
            # Try a small segment move
            seg_len = rng.choice([2, 3])
            if n <= seg_len:
                continue
            start = rng.randint(0, n - seg_len - 1)
            # This is a simplified move; just do a pivot for now
            pivot = start
            new_chain = _pivot_move(current, pivot, rng, dimension)
            if new_chain is None:
                continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta <= 0:
            current = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best = list(new_chain)
                best_energy = new_energy
        else:
            if temp > 0.01:
                prob = math.exp(-delta / temp)
                if rng.random() < prob:
                    current = new_chain
                    current_energy = new_energy
        
        temp *= cooling
    
    return best


def fold(sequence, dimension):
    """Fold the sequence using beam search followed by local search refinement."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(12345 + len(sequence))
    
    # Time budget allocation
    # Beam search is the primary method, local search is refinement
    remaining = _DEADLINE - time.time()
    if remaining < 5.0:
        # Very little time, just do a quick random walk
        chain = _random_walk(sequence, dimension, rng)
        if chain is None:
            chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in chain]
    
    # Allocate time: 60% for beam search, 40% for local search refinement
    beam_time = remaining * 0.6
    local_time = remaining * 0.4
    
    # Beam width scales with sequence length but capped
    # For short sequences (48), use wider beam; for long (136), use narrower
    if n <= 50:
        beam_width = min(200, max(50, int(beam_time * 10)))
    elif n <= 100:
        beam_width = min(150, max(30, int(beam_time * 5)))
    else:
        beam_width = min(100, max(20, int(beam_time * 3)))
    
    # Run beam search
    best_chain = _beam_search(sequence, dimension, beam_width, beam_time, rng)
    
    if best_chain is None or len(best_chain) != n:
        # Fallback to random walk
        best_chain = _random_walk(sequence, dimension, rng)
        if best_chain is None:
            best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    # Verify the chain is valid
    if len(set(best_chain)) != n:
        best_chain = _random_walk(sequence, dimension, rng)
        if best_chain is None:
            best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    # Run local search refinement on the beam search result
    time_left = _DEADLINE - time.time()
    if time_left > 1.0:
        refined = _local_search(sequence, best_chain, rng, min(local_time, time_left - 0.5), dimension)
        if refined is not None and len(refined) == n and len(set(refined)) == n:
            best_chain = refined
    
    # Final verification
    if len(best_chain) != n:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    if len(set(best_chain)) != n:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
