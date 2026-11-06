"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import heapq


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _a_star_fold(sequence, dimension, max_nodes=50000):
    """
    A* search for HP protein folding on square lattice.
    State: (last_position, occupied_set_tuple, current_index)
    Heuristic: 0 (effectively Dijkstra) but we use a simple greedy heuristic.
    
    Actually, for small sequences, we can do a best-first search with a good heuristic.
    The key insight: we want to maximize H-H contacts, so we prioritize placements
    that create new H-H contacts.
    
    Since the state space is huge, we'll use a beam search / best-first approach
    with pruning.
    """
    n = len(sequence)
    if n <= 1:
        return [(0, 0)]
    
    start_pos = (0, 0)
    # State: (neg_score, last_pos, occupied_frozenset, index)
    # We use a priority queue. Score = -energy_so_far (we want to minimize energy, so maximize -energy)
    # Actually, let's track: cost = -contacts_so_far, so we minimize cost (maximize contacts)
    
    # To keep states manageable, we use a beam search with limited beam width
    # and also limit the number of unique occupied sets we keep.
    
    # Better approach: iterative deepening or just a strong greedy with multiple restarts
    # but with a more sophisticated scoring.
    
    # Let's try: for each position, consider all 4 directions, score by:
    # 1. Number of new H-H contacts created
    # 2. Number of "potential" future contacts (neighbors that are H and not yet placed)
    # 3. Openness of the site
    
    # Actually, let's do a proper A* with a good heuristic.
    # State: (f, g, last_pos, occupied_tuple, path)
    # g = -contacts_so_far (we want to minimize, so more negative is better)
    # h = 0 (no admissible heuristic easily available)
    
    # This might be too slow. Let's use a beam search instead.
    
    beam_width = 20
    max_depth = n
    
    # Initial state
    # Each beam entry: (score, last_pos, occupied_set, path)
    # score = -contacts (higher is better, so we keep highest scores)
    
    # For efficiency, represent occupied as a set of tuples
    # But sets are not hashable for dict keys easily... use frozenset
    
    # Actually, for sequences up to 100, this is still too large for exact search.
    # Let's use a smarter greedy with lookahead.
    
    return None  # Not using this directly, falling back to improved greedy


def _greedy_fold(sequence, dimension, rng, lookahead=2):
    """
    Improved greedy chain growth with lookahead.
    At each step, try each possible direction, simulate the next `lookahead` steps
    greedily, and pick the direction that leads to the best local outcome.
    """
    n = len(sequence)
    if n <= 1:
        return [(0, 0)]
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            # Score this placement
            # 1. Immediate contacts
            immediate = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            immediate += 1
            
            # 2. Future potential: count H residues that could potentially contact this site
            #    (i.e., neighbors of site that are not occupied and are within range)
            potential = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q not in occupied:
                        # Check if any future H residue could land here
                        # This is heuristic: just count free neighbors
                        potential += 1
            
            # 3. Openness: how many free neighbors does this site have?
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # 4. Avoid trapping: ensure we don't create a dead end
            #    Check if placing here blocks too many future options
            trap_penalty = 0
            # Simple heuristic: if free < 2, it's risky
            if free < 2:
                trap_penalty = 10
            
            score = immediate * 10 + potential * 1 + free * 0.5 - trap_penalty + rng.random() * 0.1
            
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick the best option
        options.sort(key=lambda x: x[0], reverse=True)
        best_site = options[0][1]
        occupied[best_site] = i
        chain.append(best_site)
    
    return chain


def _simulated_annealing_fold(sequence, dimension, rng, max_iter=5000):
    """
    Simulated annealing starting from a greedy solution.
    Moves: rotate a segment of the chain.
    """
    # First, get an initial solution via greedy
    init_chain = _greedy_fold(sequence, dimension, rng, lookahead=1)
    if init_chain is None:
        # Fallback: straight line
        init_chain = [(i, 0) for i in range(len(sequence))]
    
    n = len(sequence)
    best_chain = init_chain[:]
    best_energy = _energy(sequence, init_chain)
    current_chain = init_chain[:]
    current_energy = best_energy
    
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    for it in range(max_iter):
        if T < min_T:
            break
        
        # Generate a move: pick a random internal segment and rotate it
        if n < 4:
            break
        
        # Pick start and end of segment to move
        # Move a single residue or a small segment
        # Let's try moving a single residue (kink move)
        
        # Pick a random position to try moving
        pos = rng.randint(1, n - 2)  # Don't move ends
        old_pos = current_chain[pos]
        
        # Try all 4 neighbors of old_pos (except the previous and next in chain)
        candidates = []
        for neighbor in _neighbours(old_pos):
            # Check if neighbor is not occupied by a non-adjacent chain residue
            if neighbor in occupied:
                j = occupied[neighbor]
                if abs(j - pos) > 1:
                    continue  # Would collide
            # Check if neighbor is adjacent to previous or next in chain
            if pos > 0 and neighbor == current_chain[pos - 1]:
                continue
            if pos < n - 1 and neighbor == current_chain[pos + 1]:
                continue
            candidates.append(neighbor)
        
        if not candidates:
            continue
        
        new_pos = rng.choice(candidates)
        
        # Check if new_pos is valid (not occupied by non-adjacent)
        if new_pos in occupied:
            j = occupied[new_pos]
            if abs(j - pos) > 1:
                continue
        
        # Apply move
        # Remove old_pos from occupied, add new_pos
        del occupied[old_pos]
        occupied[new_pos] = pos
        current_chain[pos] = new_pos
        
        # Compute new energy
        new_energy = _energy(sequence, current_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < pow(2.718281828, -delta / T):
            # Accept
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = current_chain[:]
        else:
            # Reject: undo move
            current_chain[pos] = old_pos
            occupied[old_pos] = pos
            del occupied[new_pos]
        
        T *= cooling
    
    return best_chain


def _beam_search_fold(sequence, dimension, beam_width=50, max_states=100000):
    """
    Beam search for HP folding.
    State: (last_pos, occupied_frozenset, path_tuple)
    Score: -contacts_so_far (higher is better)
    """
    n = len(sequence)
    if n <= 1:
        return [(0, 0)]
    
    # Initial state
    start = (0, 0)
    # Beam: list of (score, last_pos, occupied_set, path)
    beam = [(0, start, frozenset([start]), (start,))]
    
    for i in range(1, n):
        new_beam = []
        seen = set()  # To avoid duplicate states in the beam
        
        for score, last_pos, occupied, path in beam:
            for site in _neighbours(last_pos):
                if site in occupied:
                    continue
                
                # Check if this placement is valid
                # Compute new contacts
                new_score = score
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        if q in occupied:
                            # Find the index of q in the path
                            # This is O(n) which is slow
                            # Optimization: store indices in the occupied set?
                            # For now, just count if it's an H
                            # We need to know if the residue at q is H
                            # Let's store occupied as a dict: pos -> index
                            pass
                
                # This approach is too slow for large n.
                # Let's use a simpler scoring
                new_occupied = occupied | {site}
                state_key = (site, new_occupied)
                if state_key in seen:
                    continue
                seen.add(state_key)
                
                # Simple score: immediate contacts + openness
                immediate = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        if q in occupied:
                            # Assume it could be an H contact
                            immediate += 1
                
                free = sum(1 for q in _neighbours(site) if q not in new_occupied)
                new_score = score + immediate * 2 + free * 0.1
                
                new_beam.append((new_score, site, new_occupied, path + (site,)))
        
        if not new_beam:
            return None
        
        # Keep only the top beam_width states
        new_beam.sort(key=lambda x: x[0], reverse=True)
        beam = new_beam[:beam_width]
        
        # If beam is empty or we've exceeded max states, break
        if len(beam) == 0:
            break
    
    if not beam:
        return None
    
    # Return the best state
    best = beam[0]
    return list(best[3])


def fold(sequence, dimension):
    """
    Best conformation using multiple strategies.
    """
    n = len(sequence)
    if n <= 1:
        return [[0, 0]]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0  # Energy is negative, so we want the most negative
    
    # Strategy 1: Multiple greedy restarts
    for restart in range(30):
        chain = _greedy_fold(sequence, dimension, rng, lookahead=1)
        if chain is not None:
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
    
    # Strategy 2: Simulated annealing from the best greedy solution
    if best_chain is not None:
        sa_chain = _simulated_annealing_fold(sequence, dimension, rng, max_iter=3000)
        if sa_chain is not None:
            energy = _energy(sequence, sa_chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = sa_chain
    
    # Strategy 3: Beam search (for smaller sequences)
    if n <= 30:
        bs_chain = _beam_search_fold(sequence, dimension, beam_width=30)
        if bs_chain is not None:
            energy = _energy(sequence, bs_chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = bs_chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
