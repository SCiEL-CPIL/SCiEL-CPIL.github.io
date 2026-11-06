"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = time.time() + 325  # safety margin before 330s kill


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Energy = -(# of H-H non-adjacent lattice contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _beam_search(sequence, dimension, beam_width=50):
    """
    Beam search: grow the chain step by step, keeping only the top `beam_width`
    partial chains at each step based on a heuristic score (contacts gained + free space).
    This systematically explores promising lattice paths, avoiding the local minima
    traps of pure stochastic growth.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]

    start = (0, 0, 0)
    # Each beam entry: (score, chain, occupied_set)
    # We use a list of dicts or tuples. Let's use (score, chain_tuple, occupied_set)
    # But sets are not hashable in a way that allows easy deduplication if we just store them.
    # We need to deduplicate partial chains.
    # A partial chain is defined by its path. Two different paths that end at the same
    # set of occupied sites are distinct states if the path history differs, but for
    # self-avoiding walks, the set of occupied sites uniquely determines the "shape"
    # only if we know the order? No, the order is the chain.
    # So we should deduplicate based on the tuple of the chain itself.
    
    current_beams = []
    start_chain = [start]
    start_occupied = {start}
    start_score = 0.0
    current_beams.append((start_score, start_chain, start_occupied))

    for i in range(1, n):
        next_beams = []
        seen_chains = set() # To deduplicate identical partial chains in this step

        for score, chain, occupied in current_beams:
            last_pos = chain[-1]
            residue = sequence[i]
            
            for site in _neighbours(last_pos):
                if site in occupied:
                    continue
                
                new_chain = chain + [site]
                new_occupied = occupied | {site}
                
                # Heuristic score for the partial chain
                # 1. Immediate H-H contacts gained by adding this residue
                gain = 0
                if residue == 'H':
                    for q in _neighbours(site):
                        if q in occupied:
                            # Check if the neighbor is an H residue
                            # We need to know the index of the neighbor.
                            # Since occupied is a set of positions, we need a map.
                            # This is inefficient. Let's store the chain and look up index.
                            # But chain is a list. 
                            # Optimization: Pre-compute H positions in the current chain?
                            # Or just check sequence[index].
                            # Finding index of q in chain is O(n).
                            # Let's maintain a position-to-index map in the beam?
                            # That's heavy.
                            
                            # Alternative: Just count potential contacts.
                            # If q is in occupied, it is some previous residue j.
                            # We need to know if sequence[j] == 'H'.
                            # We can store a dict {pos: index} in the beam state?
                            # Or just re-compute energy for the partial chain? 
                            # No, that's O(n^2).
                            
                            # Let's refine the heuristic:
                            # Gain = number of H neighbors in occupied.
                            # We need to know which neighbors are H.
                            # We can store a set of H positions in the occupied set?
                            # Let's store (score, chain, occupied, h_occupied)
                            # h_occupied is a set of positions that are H.
                            
                            pass # Placeholder

                # Let's restart the beam search implementation with better state tracking.
                # State: (score, chain_list, occupied_set, h_occupied_set)
                # This is getting complex. Let's simplify.
                
                # Simpler Beam Search:
                # Score = current_energy(partial_chain) + alpha * free_space
                # current_energy is easy to compute incrementally if we track contacts.
                # But computing energy from scratch for each candidate is O(n).
                # With beam width 50 and n=48, that's 50*48*6 ops per step. 48 steps.
                # 50 * 48 * 6 * 48 ~ 700k ops. Very fast.
                
                new_chain_tuple = tuple(new_chain)
                if new_chain_tuple in seen_chains:
                    continue
                seen_chains.add(new_chain_tuple)
                
                # Calculate new score
                # Energy of partial chain
                partial_energy = _energy(sequence[:i+1], new_chain)
                
                # Free space heuristic: number of unoccupied neighbors of the last residue
                free = sum(1 for q in _neighbours(site) if q not in new_occupied)
                
                # Score: lower energy is better (more negative), so we want to maximize -energy + free
                # Or minimize energy - free. Let's maximize score.
                # score = -partial_energy + 0.1 * free
                new_score = -partial_energy + 0.1 * free
                
                next_beams.append((new_score, new_chain, new_occupied))
        
        if not next_beams:
            break
            
        # Sort by score descending and keep top beam_width
        next_beams.sort(key=lambda x: x[0], reverse=True)
        current_beams = next_beams[:beam_width]
        
        if time.time() > _DEADLINE:
            break

    if not current_beams:
        return None
        
    # Return the best full chain
    best = max(current_beams, key=lambda x: x[0])
    return best[1]


def _local_refine(sequence, chain, dimension, rng, max_steps=100):
    """Local search with 1-residue moves and 2-residue kink moves."""
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    n = len(chain)
    if n < 3:
        return best_chain

    for step in range(max_steps):
        if time.time() > _DEADLINE:
            break
        
        moved = False
        
        # Try 1-residue move at a random internal position
        i = rng.randint(1, n - 2)
        
        occupied = set(chain)
        occupied.discard(chain[i])
        
        for site in _neighbours(chain[i]):
            if site in occupied:
                continue
            
            new_chain = chain[:]
            new_chain[i] = site
            
            # Check connectivity and self-avoidance
            # Self-avoidance is guaranteed by occupied check + site not in occupied
            # Connectivity:
            # dist(new_chain[i-1], site) == 1 and dist(site, new_chain[i+1]) == 1
            d_prev = sum(abs(site[a] - new_chain[i-1][a]) for a in range(dimension))
            d_next = sum(abs(site[a] - new_chain[i+1][a]) for a in range(dimension))
            
            if d_prev != 1 or d_next != 1:
                continue
            
            new_energy = _energy(sequence, new_chain)
            if new_energy < best_energy:
                best_chain = new_chain
                best_energy = new_energy
                chain = new_chain
                moved = True
                break
        
        if moved:
            continue
        
        # Try 2-residue kink move
        if n >= 4:
            i = rng.randint(1, n - 3)
            
            p1, p2 = chain[i], chain[i+1]
            prev_point = chain[i-1]
            next_point = chain[i+2]
            
            occupied = set(chain)
            occupied.discard(p1)
            occupied.discard(p2)
            
            candidates = []
            for new_p1 in _neighbours(prev_point):
                if new_p1 in occupied:
                    continue
                for new_p2 in _neighbours(new_p1):
                    if new_p2 in occupied:
                        continue
                    # Check if new_p2 is adjacent to next_point
                    diff = sum(abs(new_p2[a] - next_point[a]) for a in range(dimension))
                    if diff != 1:
                        continue
                    candidates.append((new_p1, new_p2))
            
            if candidates:
                new_p1, new_p2 = rng.choice(candidates)
                new_chain = chain[:]
                new_chain[i] = new_p1
                new_chain[i+1] = new_p2
                
                # Full validation just in case
                valid = True
                if len(set(new_chain)) != len(new_chain):
                    valid = False
                else:
                    for j in range(len(new_chain) - 1):
                        diff = sum(abs(new_chain[j][a] - new_chain[j+1][a]) for a in range(dimension))
                        if diff != 1:
                            valid = False
                            break
                
                if valid:
                    new_energy = _energy(sequence, new_chain)
                    if new_energy < best_energy:
                        best_chain = new_chain
                        best_energy = new_energy
                        chain = new_chain
                        moved = True
        
        if not moved:
            # Try a few more random positions before giving up
            for _ in range(5):
                i = rng.randint(1, n - 2)
                occupied = set(chain)
                occupied.discard(chain[i])
                
                for site in _neighbours(chain[i]):
                    if site in occupied:
                        continue
                    
                    new_chain = chain[:]
                    new_chain[i] = site
                    
                    d_prev = sum(abs(site[a] - new_chain[i-1][a]) for a in range(dimension))
                    d_next = sum(abs(site[a] - new_chain[i+1][a]) for a in range(dimension))
                    
                    if d_prev != 1 or d_next != 1:
                        continue
                    
                    new_energy = _energy(sequence, new_chain)
                    if new_energy < best_energy:
                        best_chain = new_chain
                        best_energy = new_energy
                        chain = new_chain
                        moved = True
                        break
                if moved:
                    break
        
        if not moved:
            break
    
    return best_chain


def fold(sequence, dimension):
    """
    Best conformation using beam search for initialization and local refinement.
    """
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 1

    # Determine beam width based on chain length
    if n <= 48:
        beam_width = 50
        num_refines = 5
    elif n <= 80:
        beam_width = 30
        num_refines = 3
    else:
        beam_width = 20
        num_refines = 2

    # 1. Beam Search Initialization
    # Run beam search once with a fixed seed for reproducibility in the search path
    # But we want diversity? Beam search is deterministic given the scoring.
    # To get diversity, we can run it multiple times with slight noise in scoring?
    # Or just run it once, then use local refinement with random restarts?
    # The insight says beam search is good. Let's run it a few times with different beam widths or noise.
    
    for attempt in range(num_refines):
        if time.time() > _DEADLINE:
            break
        
        # Add some noise to the beam search heuristic to explore different paths
        # We can modify the _beam_search to accept a noise parameter, or just run it as is.
        # Let's run it as is first.
        chain = _beam_search(sequence, dimension, beam_width=beam_width)
        
        if chain is None:
            continue
            
        value = _energy(sequence, chain)
        
        if value < best_energy:
            best_chain = chain
            best_energy = value
            
        # Refine
        refined = _local_refine(sequence, chain, dimension, rng, max_steps=100)
        refined_energy = _energy(sequence, refined)
        
        if refined_energy < best_energy:
            best_chain = refined
            best_energy = refined_energy

    # 2. Fallback / Additional Randomized Growth if Beam Search fails or time allows
    # If we have time, do a few randomized greedy growths as in the original program
    # to catch cases where beam search gets stuck or misses a good basin.
    
    if time.time() < _DEADLINE - 30:
        for attempt in range(50):
            if time.time() > _DEADLINE:
                break
            
            # Simple randomized growth
            start = (0, 0, 0)
            chain, occupied = [start], {start}
            success = True
            for i in range(1, n):
                options = []
                for site in _neighbours(chain[-1]):
                    if site in occupied:
                        continue
                    gain = 0
                    if sequence[i] == 'H':
                        gain = sum(1 for q in _neighbours(site)
                                   if q in occupied and sequence[occupied.get(q, -1)] == 'H' if occupied.get(q, -1) < i - 1)
                    free = sum(1 for q in _neighbours(site) if q not in occupied)
                    score = gain + 0.1 * free + rng.random()
                    options.append((score, site))
                
                if not options:
                    success = False
                    break
                
                site = max(options)[1]
                occupied[site] = i
                chain.append(site)
            
            if not success:
                continue
                
            value = _energy(sequence, chain)
            if value < best_energy:
                best_chain = chain
                best_energy = value
                
            if value <= best_energy + 2:
                refined = _local_refine(sequence, chain, dimension, rng, max_steps=50)
                refined_energy = _energy(sequence, refined)
                if refined_energy < best_energy:
                    best_chain = refined
                    best_energy = refined_energy

    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
