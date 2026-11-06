# EVOLVE-BLOCK-START
import time
import heapq
import math


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


def _beam_search_fold(sequence, dimension, beam_width=200, time_limit=300.0):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    start = (0, 0, 0)
    # State: (current_pos, occupied_set, path_list, energy_so_far, max_possible_gain)
    # We use a max-heap on (energy_so_far + max_possible_gain)
    
    # Precompute max possible gain from each position i
    # max_gain[i] = number of H residues in sequence[i+1:] 
    # This is a very loose upper bound, but admissible.
    # A tighter bound: for each H at i, count H's in i+1..n-1.
    # Let's precompute suffix counts of H.
    suffix_h = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_h[i] = suffix_h[i + 1] + (1 if sequence[i] == 'H' else 0)
    
    # For a state ending at residue i (0-indexed), the max additional contacts is bounded by 
    # (number of H in prefix) * (number of H in suffix) ... actually no.
    # Max additional contacts is bounded by the number of H-H pairs that can still be formed.
    # A simple admissible heuristic: 
    # h(i) = (number of H in sequence[0..i]) * (number of H in sequence[i+1..n-1])
    # But we already counted some.
    # Let's just use: h(i) = suffix_h[i+1] * (i+1) # very loose
    # Better: h(i) = number of H in suffix * (number of H in prefix)
    # Even better: just use the number of H in the suffix, assuming each can make 1 contact.
    # Actually, the max number of NEW contacts added by the remaining chain is bounded by 
    # the number of H in the remaining chain * (number of H already placed).
    # Let's use: h(i) = suffix_h[i+1] * (i+1) # Still loose but simple.
    
    # Let's try a simpler approach: 
    # Score = current_energy + (number of H in suffix) * 2 # heuristic
    # Or just current_energy + suffix_h[i+1]
    
    # Let's use: score = current_energy + suffix_h[i+1] * 2
    # This is not strictly admissible (could overestimate), but it's a good heuristic for beam search.
    # For strict A*, we need admissible. 
    # Max contacts = total H-H pairs possible.
    # Let's just use beam search with a heuristic that prioritizes high current energy.
    
    start_state = (start, {start}, [start], 0)
    # Heap items: (-heuristic_score, unique_id, state)
    # We want to maximize heuristic_score, so we minimize -heuristic_score.
    
    def get_heuristic(energy, i):
        # i is the index of the last placed residue (0-indexed)
        # Max possible additional energy is bounded by the number of H in the suffix
        # Let's use: energy + suffix_h[i+1] * 2
        return energy + suffix_h[i + 1] * 2
    
    counter = 0
    heap = []
    h0 = get_heuristic(0, 0)
    heapq.heappush(heap, (-h0, counter, start_state))
    counter += 1
    
    start_time = time.time()
    best_chain = None
    best_energy = -10**9
    
    # Beam search: keep top K states at each step
    # But with a heap, we can just pop the best one and expand, but we need to keep a frontier.
    # Standard beam search: keep a list of states, sort by score, keep top K.
    # Let's do that.
    
    frontier = [start_state]
    
    for i in range(1, n):
        next_frontier = []
        # Expand all states in frontier
        for state in frontier:
            last_pos, occupied, path, energy = state
            
            for neighbor in _neighbours(last_pos):
                if neighbor in occupied:
                    continue
                
                new_occupied = set(occupied)
                new_occupied.add(neighbor)
                new_path = path + [neighbor]
                
                # Calculate new energy
                new_energy = energy
                if sequence[i] == 'H':
                    # Check if this new H makes contacts with previous H's
                    for q in _neighbours(neighbor):
                        if q in occupied:
                            # Find index of q in path
                            # This is O(N) which is slow.
                            # We can store a map pos->index in the state?
                            # That would make states larger.
                            # Let's just recalc energy from scratch? No, too slow.
                            # Let's assume we have a map.
                            pass
                # Recalculating energy incrementally is tricky without a map.
                # Let's use a different state representation that includes a map?
                # Or just accept the O(N) cost for now and see if it's fast enough.
                # N is up to 136. 136 * 6 = 816 checks.
                # Beam width 200. 200 * 6 * 816 = ~1M ops per step.
                # 136 steps * 1M = 136M ops. Might be too slow in Python.
                
                # Let's try to optimize energy calculation.
                # We can store the set of H positions in the state?
                # state: (last_pos, occupied_set, h_positions_set, path, energy)
                
                # Let's restart with a better state definition.
                pass
        
        # This loop is getting complicated. Let's simplify.
        # Given the time constraints and the complexity of implementing an efficient beam search with incremental energy,
        # let's fall back to a robust local search with multiple random starts, but with a better heuristic.
        # The inspirations suggest beam search is key. Let's try a simpler beam search that recalculates energy.
        
        # Actually, let's just use the existing pivot optimize but with more restarts and a better growth.
        # The current program already does this.
        # Let's try to implement the beam search properly.
        
        break # Exit the broken loop for now
        
    # Fallback to a simpler strategy if beam search is too complex to implement quickly
    return _simple_fold(sequence, dimension, time_limit)


def _simple_fold(sequence, dimension, time_limit=300.0):
    n = len(sequence)
    start_time = time.time()
    
    # Multi-start local search
    # 1. Generate a random valid chain (greedy growth)
    # 2. Optimize with pivot moves
    # 3. Repeat
    
    def _grow():
        import random
        rng = random.Random()
        start = (0, 0, 0)
        chain, occupied = [start], {start: 0}
        for i in range(1, n):
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

    def _pivot_optimize(chain, steps):
        import random
        rng = random.Random()
        current_energy = _energy(sequence, chain)
        best_chain = list(chain)
        best_energy = current_energy
        n_local = len(chain)
        
        for _ in range(steps):
            if time.time() - start_time > time_limit - 5:
                break
            if n_local < 4:
                break
            pivot_idx = rng.randint(1, n_local - 2)
            p_prev = chain[pivot_idx - 1]
            p_pivot = chain[pivot_idx]
            
            v = [p_pivot[i] - p_prev[i] for i in range(dimension)]
            bond_axis = -1
            for i in range(dimension):
                if v[i] != 0:
                    bond_axis = i
                    break
            
            rot_axes = [i for i in range(dimension) if i != bond_axis]
            if not rot_axes:
                continue
            r_axis = rng.choice(rot_axes)
            angle = rng.choice([1, -1])
            
            def rotate_point(point, center, axis, angle):
                p = list(point)
                c = list(center)
                p[0] -= c[0]
                p[1] -= c[1]
                p[2] -= c[2]
                
                if axis == 0:
                    if angle == 1:
                        p[1], p[2] = -p[2], p[1]
                    else:
                        p[1], p[2] = p[2], -p[1]
                elif axis == 1:
                    if angle == 1:
                        p[0], p[2] = p[2], -p[0]
                    else:
                        p[0], p[2] = -p[2], p[0]
                elif axis == 2:
                    if angle == 1:
                        p[0], p[1] = -p[1], p[0]
                    else:
                        p[0], p[1] = p[1], -p[0]
                
                p[0] += c[0]
                p[1] += c[1]
                p[2] += c[2]
                return tuple(p)
            
            head_set = set(chain[:pivot_idx])
            valid = True
            new_tail_points = []
            occupied_tail = set()
            
            for i in range(pivot_idx + 1, n_local):
                orig_point = chain[i]
                new_point = rotate_point(orig_point, p_pivot, r_axis, angle)
                
                if new_point in head_set:
                    valid = False
                    break
                if new_point in occupied_tail:
                    valid = False
                    break
                if new_tail_points:
                    prev_new = new_tail_points[-1]
                    dist = sum(abs(new_point[k] - prev_new[k]) for k in range(dimension))
                    if dist != 1:
                        valid = False
                        break
                occupied_tail.add(new_point)
                new_tail_points.append(new_point)
            
            if not valid:
                continue
            
            candidate_chain = chain[:pivot_idx + 1] + new_tail_points
            new_energy = _energy(sequence, candidate_chain)
            
            if new_energy < best_energy:
                best_chain = candidate_chain
                best_energy = new_energy
                
        return best_chain, best_energy

    best_chain = None
    best_energy = 0
    
    # Try multiple restarts
    max_restarts = 50
    for _ in range(max_restarts):
        if time.time() - start_time > time_limit - 10:
            break
        
        chain = _grow()
        if chain is None:
            continue
        
        opt_steps = 100 + n
        optimized_chain, opt_energy = _pivot_optimize(chain, opt_steps)
        
        if optimized_chain is not None:
            if best_chain is None or opt_energy < best_energy:
                best_chain = optimized_chain
                best_energy = opt_energy

    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
    return best_chain


def fold(sequence, dimension):
    """Best conformation of a fixed number of randomized greedy growths and local optimizations."""
    start_time = time.time()
    deadline = start_time + 300 # Safety margin
    
    # Try beam search first? No, let's stick to the robust local search for now.
    # The directive suggests a different strategy. Let's implement a proper beam search.
    
    # Actually, let's just use the _simple_fold which is similar to the current program but cleaner.
    # To be "materially different", let's try a simulated annealing approach?
    # Or a genetic algorithm?
    
    # Let's stick to the beam search idea but implement it correctly this time.
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    # Beam Search Implementation
    # State: (last_pos, occupied_set, h_positions_set, path, energy)
    # We keep a beam of width W.
    
    W = 100 # Beam width
    
    start = (0, 0, 0)
    start_state = (start, {start}, {start} if sequence[0] == 'H' else set(), [start], 0)
    
    beam = [start_state]
    
    for i in range(1, n):
        next_beam = []
        for state in beam:
            last_pos, occupied, h_positions, path, energy = state
            
            for neighbor in _neighbours(last_pos):
                if neighbor in occupied:
                    continue
                
                new_occupied = set(occupied)
                new_occupied.add(neighbor)
                
                new_h_positions = set(h_positions)
                new_energy = energy
                
                if sequence[i] == 'H':
                    new_h_positions.add(neighbor)
                    # Check for new contacts
                    for q in _neighbours(neighbor):
                        if q in h_positions:
                            new_energy -= 1 # One new contact
                
                new_path = path + [neighbor]
                next_beam.append((neighbor, new_occupied, new_h_positions, new_path, new_energy))
        
        # Sort next_beam by energy (most negative is best)
        next_beam.sort(key=lambda x: x[4])
        beam = next_beam[:W]
        
        if time.time() - start_time > deadline - 10:
            break
    
    if beam:
        # Pick the best state
        best_state = min(beam, key=lambda x: x[4])
        best_chain = best_state[3]
    else:
        # Fallback
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
