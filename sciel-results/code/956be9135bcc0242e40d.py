# EVOLVE-BLOCK-START
import random
import time

def _compute_energy(sequence, chain):
    """Compute HP contact energy: -number of non-adjacent H-H contacts."""
    n = len(chain)
    if n < 2:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            # Check 4 neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                q = (p[0] + dx, p[1] + dy)
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _is_valid_chain(chain):
    """Check if the chain is self-avoiding and has unit steps."""
    if not chain:
        return True
    occupied = set()
    for i, p in enumerate(chain):
        if p in occupied:
            return False
        occupied.add(p)
        if i > 0:
            prev = chain[i - 1]
            if abs(p[0] - prev[0]) + abs(p[1] - prev[1]) != 1:
                return False
    return True

def _local_search(sequence, chain, time_budget, rng):
    """
    Memetic local search using move and reverse operators.
    Greedy hill-climbing with occasional random restarts if stuck.
    """
    n = len(chain)
    if n <= 2:
        return chain

    current_chain = list(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 500
    stuck_threshold = 0.05 # seconds
    
    while time.time() - start_time < time_budget:
        if no_improve_count > max_no_improve:
            # If stuck for a long time, try a random perturbation or restart from best
            # Here we just restart the local search process from best_chain with a small random kick
            current_chain = list(best_chain)
            current_energy = best_energy
            no_improve_count = 0
            # Apply a random valid move to kick it out
            if n > 3:
                idx = rng.randint(1, n - 2)
                # Try to move a small segment
                # Simple kick: reverse a small segment
                seg_len = rng.randint(2, min(5, n - idx))
                end_idx = idx + seg_len - 1
                if end_idx < n:
                    new_chain = current_chain[:idx] + list(reversed(current_chain[idx:end_idx+1])) + current_chain[end_idx+1:]
                    if _is_valid_chain(new_chain):
                        current_chain = new_chain
                        current_energy = _compute_energy(sequence, new_chain)

        # Choose move: 50% move, 50% reverse
        move_applied = False
        if rng.random() < 0.5:
            # Move: select a segment and shift it by 1 in a random direction
            # Segment must be at least length 1, but moving single residues is often trivial or invalid.
            # Let's pick a random segment [i, j]
            i = rng.randint(1, n - 2) # Start index (0 is fixed head, n-1 is tail, but we can move any internal part)
            # Actually, moving the tail is also valid. Let's pick i from 1 to n-1.
            # If i == n-1, segment is just the last residue.
            j = rng.randint(i, n - 1)
            
            # Directions
            dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            rng.shuffle(dirs)
            
            for dx, dy in dirs:
                # Check if moving segment [i, j] by (dx, dy) is valid
                # 1. New positions must not overlap with fixed parts (0..i-1 and j+1..n-1)
                # 2. The connections at i-1/i and j/j+1 must be unit steps.
                
                # Optimization: Check collisions first
                # The segment moves as a rigid body.
                # Occupied set of fixed parts
                fixed_occupied = set(current_chain[:i]) | set(current_chain[j+1:])
                
                valid = True
                new_positions = []
                for k in range(i, j + 1):
                    ox, oy = current_chain[k]
                    nx, ny = ox + dx, oy + dy
                    # Check collision with fixed parts
                    if (nx, ny) in fixed_occupied:
                        valid = False
                        break
                    new_positions.append((nx, ny))
                
                if not valid:
                    continue
                    
                # Check internal collisions of the moved segment?
                # Since it's a rigid translation, internal relative positions don't change.
                # But we need to ensure the moved segment doesn't collide with itself?
                # It didn't before, and translation preserves distances, so no self-collision.
                # However, we must ensure the moved segment doesn't collide with the *other* part of the moved segment?
                # Wait, the moved segment is contiguous. Translation is rigid. No self-collision.
                
                # Check connectivity
                # Connection at i-1 to i
                if i > 0:
                    prev = current_chain[i-1]
                    first_new = new_positions[0]
                    if abs(prev[0] - first_new[0]) + abs(prev[1] - first_new[1]) != 1:
                        continue
                # Connection at j to j+1
                if j < n - 1:
                    next_ = current_chain[j+1]
                    last_new = new_positions[-1]
                    if abs(last_new[0] - next_[0]) + abs(last_new[1] - next_[1]) != 1:
                        continue
                
                # If all checks pass, apply move
                new_chain = current_chain[:i] + new_positions + current_chain[j+1:]
                # Note: We already checked collisions with fixed parts.
                # Is it possible that the new positions collide with each other? No, rigid translation.
                # Is it possible that the new positions collide with the *old* positions of the segment that are now empty?
                # The new positions are the old positions shifted.
                # We checked against fixed_occupied.
                # We did NOT check if new_positions intersect each other?
                # Since the original segment was self-avoiding, and we translated it rigidly, the new segment is self-avoiding.
                
                new_energy = _compute_energy(sequence, new_chain)
                current_chain = new_chain
                current_energy = new_energy
                move_applied = True
                break # Try next move if this one failed
        
        else:
            # Reverse: reverse segment [i, j]
            if n > 3:
                i = rng.randint(1, n - 2)
                j = rng.randint(i, n - 1)
                if j > i: # Need at least 2 elements to reverse meaningfully? 
                         # Reversing 1 element is identity.
                         # Reversing 2 elements swaps them.
                         # Let's allow j >= i. If i==j, it's identity, skip.
                         # Actually, standard reverse move reverses [i, j].
                         # If i==j, chain is unchanged.
                         # If i < j, chain changes.
                         pass
                
                if i < j:
                    new_chain = current_chain[:i] + list(reversed(current_chain[i:j+1])) + current_chain[j+1:]
                    if _is_valid_chain(new_chain):
                        new_energy = _compute_energy(sequence, new_chain)
                        current_chain = new_chain
                        current_energy = new_energy
                        move_applied = True

        if not move_applied:
            continue

        if current_energy < best_energy:
            best_chain = list(current_chain)
            best_energy = current_energy
            no_improve_count = 0
        else:
            no_improve_count += 1

    return best_chain

def _generate_initial_chain(sequence, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n <= 2:
        return [(0, 0), (1, 0)][:n]
    
    chain = [(0, 0)]
    occupied = {(0, 0)}
    # Start with a random direction
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    rng.shuffle(dirs)
    current_dir = dirs[0]
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for dx, dy in dirs:
            # Avoid immediate backtracking (optional, but helps SAW generation)
            # Actually, for SAW, we just need to avoid occupied.
            # Backtracking is always occupied (the previous cell), so it's automatically filtered.
            new_pos = (last[0] + dx, last[1] + dy)
            if new_pos not in occupied:
                candidates.append(new_pos)
        
        if not candidates:
            return None
        
        # Prefer straight or slight turns to avoid dead ends?
        # Random choice is fine for initialization.
        new_pos = rng.choice(candidates)
        chain.append(new_pos)
        occupied.add(new_pos)
        
    return chain

def _crossover(parent1, parent2, rng):
    """
    Crossover two valid chains.
    1. Pick a cut point k.
    2. Take head of P1 (0..k) and tail of P2 (k+1..n-1).
    3. The tail of P2 is in the coordinate system of P2.
    4. We need to translate/rotate the tail of P2 so that it connects to the end of P1's head.
    """
    n = len(parent1)
    if n < 4:
        return list(parent1)
    
    # Cut point
    k = rng.randint(1, n - 2)
    
    head1 = parent1[:k+1]
    tail2 = parent2[k+1:]
    
    # The last point of head1 is p_head_end
    # The first point of tail2 is p_tail_start
    # We need to transform tail2 such that p_tail_start' == p_head_end + direction
    # Wait, the connection must be a unit step.
    # In parent2, p_tail_start is connected to parent2[k] (which is p_tail_prev).
    # The vector from p_tail_prev to p_tail_start is v.
    # We want the vector from p_head_end to p_tail_start' to be v' (rotated/translated).
    # Actually, simpler:
    # We want to align the "exit" of head1 with the "entry" of tail2.
    
    p_head_end = head1[-1]
    p_head_prev = head1[-2] if len(head1) > 1 else (p_head_end[0]-1, p_head_end[1])
    # The direction we came from in head1 is not strictly relevant for the exit, 
    # but the exit direction is determined by where we place the first step of the tail.
    
    # Let's define the relative geometry of the tail.
    # tail2[0] is the first node of the tail.
    # tail2 is a valid SAW segment.
    # We need to apply a rigid transformation (rotation + translation) to tail2
    # such that tail2[0] becomes adjacent to p_head_end.
    
    # There are 4 possible positions for tail2[0] relative to p_head_end.
    # Let's pick one randomly, or try all 4?
    # Let's pick one randomly from the 4 neighbors of p_head_end that are not in head1.
    
    candidates = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        pos = (p_head_end[0] + dx, p_head_end[1] + dy)
        if pos not in head1: # Rough check, full check later
            candidates.append(pos)
            
    if not candidates:
        return list(parent1)
        
    target_pos = rng.choice(candidates)
    
    # We need to rotate/translate tail2 so that tail2[0] maps to target_pos.
    # Let's compute the transformation.
    # Original tail2[0] is at origin of tail's local frame? No, it's at some (x0, y0).
    # Let's shift tail2 so tail2[0] is at (0,0).
    x0, y0 = tail2[0]
    tail_local = [(x - x0, y - y0) for x, y in tail2]
    
    # Now tail_local[0] is (0,0).
    # We need to rotate tail_local by 0, 90, 180, 270 degrees.
    # Then translate by target_pos.
    
    rotations = [
        lambda x, y: (x, y),       # 0
        lambda x, y: (-y, x),      # 90
        lambda x, y: (-x, -y),     # 180
        lambda x, y: (y, -x)       # 270
    ]
    
    rng.shuffle(rotations)
    
    for rot_func in rotations:
        rotated_tail = [rot_func(x, y) for x, y in tail_local]
        translated_tail = [(x + target_pos[0], y + target_pos[1]) for x, y in rotated_tail]
        
        # Check if this tail connects to head1
        # The first point of translated_tail is target_pos.
        # It must be adjacent to p_head_end. (It is, by construction)
        # It must not collide with head1.
        # It must be self-avoiding (it is, from parent2).
        # It must not collide with itself (it is).
        
        # Check collision with head1
        valid = True
        head_set = set(head1)
        # The first point is target_pos, which we checked is not in head1.
        # Check the rest
        for p in translated_tail[1:]:
            if p in head_set:
                valid = False
                break
        
        if valid:
            # Also check that the connection is unit step (it is)
            # And that the whole chain is valid?
            # The tail is a valid SAW. The head is a valid SAW.
            # The only potential issue is collision between head and tail.
            # We checked tail[1:] against head.
            # We checked tail[0] against head.
            # So no collision.
            
            new_chain = head1 + translated_tail
            # Final sanity check
            if _is_valid_chain(new_chain):
                return new_chain
                
    return list(parent1) # Fallback if no valid crossover found

def fold(sequence, dimension):
    n = len(sequence)
    if n <= 2:
        return [[0, 0], [1, 0]][:n]
    
    rng = random.Random(42)
    
    # Time budget
    # 120s total. 10 sequences.
    # Let's allocate time based on n.
    time_budget = 10.0 # seconds per sequence
    
    # Population size
    pop_size = 20
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        init = _generate_initial_chain(sequence, rng)
        if init is None:
            init = [(i, 0) for i in range(n)]
        population.append(init)
        
    # Evaluate initial population
    energies = [_compute_energy(sequence, c) for c in population]
    
    best_chain = population[energies.index(min(energies))]
    best_energy = min(energies)
    
    start_time = time.time()
    generation = 0
    max_generations = 1000 # Safety cap
    
    while time.time() - start_time < time_budget and generation < max_generations:
        # Selection: Tournament selection
        def tournament_select():
            idx1 = rng.randint(0, pop_size - 1)
            idx2 = rng.randint(0, pop_size - 1)
            if energies[idx1] < energies[idx2]:
                return population[idx1]
            else:
                return population[idx2]
                
        new_population = []
        
        # Elitism: keep the best 2
        sorted_indices = sorted(range(pop_size), key=lambda i: energies[i])
        for i in sorted_indices[:2]:
            new_population.append(list(population[i]))
            new_population.append(list(population[i])) # Duplicate to keep size? Or just add once?
            # Let's just add the best one to ensure it's in the new pop.
            # Standard elitism: copy top K to new pop.
            
        # Reset new_population to just the best 1 or 2
        new_population = []
        for i in sorted_indices[:2]:
            new_population.append(list(population[i]))
            
        # Fill the rest with offspring
        while len(new_population) < pop_size:
            p1 = tournament_select()
            p2 = tournament_select()
            
            if rng.random() < 0.5:
                child = _crossover(p1, p2, rng)
            else:
                child = list(p1)
                # Mutation: apply a random local move?
                # Or just copy.
                # Let's apply a small random perturbation if we don't crossover
                # Actually, GA relies on crossover. Mutation can be local search.
                
            # Local search on the child
            # Allocate small time to each local search
            # Total time left / remaining children
            time_left = time_budget - (time.time() - start_time)
            remaining_children = pop_size - len(new_population)
            if remaining_children > 0:
                time_per_ls = max(0.01, time_left / (remaining_children * 2)) # Divide by 2 to be safe
            else:
                time_per_ls = 0.01
                
            improved_child = _local_search(sequence, child, time_per_ls, rng)
            new_population.append(improved_child)
            
        # Update population
        population = new_population
        energies = [_compute_energy(sequence, c) for c in population]
        
        current_best = min(energies)
        if current_best < best_energy:
            best_energy = current_best
            best_chain = population[energies.index(current_best)]
            
        generation += 1
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
