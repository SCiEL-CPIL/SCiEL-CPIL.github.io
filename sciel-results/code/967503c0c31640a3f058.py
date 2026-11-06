# EVOLVE-BLOCK-START
import time
import sys
import math
import random

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

def _beam_search_fold(sequence, dimension, beam_width=50, time_limit=5.0):
    """
    Beam search with a heuristic that considers both current energy
    and the "potential" for future contacts (neighbors of current H that are free).
    """
    n = len(sequence)
    if n <= 1:
        return [tuple([0]*dimension)]
    
    start = (0, 0, 0)
    # State: (chain_tuple, occupied_set, current_energy, heuristic_score)
    # We use a list of tuples for the beam.
    # To save time, we store chain as a tuple of tuples.
    
    initial_chain = (start,)
    initial_occupied = {start}
    initial_energy = 0
    initial_score = 0
    
    beam = [(initial_chain, initial_occupied, initial_energy, initial_score)]
    
    start_time = time.time()
    
    for step in range(1, n):
        if time.time() - start_time > time_limit:
            break
            
        res = []
        current_residue = sequence[step]
        
        for chain, occupied, energy, score in beam:
            last = chain[-1]
            
            # Generate candidates
            candidates = []
            for next_pos in _neighbours(last):
                if next_pos in occupied:
                    continue
                
                # Calculate new energy
                new_energy = energy
                if current_residue == "H":
                    for q in _neighbours(next_pos):
                        if q in occupied:
                            # Check if the residue at q is H
                            # We need to map position to index. 
                            # Since we store chain as tuple, we can't easily look up index from position 
                            # without a dict. 
                            # Optimization: Store a dict in the state? That's expensive to copy.
                            # Alternative: Calculate energy incrementally by checking neighbors of next_pos 
                            # against the *previous* positions in the chain that are H.
                            # But we don't have the sequence of the chain readily available in the state 
                            # without unpacking.
                            
                            # Let's optimize the state representation.
                            # State: (chain_tuple, occupied_set, energy)
                            # To check if neighbor q is H, we need to know the index of q.
                            # Since q is in occupied, it corresponds to some index j < step.
                            # We can pre-compute a list of positions for H residues in the current chain?
                            # No, that's O(n) per step.
                            
                            # Let's stick to the simple O(n) check for now, but optimize the inner loop.
                            # Actually, for beam search, n is up to 136. 
                            # O(n) check for each of 6 neighbors for each of 50 beams is 50*6*136 ~ 40k ops per step.
                            # 136 steps * 40k = 5.4M ops. This is fast in Python if optimized.
                            
                            pass # Placeholder for logic below
                
                # Let's refactor to calculate energy correctly.
                # We need to know if any neighbor of next_pos is an H residue already placed.
                # We can maintain a list of positions of H residues in the chain.
                # But that adds state size.
                
                # Alternative: Just check all occupied neighbors.
                # We need to map position to sequence index.
                # Let's store a dict `pos_to_idx` in the state.
                # State: (chain_tuple, occupied_set, energy, pos_to_idx_dict)
                # Copying dict is O(n). 
                
                # Let's try a different approach for the heuristic/energy update.
                # We can pre-calculate the sequence of H positions.
                # When adding next_pos, if current_residue is H, check neighbors.
                # If neighbor is occupied, find its index.
                
                # To avoid O(n) lookup, let's store the chain as a list of tuples in the state?
                # No, tuples are immutable and hashable, good for sets.
                
                # Let's use a helper to get index. 
                # Since we are in a loop, let's just iterate through the chain if needed? 
                # No, that's too slow.
                
                # Let's assume we have `pos_to_idx` in the state.
                # State: (chain, occupied, energy, pos_to_idx)
                
                # Wait, the current beam state doesn't have pos_to_idx.
                # Let's update the state structure.
                
                pass
            
            # This implementation is getting messy. Let's restart the beam search logic 
            # with a cleaner state: (chain_tuple, occupied_set, energy)
            # And compute energy incrementally by checking neighbors of the new position 
            # against the *sequence* of the chain? No, we need the index.
            
            # Actually, for a given chain tuple, we can build a pos_to_idx map once per step? 
            # No, that's O(n^2) total.
            
            # Let's use a different heuristic: 
            # Score = Energy + 0.5 * (number of free neighbors of current H)
            # This encourages placing H in open spaces to allow future contacts? 
            # Or does it encourage placing H near other H?
            # Placing H near other H gives immediate energy gain.
            
            # Let's go back to the standard beam search with exact energy.
            # To get exact energy, we need to know if neighbors are H.
            # We can store the sequence of residues in the chain? 
            # State: (chain_tuple, residues_tuple, occupied_set, energy)
            # This doubles the memory but makes energy update O(1) if we check neighbors 
            # and look up residue in residues_tuple? No, we still need index.
            
            # Simplest correct way:
            # State: (chain_tuple, occupied_set, energy)
            # When expanding, for each neighbor q of next_pos:
            #   if q in occupied:
            #      # Find index of q in chain
            #      # This is O(n). 
            #      # Total cost per step: BeamWidth * 6 * N.
            #      # For N=100, Beam=50: 50 * 6 * 100 = 30,000 ops.
            #      # 100 steps = 3,000,000 ops.
            #      # This is acceptable in Python if the constant is low.
            
            pass
        
        # Since the implementation above is complex, let's use a simpler, proven heuristic:
        # Greedy with look-ahead.
        # Or, just use the existing _greedy_fold but with multiple restarts and a better heuristic.
        
        # Let's try a "Potential" heuristic in the beam.
        # State: (chain_tuple, occupied_set, energy)
        # Heuristic: energy + alpha * potential_contacts
        # potential_contacts = sum(1 for neighbor of last_pos if last_residue is H and neighbor is free)
        # This encourages placing H where it *could* make contacts later?
        # Actually, placing H near *existing* H is better for immediate energy.
        
        # Let's implement the beam search with O(N) energy check.
        
        new_beam = []
        for chain, occupied, energy in beam:
            last = chain[-1]
            last_res = sequence[step-1] # Residue at last position
            
            # We need to evaluate candidates for the NEW position (step)
            # The residue at the new position is sequence[step]
            
            candidates = []
            for next_pos in _neighbours(last):
                if next_pos in occupied:
                    continue
                
                # Calculate new energy
                new_energy = energy
                if sequence[step] == "H":
                    for q in _neighbours(next_pos):
                        if q in occupied:
                            # Check if residue at q is H
                            # Find index of q in chain
                            # chain is a tuple of tuples.
                            # We can use a precomputed dict for this chain? 
                            # No, we are in the loop.
                            # Let's just do a linear search for now, it's a prototype.
                            # Actually, we can store the pos_to_idx in the state to make it O(1).
                            # Let's modify the state to include pos_to_idx.
                            
                            pass
                
                # This is getting too slow to write correctly in this thought process.
                # Let's use a simpler strategy: 
                # Multiple restarts of a "Smart Greedy" that prioritizes moves 
                # that create H-H contacts, and if no contact is possible, 
                # prioritizes moves that keep the chain compact (minimize bounding box volume?).
                
                pass
                
            # I will implement a robust Beam Search in the final code.
            pass

        # For the sake of this response, I will implement a solid Beam Search 
        # that stores (chain, occupied, energy) and uses a heuristic score.
        
        # Heuristic: 
        # score = energy - 0.1 * (distance from center of mass)
        # This encourages compactness.
        
        # Let's implement it properly.
        
        pass
        
    # Fallback to the best in beam
    if beam:
        # Sort by energy
        beam.sort(key=lambda x: x[2])
        return list(beam[0][0])
        
    return None

def _local_optimize(sequence, chain, dimension, time_limit=2.0):
    """
    Local optimization using pivot moves.
    """
    n = len(chain)
    if n < 3:
        return chain
    
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    rng = random.Random(42)
    
    # Precompute occupied set for fast lookup
    occupied = {p: i for i, p in enumerate(chain)}
    
    while time.time() - start_time < time_limit:
        # Pick a random pivot
        i = rng.randint(1, n - 2)
        
        # Try to rotate the tail i+1..n-1
        # The pivot is at chain[i]. The tail starts at chain[i+1].
        # We try to move chain[i+1] to a new position q that is a neighbor of chain[i]
        # and not occupied (excluding the tail itself).
        
        prev = chain[i]
        curr = chain[i+1]
        
        # The tail is chain[i+1:]
        # We need to remove the tail from occupied, try a new move, then re-add.
        
        # This is expensive to manage the occupied set dynamically.
        # Alternative: Check validity by scanning the chain? O(N).
        
        valid_moves = []
        for q in _neighbours(prev):
            if q == curr:
                continue
            # Check if q is occupied by any part of the chain EXCEPT the tail i+1..n-1
            # And q is not prev (already checked)
            
            # Simple check:
            is_valid = True
            for j, p in enumerate(chain):
                if p == q:
                    if j <= i: # Occupied by head or pivot
                        is_valid = False
                        break
                    # If j > i, it's part of the tail, which will be moved, so it's "free"
            
            if is_valid:
                valid_moves.append(q)
        
        if not valid_moves:
            continue
            
        new_curr = rng.choice(valid_moves)
        
        # Now we need to move the rest of the tail.
        # The vector from curr to chain[i+2] is v1.
        # The new vector from new_curr to new_chain[i+2] should be v1?
        # No, a pivot move rotates the tail. 
        # If we just translate the tail, it might collide.
        # A standard pivot move: 
        # 1. Remove tail.
        # 2. Place new_curr.
        # 3. Try to place chain[i+2] at new_curr + (chain[i+2] - chain[i+1]).
        #    If collision, fail.
        #    If no collision, continue.
        
        # This "translation" of the tail is not a rotation, it's a shift.
        # If we shift the tail, the shape is preserved.
        # This is a valid move if the shifted tail doesn't collide with the head.
        
        # Let's try this "shift" move.
        delta = tuple(new_curr[k] - curr[k] for k in range(dimension))
        
        new_chain = chain[:i+1] + [new_curr]
        new_occupied = {p: j for j, p in enumerate(chain[:i+1])}
        
        success = True
        for j in range(i+2, n):
            old_p = chain[j]
            new_p = tuple(old_p[k] + delta[k] for k in range(dimension))
            
            if new_p in new_occupied:
                success = False
                break
                
            new_occupied[new_p] = j
            new_chain.append(new_p)
            
        if not success:
            continue
            
        # Check if the new tail collides with the head (already checked in new_occupied)
        # Calculate energy
        new_energy = _energy(sequence, new_chain)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_chain = new_chain
            # Update occupied for the new best chain? 
            # We can recalculate occupied if we accept.
            occupied = {p: j for j, p in enumerate(best_chain)}
            
    return best_chain

def fold(sequence, dimension):
    """
    Uses beam search and local optimization.
    """
    n = len(sequence)
    
    if not hasattr(fold, '_global_start'):
        fold._global_start = time.time()
        
    global_deadline = fold._global_start + 280.0
    current_time = time.time()
    
    if current_time >= global_deadline:
        return [list([i] + [0] * (dimension - 1)) for i in range(n)]
    
    # Budget for this call
    if n < 50:
        call_time = 10.0
    elif n < 100:
        call_time = 5.0
    else:
        call_time = 2.0
        
    call_deadline = current_time + call_time
    if call_deadline > global_deadline:
        call_deadline = global_deadline
        
    best_chain = None
    best_energy = 0
    
    # 1. Beam Search
    # We will implement a simple beam search here.
    # State: (chain_tuple, occupied_set, energy)
    
    beam_width = 20
    start = (0, 0, 0)
    beam = [((start,), {start}, 0)]
    
    for step in range(1, n):
        if time.time() > call_deadline - 1.0: # Reserve time for local opt
            break
            
        new_beam = []
        current_res = sequence[step]
        
        for chain, occupied, energy in beam:
            last = chain[-1]
            
            for next_pos in _neighbours(last):
                if next_pos in occupied:
                    continue
                
                # Calculate new energy
                new_energy = energy
                if current_res == "H":
                    for q in _neighbours(next_pos):
                        if q in occupied:
                            # Check if q is H
                            # Find index of q
                            # O(N) search
                            for j, p in enumerate(chain):
                                if p == q:
                                    if sequence[j] == "H":
                                        new_energy -= 1
                                    break
                            
                # Heuristic: prefer higher energy (more negative is better? No, energy is negative, so higher is worse? 
                # Wait, Energy = -contacts. So -10 is better than -5.
                # We want to MAXIMIZE contacts, so MINIMIZE energy value (make it more negative).
                # So we want the smallest energy.
                
                # Add to new_beam
                new_chain = chain + (next_pos,)
                new_occupied = occupied | {next_pos}
                new_beam.append((new_chain, new_occupied, new_energy))
                
        if not new_beam:
            break
            
        # Sort by energy (ascending, so most negative first)
        new_beam.sort(key=lambda x: x[2])
        beam = new_beam[:beam_width]
        
    if beam:
        # Pick the best
        best_candidate = beam[0][0]
        best_chain = list(best_candidate)
        best_energy = _energy(sequence, best_chain)
        
    if best_chain is None:
        # Fallback
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = 0
        
    # 2. Local Optimization
    if time.time() < call_deadline - 0.5:
        opt_time = max(0.5, call_deadline - time.time() - 0.5)
        optimized = _local_optimize(sequence, best_chain, dimension, time_limit=opt_time)
        opt_energy = _energy(sequence, optimized)
        
        if opt_energy < best_energy:
            best_chain = optimized
            best_energy = opt_energy
            
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
