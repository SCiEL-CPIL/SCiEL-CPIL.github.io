"""
HP lattice protein folding using Beam Search over partial self-avoiding walks.
This strategy maintains a population of the K lowest-energy partial conformations
at each step, pruning inferior paths to manage the combinatorial explosion.
"""
# EVOLVE-BLOCK-START
import time
import heapq
import math

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    # Safety margin: 210s limit, leave 10s buffer -> 200s usable
    return 200.0 - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _fold_beam_search(sequence, dimension, beam_width=50, time_limit=10.0):
    """
    Perform Beam Search for the given sequence.
    Returns the best full conformation found.
    """
    n = len(sequence)
    is_h = [c == 'H' for c in sequence]
    
    # State: (energy, chain_tuple, occupied_set, last_index)
    # We use a min-heap for efficiency, but since we prune to beam_width,
    # a sorted list or heap is fine. We'll use a list and sort each step.
    
    # Initial state
    start_pos = (0, 0)
    initial_chain = [start_pos]
    initial_occupied = {start_pos}
    initial_energy = 0
    
    # Beam: list of (energy, chain, occupied)
    # We store chain as a list of tuples for easy extension
    beam = [(0, initial_chain, initial_occupied)]
    
    best_full_chain = None
    best_full_energy = float('inf')
    
    start_time = time.time()
    
    for i in range(1, n):
        new_beam = []
        current_residue = sequence[i]
        is_current_h = is_h[i]
        
        # Expand all states in the current beam
        for energy, chain, occupied in beam:
            last_pos = chain[-1]
            
            # Try all 4 neighbors
            for nb in _neighbours(last_pos):
                if nb in occupied:
                    continue
                
                new_chain = chain + [nb]
                new_occupied = occupied | {nb}
                
                # Calculate energy delta
                delta_e = 0
                if is_current_h:
                    # Check neighbors of nb for H contacts
                    for q in _neighbours(nb):
                        if q in new_occupied:
                            # Find the index of q in the chain
                            # Optimization: We can maintain a position map, 
                            # but for beam search with small n, linear search 
                            # or a dict update is acceptable. 
                            # Let's use a dict for position lookup for speed.
                            pass 
                
                # To speed up energy calculation, we should maintain a position map 
                # or simply recompute. Given the beam width is small (50-100) and 
                # n is up to 100, recomputing energy from scratch for each candidate 
                # is O(N^2) per candidate, which might be slow.
                # Better: Incremental energy.
                # We need to know if 'nb' forms an H-H contact.
                # Contacts are with previous H residues that are spatially adjacent.
                
                # Let's refactor to maintain a 'h_positions' set or similar?
                # Actually, simple check:
                # If current is H, check its 4 neighbors. If any neighbor is occupied 
                # by an H residue at index < i-1, we have a contact.
                
                # We need to know if the occupied neighbor is H and not the immediate predecessor.
                # Let's maintain a dict {pos: index} for the chain? 
                # That's expensive to update. 
                # Alternative: Just check the 4 neighbors of 'nb'.
                # If 'nb' is H:
                if is_current_h:
                    for q in _neighbours(nb):
                        if q in new_occupied:
                            # Is q the immediate predecessor (chain[-2])?
                            if len(new_chain) >= 2 and q == new_chain[-2]:
                                continue
                            # Is the residue at q Hydrophobic?
                            # We need to know the type of the residue at q.
                            # We can store the chain types or just look up index.
                            # Let's assume we have a helper to get index.
                            # For speed in beam search, let's precompute a list of H indices?
                            # No, the position of H changes with the path.
                            
                            # Efficient check:
                            # We can maintain a set of 'H positions' in the state.
                            # State: (energy, chain, occupied, h_positions_set)
                            # This adds memory but makes delta E O(1) or O(4).
                            
                            pass
                # For the sake of this implementation, I will use a slightly 
                # heavier state that includes a set of H positions to allow O(1) 
                # contact checking.
                
                # Re-evaluating strategy for speed:
                # State: (energy, chain_list, occupied_set, h_positions_set)
                # When adding new residue at 'nb':
                # 1. Add 'nb' to occupied.
                # 2. If is_current_h:
                #    a. Add 'nb' to h_positions.
                #    b. Check neighbors of 'nb'. For each neighbor q in h_positions:
                #       if q is not chain[-2] (predecessor), delta_e += 1.
                # 3. If not is_current_h:
                #    a. h_positions remains same.
                #    b. delta_e = 0.
                
                # Wait, the state in the loop above didn't have h_positions.
                # I will rewrite the loop to include it.
                
                pass # Placeholder to restart logic with correct state
                
        # The previous loop was pseudocode. Let's implement the real one below.
        
        # Break outer loop structure to implement correctly
        break
        
    # Real Implementation:
    
    beam = [(0, [start_pos], {start_pos}, set())] # (energy, chain, occupied, h_pos_set)
    
    if is_h[0]:
        beam[0] = (0, [start_pos], {start_pos}, {start_pos})
        
    for i in range(1, n):
        new_beam = []
        is_current_h = is_h[i]
        
        for state in beam:
            energy, chain, occupied, h_pos = state
            last_pos = chain[-1]
            pred_pos = chain[-2] if len(chain) > 1 else None
            
            for nb in _neighbours(last_pos):
                if nb in occupied:
                    continue
                
                new_energy = energy
                new_h_pos = h_pos
                new_occupied = occupied | {nb}
                
                if is_current_h:
                    new_h_pos = h_pos | {nb}
                    # Check contacts
                    for q in _neighbours(nb):
                        if q in h_pos:
                            # q is an H residue.
                            # Check if it's the immediate predecessor
                            if q != pred_pos:
                                new_energy -= 1 # Energy is -contacts
                                # Note: multiple contacts with same q? No, q is a single site.
                                # But q can be one of 4 neighbors.
                
                new_chain = chain + [nb]
                new_beam.append((new_energy, new_chain, new_occupied, new_h_pos))
                
        # Prune: Keep only the best 'beam_width' states
        # Sort by energy (min heap order)
        new_beam.sort(key=lambda x: x[0])
        
        # If we are near the end, we might want to keep more or less?
        # Standard beam search keeps fixed width.
        beam = new_beam[:beam_width]
        
        # Time check
        if time.time() - start_time > time_limit:
            break
            
    # After the loop, find the best full chain
    if beam:
        # The beam contains partial chains if we broke early, or full chains.
        # If we completed the loop, all chains in beam are length n.
        # If we broke early, we need to extend them greedily or pick the best partial?
        # The task requires a full conformation.
        # If time runs out, we should return the best *valid full* conformation found.
        # Since Beam Search builds sequentially, we only have full conformations 
        # if we finished the loop.
        # If we broke early, we have partial chains. We must extend them to full length.
        
        best_state = min(beam, key=lambda x: x[0])
        chain, occupied, h_pos = best_state[1], best_state[2], best_state[3]
        
        # If chain is not full length, extend it greedily to avoid invalid output
        if len(chain) < n:
            for j in range(len(chain), n):
                is_j_h = is_h[j]
                last_pos = chain[-1]
                pred_pos = chain[-2] if len(chain) > 1 else None
                best_nb = None
                best_e = 0
                for nb in _neighbours(last_pos):
                    if nb in occupied:
                        continue
                    e = 0
                    if is_j_h:
                        for q in _neighbours(nb):
                            if q in h_pos and q != pred_pos:
                                e -= 1
                    if best_nb is None or e < best_e:
                        best_e = e
                        best_nb = nb
                if best_nb is None:
                    # Dead end, pick any valid neighbor if possible, else break
                    # This shouldn't happen often if beam width is decent
                    for nb in _neighbours(last_pos):
                        if nb not in occupied:
                            best_nb = nb
                            break
                if best_nb is None:
                    break
                chain.append(best_nb)
                occupied.add(best_nb)
                if is_j_h:
                    h_pos.add(best_nb)
                    
        return [list(p) for p in chain]
    else:
        # Fallback
        return [list(p) for p in [(i, 0) for i in range(n)]]

def fold(sequence, dimension):
    """
    Fold the sequence using Beam Search.
    """
    _init_time()
    n = len(sequence)
    
    # Adjust beam width based on time remaining and sequence length
    # Longer sequences need more time, but also explode faster.
    # Short sequences can use wider beams.
    if n <= 20:
        beam_width = 200
        time_limit = 5.0
    elif n <= 50:
        beam_width = 100
        time_limit = 10.0
    else:
        beam_width = 50
        time_limit = 15.0
        
    # We have a total budget of ~195s.
    # We can run multiple beam searches with different beam widths or randomizations?
    # Beam search is deterministic. 
    # To get diversity, we can randomize the initial direction or use different beam widths.
    
    best_chain = None
    best_energy = 0 # 0 is neutral, we want negative
    
    # Strategy: Run a few beam searches with slightly different parameters or seeds?
    # Beam search doesn't use random seeds directly unless we randomize tie-breaking.
    # Let's randomize the tie-breaking in the pruning step to explore different topologies.
    
    import random
    
    for run in range(5):
        if _time_remaining() < 10:
            break
            
        rng = random.Random(12345 + run)
        
        # Modify the beam search to accept a randomizer for tie-breaking
        # Or just run the same beam search. Since it's deterministic, 
        # running it 5 times yields the same result.
        # We need to inject randomness into the pruning or expansion.
        
        # Let's modify the internal logic to use rng for tie-breaking in sort.
        # Python's sort is stable, but if energies are equal, order is arbitrary.
        # We can shuffle equal-energy states before pruning.
        
        chain = _fold_beam_search_randomized(sequence, dimension, beam_width, time_limit, rng)
        
        if chain is not None and len(chain) == n:
            # Validate and compute energy
            # Simple validation
            valid = True
            occupied = set()
            for i, p in enumerate(chain):
                if p in occupied:
                    valid = False
                    break
                occupied.add(p)
                if i > 0:
                    prev = chain[i-1]
                    if abs(p[0]-prev[0]) + abs(p[1]-prev[1]) != 1:
                        valid = False
                        break
            
            if valid:
                # Calculate energy
                is_h = [c == 'H' for c in sequence]
                energy = 0
                pos_to_idx = {p: i for i, p in enumerate(chain)}
                for i, p in enumerate(chain):
                    if is_h[i]:
                        for q in _neighbours(p):
                            j = pos_to_idx.get(q)
                            if j is not None and j > i + 1 and is_h[j]:
                                energy -= 1
                                
                if best_chain is None or energy < best_energy:
                    best_chain = chain
                    best_energy = energy
                    
    if best_chain is None:
        # Fallback: Straight line
        best_chain = [list(p) for p in [(i, 0) for i in range(n)]]
        
    return best_chain

def _fold_beam_search_randomized(sequence, dimension, beam_width, time_limit, rng):
    n = len(sequence)
    is_h = [c == 'H' for c in sequence]
    start_pos = (0, 0)
    
    # State: (energy, chain, occupied, h_pos)
    initial_chain = [start_pos]
    initial_occupied = {start_pos}
    initial_h = {start_pos} if is_h[0] else set()
    initial_energy = 0
    
    beam = [(initial_energy, initial_chain, initial_occupied, initial_h)]
    
    start_time = time.time()
    
    for i in range(1, n):
        new_beam = []
        is_current_h = is_h[i]
        
        for state in beam:
            energy, chain, occupied, h_pos = state
            last_pos = chain[-1]
            pred_pos = chain[-2] if len(chain) > 1 else None
            
            for nb in _neighbours(last_pos):
                if nb in occupied:
                    continue
                
                new_energy = energy
                new_h_pos = h_pos
                new_occupied = occupied | {nb}
                
                if is_current_h:
                    new_h_pos = h_pos | {nb}
                    for q in _neighbours(nb):
                        if q in h_pos:
                            if q != pred_pos:
                                new_energy -= 1
                
                new_chain = chain + [nb]
                new_beam.append((new_energy, new_chain, new_occupied, new_h_pos))
                
        # Prune with randomization for ties
        # Sort by energy
        new_beam.sort(key=lambda x: x[0])
        
        # Randomize order of states with the same energy to explore different paths
        # Find the max energy in the top beam_width
        if len(new_beam) > beam_width:
            cutoff_energy = new_beam[beam_width - 1][0]
            
            # Separate states into 'definitely keep' (energy < cutoff) and 'maybe keep' (energy == cutoff)
            keep = []
            maybe = []
            for state in new_beam:
                if state[0] < cutoff_energy:
                    keep.append(state)
                elif state[0] == cutoff_energy:
                    maybe.append(state)
                else:
                    break # Since sorted, rest are worse
            
            # Need to fill remaining slots from 'maybe'
            slots_needed = beam_width - len(keep)
            if slots_needed > 0:
                rng.shuffle(maybe)
                keep.extend(maybe[:slots_needed])
            
            beam = keep
        else:
            beam = new_beam
            
        if time.time() - start_time > time_limit:
            break
            
    if not beam:
        return None
        
    # Pick the best state from the final beam
    best_state = min(beam, key=lambda x: x[0])
    chain = best_state[1]
    occupied = best_state[2]
    h_pos = best_state[3]
    
    # If we broke early, extend greedily
    if len(chain) < n:
        for j in range(len(chain), n):
            is_j_h = is_h[j]
            last_pos = chain[-1]
            pred_pos = chain[-2] if len(chain) > 1 else None
            best_nb = None
            best_e = float('inf')
            
            candidates = []
            for nb in _neighbours(last_pos):
                if nb in occupied:
                    continue
                e = 0
                if is_j_h:
                    for q in _neighbours(nb):
                        if q in h_pos and q != pred_pos:
                            e -= 1
                candidates.append((e, nb))
                
            if candidates:
                candidates.sort(key=lambda x: x[0])
                # Pick best, or random among best if tie
                best_e_val = candidates[0][0]
                best_opts = [nb for e, nb in candidates if e == best_e_val]
                best_nb = rng.choice(best_opts)
                
                chain.append(best_nb)
                occupied.add(best_nb)
                if is_j_h:
                    h_pos.add(best_nb)
            else:
                # Dead end, return None to signal failure? 
                # Or just break and return partial? 
                # Task requires full length. 
                # If we hit a dead end, this path is invalid. 
                # Since we are in a greedy extension of a beam state, 
                # it's better to return None so the caller tries another beam state or run.
                return None
                
    return chain
# EVOLVE-BLOCK-END
