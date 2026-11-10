"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import heapq
import copy


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

def _beam_search_fold(sequence, dimension, beam_width=100, rng=None):
    if rng is None:
        rng = random.Random()
        
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,)*dimension]
        
    # State: (score, chain, occupied_set, last_point)
    # We want to minimize score.
    # Score = Energy - heuristic_bonus
    # Energy is negative.
    
    start_point = (0,) * dimension
    start_chain = [start_point]
    start_occupied = {start_point}
    
    # Initial state
    # Heuristic: 0
    beam = [(0, start_chain, start_occupied, start_point)]
    
    for i in range(1, n):
        res = sequence[i]
        new_beam = []
        
        for score, chain, occupied, last_point in beam:
            # Try all neighbors
            for nb in _neighbours(last_point):
                if nb in occupied:
                    continue
                
                new_chain = chain + [nb]
                new_occupied = occupied | {nb}
                
                # Calculate incremental energy
                # Only check contacts involving the new residue 'nb'
                # New contacts are H-H pairs where one is 'nb' and the other is in chain[:-1]
                new_energy_delta = 0
                if res == "H":
                    for q in _neighbours(nb):
                        if q in occupied:
                            # Find index of q in chain
                            # Since we don't have index map in state, we can check sequence
                            # But we need to know if the neighbor is H and not adjacent in chain.
                            # The neighbor q is in occupied, so it's in chain.
                            # We need to ensure it's not the previous residue (chain[-1]).
                            # And it must be H.
                            if q != last_point:
                                # Find index of q
                                # This is O(N). We can optimize by keeping index map in state?
                                # That's heavy. Let's just check if sequence has H at that position.
                                # We can find index by iterating? No.
                                # Let's store index in occupied? occupied: {point: index}
                                # This is better.
                                pass
                
                # Let's refactor state to include index map for efficiency
                # State: (score, chain, occupied_dict, last_point)
                # occupied_dict: {point: index}
                
                # Re-evaluate with index map
                new_occupied_dict = dict(occupied)
                new_occupied_dict[nb] = i
                
                if res == "H":
                    for q in _neighbours(nb):
                        if q in new_occupied_dict:
                            j = new_occupied_dict[q]
                            if j != i - 1: # Not adjacent in chain
                                if sequence[j] == "H":
                                    new_energy_delta += 1
                
                current_energy = score[0] # Wait, score is not energy.
                # Let's store energy separately or calculate it.
                # It's better to store (energy, chain, occupied_dict, last_point, heuristic)
                
                # Let's restart the state definition for clarity in the loop.
                pass
        
        # The logic above is getting messy with inline definitions. 
        # Let's rewrite the loop properly.
        
        new_beam_candidates = []
        
        for state in beam:
            energy_so_far = state['energy']
            chain = state['chain']
            occupied_dict = state['occupied']
            last_point = state['last_point']
            
            for nb in _neighbours(last_point):
                if nb in occupied_dict:
                    continue
                
                new_chain = chain + [nb]
                new_occupied_dict = occupied_dict.copy()
                new_occupied_dict[nb] = i
                
                # Calculate energy delta
                delta_e = 0
                if res == "H":
                    for q in _neighbours(nb):
                        if q in new_occupied_dict:
                            j = new_occupied_dict[q]
                            if j != i - 1:
                                if sequence[j] == "H":
                                    delta_e += 1
                
                new_energy = energy_so_far - delta_e
                
                # Heuristic: Compactness
                # Count free neighbors of nb
                free_count = 0
                for q in _neighbours(nb):
                    if q not in new_occupied_dict:
                        free_count += 1
                
                # Score: Lower is better
                # We want low energy (negative) and low free count (compact)
                # score = new_energy - 0.1 * free_count
                # But energy is negative, so -0.1 * free_count makes it more negative if free_count is low?
                # No, if free_count is low, we want to prefer it.
                # Let's use: score = new_energy + 0.1 * free_count
                # If energy is -10, free is 2 -> score -9.8
                # If energy is -10, free is 5 -> score -9.5
                # -9.8 < -9.5, so we prefer lower free count. Correct.
                
                heuristic_weight = 0.1
                score = new_energy + heuristic_weight * free_count
                
                new_state = {
                    'score': score,
                    'energy': new_energy,
                    'chain': new_chain,
                    'occupied': new_occupied_dict,
                    'last_point': nb
                }
                new_beam_candidates.append(new_state)
        
        if not new_beam_candidates:
            break
            
        # Sort by score and keep top beam_width
        new_beam_candidates.sort(key=lambda x: x['score'])
        beam = new_beam_candidates[:beam_width]
        
    if not beam:
        return []
        
    # Find the best energy in the final beam
    best_state = min(beam, key=lambda x: x['energy'])
    return best_state['chain']

def fold(sequence, dimension):
    """
    Use Beam Search to construct the chain.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]*dimension]
        
    # Beam width can be adaptive or fixed.
    # For short chains, larger beam. For long, smaller to save time.
    if n < 20:
        bw = 200
    elif n < 50:
        bw = 150
    else:
        bw = 100
        
    # Run multiple beam searches with different randomizations?
    # Beam search is deterministic if tie-breaking is stable.
    # We can add noise to the score to explore different paths.
    
    best_chain = None
    best_energy = 0
    
    rng = random.Random(12345)
    
    # Run a few times with slight heuristic variations or noise
    for run in range(3):
        # Add noise to heuristic weight
        hw = 0.1 + rng.random() * 0.1
        
        # Modify _beam_search_fold to accept hw? 
        # Or just implement the loop here with hw.
        
        # Let's just call the function with fixed hw for simplicity first, 
        # but we can't easily pass hw without refactoring.
        # Let's refactor _beam_search_fold to take hw.
        
        chain = _beam_search_fold_heuristic(sequence, dimension, bw, hw)
        
        if chain:
            e = _energy(sequence, chain)
            if best_chain is None or e < best_energy:
                best_chain = chain
                best_energy = e
                
    if best_chain is None:
        # Fallback
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
    return [list(p) for p in best_chain]

def _beam_search_fold_heuristic(sequence, dimension, beam_width, hw):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,)*dimension]
        
    start_point = (0,) * dimension
    start_chain = [start_point]
    start_occupied = {start_point: 0}
    
    beam = [{
        'score': 0,
        'energy': 0,
        'chain': start_chain,
        'occupied': start_occupied,
        'last_point': start_point
    }]
    
    for i in range(1, n):
        res = sequence[i]
        new_beam_candidates = []
        
        for state in beam:
            energy_so_far = state['energy']
            chain = state['chain']
            occupied_dict = state['occupied']
            last_point = state['last_point']
            
            for nb in _neighbours(last_point):
                if nb in occupied_dict:
                    continue
                
                new_chain = chain + [nb]
                new_occupied_dict = occupied_dict.copy()
                new_occupied_dict[nb] = i
                
                delta_e = 0
                if res == "H":
                    for q in _neighbours(nb):
                        if q in new_occupied_dict:
                            j = new_occupied_dict[q]
                            if j != i - 1:
                                if sequence[j] == "H":
                                    delta_e += 1
                
                new_energy = energy_so_far - delta_e
                
                free_count = 0
                for q in _neighbours(nb):
                    if q not in new_occupied_dict:
                        free_count += 1
                
                score = new_energy + hw * free_count
                
                new_state = {
                    'score': score,
                    'energy': new_energy,
                    'chain': new_chain,
                    'occupied': new_occupied_dict,
                    'last_point': nb
                }
                new_beam_candidates.append(new_state)
        
        if not new_beam_candidates:
            break
            
        new_beam_candidates.sort(key=lambda x: x['score'])
        beam = new_beam_candidates[:beam_width]
        
    if not beam:
        return []
        
    best_state = min(beam, key=lambda x: x['energy'])
    return best_state['chain']

# EVOLVE-BLOCK-END
