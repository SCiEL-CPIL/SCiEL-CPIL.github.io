# EVOLVE-BLOCK-START
import time
import math
import heapq
import sys

# Global deadline for the entire process (240s budget, keep 30s safety margin)
# The evaluator kills at 330s, but the contract says 240s CPU budget.
# We will be conservative to ensure we finish within the strictest interpretation.
GLOBAL_START = time.time()
GLOBAL_DEADLINE = GLOBAL_START + 230.0

def _neighbours(point):
    """Yield the 6 neighbors of a 3D point."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)

def _compute_energy(sequence, chain):
    """
    Compute the energy of a chain.
    Energy = - (number of H-H contacts between non-adjacent residues).
    """
    n = len(chain)
    if n == 0:
        return 0
    
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    
    for i, p in enumerate(chain):
        if sequence[i] != 'H':
            continue
        for q in _neighbours(p):
            j = pos_to_idx.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
                
    return -contacts

def _beam_search_fold(sequence, dimension, beam_width=100, time_limit=5.0):
    """
    Deterministic Beam Search for HP protein folding.
    
    State: (chain, occupied_set, energy, last_pos)
    Heuristic: 
      - Primary: Maximize current H-H contacts (minimize energy).
      - Secondary: Maximize 'openness' of the tip to avoid dead ends.
      - Tertiary: Random tie-breaker for diversity in beam expansion if needed, 
        but we keep it deterministic based on sorted order for stability.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    # Initialize beam with a few diverse starting directions
    # Start at (0,0,0) with first residue.
    # We can start by placing the second residue in one of the 6 directions.
    # To be efficient, we start the beam with all 6 possible second positions.
    
    start_time = time.time()
    
    # Beam elements: (score, chain_tuple, occupied_set, last_pos)
    # We use a min-heap, so we negate the score to get max-heap behavior.
    # Score = -energy + openness_bonus. Since energy is negative, -energy is positive.
    # Higher score is better.
    
    beam = []
    
    # Initial state: Residue 0 at (0,0,0)
    init_occupied = {(0, 0, 0)}
    init_chain = [(0, 0, 0)]
    
    # Expand to length 2
    for nb in _neighbours((0, 0, 0)):
        if nb in init_occupied:
            continue
        chain = [(0, 0, 0), nb]
        occ = init_occupied | {nb}
        
        # Calculate initial energy (should be 0 for length 2)
        e = _compute_energy(sequence, chain)
        
        # Calculate openness of the tip
        openness = 0
        for q in _neighbours(nb):
            if q not in occ:
                openness += 1
                
        # Score: We want to maximize contacts (minimize energy) and openness.
        # Energy is negative. Let's use score = -energy * 10 + openness.
        # This heavily weights contacts.
        score = -e * 10 + openness
        
        # Push to beam
        # We need to keep track of the chain to reconstruct later.
        # Storing the full chain in the heap is memory expensive for large N.
        # For N up to 136, storing tuples of length N in a beam of width 100
        # is ~13,600 integers per element. 100 elements = 1.3M integers.
        # This is manageable in Python but might be slow.
        # Optimization: Store only the last few steps? No, we need full chain for collision check.
        # Alternative: Store the chain as a tuple of tuples.
        
        heapq.heappush(beam, (-score, id(chain), chain, occ))
        
    if not beam:
        return None

    # Main loop
    for step in range(2, n):
        if time.time() > time_limit:
            break
            
        new_beam = []
        
        # Process all candidates in the current beam
        # We need to expand each candidate in the beam
        # To avoid processing duplicates or too many, we can limit expansion.
        
        # Sort beam by score (best first)
        # heapq is a min-heap, so we pop the best (most negative -score)
        # But we need to expand ALL of them? No, beam search usually expands
        # the top K. Here, the beam IS the top K from the previous step.
        
        candidates_to_expand = []
        while beam:
            neg_score, _, chain, occ = heapq.heappop(beam)
            candidates_to_expand.append((chain, occ))
            
        # Expand each candidate
        for chain, occ in candidates_to_expand:
            last_pos = chain[-1]
            
            # Try all 6 neighbors
            for nb in _neighbours(last_pos):
                if nb in occ:
                    continue
                
                new_chain = chain + (nb,)
                new_occ = occ | {nb}
                
                # Calculate energy delta
                # Only the new residue (step) can form new contacts.
                # It can contact any previous H residue that is a neighbor.
                delta_e = 0
                if sequence[step] == 'H':
                    for q in _neighbours(nb):
                        if q in occ:
                            # Find index of q
                            # We don't have pos_to_idx here. 
                            # We can check if the residue at that position is H.
                            # To do this efficiently, we need to map pos->idx.
                            # Rebuilding map every step is O(N). 
                            # N=136, beam=100, steps=136. 136*100*136 = ~1.8M ops.
                            # This is fine.
                            pass # We'll just recompute full energy for simplicity first, 
                                 # or optimize if slow. 
                                 # Actually, let's just compute full energy. It's O(N*6).
                                 # 100 candidates * 6 neighbors * O(N) energy = 600 * 136 = 81k ops per step.
                                 # 136 steps = ~11M ops. Very fast.
                
                e = _compute_energy(sequence, new_chain)
                
                # Calculate openness
                openness = 0
                for q in _neighbours(nb):
                    if q not in new_occ:
                        openness += 1
                
                score = -e * 10 + openness
                
                heapq.heappush(new_beam, (-score, id(new_chain), new_chain, new_occ))
                
                # Optimization: If the beam gets too large, we should prune.
                # But we only expand from the current beam size.
                # If beam_width is 100, new_beam will have up to 600 elements.
                # We should prune to beam_width immediately?
                # Standard beam search: generate children, keep top K.
                
        # Prune new_beam to beam_width
        if len(new_beam) > beam_width:
            # heapq doesn't have a direct "keep top K" for a heap that's already built.
            # We can pop the worst (largest -score, i.e., lowest score)
            # Or use nlargest.
            # Since we have a heap of -scores, the largest -score is the worst.
            # We want to keep the smallest -scores (best scores).
            
            # Efficient pruning:
            # Pop the worst elements until we have beam_width left.
            # Worst is at the end? No, heap property: min is at root.
            # Max is anywhere. 
            # Better: Convert to list, sort, take top K.
            # List size is <= 600. Sorting 600 items is fast.
            
            new_beam_list = []
            while new_beam:
                new_beam_list.append(heapq.heappop(new_beam))
            
            # new_beam_list contains (-score, id, chain, occ)
            # Sort by -score (ascending, so best scores first)
            new_beam_list.sort(key=lambda x: x[0])
            
            # Keep top beam_width
            new_beam = new_beam_list[:beam_width]
            # Re-heapify
            heapq.heapify(new_beam)
            
        beam = new_beam
        
        if not beam:
            break
            
    if not beam:
        return None
        
    # Return the best chain
    # The best is at the root of the heap (smallest -score)
    best_neg_score, _, best_chain, _ = heapq.heappop(beam)
    
    return list(best_chain)

def _local_search(sequence, chain, time_limit, rng=None):
    """
    Fast local search using pivot moves.
    """
    import random
    if rng is None:
        rng = random.Random(42)
        
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
    
    current_chain = [tuple(p) for p in chain]
    current_energy = _compute_energy(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    T = 1.5
    alpha = 0.98
    
    is_h = [res == 'H' for res in sequence]
    
    while time.time() < time_limit:
        pivot_idx = rng.randint(1, n - 2)
        
        p_prev = current_chain[pivot_idx - 1]
        p_pivot = current_chain[pivot_idx]
        
        bond_vector = [p_pivot[i] - p_prev[i] for i in range(3)]
        bond_axis = -1
        for i in range(3):
            if bond_vector[i] != 0:
                bond_axis = i
                break
        if bond_axis == -1:
            continue
            
        rot_axes = [i for i in range(3) if i != bond_axis]
        r_axis = rot_axes[rng.randint(0, 1)]
        angle = 1 if rng.random() < 0.5 else -1
        
        head_set = set(current_chain[:pivot_idx + 1])
        new_tail = []
        valid = True
        occupied_tail = set()
        
        for i in range(pivot_idx + 1, n):
            orig = current_chain[i]
            dx = orig[0] - p_pivot[0]
            dy = orig[1] - p_pivot[1]
            dz = orig[2] - p_pivot[2]
            
            if r_axis == 0:
                if angle == 1: nx, ny, nz = dx, -dz, dy
                else: nx, ny, nz = dx, dz, -dy
            elif r_axis == 1:
                if angle == 1: nx, ny, nz = dz, dy, -dx
                else: nx, ny, nz = -dz, dy, dx
            else:
                if angle == 1: nx, ny, nz = -dy, dx, dz
                else: nx, ny, nz = dy, -dx, dz
                
            new_point = (p_pivot[0] + nx, p_pivot[1] + ny, p_pivot[2] + nz)
            
            if new_point in head_set or new_point in occupied_tail:
                valid = False
                break
                
            occupied_tail.add(new_point)
            new_tail.append(new_point)
            
        if not valid:
            continue
            
        new_chain = current_chain[:pivot_idx + 1] + new_tail
        new_energy = _compute_energy(sequence, new_chain)
        
        delta_e = new_energy - current_energy
        
        if delta_e < 0:
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        else:
            if delta_e == 0:
                if rng.random() < 0.5:
                    current_chain = new_chain
                    current_energy = new_energy
            else:
                prob = math.exp(-delta_e / T)
                if rng.random() < prob:
                    current_chain = new_chain
                    current_energy = new_energy
                    
        T *= alpha
        
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Beam Search + Local Search.
    """
    start_time = time.time()
    n = len(sequence)
    
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    # Allocate time
    # Total budget 240s. 26 sequences.
    # ~9s per sequence.
    # Beam search is deterministic and fast.
    # Local search is stochastic and can run until deadline.
    
    # Beam search time: Fixed 2s? Or dynamic?
    # Beam search with width 100 for N=136 is very fast (<0.1s usually).
    # Let's give it 1s to be safe, then use the rest for local search.
    
    bs_time_limit = 1.0
    ls_time_limit = 7.0 # 8s total per sequence roughly
    
    # Check global deadline
    remaining_global = GLOBAL_DEADLINE - time.time()
    if remaining_global < 0:
        return [[i, 0, 0] for i in range(n)]
        
    # Cap local search time by global deadline
    ls_time_limit = min(ls_time_limit, remaining_global - 0.5)
    
    # 1. Beam Search
    best_chain = _beam_search_fold(sequence, dimension, beam_width=200, time_limit=bs_time_limit)
    
    if best_chain is None:
        # Fallback to simple random walk or straight line
        # Try a few random walks
        best_chain = None
        for _ in range(10):
            # Simple greedy with randomness
            chain = [(0,0,0)]
            occupied = {(0,0,0)}
            valid = True
            for i in range(1, n):
                last = chain[-1]
                nbs = [nb for nb in _neighbours(last) if nb not in occupied]
                if not nbs:
                    valid = False
                    break
                # Pick random
                nb = nbs[int(time.time()*1000000) % len(nbs)] # Pseudo random for speed
                chain.append(nb)
                occupied.add(nb)
            if valid:
                best_chain = chain
                break
        if best_chain is None:
            best_chain = [[i, 0, 0] for i in range(n)]
    else:
        best_chain = [list(p) for p in best_chain]
        
    # 2. Local Search
    if ls_time_limit > 0.1:
        opt_chain, opt_energy = _local_search(sequence, best_chain, ls_time_limit)
        best_chain = [list(p) for p in opt_chain]
        
    # Final check against global deadline
    if time.time() > GLOBAL_DEADLINE - 1.0:
         # Ensure we return something valid quickly
         pass
         
    return best_chain

# EVOLVE-BLOCK-END
