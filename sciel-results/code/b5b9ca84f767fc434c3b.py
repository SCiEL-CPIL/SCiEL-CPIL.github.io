"""HP lattice protein folding using greedy growth followed by simulated annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = time.time()
DEADLINE = 170.0  # Safety margin before 180s kill (and 120s CPU budget consideration)

def _remaining_time():
    return DEADLINE - time.time()

def _neighbours(point):
    """Generate 4-neighbours in 2D."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _energy(sequence, chain):
    """Compute energy as -contacts. Contacts are H-H pairs non-adjacent in chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                    # Count each pair once
                    if i < j:
                        contacts += 1
    return -contacts

def _grow(sequence, dimension, rng):
    """
    Fast randomized greedy growth.
    Returns a list of tuples representing the chain.
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        last_pos = chain[-1]
        for site in _neighbours(last_pos):
            if site in occupied:
                continue
            # Simple bias: prefer sites that might create contacts if current is H
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        idx = occupied[q]
                        if idx < i - 1 and sequence[idx] == "H":
                            gain += 1
            # Add some randomness to avoid getting stuck
            score = gain + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
            
        # Pick best option
        options.sort(reverse=True)
        # Occasionally pick a random valid option to increase diversity
        if rng.random() < 0.2:
            site = options[rng.randint(0, len(options)-1)][1]
        else:
            site = options[0][1]
            
        occupied[site] = i
        chain.append(site)
        
    return chain

def _simulate_annealing(sequence, initial_chain, rng):
    """
    Perform simulated annealing to improve the conformation.
    Returns the best chain found.
    """
    chain = initial_chain[:]
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    n = len(chain)
    # Precompute positions for quick lookup
    pos_map = {p: i for i, p in enumerate(chain)}
    
    T = 1.0
    T_min = 0.01
    alpha = 0.995
    # Estimate steps based on time, but cap it
    # We check time in the loop
    
    while _remaining_time() > 1.0 and T > T_min:
        # Generate a random move
        move_type = rng.random()
        
        if move_type < 0.5:
            # Flip move: change direction of one bond
            if n < 3:
                T *= alpha
                continue
            i = rng.randint(1, n - 2)
            p_prev = chain[i-1]
            p_curr = chain[i]
            p_next = chain[i+1]
            
            # Find neighbours of p_curr that are not p_prev and not p_next
            candidates = []
            for nb in _neighbours(p_curr):
                if nb != p_prev and nb != p_next:
                    # Check if nb is occupied by other parts of chain
                    if nb not in pos_map:
                        candidates.append(nb)
                    else:
                        # If occupied, it must be the same index? No, pos_map maps pos to index.
                        # If nb is in pos_map, it's occupied.
                        # We can only move if nb is empty.
                        pass
            
            if not candidates:
                T *= alpha
                continue
                
            new_pos = candidates[rng.randint(0, len(candidates)-1)]
            
            # Remove old pos, add new pos
            del pos_map[p_curr]
            pos_map[new_pos] = i
            chain[i] = new_pos
            
            # Calculate energy change
            # This is expensive if we recalculate full energy.
            # For simplicity and robustness, we recalculate. 
            # Optimization: Only recalculate if move is accepted or if we are near end?
            # Given 120s budget, full recalc might be slow for large N.
            # Let's try to be smarter or just accept the cost if N is small enough.
            # Benchmark N is up to 100. 100 steps * 10000 moves = 1M ops. Python is slow.
            # We need incremental energy or faster moves.
            
            # Let's use a faster energy update logic.
            # Energy depends on H-H contacts.
            # Moving chain[i] breaks contacts involving i and makes new ones.
            
            # For now, let's stick to full recalc but limit iterations or use a faster heuristic?
            # Actually, let's implement a quick check.
            # If we do full recalc every step, it's O(N).
            # N=100, 100k steps = 10M ops. Might be tight in Python.
            
            # Let's try a different move set that is easier to update or just accept the risk.
            # To be safe, let's reduce the frequency of full recalc or use a simpler move.
            
            # Alternative: Use "pivot" moves which are also O(N) to update if done naively.
            
            # Let's optimize _energy to be as fast as possible.
            new_energy = _energy(sequence, chain)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
            else:
                # Revert
                pos_map[p_curr] = i
                del pos_map[new_pos]
                chain[i] = p_curr
                
        else:
            # Pivot move: rotate a segment
            if n < 4:
                T *= alpha
                continue
            i = rng.randint(1, n - 2)
            length = rng.randint(1, n - i - 1)
            if length > 1:
                # Segment is chain[i : i+length]
                # Pivot around chain[i-1]
                # This is complex to implement correctly with pos_map updates.
                # Let's stick to flips for simplicity and reliability, but maybe increase T or steps.
                pass
        
        T *= alpha
        
        # Periodic check for time
        if rng.random() < 0.01:
             if _remaining_time() < 2.0:
                 break

    return best_chain

def fold(sequence, dimension):
    """
    Main entry point.
    """
    if dimension != 2:
        raise ValueError("Only 2D is supported")
    
    rng = random.Random(42)
    best_chain = None
    best_energy = 0 # Energy is negative, so 0 is worse than -1
    
    # Phase 1: Generate multiple initial conformations
    num_starts = 50
    for i in range(num_starts):
        if _remaining_time() < 5.0:
            break
            
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
            
        e = _energy(sequence, chain)
        if best_chain is None or e < best_energy:
            best_chain = chain
            best_energy = e
            
        # If we find a very good energy, we can stop early?
        # Theoretical min energy is -N/2 roughly.
        # If we hit a known good value, we might still improve.
        
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(len(sequence))]
        return [list(p) for p in best_chain]

    # Phase 2: Polish with Simulated Annealing
    # We have a time budget. Let's use the remaining time.
    if _remaining_time() > 2.0:
        # Run SA on the best chain found
        # We might want to run SA on multiple good chains, but time is tight.
        # Let's run SA on the single best one.
        polished = _simulate_annealing(sequence, best_chain, rng)
        e_polished = _energy(sequence, polished)
        
        if e_polished < best_energy:
            best_chain = polished
            best_energy = e_polished
            
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
