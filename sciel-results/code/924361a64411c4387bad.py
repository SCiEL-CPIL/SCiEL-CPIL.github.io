import random
import time
import math

# Global deadline guard: must finish well before 330s wall-clock.
_TOTAL_START = time.time()
_SAFETY_MARGIN = 10.0  # seconds to leave before hard kill

def _remaining_time():
    """Return remaining seconds before our safety deadline."""
    return _TOTAL_START + 330.0 - _SAFETY_MARGIN - time.time()

def _neighbours(point):
    """Generate 6 neighbors of a point in 3D lattice."""
    for axis in range(3):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Calculate energy: -number of H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth."""
    start = (0, 0, 0)
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            # Simple heuristic: prefer sites with more free neighbors to avoid dead ends
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Add some randomness
            options.append((free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _simulated_annealing(sequence, initial_chain, rng, time_limit):
    """
    Simulated Annealing with 2-move and 3-move perturbations.
    """
    n = len(sequence)
    if n < 5:
        return initial_chain, _energy(sequence, initial_chain)
    
    chain = list(initial_chain)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    occupied = {p: i for i, p in enumerate(chain)}
    
    # Cooling schedule parameters
    T_start = 2.0
    T_end = 0.01
    # Estimate number of iterations we might do, but we are time-limited
    # We will adjust cooling based on actual steps taken
    cooling_rate = 0.995
    
    T = T_start
    steps = 0
    start_time = time.time()
    
    while True:
        if time.time() > time_limit:
            break
        if T <= T_end:
            break
            
        # Choose a move type: 2-move or 3-move
        # 2-moves are cheaper and more common, 3-moves help escape
        if rng.random() < 0.7:
            move_type = 2
        else:
            move_type = 3
            
        # Pick a random interior segment to move
        # For 2-move: pick i such that i-1, i, i+1, i+2 are valid
        # Actually, a 2-move rotates the bond between i and i+1.
        # We need i >= 1 and i+1 <= n-2. So i in [1, n-2]
        # The residues involved are i and i+1.
        # Wait, standard 2-move: pick i, move residue i+1 to a new position
        # such that it is still connected to i and i+2? No, that's a 3-move context.
        
        # Let's define moves precisely:
        # 2-move: Pick i (1 <= i <= n-2). Move residue i to a new position.
        # It must stay connected to i-1 and i+1.
        # This is effectively a rotation of the (i-1, i, i+1) triad? No.
        # A 2-move usually refers to moving a single residue i such that it remains
        # connected to i-1 and i+1. This is possible if the new position is a common
        # neighbor of i-1 and i+1 (excluding the old position).
        # In 3D, two points (i-1 and i+1) that are distance 2 apart (Manhattan)
        # have exactly 2 common neighbors (one is the old position, one is the new).
        # So a 2-move is deterministic given i: it's a "flip" of the residue.
        
        # 3-move: Pick i (1 <= i <= n-3). Move residues i and i+1?
        # Or move residue i+1 such that it stays connected to i and i+2,
        # but also allowing i to move?
        # Let's stick to:
        # 1. Single Residue Move (1-move): Move i to a neighbor of i-1 or i+1?
        #    No, must stay connected to BOTH.
        #    If i-1 and i+1 are distance 2 apart, there is exactly 1 other valid spot.
        #    This is a "flip".
        # 2. 2-Residue Move (2-move): Move i and i+1.
        #    This is more complex.
        
        # Let's use a simpler, effective set of moves for 3D HP:
        # A. Flip (1-residue): If i-1 and i+1 are distance 2 apart, i can move to the other common neighbor.
        # B. 2-Residue Move: Pick i. Try to move i and i+1 to new positions.
        #    This is computationally expensive to enumerate.
        
        # Alternative: Use "Crankshaft" moves (2-move) and "3-move".
        # Crankshaft: Pick i. The residues i-1, i, i+1 form a path.
        # We can rotate the bond (i, i+1) around the axis defined by (i-1, i)?
        # No, lattice moves are discrete.
        
        # Let's implement:
        # 1. 1-residue flip: Valid if dist(i-1, i+1) == 2. New pos is the other common neighbor.
        # 2. 2-residue move: Pick i. Move residue i to a new pos p_new such that
        #    it is connected to i-1. Then move residue i+1 to a new pos q_new such that
        #    it is connected to p_new and i+2.
        #    This is a sequential move.
        
        # To keep it simple and fast, let's use:
        # 1. 1-residue flips (high probability, low cost)
        # 2. 2-residue moves (lower probability, higher mobility)
        
        if move_type == 1:
            # Flip
            i = rng.randrange(1, n - 1)
            p_prev = chain[i-1]
            p_next = chain[i+1]
            p_old = chain[i]
            
            # Check if p_prev and p_next are distance 2 apart
            dist = abs(p_prev[0]-p_next[0]) + abs(p_prev[1]-p_next[1]) + abs(p_prev[2]-p_next[2])
            if dist != 2:
                continue # Cannot flip if not a straight line or L-shape? 
                         # In 3D, if dist is 2, they are either straight (1,0,0)-(1,1,0) no, that's dist 2.
                         # Common neighbors exist if they share a face? No.
                         # If dist(p_prev, p_next) == 2, there are exactly 2 common neighbors.
                         # One is p_old. The other is p_new.
            
            # Find the other common neighbor
            p_new = None
            for q in _neighbours(p_prev):
                if q == p_old:
                    continue
                if abs(q[0]-p_next[0]) + abs(q[1]-p_next[1]) + abs(q[2]-p_next[2]) == 1:
                    p_new = q
                    break
            
            if p_new is None or p_new in occupied:
                continue
            
            # Calculate energy delta
            delta_e = 0
            if sequence[i] == "H":
                # Lose contacts at p_old
                for q in _neighbours(p_old):
                    j = occupied.get(q)
                    if j is not None and j != i and sequence[j] == "H" and abs(i-j) > 1:
                        delta_e += 1
                # Gain contacts at p_new
                for q in _neighbours(p_new):
                    j = occupied.get(q)
                    if j is not None and j != i and sequence[j] == "H" and abs(i-j) > 1:
                        delta_e -= 1
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                # Accept
                occupied.pop(p_old)
                occupied[p_new] = i
                chain[i] = p_new
                current_energy += delta_e
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
            
        else:
            # 2-Residue Move (Crankshaft-like)
            # Pick i. Move i and i+1.
            # i must be in [1, n-3]
            i = rng.randrange(1, n - 2)
            p_i_old = chain[i]
            p_ip1_old = chain[i+1]
            p_im1 = chain[i-1]
            p_ip2 = chain[i+2]
            
            # Generate possible new positions for i
            # Must be neighbor of p_im1
            candidates_i = [q for q in _neighbours(p_im1) if q != p_i_old and q != p_ip1_old and q not in occupied]
            
            if not candidates_i:
                continue
            
            # For each candidate for i, generate candidates for i+1
            # Must be neighbor of new_i and p_ip2
            best_delta_for_move = float('inf')
            best_new_i = None
            best_new_ip1 = None
            best_delta_val = 0
            
            for new_i in candidates_i:
                # Check if new_i is valid (not occupied, not p_ip1_old)
                # Already checked in list comprehension
                
                # Candidates for i+1
                candidates_ip1 = [q for q in _neighbours(new_i) 
                                  if q != p_i_old and q != p_ip1_old 
                                  and q not in occupied 
                                  and abs(q[0]-p_ip2[0]) + abs(q[1]-p_ip2[1]) + abs(q[2]-p_ip2[2]) == 1]
                
                if not candidates_ip1:
                    continue
                
                for new_ip1 in candidates_ip1:
                    # Calculate delta energy
                    delta_e = 0
                    # Residue i
                    if sequence[i] == "H":
                        for q in _neighbours(p_i_old):
                            j = occupied.get(q)
                            if j is not None and j not in [i, i+1] and sequence[j] == "H" and abs(i-j) > 1:
                                delta_e += 1
                        for q in _neighbours(new_i):
                            j = occupied.get(q)
                            if j is not None and j not in [i, i+1] and sequence[j] == "H" and abs(i-j) > 1:
                                delta_e -= 1
                        # Contact between i and i+1? No, they are adjacent, no contact energy.
                        # But wait, if i and i+1 are H, they don't form a contact because they are chain neighbors.
                        # We only count non-adjacent H-H.
                    
                    # Residue i+1
                    if sequence[i+1] == "H":
                        for q in _neighbours(p_ip1_old):
                            j = occupied.get(q)
                            if j is not None and j not in [i, i+1] and sequence[j] == "H" and abs(i+1-j) > 1:
                                delta_e += 1
                        for q in _neighbours(new_ip1):
                            j = occupied.get(q)
                            if j is not None and j not in [i, i+1] and sequence[j] == "H" and abs(i+1-j) > 1:
                                delta_e -= 1
                    
                    # Check if new_i and new_ip1 create a contact with each other?
                    # They are adjacent in chain, so no.
                    
                    # Check if new_i creates contact with i+1's old neighbors? Handled above.
                    # Check if new_ip1 creates contact with i's old neighbors? Handled above.
                    
                    # Important: Check if new_i and new_ip1 are "close" to each other in a way that matters?
                    # No, just lattice adjacency.
                    
                    # One more check: Did we create a contact between i and i+1? No.
                    # Did we lose a contact between i and some other residue that is now blocked?
                    # The delta calculation above accounts for all neighbors of the moved residues.
                    # However, we must ensure that the new positions don't overlap with each other
                    # (checked in candidates_ip1 generation: q != p_i_old, but we need q != new_i? 
                    # Yes, new_i is in occupied temporarily? No, we haven't updated occupied yet.
                    # But new_ip1 must not be new_i.
                    # In candidates_ip1: q is neighbor of new_i. So q != new_i automatically?
                    # No, a point is not its own neighbor. So q != new_i is guaranteed.
                    
                    if delta_e < best_delta_for_move:
                        best_delta_for_move = delta_e
                        best_new_i = new_i
                        best_new_ip1 = new_ip1
                        best_delta_val = delta_e
            
            if best_new_i is not None:
                if best_delta_val < 0 or rng.random() < math.exp(-best_delta_val / T):
                    # Apply move
                    occupied.pop(p_i_old)
                    occupied.pop(p_ip1_old)
                    occupied[best_new_i] = i
                    occupied[best_new_ip1] = i + 1
                    chain[i] = best_new_i
                    chain[i+1] = best_new_ip1
                    current_energy += best_delta_val
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
        
        T *= cooling_rate
        steps += 1
        
        # Optional: If we have many steps without improvement, reheat?
        # For now, simple cooling.
        
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Best conformation using multiple SA restarts.
    """
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    
    n = len(sequence)
    
    # Budget allocation
    # We have ~320s.
    # Let's do as many restarts as possible.
    # Each restart: Grow (fast) + SA (variable time).
    # We will run SA for a fixed duration per restart, e.g., 2 seconds, 
    # or until T drops, whichever is sooner.
    # Or better: Run SA until time remaining is low.
    
    # Strategy:
    # 1. Generate a few initial chains.
    # 2. Run SA on the best one for a long time.
    # 3. Or run multiple short SAs.
    
    # Given the complexity, let's run SA on the best greedy chain for the majority of the time,
    # and use the remaining time for a few more restarts if time allows.
    
    # Phase 1: Get a good initial chain
    init_chains = []
    for _ in range(10):
        if _remaining_time() < 5.0:
            break
        c = _grow(sequence, dimension, rng)
        if c:
            e = _energy(sequence, c)
            init_chains.append((e, c))
    
    if not init_chains:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
        return [list(p) for p in best_chain]
    
    # Sort by energy
    init_chains.sort(key=lambda x: x[0])
    current_best_chain = init_chains[0][1]
    current_best_energy = init_chains[0][0]
    
    # Phase 2: SA on the best chain
    # Reserve 10s for final checks.
    time_limit = time.time() + _remaining_time() - 10.0
    
    # If we have multiple good initial chains, we could run SA on them in parallel? No, single process.
    # Let's run SA on the best one.
    # If time remains, run SA on the 2nd best one?
    
    # Let's just run SA on the best one for the available time.
    # To improve diversity, we can "kick" the chain by accepting worse moves more aggressively at the start.
    
    improved_chain, improved_energy = _simulated_annealing(sequence, current_best_chain, rng, time_limit)
    
    if improved_energy < current_best_energy:
        current_best_chain = improved_chain
        current_best_energy = improved_energy
    
    # If we still have time, try a second restart from the 2nd best initial chain
    if _remaining_time() > 5.0 and len(init_chains) > 1:
        # Run a shorter SA on the 2nd best
        time_limit2 = time.time() + _remaining_time() - 2.0
        improved_chain2, improved_energy2 = _simulated_annealing(sequence, init_chains[1][1], rng, time_limit2)
        if improved_energy2 < current_best_energy:
            current_best_chain = improved_chain2
            current_best_energy = improved_energy2
    
    return [list(p) for p in current_best_chain]
