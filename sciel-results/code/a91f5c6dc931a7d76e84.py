"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global start time to manage the 240s budget for all sequences
_START_TIME = time.time()
# We have 26 sequences. Budget is 240s. 
# To be safe, we allow ~9s per sequence. 
# However, the prompt says "240 s CPU budget for all sequences together".
# If we use a fixed time limit per call, and the evaluator calls them sequentially, 
# we should be careful. Let's use a conservative per-call limit.
TIME_LIMIT_PER_CALL = 8.5

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Weight gain heavily
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _calc_energy(sequence, chain):
    """Calculate total energy of a chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _pivot_move(chain, occupied, i, sequence, rng):
    """
    Perform a pivot move at bond i (rotating chain[i+1:] around chain[i]).
    Returns (new_chain, new_occupied, energy_delta, success)
    """
    n = len(chain)
    if i <= 0 or i >= n - 1:
        return None, None, 0, False
    
    # The part that moves
    moving_part = chain[i+1:]
    # The pivot point
    pivot = chain[i]
    
    # Generate 24 rotations in 3D (or 8 in 2D, but we assume 3D mostly, though dim is param)
    # For simplicity and robustness, we generate a random rotation vector and check validity.
    # A better way: pick a random axis and angle? 
    # Standard pivot moves in 3D: rotate the sub-chain by 90, 180, 270 degrees around the bond axis?
    # No, pivot move rotates around the bond vector.
    # Actually, simpler: pick a random new direction for the bond (i, i+1) and see if the rest fits?
    # That's complex.
    
    # Let's use the standard "pivot" definition: 
    # Rotate the sub-chain $i+1 \dots N-1$ around the bond $(i, i+1)$.
    # In a cubic lattice, there are 3 possible 90-degree rotations around the bond axis? 
    # Actually, the bond is fixed. The rest of the chain can be rotated.
    # There are 3 axes perpendicular to the bond. For each axis, 3 non-identity rotations.
    # Total 9 moves? Or just pick a random rotation matrix?
    
    # Simpler approach: "Single Residue Move" and "Pivot" via re-growth? No.
    # Let's implement a "Pivot" by picking a random bond i, and trying to move the end of the chain?
    # No, that's not a pivot.
    
    # Let's implement a robust "Pivot" by rotating the sub-chain around the pivot point chain[i].
    # We try all 6 neighbors of chain[i+1] that are not chain[i] and not occupied by the static part?
    # No, the sub-chain is rigid.
    
    # Alternative: Use "Rotation Move" (pick i, rotate sub-chain i..N-1 around chain[i]).
    # And "Pivot Move" (pick i, rotate sub-chain i+1..N-1 around bond i-i+1).
    
    # For efficiency in Python, let's implement a "Move Single Residue" and "Pivot End".
    # Actually, the most effective simple move is "Pivot" where we pick a random bond, 
    # and rotate the tail by 90 degrees around the bond axis.
    
    # Let's stick to a simpler, proven move: "Pivot" by picking a random index i, 
    # and moving chain[i] to a random empty neighbor, then re-growing the rest? 
    # That's expensive.
    
    # Let's use the "Rotation Move" on a sub-chain.
    # Pick i. Rotate chain[i:] around chain[i].
    # We need to check if the new positions of chain[i+1:] are empty.
    
    # To keep it simple and fast, let's use a "Single Residue Move" for small chains 
    # and "Pivot" for larger.
    
    # Actually, let's implement a generic "Pivot" by trying to rotate the sub-chain 
    # around the bond (i, i+1) by 90 degrees around the 3 axes perpendicular to the bond.
    
    # Bond vector
    v = [chain[i+1][k] - chain[i][k] for k in range(dimension)]
    
    # Find 3 axes perpendicular to v.
    # This is getting complicated. 
    # Let's use a different strategy: "Random Walk Move"
    # Pick a random residue i (not 0 or N-1). Move it to a random empty neighbor.
    # Check if the chain remains connected and self-avoiding.
    # This is O(N) to check connectivity? No, just check if the new position is adjacent to old neighbors.
    # This is a "Crankshaft" or "Single Move".
    
    # Let's implement a "Pivot" properly.
    # We will rotate the sub-chain chain[i+1:] around the point chain[i].
    # We pick a random rotation from the set of 24 proper rotations in 3D?
    # Or just 3 rotations around the bond axis?
    
    # Given the time, let's use a simpler, effective heuristic:
    # 1. Try to move the last residue (end) to a new neighbor. (Fast)
    # 2. Try to pivot a random internal bond.
    
    # Let's implement a "Pivot" that rotates the tail around a random bond.
    # To avoid complex geometry, we will simply try to move the *entire* tail 
    # to a different configuration by picking a new direction for the first bond of the tail?
    # No, that breaks the rigid body.
    
    # Let's use "Rotation Move" at index i.
    # Rotate chain[i:] around chain[i].
    # We generate a new position for chain[i+1] by picking a random neighbor of chain[i] 
    # that is not chain[i-1] (if i>0) and not occupied.
    # Then we check if the rest of the chain can be placed? No, it's a rigid rotation.
    
    # Okay, I will implement a "Single Residue Move" and a "Pivot End".
    # "Pivot End": Move the last residue to a random empty neighbor.
    # "Single Move": Move residue i to a random empty neighbor, ensuring it stays connected to i-1 and i+1.
    
    # Actually, the inspiration mentions "Pivot moves" and "Rotation moves".
    # Let's implement a "Pivot" by picking a bond (i, i+1) and rotating the sub-chain i+1..N-1 
    # by 90 degrees around the bond axis.
    
    # Bond axis: v = chain[i+1] - chain[i]
    # We need 3 vectors perpendicular to v.
    # If v is (1,0,0), perp axes are (0,1,0), (0,0,1), (1,1,0)? No, (0,1,0) and (0,0,1) are perp.
    # Rotations around these axes by 90, 180, 270.
    
    # This is too complex to code robustly in a short time without bugs.
    # Let's use a "Re-growth" strategy for the tail?
    # Pick a cut point i. Remove chain[i:]. Re-grow chain[i:] from chain[i-1] using greedy?
    # This is "Local Re-optimization".
    
    # Let's go with "Simulated Annealing with Single Residue Moves and Pivot Ends".
    # This is simpler and often effective.
    
    # Move 1: Pivot End. Move chain[-1] to a random empty neighbor.
    # Move 2: Single Move. Pick random i (1..N-2). Move chain[i] to random empty neighbor.
    #          Must be adjacent to chain[i-1] and chain[i+1].
    
    # Let's implement these two.
    
    return None, None, 0, False # Placeholder

def _try_move_single(chain, occupied, i, sequence, rng):
    """Try to move residue i to a random empty neighbor."""
    n = len(chain)
    if i <= 0 or i >= n - 1:
        return None, None, 0, False
        
    old_pos = chain[i]
    # Find empty neighbors
    candidates = [q for q in _neighbours(old_pos) if q not in occupied]
    if not candidates:
        return None, None, 0, False
        
    new_pos = rng.choice(candidates)
    
    # Check connectivity: new_pos must be adjacent to chain[i-1] and chain[i+1]
    prev_pos = chain[i-1]
    next_pos = chain[i+1]
    
    # Distance check
    def dist(a, b):
        return sum(abs(a[k]-b[k]) for k in range(len(a)))
        
    if dist(new_pos, prev_pos) != 1 or dist(new_pos, next_pos) != 1:
        return None, None, 0, False
        
    # Check self-avoidance: new_pos is already checked to be empty.
    # But we must ensure that moving i doesn't cause collision with i-1 or i+1? 
    # No, i-1 and i+1 are fixed. We only move i.
    # The only collision risk is with other residues, which is handled by `occupied` check.
    
    # Update energy
    # Remove old contacts involving i
    # Add new contacts involving i
    
    # This requires knowing which residues are H and their positions.
    # We will calculate delta energy by removing old and adding new.
    
    # For simplicity in this implementation, we will recalculate energy if the move is accepted.
    # This is O(N). With N=136, it's fast enough.
    
    # Perform move
    new_chain = chain[:]
    new_chain[i] = new_pos
    new_occupied = occupied.copy()
    new_occupied.remove(old_pos)
    new_occupied.add(new_pos)
    
    return new_chain, new_occupied, 0, True

def _try_move_pivot_end(chain, occupied, sequence, rng):
    """Try to move the last residue to a random empty neighbor."""
    n = len(chain)
    if n == 0:
        return None, None, 0, False
        
    old_pos = chain[-1]
    candidates = [q for q in _neighbours(old_pos) if q not in occupied]
    if not candidates:
        return None, None, 0, False
        
    new_pos = rng.choice(candidates)
    
    new_chain = chain[:]
    new_chain[-1] = new_pos
    new_occupied = occupied.copy()
    new_occupied.remove(old_pos)
    new_occupied.add(new_pos)
    
    return new_chain, new_occupied, 0, True

def _anneal(sequence, dimension, initial_chain, rng, time_limit):
    """Simulated Annealing loop."""
    if not initial_chain:
        return None
        
    chain = initial_chain
    occupied = {p: i for i, p in enumerate(chain)}
    # Note: occupied is a set for fast lookup in move checks, but we need index for energy?
    # Let's use a set for occupied sites.
    occupied_set = set(chain)
    
    current_energy = _calc_energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    start_time = time.time()
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.99
    
    n = len(chain)
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
            
        # Choose a move
        if rng.random() < 0.5:
            # Pivot End
            new_chain, new_occupied_set, _, success = _try_move_pivot_end(chain, occupied_set, sequence, rng)
        else:
            # Single Move
            i = rng.randint(1, n-2)
            new_chain, new_occupied_set, _, success = _try_move_single(chain, occupied_set, i, sequence, rng)
            
        if not success:
            continue
            
        new_energy = _calc_energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            # Accept
            chain = new_chain
            occupied_set = new_occupied_set
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
                
        T *= cooling_rate
        
    return best_chain

def fold(sequence, dimension):
    """Best conformation of a fixed number of randomized greedy growths + annealing."""
    rng = random.Random(12345 + len(sequence))
    
    # Determine time budget for this call
    # We have 240s total. 26 sequences. ~9s each.
    # Let's use 8s to be safe.
    time_budget = TIME_LIMIT_PER_CALL
    
    best_chain = None
    best_energy = 0 # 0 is worst (unfolded)
    
    # Run multiple restarts
    # We will allocate time to restarts.
    # Let's do 10 restarts, each with a small time slice, or 1 long anneal?
    # Multi-start is better.
    num_restarts = 10
    time_per_restart = time_budget / num_restarts
    
    for r in range(num_restarts):
        # Grow initial chain
        init_chain = _grow(sequence, dimension, rng)
        if init_chain is None:
            continue
            
        # Anneal
        annealed_chain = _anneal(sequence, dimension, init_chain, rng, time_per_restart)
        
        if annealed_chain:
            energy = _calc_energy(sequence, annealed_chain)
            if best_chain is None or energy < best_energy:
                best_chain = annealed_chain
                best_energy = energy
                
        # Check global time
        if time.time() - _START_TIME > 230: # Safety margin
            break
            
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
