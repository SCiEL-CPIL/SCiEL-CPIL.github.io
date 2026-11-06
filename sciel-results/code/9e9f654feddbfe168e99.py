"""
Simulated Annealing for HP Lattice Protein Folding.
Strategy:
1. Generate an initial self-avoiding walk (SAW) using a random growth algorithm.
2. Optimize the conformation using Simulated Annealing.
3. Move set includes:
   - Single step moves (pivot one residue).
   - Crankshaft moves (rotate a segment of 3 residues).
   - Segment shifts (move a segment by 1 unit if space allows).
4. Use a geometric cooling schedule.
5. Multiple restarts to escape global local minima.
"""

# EVOLVE-BLOCK-START
import time
import random
import math
import sys

# Global deadline: evaluator kills after 210s wall-clock. Use a safety margin.
_DEADLINE = time.time() + 190.0  # leave 20s buffer for overhead/sandbox


def _remaining():
    return _DEADLINE - time.time()


def _neighbours(point):
    """Yield the 4 neighbours of a lattice point."""
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _energy(sequence, chain):
    """Compute the HP energy: -1 for each non-consecutive H-H contact."""
    if chain is None or len(chain) < 2:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _generate_initial_walk(sequence, rng):
    """
    Generate a random self-avoiding walk.
    Returns a list of points or None if it fails (traps itself).
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    pos_to_idx = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        neighbors = _neighbours(last)
        rng.shuffle(neighbors)
        
        placed = False
        for nb in neighbors:
            if nb in pos_to_idx:
                continue
            
            # Optional: Avoid dead ends (simple check)
            # If we place here, does the next step have options?
            # For speed, we just try to place. If we get stuck, we return None.
            
            chain.append(nb)
            pos_to_idx[nb] = i
            placed = True
            break
        
        if not placed:
            return None
            
    return chain


def _sa_fold(sequence, rng, time_limit):
    """
    Perform Simulated Annealing starting from a random walk.
    """
    n = len(sequence)
    if n == 0:
        return [], 0
    if n == 1:
        return [(0, 0)], 0

    # Try to get a valid initial walk
    chain = None
    for _ in range(100):
        chain = _generate_initial_walk(sequence, rng)
        if chain is not None:
            break
    
    if chain is None:
        return None, 0

    pos_to_idx = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    
    # SA Parameters
    # Temperature should be high enough to allow moves that increase energy (fewer contacts)
    # Max energy change per move is roughly +1 or +2 (breaking a contact) or large if breaking multiple?
    # Actually, a single move breaks at most 1 contact (the one involving the moved residue) and creates at most 1.
    # Wait, moving a residue can break contacts with neighbors that are not the chain neighbors.
    # Max loss is usually small. T=2.0 is a good starting point.
    T = 2.5
    T_min = 0.05
    cooling = 0.999  # Adjust based on expected iterations
    
    # We will run until time_limit is reached.
    # We need to estimate how many moves we can do.
    # Each move involves checking neighbors (O(1)) and updating energy (O(1) if we track deltas, or O(N) if recompute).
    # Recomputing energy is O(N * 4). For N=100, that's 400 ops.
    # In Python, we can probably do 10,000 - 50,000 moves per second.
    # Let's just loop and check time.
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Generate a random move
        move_type = rng.random()
        
        if move_type < 0.5:
            # Single residue pivot
            # Pick a random index i (1 to n-2 to keep ends fixed? Or allow ends to move? 
            # Moving ends is easy. Moving middle residues is harder but crankshaft handles it.)
            # Let's pick i in [1, n-2] for pivots, and handle ends separately if needed.
            # Actually, pivoting the first or last residue is just a step.
            i = rng.randint(1, n - 2)
            prev = chain[i-1]
            curr = chain[i]
            next_pos = chain[i+1]
            
            # Possible new positions for curr
            candidates = [p for p in _neighbours(prev) if p != curr and p != next_pos]
            if not candidates:
                continue
            
            new_pos = rng.choice(candidates)
            
            # Check self-avoidance
            if new_pos in pos_to_idx:
                # Check if it collides with a non-adjacent part
                # The only adjacent parts are prev and next_pos.
                # If new_pos is prev or next_pos, we already excluded them.
                # So if it's in pos_to_idx, it's a collision.
                continue
            
            # Calculate energy delta
            # Remove curr from pos_to_idx temporarily? No, we can calculate delta directly.
            # Old contacts involving i:
            # i is at curr. Neighbors of curr in lattice:
            # 1. prev (chain neighbor) - not a contact
            # 2. next_pos (chain neighbor) - not a contact
            # 3. Other lattice neighbors that are H and part of chain.
            
            old_delta = 0
            if sequence[i] == 'H':
                for nb in _neighbours(curr):
                    if nb == prev or nb == next_pos:
                        continue
                    j = pos_to_idx.get(nb)
                    if j is not None and sequence[j] == 'H':
                        old_delta += 1
            
            # New contacts involving i:
            new_delta = 0
            if sequence[i] == 'H':
                for nb in _neighbours(new_pos):
                    if nb == prev or nb == next_pos:
                        continue
                    j = pos_to_idx.get(nb)
                    if j is not None and sequence[j] == 'H':
                        new_delta += 1
            
            delta_e = new_delta - old_delta # Energy change = -new_contacts - (-old_contacts) = old - new? 
            # Energy = -contacts.
            # E_new - E_old = -new_total - (-old_total) = old_total - new_total.
            # old_total = old_other + old_delta (where old_delta is contacts involving i)
            # new_total = new_other + new_delta
            # Since i moves, only contacts involving i change.
            # So delta_e = -new_delta - (-old_delta) = old_delta - new_delta.
            # Wait.
            # E = -C.
            # C_old = C_fixed + C_i_old
            # C_new = C_fixed + C_i_new
            # Delta E = E_new - E_old = -(C_fixed + C_i_new) - (-(C_fixed + C_i_old)) = C_i_old - C_i_new.
            # So delta_e = old_delta - new_delta.
            
            # If delta_e < 0, energy decreases (good).
            # If delta_e > 0, energy increases (bad).
            
            # Acceptance probability
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                # Accept move
                pos_to_idx.pop(curr)
                pos_to_idx[new_pos] = i
                chain[i] = new_pos
                current_energy += delta_e
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                    
        elif move_type < 0.9:
            # Crankshaft move
            # Rotate a segment of 3 residues: i-1, i, i+1 around i-1? 
            # No, crankshaft usually moves i-1 and i+1 while keeping i fixed? 
            # Or rotates the vector (i-1 -> i) and (i -> i+1).
            # Standard crankshaft: Pick i. Keep chain[i] fixed. 
            # Move chain[i-1] to a new position adjacent to chain[i] (not chain[i-2]).
            # Move chain[i+1] to a new position adjacent to chain[i] (not chain[i+2]).
            # This allows the chain to "turn" without moving the center.
            
            if n < 4:
                continue
                
            i = rng.randint(1, n - 2)
            center = chain[i]
            prev = chain[i-1]
            next_p = chain[i+1]
            
            # Candidates for prev
            prev_cands = [p for p in _neighbours(center) if p != prev and p != chain[i-2]]
            # Candidates for next
            next_cands = [p for p in _neighbours(center) if p != next_p and p != chain[i+2]]
            
            if not prev_cands or not next_cands:
                continue
                
            new_prev = rng.choice(prev_cands)
            new_next = rng.choice(next_cands)
            
            # Check collisions
            # new_prev must not collide with anything except center and i-2 (excluded)
            # It must not be i+1 (next_p) or i-2.
            # It must not be any other chain point.
            # Similarly for new_next.
            # Also new_prev != new_next.
            
            if new_prev == new_next:
                continue
                
            # Check new_prev against pos_to_idx
            # Exclude center, prev, i-2
            # Actually, just check if it's in pos_to_idx and not one of the allowed neighbors.
            # Allowed neighbors for new_prev: center (i), and i-2 is the old prev's neighbor? No.
            # The chain is ... i-2, i-1, i, i+1 ...
            # New chain: ... i-2, new_prev, center, new_next, i+2 ...
            # new_prev is adjacent to i-2 and center.
            # So new_prev can be i-2? No, i-2 is already occupied.
            # So new_prev must not be in pos_to_idx.
            # Wait, i-2 is in pos_to_idx. So if new_prev == i-2, it's a collision? 
            # No, i-2 is the residue before prev. 
            # In the new config, new_prev is connected to i-2.
            # So new_prev is NOT i-2.
            # So new_prev must not be in pos_to_idx.
            
            if new_prev in pos_to_idx or new_next in pos_to_idx:
                continue
                
            # Also check if new_prev == next_p or new_next == prev?
            # If new_prev == next_p, then chain is ... i-2, next_p, center, new_next ...
            # This is a valid topology if next_p was not occupied? But next_p IS occupied (by i+1).
            # So new_prev cannot be next_p.
            # My candidate generation excluded `prev` (old position) and `chain[i-2]`.
            # It did NOT exclude `next_p`.
            # So I must check `new_prev != next_p`.
            if new_prev == next_p or new_next == prev:
                continue
            
            # Calculate Energy Delta
            # Residues affected: i-1, i+1. (i is fixed).
            # Old contacts involving i-1 and i+1.
            
            def get_contacts(idx, pos):
                count = 0
                if sequence[idx] == 'H':
                    for nb in _neighbours(pos):
                        j = pos_to_idx.get(nb)
                        if j is not None and j != idx - 1 and j != idx + 1:
                            if sequence[j] == 'H':
                                count += 1
                return count

            old_c_prev = get_contacts(i-1, prev)
            old_c_next = get_contacts(i+1, next_p)
            
            # Also need to check if prev and next_p were in contact with EACH OTHER?
            # They are not adjacent in chain (distance 2).
            # If they are spatial neighbors, that's a contact.
            # Let's check if prev and next_p are neighbors.
            if abs(prev[0]-next_p[0]) + abs(prev[1]-next_p[1]) == 1:
                if sequence[i-1] == 'H' and sequence[i+1] == 'H':
                    old_c_prev += 1 # This contact is counted in prev's neighbors?
                    # In get_contacts(i-1, prev), we check neighbors of prev.
                    # If next_p is a neighbor of prev, and sequence[i+1]=='H', and i+1 != i-1+1 (which is i) and i+1 != i-1-1 (which is i-2).
                    # So yes, it would be counted.
                    # Similarly for next_p.
                    # So we don't need to add it separately if get_contacts handles it.
                    # BUT, if we move them, we must ensure we don't double count or miss it.
                    # The function get_contacts looks at ALL neighbors in pos_to_idx except chain neighbors.
                    # So it correctly counts the i-1/i+1 contact if it exists.
            
            # New positions
            # Temporarily update pos_to_idx to calculate new contacts?
            # Or just calculate manually.
            # It's safer to simulate the move in a local dict or just calculate deltas.
            
            # Let's remove prev and next_p from pos_to_idx, add new_prev and new_next, then calc.
            # This is O(1) updates.
            
            pos_to_idx.pop(prev)
            pos_to_idx.pop(next_p)
            pos_to_idx[new_prev] = i-1
            pos_to_idx[new_next] = i+1
            
            new_c_prev = get_contacts(i-1, new_prev)
            new_c_next = get_contacts(i+1, new_next)
            
            # Revert pos_to_idx for now (if we reject)
            # We'll keep it updated if we accept.
            
            delta_e = (old_c_prev + old_c_next) - (new_c_prev + new_c_next)
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                # Accept
                chain[i-1] = new_prev
                chain[i+1] = new_next
                current_energy += delta_e
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
            else:
                # Reject: Revert pos_to_idx
                pos_to_idx.pop(new_prev)
                pos_to_idx.pop(new_next)
                pos_to_idx[prev] = i-1
                pos_to_idx[next_p] = i+1
        else:
            # Segment shift? Or just another pivot.
            # Let's stick to pivots and crankshafts for simplicity and robustness.
            pass
            
        # Cooling
        T *= cooling
        # Ensure T doesn't go to 0 too fast
        if T < T_min:
            break

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Find a low-energy conformation for the given HP sequence on a square lattice.
    Uses multiple Simulated Annealing restarts.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    rng = random.Random(42 + len(sequence))

    best_chain = None
    best_energy = 0 # Worst case is 0 (no contacts), we want negative

    # Allocate time budget
    # 10 sequences, 120s total. ~12s per sequence.
    
    start_time = time.time()
    
    # For small sequences, we can afford more restarts.
    # For large sequences, SA is very effective, so fewer restarts might be needed,
    # but each restart should run longer.
    
    if n <= 25:
        max_restarts = 50
        time_per_restart = 0.2
    elif n <= 40:
        max_restarts = 30
        time_per_restart = 0.3
    elif n <= 60:
        max_restarts = 20
        time_per_restart = 0.5
    else:
        max_restarts = 10
        time_per_restart = 1.0

    for restart in range(max_restarts):
        if _remaining() < 1.0:
            break
        
        # Check if we have enough time for this restart
        if time.time() - start_time + time_per_restart > _DEADLINE - 5:
             # Adjust time limit for this restart
             current_time = time.time()
             time_limit = _DEADLINE - 5 - current_time
             if time_limit < 0.1:
                 break
        else:
             time_limit = time_per_restart

        chain, energy = _sa_fold(sequence, rng, time_limit)
        
        if chain is not None and energy < best_energy:
            best_chain = chain
            best_energy = energy
            
    # If no chain was found, return a straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
