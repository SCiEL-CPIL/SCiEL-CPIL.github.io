"""HP lattice protein folding on the cubic lattice - improved simulated annealing with incremental energy."""
# EVOLVE-BLOCK-START
import random
import math
import time


def _manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])


def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D."""
    yield (point[0] + 1, point[1], point[2])
    yield (point[0] - 1, point[1], point[2])
    yield (point[0], point[1] + 1, point[2])
    yield (point[0], point[1] - 1, point[2])
    yield (point[0], point[1], point[2] + 1)
    yield (point[0], point[1], point[2] - 1)


def _compute_initial_energy(sequence, chain, occupied):
    """Compute the initial energy."""
    energy = 0
    n = len(chain)
    h_indices = [i for i in range(n) if sequence[i] == 'H']
    for i in h_indices:
        p = chain[i]
        for q in _neighbours(p):
            j = occupied.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                energy -= 1
    return energy


def _sa_fold(sequence, dimension, time_budget, rng):
    n = len(sequence)
    if n == 0:
        return []
    
    # 1. Initialization: Random Self-Avoiding Walk
    # Start at origin
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    # Try to build a valid SAW
    success = False
    for attempt in range(100):
        chain = [(0, 0, 0)]
        occupied = {(0, 0, 0): 0}
        success = True
        for i in range(1, n):
            current = chain[-1]
            neighbors = list(_neighbours(current))
            rng.shuffle(neighbors)
            moved = False
            for nb in neighbors:
                if nb not in occupied:
                    chain.append(nb)
                    occupied[nb] = i
                    moved = True
                    break
            if not moved:
                success = False
                break
        if success:
            break
    
    if not success:
        # Fallback: Straight line (might not be optimal but valid)
        chain = [(i, 0, 0) for i in range(n)]
        occupied = {p: i for i, p in enumerate(chain)}

    current_energy = _compute_initial_energy(sequence, chain, occupied)
    best_chain = list(chain)
    best_energy = current_energy
    
    h_indices = [i for i in range(n) if sequence[i] == 'H']
    
    start_time = time.time()
    t = 10.0
    t_min = 0.01
    cooling_rate = 0.9999
    
    # Pre-compute neighbor offsets for speed
    offsets = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]

    while time.time() - start_time < time_budget:
        # Pick a move type
        # 80% Single Residue, 20% Segment Shift (length 2-4)
        if rng.random() < 0.8:
            # Single Residue Move
            i = rng.randint(0, n - 1)
            old_pos = chain[i]
            
            # Try random neighbor
            nb = rng.choice(offsets)
            new_pos = (old_pos[0] + nb[0], old_pos[1] + nb[1], old_pos[2] + nb[2])
            
            if new_pos in occupied:
                continue
            
            # Check connectivity
            valid = True
            if i > 0:
                if _manhattan(new_pos, chain[i-1]) != 1:
                    valid = False
            if valid and i < n - 1:
                if _manhattan(new_pos, chain[i+1]) != 1:
                    valid = False
            
            if not valid:
                continue
            
            # Calculate Delta E
            delta_e = 0
            if sequence[i] == 'H':
                # Remove old contacts
                for q in _neighbours(old_pos):
                    j = occupied.get(q)
                    if j is not None and j != i-1 and j != i+1 and sequence[j] == 'H':
                        # This was a contact
                        # Check if j is still a neighbor after move?
                        # No, we are moving i away from old_pos.
                        # So this contact is broken.
                        delta_e += 1 # Energy increases (less negative)
                
                # Add new contacts
                for q in _neighbours(new_pos):
                    j = occupied.get(q)
                    if j is not None and j != i-1 and j != i+1 and sequence[j] == 'H':
                        # This is a new contact
                        # Check if this contact existed before?
                        # No, because new_pos was free.
                        delta_e -= 1 # Energy decreases (more negative)
            
            # Acceptance
            if delta_e < 0 or rng.random() < math.exp(-delta_e / t):
                # Accept
                occupied.pop(old_pos)
                occupied[new_pos] = i
                chain[i] = new_pos
                current_energy += delta_e
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
        else:
            # Segment Shift Move
            # Pick small segment length
            seg_len = rng.randint(2, 4)
            if n < seg_len:
                continue
            i = rng.randint(0, n - seg_len)
            j = i + seg_len - 1
            
            # Pick shift vector
            # Try small shifts
            shift = rng.choice(offsets)
            
            # Check if all new positions are free
            new_positions = []
            valid = True
            for k in range(i, j + 1):
                old_p = chain[k]
                new_p = (old_p[0] + shift[0], old_p[1] + shift[1], old_p[2] + shift[2])
                # Check collision
                # Note: We are moving the whole segment.
                # We must ensure new_p is not occupied by *any* other part of the chain.
                # And new_p must not be one of the *old* positions of the segment itself (unless it's a swap, which shift doesn't do).
                if new_p in occupied:
                    # Check if it's part of the segment being moved
                    # If it is, it's fine as long as we handle it correctly?
                    # Actually, if new_p is in occupied, and it belongs to the segment, 
                    # it means the segment is colliding with itself or shifting into its own old spot.
                    # Since it's a shift, old spots are vacated.
                    # But we check `new_p in occupied` which includes the segment's old spots.
                    # So we need to exclude the segment's old spots from the collision check.
                    # Let's collect old spots first.
                    pass
            
            # Refine collision check
            old_spots = set(chain[k] for k in range(i, j + 1))
            for k in range(i, j + 1):
                old_p = chain[k]
                new_p = (old_p[0] + shift[0], old_p[1] + shift[1], old_p[2] + shift[2])
                if new_p in occupied and new_p not in old_spots:
                    valid = False
                    break
                new_positions.append(new_p)
            
            if not valid:
                continue
            
            # Check connectivity at boundaries
            if i > 0:
                if _manhattan(new_positions[0], chain[i-1]) != 1:
                    valid = False
            if valid and j < n - 1:
                if _manhattan(new_positions[-1], chain[j+1]) != 1:
                    valid = False
            
            if not valid:
                continue
            
            # Calculate Delta E
            delta_e = 0
            for k in range(i, j + 1):
                if sequence[k] == 'H':
                    old_p = chain[k]
                    new_p = new_positions[k - i]
                    
                    # Remove old contacts
                    for q in _neighbours(old_p):
                        idx = occupied.get(q)
                        if idx is not None:
                            # Exclude chain neighbors
                            if idx == k - 1 or idx == k + 1:
                                continue
                            # Exclude neighbors within the segment that are moving
                            # If idx is in [i, j], it's part of the segment.
                            # But wait, if idx is in the segment, and q is a neighbor of old_p,
                            # and idx is a chain neighbor of k, we skipped it.
                            # If idx is in the segment but NOT a chain neighbor, it's a contact.
                            # Does this contact break?
                            # Yes, because k moves.
                            # But does the new position form a contact with the *new* position of idx?
                            # We handle that in the "Add new contacts" step.
                            if sequence[idx] == 'H':
                                delta_e += 1
                    
                    # Add new contacts
                    for q in _neighbours(new_p):
                        idx = occupied.get(q)
                        if idx is not None:
                            # Exclude chain neighbors in the *new* configuration
                            # Chain neighbors of k are k-1 and k+1.
                            # Their positions will be new_positions[k-1-i] and new_positions[k+1-i].
                            # But `occupied` still has OLD positions for the segment.
                            # So we must be careful.
                            
                            # We check if q is a valid H neighbor.
                            # If idx is in [i, j], it is part of the segment.
                            # If idx is a chain neighbor of k, skip.
                            if idx == k - 1 or idx == k + 1:
                                continue
                            
                            # If idx is in the segment, we need to check if the NEW position of idx is q?
                            # No, q is a neighbor of new_p (new position of k).
                            # We are checking if there is an H residue at q.
                            # But `occupied` maps OLD positions to indices.
                            # So `occupied.get(q)` returns the index of the residue *currently* at q.
                            # If q is the OLD position of some segment member, it will be returned.
                            # But that member is moving away.
                            # So we should not count it as a contact from the new position.
                            
                            # Correct logic:
                            # A contact is formed between new_p and q if:
                            # 1. There is an H residue at q in the FINAL configuration.
                            # 2. The residue at q is not a chain neighbor of k.
                            
                            # How to know if there is an H residue at q in the FINAL configuration?
                            # If q is not in old_spots, then the residue at q is not moving.
                            #   So if sequence[occupied.get(q)] == 'H', it's a contact.
                            # If q is in old_spots, then the residue at q IS moving.
                            #   Let idx_old = occupied.get(q). This is the index of the residue moving FROM q.
                            #   But we care about the residue at q in the NEW config.
                            #   The residue at q in the new config is the one that moved TO q.
                            #   Let's find if any new_positions[m] == q.
                            #   If so, let idx_new be that index.
                            #   If sequence[idx_new] == 'H' and idx_new is not chain neighbor of k, it's a contact.
                            
                            # This is getting complicated.
                            # Simplification: 
                            # Only count contacts with residues NOT in the moving segment.
                            # Contacts within the moving segment are preserved (relative geometry is shifted, but neighbors are neighbors).
                            # Wait, if I shift a segment, the internal distances are preserved.
                            # So any H-H contact *within* the segment remains a contact.
                            # So we only need to calculate Delta E for contacts between the segment and the rest of the chain.
                            
                            # So:
                            # Remove contacts between segment and rest (at old positions).
                            # Add contacts between segment and rest (at new positions).
                            
                            # Let's re-evaluate Delta E based on this.
                            pass
            
            # Re-implementing Delta E for Segment Shift correctly
            delta_e = 0
            # Remove old contacts between segment and non-segment
            for k in range(i, j + 1):
                if sequence[k] == 'H':
                    old_p = chain[k]
                    for q in _neighbours(old_p):
                        idx = occupied.get(q)
                        if idx is not None:
                            if idx < i or idx > j: # Non-segment
                                if sequence[idx] == 'H':
                                    delta_e += 1 # Breaking a contact
                            
            # Add new contacts between segment and non-segment
            for k in range(i, j + 1):
                if sequence[k] == 'H':
                    new_p = new_positions[k - i]
                    for q in _neighbours(new_p):
                        # Check if q is occupied by a non-segment H residue
                        # If q is in old_spots, it's a segment member (old pos).
                        # If q is not in old_spots, it's a non-segment member.
                        if q not in old_spots:
                            idx = occupied.get(q)
                            if idx is not None:
                                if sequence[idx] == 'H':
                                    # Check if idx is chain neighbor of k?
                                    # If idx is non-segment, it can't be chain neighbor of k (which is in segment)
                                    # UNLESS i=0 or j=n-1? No, chain neighbors are i-1, i+1, etc.
                                    # If idx is non-segment, it is not k-1 or k+1 (which are in segment).
                                    # So it's a valid contact.
                                    delta_e -= 1 # Forming a contact
            
            # Acceptance
            if delta_e < 0 or rng.random() < math.exp(-delta_e / t):
                # Accept
                # Update occupied and chain
                for k in range(i, j + 1):
                    old_p = chain[k]
                    new_p = new_positions[k - i]
                    occupied.pop(old_p)
                    chain[k] = new_p
                    occupied[new_p] = k
                
                current_energy += delta_e
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
        
        # Cooling
        t *= cooling_rate
        if t < t_min:
            t = t_min

    return best_chain


def fold(sequence, dimension):
    """
    Fold a protein sequence on the cubic lattice using simulated annealing.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n == 0:
        return []
    
    rng = random.Random(12345 + len(sequence))
    
    # Allocate time
    # Total budget 240s for 26 sequences.
    # Average ~9s.
    # Give longer sequences more time.
    base_time = 2.0
    time_budget = base_time + (n / 100.0) * 4.0
    time_budget = min(time_budget, 15.0)
    
    chain = _sa_fold(sequence, dimension, time_budget, rng)
    
    return [list(p) for p in chain]
# EVOLVE-BLOCK-END
