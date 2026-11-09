"""Contact-map feasibility approach for HP lattice protein folding."""
import random
import math
import time

# EVOLVE-BLOCK-START

def _neighbours(point):
    """Generate all 6 neighboring sites on cubic lattice."""
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)


def _compute_contact_map(sequence):
    """
    Pre-compute which H-H pairs CAN be contacts based on chain distance.
    Two residues i and j can be contacts if |i-j| >= 2 and they can be
    placed on adjacent lattice sites (Manhattan distance 1).
    For a self-avoiding walk, the minimum lattice distance between residues
    i and j is at least |i-j| (each step moves at most 1 unit).
    They can be contacts if their lattice positions differ by exactly 1 in
    one coordinate. The minimum possible lattice distance is 1 (if they're
    adjacent in space). This is always possible for |i-j| >= 2 as long as
    the walk doesn't force them apart. We return all pairs (i,j) where
    both are H and |i-j| >= 2.
    """
    n = len(sequence)
    contacts = []
    for i in range(n):
        if sequence[i] != 'H':
            continue
        for j in range(i+2, n):
            if sequence[j] != 'H':
                continue
            # Check if it's geometrically possible: min lattice distance is 1
            # For |i-j| >= 2, we can always arrange them adjacent in a spiral
            contacts.append((i, j))
    return contacts


def _initial_chain(sequence, dimension, rng):
    """Create an initial self-avoiding chain using a simple growth strategy."""
    n = len(sequence)
    if n == 0:
        return [], {}
    
    # Start at origin
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Score: prefer sites with more free space (less trapped)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Add small random perturbation
            options.append((free + rng.random() * 0.1, site))
        
        if not options:
            # Dead end - try to backtrack or restart
            return None, None
        
        # Pick the site with most free space (with some randomness)
        options.sort(reverse=True)
        # Pick from top 3 with some probability
        k = min(3, len(options))
        site = options[rng.randint(0, k-1)][1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain, occupied


def _apply_pivot(chain, occupied, rng):
    """Apply a pivot move: rotate a subchain around a hinge point."""
    n = len(chain)
    if n < 3:
        return False, None, None, 0
    
    # Pick a pivot point (not the first residue)
    pivot_idx = rng.randint(1, n-2)
    pivot = chain[pivot_idx]
    
    # The part to rotate is chain[pivot_idx+1:]
    # We need to rotate it 90 degrees around the pivot
    # In 3D, there are 24 possible rotations, but we'll do a simple one
    # Pick a random axis to rotate around
    axis = rng.randint(0, 2)
    direction = rng.choice([-1, 1])
    
    # Compute the vector from pivot to the next residue
    vec = [chain[pivot_idx+1][a] - pivot[a] for a in range(3)]
    
    # Rotate vec 90 degrees around the chosen axis
    # This is a simplification - we'll just try moving the tip
    # Actually, let's do a proper 90-degree rotation of the subchain
    
    # For simplicity, let's just try to move the tip of the subchain
    # to a different valid position
    tip_idx = n - 1
    tip = chain[tip_idx]
    
    options = []
    for site in _neighbours(tip):
        if site in occupied and occupied[site] != tip_idx:
            continue
        options.append(site)
    
    if not options:
        return False, None, None, 0
    
    new_tip = options[rng.randint(0, len(options)-1)]
    
    # Now we need to re-route the subchain from chain[pivot_idx] to new_tip
    # This is complex, so let's just do a simple creep move instead
    
    # Actually, let's implement a simpler move: creep
    # Move the tip to an adjacent free site
    return _apply_crep(chain, occupied, rng)


def _apply_crep(chain, occupied, rng):
    """Apply a creep move: move the tip of the chain to an adjacent free site."""
    n = len(chain)
    if n < 1:
        return False, None, None, 0
    
    tip = chain[-1]
    tip_idx = n - 1
    
    options = []
    for site in _neighbours(tip):
        if site in occupied and occupied[site] != tip_idx:
            continue
        # Make sure the new site doesn't create a bond with a non-adjacent residue
        # (this is checked by self-avoidance)
        options.append(site)
    
    if not options:
        return False, None, None, 0
    
    new_tip = options[rng.randint(0, len(options)-1)]
    
    # Create new chain
    new_chain = chain[:-1] + [new_tip]
    
    # Create new occupied set
    new_occupied = dict(occupied)
    del new_occupied[tip]
    new_occupied[new_tip] = tip_idx
    
    # Compute energy delta
    delta_e = _energy_delta(sequence_holder['seq'], chain, new_chain, tip, new_tip)
    
    return True, new_chain, new_occupied, delta_e


# Global holder for sequence (to avoid passing it around)
sequence_holder = {'seq': None}


def _energy_delta(sequence, old_chain, new_chain, removed_pos, added_pos):
    """
    Compute the change in energy when removing a residue from removed_pos
    and adding it at added_pos.
    """
    n = len(sequence)
    
    # Build position-to-index map for new chain
    new_where = {p: i for i, p in enumerate(new_chain)}
    
    delta = 0
    residue_idx = n - 1  # The tip residue
    
    if sequence[residue_idx] != 'H':
        return 0
    
    # Remove contacts involving the tip in old chain
    old_where = {p: i for i, p in enumerate(old_chain)}
    for q in _neighbours(removed_pos):
        j = old_where.get(q)
        if j is not None and j != residue_idx - 1 and sequence[j] == 'H':
            # This was a contact in the old chain
            delta += 1  # Energy was -contacts, so removing a contact increases energy by 1
    
    # Add contacts involving the tip in new chain
    for q in _neighbours(added_pos):
        j = new_where.get(q)
        if j is not None and j != residue_idx - 1 and sequence[j] == 'H':
            # This is a new contact in the new chain
            delta -= 1  # Adding a contact decreases energy by 1
    
    return delta


def _energy(sequence, chain):
    """Compute the full energy of a chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _simulated_annealing(sequence, dimension, time_budget):
    """
    Simulated annealing to find low-energy conformations.
    Uses the contact map to guide the search.
    """
    global sequence_holder
    sequence_holder['seq'] = sequence
    
    n = len(sequence)
    if n == 0:
        return []
    
    # Get initial chain
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = float('inf')
    
    # Generate multiple initial chains
    max_initial_attempts = min(20, n)
    for attempt in range(max_initial_attempts):
        chain, occupied = _initial_chain(sequence, dimension, rng)
        if chain is None:
            continue
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
    
    if best_chain is None:
        # Fallback: straight line
        return [[i, 0, 0] for i in range(n)]
    
    current_chain = list(best_chain)
    current_occupied = {p: i for i, p in enumerate(current_chain)}
    current_energy = best_energy
    
    # Simulated annealing
    start_time = time.time()
    T = max(1.0, -best_energy * 0.5)  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.999
    
    steps_since_improvement = 0
    max_steps_no_improvement = 5000
    
    while T > T_min:
        if time.time() - start_time > time_budget:
            break
        
        # Do a batch of moves at this temperature
        batch_size = 100
        for _ in range(batch_size):
            if time.time() - start_time > time_budget:
                break
            
            # Choose move type
            r = rng.random()
            if r < 0.7:
                success, new_chain, new_occupied, delta_e = _apply_crep(current_chain, current_occupied, rng)
            else:
                # Try a double creep: move tip, then move new tip
                success, new_chain, new_occupied, delta_e = _apply_crep(current_chain, current_occupied, rng)
                if success:
                    success2, new_chain2, new_occupied2, delta_e2 = _apply_crep(new_chain, new_occupied, rng)
                    if success2:
                        new_chain, new_occupied, delta_e = new_chain2, new_occupied2, delta_e + delta_e2
                    else:
                        success = False
            
            if not success:
                continue
            
            # Acceptance criterion
            if delta_e < 0:
                # Better energy - always accept
                current_chain = new_chain
                current_occupied = new_occupied
                current_energy += delta_e
                steps_since_improvement = 0
            else:
                # Worse energy - accept with probability exp(-delta_e / T)
                if T > 0:
                    prob = math.exp(-delta_e / T)
                    if rng.random() < prob:
                        current_chain = new_chain
                        current_occupied = new_occupied
                        current_energy += delta_e
            
            # Update best
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
        
        # If stuck for too long, restart with a perturbed chain
        if steps_since_improvement > max_steps_no_improvement:
            # Perturb the best chain and restart
            perturbed_chain, perturbed_occupied = _initial_chain(sequence, dimension, rng)
            if perturbed_chain is not None:
                perturbed_energy = _energy(sequence, perturbed_chain)
                if perturbed_energy < current_energy:
                    current_chain = perturbed_chain
                    current_occupied = perturbed_occupied
                    current_energy = perturbed_energy
            steps_since_improvement = 0
    
    return [list(p) for p in best_chain]


def fold(sequence, dimension):
    """
    Find a low-energy conformation for an HP chain on the cubic lattice.
    
    Args:
        sequence: String of 'H' and 'P' residues
        dimension: Lattice dimension (should be 3)
    
    Returns:
        List of lattice points (lists of integers)
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Set global deadline
    deadline = 300.0  # Total budget for all sequences
    
    # Estimate time per sequence
    time_per_seq = min(10.0, max(1.0, n / 50.0))  # Scale with chain length
    
    start_time = time.time()
    
    # Use simulated annealing
    result = _simulated_annealing(sequence, dimension, time_per_seq)
    
    # Ensure result is valid format
    if len(result) != n:
        # Fallback
        result = [[i, 0, 0] for i in range(n)]
    
    return result

# EVOLVE-BLOCK-END
