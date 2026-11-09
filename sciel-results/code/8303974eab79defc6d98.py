# EVOLVE-BLOCK-START
import random
import time
import math

def _compute_energy(sequence, chain):
    """Compute energy = -number of H-H contacts."""
    n = len(chain)
    if n == 0:
        return 0
    
    # Map position to index for O(1) lookup
    pos_to_idx = {chain[i]: i for i in range(n)}
    
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            x, y, z = chain[i]
            # Check 6 neighbors
            for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                nb = (x+dx, y+dy, z+dz)
                j = pos_to_idx.get(nb)
                if j is not None and j > i and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _initial_fold(sequence):
    """
    Generate an initial compact fold using a spiral pattern.
    This ensures a valid self-avoiding walk with high density.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Simple spiral in 3D (layered 2D spirals)
    # Start at (0,0,0)
    # Move right, up, left, down, expand, etc.
    # For simplicity and speed, we can use a "snake" in a box or a standard spiral.
    # A standard 2D spiral projected to 3D (z=0) is valid but might not be optimal for 3D packing.
    # Let's try a 3D spiral.
    
    # A simpler robust initialization:
    # Place residues in a compact block.
    # We can generate a valid SAW by random walk with backtracking if needed, 
    # but for HP, a "spiral" is often a good start.
    
    # Let's use a deterministic spiral for reproducibility and compactness.
    # 3D Spiral:
    # Layer 0: z=0. 2D spiral.
    # Layer 1: z=1. 2D spiral shifted.
    # Connect them.
    
    # Actually, a very robust and fast initializer is a random walk that tries to stay compact.
    # But let's stick to a structured spiral for better initial contacts.
    
    # Generate a 2D spiral of size k x k, then stack layers.
    # For N residues, estimate k = ceil(sqrt(N/2)) or similar.
    
    # Simpler: Just do a random walk with a bias towards the center.
    # But to ensure validity, let's use a simple recursive backtracker for the first 50 steps if needed?
    # No, let's use a "spiral" in a 3D grid.
    
    # Algorithm:
    # 1. Determine box size.
    # 2. Fill box in a spiral order.
    # 3. If N > box capacity, expand.
    
    # Let's try a simpler "Greedy Compact" init.
    # Start at (0,0,0).
    # At each step, pick a neighbor that minimizes distance to the centroid of existing chain,
    # with a random tie-breaker.
    
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0)}
    
    cx, cy, cz = 0, 0, 0
    count = 1
    
    for i in range(1, n):
        last = chain[-1]
        x, y, z = last
        
        candidates = []
        for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            nb = (x+dx, y+dy, z+dz)
            if nb not in occupied:
                # Score: distance to centroid
                dist = (nb[0] - cx/count)**2 + (nb[1] - cy/count)**2 + (nb[2] - cz/count)**2
                # Add slight randomness to break ties
                score = dist + random.random() * 0.1
                candidates.append((score, nb))
        
        if not candidates:
            # Dead end, this shouldn't happen with 3D spiral logic if we are careful, 
            # but if it does, we might need to backtrack. 
            # For a simple init, we'll just pick a random valid move if available, 
            # or restart init if stuck (rare for small N).
            # Let's just break and return what we have, or try a different init.
            # To be safe, let's just return a linear chain if we get stuck, 
            # but the SA will fix it.
            # Actually, let's just pick the first valid one if any, else break.
            # With 3D, dead ends are very rare in early stages.
            break
            
        candidates.sort()
        best_nb = candidates[0][1]
        chain.append(best_nb)
        occupied.add(best_nb)
        
        # Update centroid incrementally
        cx += best_nb[0]
        cy += best_nb[1]
        cz += best_nb[2]
        count += 1
        
    # If we didn't reach N, pad with a linear extension (shouldn't happen)
    if len(chain) < n:
        last = chain[-1]
        x, y, z = last
        for i in range(len(chain), n):
            # Just go forward in x
            nb = (x+1, y, z)
            while nb in occupied:
                # Try to go up
                nb = (x+1, y+1, z)
                if nb in occupied:
                     nb = (x+1, y, z+1)
            chain.append(nb)
            occupied.add(nb)
            x, y, z = nb
            
    return chain

def _flip_move(sequence, chain, i, j, rng):
    """
    Attempt a flip move between indices i and j.
    Returns (new_chain, delta_energy) or (None, 0) if invalid.
    """
    n = len(chain)
    if i == j:
        return None, 0
    
    # A flip is valid if:
    # 1. The reversed segment does not self-intersect.
    # 2. The boundaries (i-1, i) and (j, j+1) remain valid (unit steps).
    #    Note: The internal steps of the reversed segment are automatically valid 
    #    because the original was valid and we are just reversing the order of positions.
    #    Wait, if we reverse the *positions* in the list, the adjacency is preserved 
    #    because the set of positions is the same, just traversed in reverse.
    #    So internal validity is guaranteed.
    #    We only need to check:
    #    a. Self-avoidance of the reversed segment against the rest of the chain.
    #    b. Connectivity at the boundaries.
    
    # Boundary check:
    # If i > 0, chain[i-1] must be adjacent to chain[j] (new position of i).
    # If j < n-1, chain[j+1] must be adjacent to chain[i] (new position of j).
    
    if i > 0:
        p_prev = chain[i-1]
        p_new_i = chain[j]
        if abs(p_prev[0]-p_new_i[0]) + abs(p_prev[1]-p_new_i[1]) + abs(p_prev[2]-p_new_i[2]) != 1:
            return None, 0
            
    if j < n-1:
        p_next = chain[j+1]
        p_new_j = chain[i]
        if abs(p_next[0]-p_new_j[0]) + abs(p_next[1]-p_new_j[1]) + abs(p_next[2]-p_new_j[2]) != 1:
            return None, 0
            
    # Self-avoidance check:
    # The segment chain[i:j+1] is reversed.
    # We need to ensure that no position in the reversed segment is occupied by the rest of the chain.
    # The rest of the chain is chain[0:i] and chain[j+1:n].
    
    # Build a set of occupied positions for the "fixed" parts
    fixed_occupied = set()
    for k in range(0, i):
        fixed_occupied.add(chain[k])
    for k in range(j+1, n):
        fixed_occupied.add(chain[k])
        
    # Check the reversed segment
    # The positions are the same, just order changed. 
    # So if the original segment was self-avoiding (which it is, as it's part of a SAW),
    # and none of its positions are in fixed_occupied, then the new segment is valid.
    # Wait, the original segment positions are chain[i], ..., chain[j].
    # Are any of these in fixed_occupied? 
    # In a valid SAW, no two positions are the same. So chain[i..j] are distinct from chain[0..i-1] and chain[j+1..n-1].
    # Therefore, the set of positions is disjoint.
    # So self-avoidance is automatically preserved for Flip moves!
    # We only need to check the boundary connectivity, which we did above.
    
    # So, if boundaries are valid, the flip is a valid move.
    
    # Calculate Delta Energy
    # Energy changes only for contacts involving residues in [i, j] and their neighbors.
    # Specifically, contacts involving:
    # 1. Residues in [i, j] and residues in [0, i-1]
    # 2. Residues in [i, j] and residues in [j+1, n-1]
    # 3. Contacts *within* [i, j] remain the same? 
    #    No, the *sequence* of residues is reversed, but the *positions* are the same.
    #    The set of pairs of residues in contact is the same, because the geometric proximity is unchanged.
    #    However, the *indices* change. 
    #    Energy is defined on the sequence. 
    #    If residue A at index i and residue B at index k (i<k) are in contact, 
    #    after flip, residue A is at index j-(k-i) and residue B is at index j-(i-i)=j? 
    #    Let's trace indices.
    #    Original: ... i, i+1, ..., j ...
    #    Flipped:  ... j, j-1, ..., i ...
    #    The pair (u, v) in the original segment (i <= u < v <= j) becomes (j-(v-i), j-(u-i)) in the new segment.
    #    The geometric positions are the same. So if u and v were in contact, they are still in contact.
    #    The energy contribution from internal contacts of the segment is UNCHANGED.
    #    
    #    Therefore, we only need to recalculate contacts between the segment [i, j] and the outside.
    
    # Let's compute the energy contribution of the segment to the outside.
    # Old Energy Contribution:
    #   Sum of H-H contacts where one index is in [i, j] and the other is NOT in [i, j].
    # New Energy Contribution:
    #   Sum of H-H contacts where one index is in [i, j] (new indices) and the other is NOT in [i, j].
    
    # To do this efficiently, we need a position-to-index map for the CURRENT chain.
    # And we need to know which residues are H.
    
    # Let's build the map for the current chain.
    pos_to_idx = {chain[k]: k for k in range(n)}
    
    # Helper to get neighbors of a position that are in the chain
    def get_chain_neighbors(pos):
        x, y, z = pos
        nbs = []
        for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            nb = (x+dx, y+dy, z+dz)
            idx = pos_to_idx.get(nb)
            if idx is not None:
                nbs.append(idx)
        return nbs
    
    # Identify the set of indices in the segment: S = {i, ..., j}
    # Identify the set of indices outside: O = {0, ..., i-1} U {j+1, ..., n-1}
    
    # We need to count H-H contacts between S and O in the OLD state.
    # And in the NEW state.
    
    # In the OLD state:
    # For each k in S:
    #   if sequence[k] == 'H':
    #     for each neighbor idx in get_chain_neighbors(chain[k]):
    #       if idx in O and sequence[idx] == 'H':
    #         old_contacts += 1
    # Note: This counts each contact twice (once from k, once from idx). We want unique pairs.
    # So we can just sum and divide by 2, or only count if idx < k (but indices are mixed).
    # Let's just count all directed contacts and divide by 2 at the end, or use a set.
    # Given N is small (<=136), a set is fine.
    
    old_cross_contacts = set()
    for k in range(i, j+1):
        if sequence[k] == 'H':
            for nb_idx in get_chain_neighbors(chain[k]):
                if nb_idx < i or nb_idx > j: # nb_idx is in O
                    if sequence[nb_idx] == 'H':
                        old_cross_contacts.add((k, nb_idx) if k < nb_idx else (nb_idx, k))
                        
    # In the NEW state:
    # The positions are the same. The indices are mapped: k -> new_k = i + j - k.
    # The outside indices O remain the same.
    # We need to count H-H contacts between the new segment indices and O.
    
    new_cross_contacts = set()
    for k in range(i, j+1):
        new_k = i + j - k
        # The residue at position chain[k] is still sequence[k].
        # Wait! The *sequence* is fixed. The *chain* is the list of positions.
        # In the flip move, we are changing the *chain* (the list of positions).
        # The sequence `sequence` is fixed.
        # So, residue `sequence[k]` moves from position `chain[k]` to position `chain[i+j-k]`?
        # NO.
        # The flip move reverses the *segment of the chain*.
        # Original chain: P_0, P_1, ..., P_{i-1}, P_i, P_{i+1}, ..., P_j, P_{j+1}, ...
        # New chain:     P_0, P_1, ..., P_{i-1}, P_j, P_{j-1}, ..., P_i, P_{j+1}, ...
        # 
        # Residue `sequence[k]` (where i <= k <= j) was at position `P_k`.
        # In the new chain, the residue at index `new_k = i + j - k` is at position `P_k`?
        # Let's check.
        # New chain index `new_k` corresponds to position `P_{i + j - new_k}`?
        # If new_k = i, pos = P_j. Residue is sequence[i].
        # If new_k = j, pos = P_i. Residue is sequence[j].
        # So, residue `sequence[k]` is now at position `P_{i + j - k}`?
        # No, the residue `sequence[k]` is attached to the *index* k.
        # The flip changes which position each index occupies.
        # Index `k` (in [i, j]) now occupies position `P_{i + j - k}`.
        # 
        # So, to calculate the new energy:
        # For each index `k` in [i, j]:
        #   Pos = chain[i + j - k]
        #   Residue = sequence[k]
        #   Check neighbors of Pos.
        #   If neighbor index `nb_idx` is in O (outside) and sequence[nb_idx] == 'H' and sequence[k] == 'H':
        #       Add to new_cross_contacts.
        
    for k in range(i, j+1):
        if sequence[k] == 'H':
            new_pos = chain[i + j - k]
            for nb_idx in get_chain_neighbors(new_pos):
                if nb_idx < i or nb_idx > j: # nb_idx is in O
                    if sequence[nb_idx] == 'H':
                        new_cross_contacts.add((k, nb_idx) if k < nb_idx else (nb_idx, k))
                        
    # Delta energy
    # Old energy contribution from cross contacts: -len(old_cross_contacts)
    # New energy contribution from cross contacts: -len(new_cross_contacts)
    # Internal contacts are unchanged.
    # So delta_energy = (-len(new_cross_contacts)) - (-len(old_cross_contacts))
    #                  = len(old_cross_contacts) - len(new_cross_contacts)
    
    delta_energy = len(old_cross_contacts) - len(new_cross_contacts)
    
    # Construct new chain
    new_chain = chain[:i] + chain[i:j+1][::-1] + chain[j+1:]
    
    return new_chain, delta_energy

def fold(sequence, dimension):
    """
    Fold the sequence using Simulated Annealing with Flip moves.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    rng = random.Random(42)
    
    # Time budget: 240s total for 26 sequences. ~9s per sequence.
    # We allocate 8.5s per sequence.
    start_time = time.time()
    time_limit = 8.5
    
    # 1. Initial Fold
    chain = _initial_fold(sequence)
    
    # Ensure length is correct
    if len(chain) != n:
        # Fallback: linear chain
        chain = [(i, 0, 0) for i in range(n)]
        
    # 2. Simulated Annealing
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    temperature = 1.0
    cooling_rate = 0.9999
    
    # Pre-compute positions for fast lookup? 
    # The _flip_move function rebuilds pos_to_idx every time. 
    # This is O(N). For N=136, this is fast enough.
    
    while time.time() - start_time < time_limit:
        # Choose a flip move
        # Prefer shorter flips for local search, longer for global.
        # Random length
        length = rng.randint(1, n)
        i = rng.randint(0, n - length - 1) if n - length - 1 >= 0 else 0
        j = i + length
        
        if i == j:
            continue
            
        new_chain, delta_energy = _flip_move(sequence, chain, i, j, rng)
        
        if new_chain is None:
            continue
            
        # Acceptance criterion
        if delta_energy < 0:
            # Better energy
            chain = new_chain
            current_energy += delta_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        else:
            # Worse energy, accept with probability
            if temperature > 0:
                prob = math.exp(-delta_energy / temperature)
                if rng.random() < prob:
                    chain = new_chain
                    current_energy += delta_energy
                    
        # Cool down
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
            
    # Return the best chain found
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
