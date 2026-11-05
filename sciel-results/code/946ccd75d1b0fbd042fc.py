"""
HP lattice protein folding using a contact-graph maximum independent set (MIS) heuristic.

Strategy:
1. Precompute all geometrically feasible H-H contacts (i, j) for the sequence length n.
   A contact is feasible if |i-j| > 1 and the Manhattan distance between the two positions
   on a self-avoiding walk is 1. This is approximated by checking if the Manhattan distance
   between the positions in a reference grid is 1, but since we don't have a fixed grid, we
   instead enumerate all pairs (i, j) with |i-j| > 1 and consider them as potential contacts.
   We then build a conflict graph where two contacts conflict if they share a residue index
   or if their geometric positions would overlap in a plausible lattice embedding.

   However, a more practical approach for this task is to:
   - Generate an initial conformation using a space-filling curve (Boustrophedon).
   - Identify all H-H contacts in this conformation.
   - Build a conflict graph where nodes are H residues and edges represent conflicts
     (two H residues cannot both be in a contact if they are adjacent in the chain or
     if their positions in the current conformation are not adjacent on the lattice).
   - Use a greedy MIS heuristic to select a maximal set of non-conflicting H residues
     that could form contacts.
   - Use this set to guide a local search that tries to bring these H residues together.

   But the most direct interpretation of the directive is to model the folding as an MIS
   problem on the contact graph. Let's do this:

   Step 1: Generate a reference grid (Boustrophedon) of size W x H.
   Step 2: Map the sequence to this grid.
   Step 3: For each pair of H residues (i, j) with |i-j| > 1, check if their positions
           in the grid are adjacent (Manhattan distance 1). If so, this is a valid contact
           in the reference conformation.
   Step 4: Build a conflict graph where nodes are H residues. Two H residues i and j are
           connected by an edge if they are adjacent in the chain (|i-j| <= 1) OR if they
           are not adjacent in the reference conformation (Manhattan distance > 1).
           Wait, this is not quite right. The MIS problem should be: we want to select
           a set of contacts (edges in the contact graph) such that no two selected contacts
           share a residue. This is a matching problem, not MIS.

   Let's reinterpret: The contact graph has H residues as nodes. An edge between i and j
   exists if i and j can be in contact (i.e., |i-j| > 1 and their positions in some
   conformation are adjacent). We want to find a maximum set of such edges such that no
   two edges share a node. This is a maximum matching problem.

   But the directive says "maximum independent set (MIS) problem on a conflict graph
   constructed from the sequence's H-residue contact possibilities, where nodes represent
   potential H-H contacts and edges represent geometric or sequential incompatibilities".

   So nodes are potential contacts (pairs (i, j)), and edges between nodes (i, j) and (k, l)
   exist if the two contacts are incompatible (share a residue or have conflicting geometry).
   We want a maximum independent set of nodes (contacts) that are mutually compatible.

   This is a valid formulation. Let's implement it.

   However, enumerating all possible contacts for n=100 is O(n^2) = 10,000 nodes, which is
   manageable. But checking incompatibility between all pairs of contacts is O(n^4) = 10^8,
   which is too slow. We need a smarter approach.

   Practical approach:
   1. Generate a reference conformation (Boustrophedon).
   2. Identify all H-H contacts in this conformation.
   3. Build a conflict graph on these contacts: two contacts conflict if they share a residue.
   4. Find a maximum independent set in this conflict graph (which is equivalent to a maximum
      matching in the contact graph, since the conflict graph is a line graph of the contact graph).
   5. Use this set of contacts to guide a local search.

   But since the reference conformation may not have many contacts, this may not be effective.

   Alternative practical approach:
   1. Generate multiple initial conformations (Boustrophedon with different orientations).
   2. For each conformation, compute the energy.
   3. Use a fast local search (pivot moves) to refine each conformation.
   4. Return the best conformation found.

   This is essentially the same as the current program. To make it different, let's use a
   different local search method: simulated annealing with a fast move set.

   But the directive specifically asks for an MIS-based approach. Let's try to implement it
   in a way that is practical:

   1. Generate a reference grid (Boustrophedon) of size W x H.
   2. Map the sequence to this grid.
   3. For each pair of H residues (i, j) with |i-j| > 1, if their positions in the grid are
      adjacent, add an edge (i, j) to the contact graph.
   4. Build the conflict graph: nodes are edges of the contact graph. Two nodes (edges) are
      connected if they share a vertex (residue).
   5. Find a maximum independent set in the conflict graph using a greedy heuristic.
   6. The selected contacts indicate which H residues should be in contact.
   7. Use this information to guide a local search that tries to bring these H residues together.

   For step 7, we can use a potential-based local search: define a potential function that
   is the sum of distances between H residues that are in the selected contact set, and
   minimize this potential using pivot moves.

   This is a different strategy from the current program, which uses a greedy local search
   on the energy function.

   Let's implement this.
"""

import random
import time

# EVOLVE-BLOCK-START

def _neighbours(point):
    """Generate the 4 neighbours of a point on the square lattice."""
    x, y = point
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

def _energy(sequence, chain):
    """Compute HP contact energy."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _generate_skeleton(length, dimension=2):
    """
    Generate a self-avoiding walk (skeleton) of the given length on a square lattice.
    Uses a space-filling curve approach (Boustrophedon) on a rectangular grid,
    then trims or pads to the exact length.
    """
    # Find grid dimensions W x H such that W*H >= length
    # Prefer roughly square grids for better contact density
    W = int(length ** 0.5)
    if W == 0:
        W = 1
    H = (length + W - 1) // W
    
    # Ensure we can fit 'length' points. If W*H < length, expand.
    while W * H < length:
        H += 1
        
    points = []
    for r in range(H):
        row = []
        for c in range(W):
            # Boustrophedon: snake back and forth
            if r % 2 == 0:
                row.append((c, r))
            else:
                row.append((W - 1 - c, r))
        points.extend(row)
        
    # Trim to exact length
    # The Boustrophedon path is contiguous. Trimming the end preserves self-avoidance.
    skeleton = points[:length]
    
    # Normalize coordinates to start at (0,0)
    min_x = min(p[0] for p in skeleton)
    min_y = min(p[1] for p in skeleton)
    skeleton = [(p[0] - min_x, p[1] - min_y) for p in skeleton]
    
    return skeleton

def _build_contact_graph(sequence, chain):
    """
    Build a contact graph where nodes are H residues and edges represent
    H-H contacts in the given conformation.
    """
    n = len(sequence)
    where = {p: i for i, p in enumerate(chain)}
    contacts = []
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts.append((i, j))
    return contacts

def _max_independent_set_greedy(conflict_graph):
    """
    Find a maximum independent set in a conflict graph using a greedy heuristic.
    conflict_graph is a dictionary mapping node -> set of conflicting nodes.
    """
    # Sort nodes by degree (ascending) for a better greedy heuristic
    nodes = sorted(conflict_graph.keys(), key=lambda x: len(conflict_graph[x]))
    independent_set = set()
    for node in nodes:
        # Check if node conflicts with any node already in the independent set
        if not any(conflict in independent_set for conflict in conflict_graph[node]):
            independent_set.add(node)
    return independent_set

def _pivot_search(sequence, chain, rng, max_iters=500, target_contacts=None):
    """
    Local search using pivot moves.
    A pivot move rotates a segment of the chain around a joint.
    If target_contacts is provided, use a potential function that encourages
    these contacts in addition to the energy function.
    """
    n = len(chain)
    if n < 5:
        return chain

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Precompute occupied set for fast lookup
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    # Define a potential function that encourages target contacts
    def potential(chain):
        if target_contacts is None:
            return 0.0
        where = {p: i for i, p in enumerate(chain)}
        pot = 0.0
        for i, j in target_contacts:
            p_i = chain[i]
            p_j = chain[j]
            dist = abs(p_i[0] - p_j[0]) + abs(p_i[1] - p_j[1])
            pot += dist
        return pot

    current_potential = potential(current_chain)
    best_potential = current_potential
    
    # We will perform random pivot moves
    # To be efficient, we only try a limited number of iterations
    # and accept moves that improve the combined objective (energy + potential)
    
    for _ in range(max_iters):
        # Choose a pivot point i (1 <= i < n-1)
        i = rng.randint(1, n - 2)
        # Choose a segment length L (1 <= L < n - i)
        max_L = n - i - 1
        L = rng.randint(1, max_L)
        
        # The segment to move is chain[i+1 : i+1+L]
        # Pivot at chain[i]
        # Original direction from i to i+1
        p_prev = current_chain[i]
        p_start = current_chain[i + 1]
        
        # Calculate vector
        dx = p_start[0] - p_prev[0]
        dy = p_start[1] - p_prev[1]
        
        # Rotate 90 degrees clockwise or counter-clockwise
        # Clockwise: (dx, dy) -> (dy, -dx)
        # Counter-clockwise: (dx, dy) -> (-dy, dx)
        
        # Try both rotations
        for rot_type in [0, 1]:
            if rot_type == 0:
                ndx, ndy = dy, -dx
            else:
                ndx, ndy = -dy, dx
                
            new_start = (p_prev[0] + ndx, p_prev[1] + ndy)
            
            # Check if new_start is valid (not occupied by non-adjacent parts)
            # The segment chain[i+1:L+i+1] is being moved.
            # We need to check if the new positions collide with the rest of the chain.
            
            # Build new segment
            # We need to rotate the whole segment around p_prev
            # This is complex to do in-place efficiently without a full simulation.
            # Simplified approach: Simulate the move.
            
            old_segment = current_chain[i+1 : i+1+L]
            
            # Check if new_start is in occupied and not part of the moving segment or immediate neighbors
            # The moving segment indices are [i+1, i+L]
            # The fixed parts are [0, i] and [i+L+1, n-1]
            
            # 1. Check new_start against fixed parts
            if new_start in occupied:
                idx = occupied[new_start]
                if not (i+1 <= idx <= i+L):
                    continue # Collision
            
            # 2. Simulate the rest of the segment
            # For a simple pivot, we just rotate the first step. 
            # Wait, a standard pivot move rotates the *entire* tail or a segment.
            # If we rotate the whole tail (L = n-i-1), it's easier.
            # Let's stick to rotating the tail for simplicity and effectiveness in small proteins.
            # If L < tail, we need to check intermediate collisions which is O(L).
            
            # Let's implement full tail rotation for i, and maybe short segments.
            # Actually, let's just do tail rotations (i, n-1-i) and random segment rotations if fast enough.
            # Given 120s budget, we can afford O(N^2) or O(N^3) per move if N is small (<=100).
            
            # Let's try to construct the new chain
            new_chain = list(current_chain[:i+1])
            
            # The rest of the chain from i+1 onwards needs to be rotated.
            # This is tricky because the rest of the chain has its own shape.
            # A true pivot move rotates the vector of the entire sub-chain.
            
            # Let's simplify: Only allow moves that rotate the *tail* (from i to end).
            # This is a standard and effective move.
            if L != n - i - 1:
                continue
                
            # Rotate tail from i to end
            # Get tail points
            tail = current_chain[i:]
            
            # Rotate all points in tail around p_prev by 90 deg
            # Note: p_prev is fixed.
            # p_k' = p_prev + R * (p_k - p_prev)
            
            new_tail = []
            valid = True
            for k in range(len(tail)):
                px, py = tail[k]
                vx, vy = px - p_prev[0], py - p_prev[1]
                if rot_type == 0:
                    nvx, nvy = vy, -vx
                else:
                    nvx, nvy = -vy, vx
                new_p = (p_prev[0] + nvx, p_prev[1] + nvy)
                new_tail.append(new_p)
                
                # Check collision
                # The new point new_p must not collide with fixed part [0, i-1]
                # It can be adjacent to p_prev (which is index i)
                # It must not collide with any other point in new_tail (handled by self-avoidance of rotation? No, rotation preserves distances, so if original was self-avoiding, new is self-avoiding internally)
                # So we only need to check against fixed part [0, i-1]
                
                # Wait, the fixed part is [0, i]. p_prev is at index i.
                # new_tail[0] is the new position of index i+1.
                # We must check new_tail[k] against occupied set, excluding indices i+1..n-1 (which are being moved).
                
                # Since we are moving the tail, the "occupied" set for collision checking
                # should only contain indices 0..i-1.
                
                # Let's precompute fixed_occupied
                # But doing this every move is slow.
                # Let's just check against the global occupied set, but ignore indices > i.
                
                if new_p in occupied:
                    idx = occupied[new_p]
                    if idx < i: # Collision with fixed head
                        valid = False
                        break
                    # If idx >= i, it's part of the tail being moved, so it's fine (internal collision is impossible due to rotation)
                    # Wait, is it possible that new_p collides with a *different* point in the new_tail?
                    # No, rotation is an isometry. If old tail was self-avoiding, new tail is self-avoiding.
                    # So we only care about head-tail collision.
                    
            if not valid:
                continue
            
            # Construct new chain
            candidate_chain = current_chain[:i] + new_tail
            
            # Calculate new energy
            # We can do a full energy calc for simplicity, N<=100 is fast
            candidate_energy = _energy(sequence, candidate_chain)
            candidate_potential = potential(candidate_chain)
            
            # Accept if energy improves or potential improves
            if candidate_energy < current_energy or candidate_potential < current_potential:
                current_chain = candidate_chain
                current_energy = candidate_energy
                current_potential = candidate_potential
                occupied = {p: idx for idx, p in enumerate(current_chain)}
                
                if current_energy < best_energy or (current_energy == best_energy and current_potential < best_potential):
                    best_energy = current_energy
                    best_potential = current_potential
                    best_chain = list(current_chain)
                    
                # Optional: Accept worse moves with probability exp(-dE/T) for SA
                # For now, greedy hill climbing.
                
    return best_chain

def fold(sequence, dimension):
    """
    Fold the sequence using a contact-graph MIS-based approach with local refinement.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Generate skeleton
    skeleton = _generate_skeleton(n, dimension)
    
    # Map sequence to skeleton
    # The skeleton is a list of points. The sequence is mapped 1-to-1.
    initial_chain = skeleton
    
    # Build contact graph and find a maximum independent set of contacts
    contacts = _build_contact_graph(sequence, initial_chain)
    
    # Build conflict graph: nodes are contacts, edges represent conflicts (shared residue)
    conflict_graph = {contact: set() for contact in contacts}
    for i, c1 in enumerate(contacts):
        for j, c2 in enumerate(contacts):
            if i < j:
                # Check if contacts share a residue
                if c1[0] == c2[0] or c1[0] == c2[1] or c1[1] == c2[0] or c1[1] == c2[1]:
                    conflict_graph[c1].add(c2)
                    conflict_graph[c2].add(c1)
    
    # Find a maximum independent set in the conflict graph
    target_contacts = _max_independent_set_greedy(conflict_graph)
    
    # Refine with pivot search
    # Use multiple restarts with different random seeds for the pivot search
    # to escape local minima.
    
    best_chain = initial_chain
    best_energy = _energy(sequence, initial_chain)
    
    # Time budget management
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin for 120s limit (or 210s wall clock if that's the hard kill, but task says 120s CPU for all sequences)
    # Task says: "120 s CPU budget for all sequences together".
    # "Each evaluation must finish within 180 s".
    # "evaluator KILLS the program after 210s wall-clock".
    # We should be safe with 110s per call if there are only a few calls, but there are 10 sequences.
    # 120s total for 10 sequences means ~12s per sequence on average.
    # Let's be conservative: 10s per sequence.
    
    time_per_seq = 10.0
    
    # Run several restarts of pivot search
    for restart in range(10):
        if time.time() - start_time > time_per_seq:
            break
            
        rng = random.Random(12345 + restart + n)
        # Perturb the initial chain? 
        # For restart 0, use skeleton directly.
        # For restart > 0, we can try to "shake" the skeleton or use a different skeleton orientation.
        # Simplest: Just run pivot search on the same skeleton but with different random choices for moves.
        # Since pivot search is greedy, it will find the same local minimum if it starts from the same state.
        # To get different results, we need to start from different states or use SA.
        
        # Let's implement a simple SA step in the pivot search loop if we have time.
        # Or, we can generate a slightly different skeleton (e.g. flipped, rotated) for each restart.
        
        # Let's try rotating/flipping the skeleton for diversity.
        if restart == 0:
            chain = skeleton
        else:
            # Apply a random symmetry operation to the skeleton
            chain = list(skeleton)
            if restart % 2 == 1:
                # Flip X
                chain = [(-p[0], p[1]) for p in chain]
                # Normalize
                min_x = min(p[0] for p in chain)
                chain = [(p[0] - min_x, p[1]) for p in chain]
            if restart % 3 == 1:
                # Flip Y
                chain = [(p[0], -p[1]) for p in chain]
                min_y = min(p[1] for p in chain)
                chain = [(p[0], p[1] - min_y) for p in chain]
            if restart % 5 == 1:
                # Swap X and Y
                chain = [(p[1], p[0]) for p in chain]
                
        refined_chain = _pivot_search(sequence, chain, rng, max_iters=200, target_contacts=target_contacts)
        refined_energy = _energy(sequence, refined_chain)
        
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain

    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
