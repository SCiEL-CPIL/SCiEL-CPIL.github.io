"""
HP lattice protein folding using a Hamiltonian path skeleton strategy.

Strategy:
1. Generate a deterministic space-filling grid (Hamiltonian path) that visits
   every cell in a bounding box exactly once. This path has a high density of
   "neighbor" pairs (non-consecutive indices that are adjacent on the lattice).
2. Map the input sequence onto this path.
3. Use a fast pivot-move local search to refine the conformation, allowing the
   chain to "breathe" and optimize H-H contacts while maintaining self-avoidance.
"""

import random
import time

# EVOLVE-BLOCK-START

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

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _pivot_search(sequence, chain, rng, max_iters=500):
    """
    Local search using pivot moves.
    A pivot move rotates a segment of the chain around a joint.
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
    
    # We will perform random pivot moves
    # To be efficient, we only try a limited number of iterations
    # and accept moves that improve energy (greedy hill climbing with random restarts)
    
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
            
            if candidate_energy < current_energy:
                current_chain = candidate_chain
                current_energy = candidate_energy
                occupied = {p: idx for idx, p in enumerate(current_chain)}
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                    
                # Optional: Accept worse moves with probability exp(-dE/T) for SA
                # For now, greedy hill climbing.
                
    return best_chain

def fold(sequence, dimension):
    """
    Fold the sequence using a skeleton-based approach with local refinement.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Generate skeleton
    skeleton = _generate_skeleton(n, dimension)
    
    # Map sequence to skeleton
    # The skeleton is a list of points. The sequence is mapped 1-to-1.
    initial_chain = skeleton
    
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
                
        refined_chain = _pivot_search(sequence, chain, rng, max_iters=200)
        refined_energy = _energy(sequence, refined_chain)
        
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain

    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
