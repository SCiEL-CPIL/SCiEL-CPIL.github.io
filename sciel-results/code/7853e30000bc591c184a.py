import time
import random
import math

def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D space."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _calculate_energy(sequence, chain):
    """Calculate the energy of the chain (number of H-H contacts)."""
    # Map position to index for O(1) lookup
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    # Only need to check H residues
    for i in range(n):
        if sequence[i] == 'H':
            p = chain[i]
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None:
                    # Check if they are not adjacent in the chain
                    if abs(i - j) > 1:
                        # Count each contact only once (i < j)
                        if i < j:
                            contacts += 1
    return -contacts

def _initial_chain(sequence, dimension):
    """Generate a random self-avoiding walk to start with."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    # Start at origin
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0)}
    
    # Directions
    dirs = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    
    # Simple random walk with backtracking is complex. 
    # A "snake" or "spiral" is better for initialization to ensure validity.
    # Let's try a simple biased random walk that prefers expanding outwards.
    
    # Actually, a straight line is always valid. Let's start with a straight line 
    # and then rely on local search to improve it. This guarantees a valid start.
    # To make it slightly better, we can start with a zig-zag.
    
    # Zig-zag initialization:
    # (0,0,0) -> (1,0,0) -> (1,1,0) -> (2,1,0) -> (2,2,0) ...
    # This is valid.
    
    x, y, z = 0, 0, 0
    for i in range(1, n):
        # Alternate directions to create a compact-ish start
        if i % 4 == 1:
            x += 1
        elif i % 4 == 2:
            y += 1
        elif i % 4 == 3:
            x += 1
        else:
            y += 1
            
        # This simple pattern might collide if we wrap around, but for linear growth it's fine.
        # Let's just use a straight line for safety, it's fast.
        pass
        
    # Re-do with straight line for absolute safety and speed
    chain = [(i, 0, 0) for i in range(n)]
    return chain

def _try_pivot_move(chain, occupied, i, rng):
    """
    Attempt a pivot move at index i.
    Returns (new_chain, new_occupied) or (None, None) if invalid.
    """
    n = len(chain)
    if i <= 0 or i >= n:
        return None, None
    
    # The pivot point is chain[i-1]. The segment to rotate is chain[i:]
    # We rotate the segment around chain[i-1].
    # In 3D, a pivot move involves rotating the vector from chain[i-1] to chain[i] 
    # and applying that rotation to the rest of the segment.
    
    # For simplicity and robustness, let's implement a simpler move:
    # "Head move" or "Tail move" is easy but limited.
    # Let's implement a simple "Crush" or "Slide" move?
    # Actually, a standard pivot in 3D:
    # 1. Pick a random axis and angle? No, lattice moves are discrete.
    # 2. A discrete pivot: Pick a random neighbor of the pivot bond vector?
    
    # Let's use a different move set that is easier to implement correctly:
    # 1. **Single Residue Move**: Move one residue (not the ends) to a random free neighbor.
    #    This is like a "kink" move.
    # 2. **Two Residue Move**: Move a segment of 2.
    
    # Let's stick to **Pivot** but implemented as:
    # Pick index i. The bond (i-1, i) is the pivot.
    # The vector v = chain[i] - chain[i-1].
    # We want to rotate the rest of the chain (i+1 to end) around the axis defined by v?
    # No, that's not a lattice move.
    
    # Correct Lattice Pivot:
    # Pick a random index i (1 to n-1).
    # The bond is chain[i-1] -> chain[i].
    # There are 5 other directions from chain[i-1] that are not chain[i].
    # Pick one randomly, say new_dir.
    # The new position for chain[i] is chain[i-1] + new_dir.
    # Then we need to move the rest of the chain (i+1...) by the same translation vector?
    # No, that's a "translation" move, which preserves the shape of the tail.
    # This is a valid move called a "Pivot" in some contexts, or "Rigid Body" move.
    
    # Let's implement **Rigid Body Move**:
    # 1. Pick a split point i.
    # 2. The head is chain[0:i], tail is chain[i:].
    # 3. The bond connecting them is chain[i-1] -> chain[i].
    # 4. We rotate the tail around the pivot chain[i-1].
    # 5. In lattice terms: The vector v = chain[i] - chain[i-1].
    # 6. We pick a new vector v' from the 5 neighbors of v (excluding v itself? No, excluding -v? 
    #    Actually, we pick any of the 6 neighbors of chain[i-1] that is not chain[i-2] (if i>1) 
    #    and not chain[i] (to avoid reversing? No, we just pick a new direction for the first bond of the tail).
    # 7. Then we apply the same "rotation" to the rest of the tail?
    #    In lattice, "rotation" is not straightforward.
    
    # Simpler and very effective: **Monte Carlo with "Move Head" and "Move Tail" and "Crush"**.
    # But "Pivot" is standard. Let's look up a simple lattice pivot implementation.
    # A common simple move is:
    # Pick i. Try to move chain[i] to a random empty neighbor.
    # If successful, check if the rest of the chain (i+1...) still fits? No, that breaks the chain.
    
    # Okay, let's use **Crush Move**:
    # Pick a segment of length L (e.g., 3 or 4).
    # Try to move the whole segment to a new location.
    # This is hard to get right.
    
    # Let's go with the simplest valid move: **Single Residue Move (Kink)**
    # Pick an index i (1 to n-2).
    # Try to move chain[i] to a random free neighbor.
    # Check if chain[i-1] and chain[i+1] are still connected?
    # No, if we move chain[i], the bonds (i-1, i) and (i, i+1) change.
    # We need to ensure the new position is adjacent to chain[i-1] AND chain[i+1]?
    # No, the chain must remain connected.
    # If we move chain[i], it must be adjacent to chain[i-1] and chain[i+1].
    # This is a "kink" removal or addition.
    # If chain[i-1] and chain[i+1] are adjacent, we can remove chain[i] (if it was a kink).
    # But we must maintain length.
    
    # Let's use **Pivot Move** properly.
    # Reference: "Protein Folding on Lattices"
    # A pivot move consists of:
    # 1. Select a random pivot bond (i-1, i).
    # 2. Rotate the tail (i, i+1, ..., n-1) around the pivot bond axis? No.
    # 3. In 3D lattice, a pivot move is:
    #    - Pick a random index i.
    #    - The vector v = chain[i] - chain[i-1].
    #    - There are 6 possible directions for the next bond from chain[i-1].
    #    - One is v (current). One is -v (backwards, usually invalid or trivial).
    #    - The other 4 are orthogonal to v.
    #    - Pick one of the 4 orthogonal directions, call it v_new.
    #    - The new position for chain[i] is chain[i-1] + v_new.
    #    - Now, the rest of the tail (chain[i+1]...) must be transformed.
    #    - The transformation is a rotation that maps v to v_new.
    #    - There are multiple such rotations. We pick one randomly.
    #    - Apply this rotation to all subsequent vectors in the tail.
    #    - Check for collisions.
    
    # This is complex to implement in Python quickly without bugs.
    
    # Alternative: **Simulated Annealing with "Head" and "Tail" moves**.
    # Head move: Move chain[0] to a random free neighbor.
    # Tail move: Move chain[n-1] to a random free neighbor.
    # These are very simple and valid. They allow the chain to explore space.
    # Combined with occasional "Crush" or "Pivot" if I can implement them.
    # Given time constraints, Head/Tail moves are safer.
    # However, Head/Tail moves are slow to mix.
    
    # Let's try a **Hybrid**:
    # 1. Head Move
    # 2. Tail Move
    # 3. "Slide" Move: Pick i, move chain[i] to a neighbor, and shift the rest of the chain? No.
    
    # Let's implement **Pivot** using the rotation logic.
    # We need to define the rotation.
    # If v = (1,0,0) and v_new = (0,1,0), the rotation is around Z-axis by 90 deg.
    # If v = (1,0,0) and v_new = (0,0,1), the rotation is around Y-axis by -90 deg.
    # We can pre-compute or generate the rotation matrix.
    
    # Simpler approach for "Pivot":
    # Just try to move the *entire tail* by a translation vector?
    # No, that's not a pivot.
    
    # Let's stick to **Head and Tail moves** and **Single Residue Kink moves** if possible.
    # Kink move:
    # If chain[i-1] and chain[i+1] are neighbors, we can try to "remove" the kink at i?
    # No, we must keep N residues.
    # We can try to move chain[i] to a different neighbor of chain[i-1] that is also a neighbor of chain[i+1]?
    # This is rare.
    
    # Okay, I will implement **Head** and **Tail** moves. They are robust.
    # To improve mixing, I will also implement a **Segment Move** of length 2.
    # Move chain[i] and chain[i+1] together.
    # This is a "crankshaft" of length 2.
    
    def _try_head_move(chain, occupied, rng):
        n = len(chain)
        if n <= 1: return None
        current_head = chain[0]
        neighbors = list(_neighbours(current_head))
        # Filter out occupied and the second residue
        valid = []
        for nb in neighbors:
            if nb in occupied: continue
            if nb == chain[1]: continue
            valid.append(nb)
        if not valid: return None
        new_head = rng.choice(valid)
        # Update chain
        new_chain = list(chain)
        new_chain[0] = new_head
        # Update occupied
        new_occupied = set(occupied)
        new_occupied.remove(current_head)
        new_occupied.add(new_head)
        return new_chain, new_occupied

    def _try_tail_move(chain, occupied, rng):
        n = len(chain)
        if n <= 1: return None
        current_tail = chain[-1]
        neighbors = list(_neighbours(current_tail))
        valid = []
        for nb in neighbors:
            if nb in occupied: continue
            if nb == chain[-2]: continue
            valid.append(nb)
        if not valid: return None
        new_tail = rng.choice(valid)
        new_chain = list(chain)
        new_chain[-1] = new_tail
        new_occupied = set(occupied)
        new_occupied.remove(current_tail)
        new_occupied.add(new_tail)
        return new_chain, new_occupied

    def _try_crankshaft_move(chain, occupied, rng):
        n = len(chain)
        if n < 4: return None
        # Pick a random start index i for a segment of length 3 or 4
        # Let's pick length 3: i, i+1, i+2
        max_i = n - 3
        if max_i < 1: return None
        i = rng.randint(1, max_i)
        # Segment is chain[i], chain[i+1], chain[i+2]
        # Pivot is chain[i-1]
        # We want to move the segment.
        # This is complex. Let's skip crankshaft for now.
        return None

    return None, None # Placeholder, will define outside

def _local_search(sequence, chain, dimension, time_limit):
    """
    Run Simulated Annealing local search.
    """
    n = len(chain)
    if n <= 1:
        return chain
    
    rng = random.Random(42)
    
    # Initial state
    current_chain = list(chain)
    current_occupied = set(current_chain)
    current_energy = _calculate_energy(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    t = 1.0
    cooling_rate = 0.999 # Adjust based on steps
    
    # Estimate steps: if we have 8 seconds, and each step takes ~1ms (optimistic), we can do 8000 steps.
    # Python is slower. Let's assume 100-500 steps/sec for complex moves.
    # Let's just run until time_limit.
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
            
        # Choose a move
        move_type = rng.random()
        new_chain = None
        new_occupied = None
        
        if move_type < 0.4:
            # Head move
            res = _try_head_move(current_chain, current_occupied, rng)
            if res[0]:
                new_chain, new_occupied = res
        elif move_type < 0.8:
            # Tail move
            res = _try_tail_move(current_chain, current_occupied, rng)
            if res[0]:
                new_chain, new_occupied = res
        else:
            # Try a simple "Kink" move or just retry head/tail
            res = _try_head_move(current_chain, current_occupied, rng)
            if res[0]:
                new_chain, new_occupied = res
            else:
                res = _try_tail_move(current_chain, current_occupied, rng)
                if res[0]:
                    new_chain, new_occupied = res

        if new_chain is None:
            continue

        new_energy = _calculate_energy(sequence, new_chain)
        
        # Simulated Annealing acceptance
        delta_e = new_energy - current_energy
        
        if delta_e < 0:
            # Better
            current_chain = new_chain
            current_occupied = new_occupied
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(new_chain)
        else:
            # Worse
            # Accept with probability exp(-delta_e / t)
            # Energy is negative, so delta_e is positive for worse (less negative).
            # Wait, Energy = -contacts.
            # Better energy is more negative.
            # If new_energy < current_energy, delta_e is negative. Correct.
            # If new_energy > current_energy, delta_e is positive.
            # Probability = exp(-delta_e / t).
            # Since delta_e > 0, exp is < 1.
            prob = math.exp(-delta_e / t)
            if rng.random() < prob:
                current_chain = new_chain
                current_occupied = new_occupied
                current_energy = new_energy
        
        # Cool down
        t *= cooling_rate
        if t < 0.01:
            t = 0.01

    return best_chain

def _try_head_move(chain, occupied, rng):
    n = len(chain)
    if n <= 1:
        return None, None
    current_head = chain[0]
    neighbors = list(_neighbours(current_head))
    valid = []
    for nb in neighbors:
        if nb in occupied:
            continue
        if nb == chain[1]:
            continue
        valid.append(nb)
    if not valid:
        return None, None
    new_head = rng.choice(valid)
    new_chain = list(chain)
    new_chain[0] = new_head
    new_occupied = set(occupied)
    new_occupied.remove(current_head)
    new_occupied.add(new_head)
    return new_chain, new_occupied

def _try_tail_move(chain, occupied, rng):
    n = len(chain)
    if n <= 1:
        return None, None
    current_tail = chain[-1]
    neighbors = list(_neighbours(current_tail))
    valid = []
    for nb in neighbors:
        if nb in occupied:
            continue
        if nb == chain[-2]:
            continue
        valid.append(nb)
    if not valid:
        return None, None
    new_tail = rng.choice(valid)
    new_chain = list(chain)
    new_chain[-1] = new_tail
    new_occupied = set(occupied)
    new_occupied.remove(current_tail)
    new_occupied.add(new_tail)
    return new_chain, new_occupied

def fold(sequence, dimension):
    """
    Fold the sequence using local search.
    """
    if len(sequence) <= 1:
        return [[0, 0, 0]] if len(sequence) == 1 else []
        
    # Determine time budget
    # Total budget is ~240s for 26 sequences.
    # Assume this function is called for one sequence.
    # We don't know how many calls are left, but we can assume a fair share.
    # If we assume 26 calls, we have ~9s each.
    # Let's use 8s to be safe.
    time_limit = 8.0
    
    # Initialize
    initial_chain = _initial_chain(sequence, dimension)
    
    # Run local search
    best_chain = _local_search(sequence, initial_chain, dimension, time_limit)
    
    # Convert tuples to lists
    return [list(p) for p in best_chain]

