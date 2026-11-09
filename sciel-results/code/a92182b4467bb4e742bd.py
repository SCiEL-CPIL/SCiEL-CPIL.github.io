"""
HP lattice protein folding on the cubic lattice.
Strategy: Hybrid of fast greedy initialization and local search (Monte Carlo moves).
We initialize with a quick greedy grow, then refine using 2-point and 3-point 
local moves that respect self-avoidance and maximize H-H contacts.
A global deadline guard ensures we stay within the 330s wall-clock limit.
"""
import random
import time

# Global deadline management
_START_TIME = time.time()
_DEADLINE = _START_TIME + 315.0  # 15s safety margin before 330s kill

def _neighbours(point):
    """Yield the 6 face-neighbours of point in 3D."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)

def _energy(sequence, chain):
    """Compute total energy: negative count of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _grow(sequence, dimension, rng):
    """Fast randomized greedy chain growth."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                # Count potential H-H contacts with existing H's
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H')
            # Prefer free space to avoid trapping
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _local_search(sequence, chain, rng, max_steps, deadline):
    """
    Refine a chain using local moves.
    Moves:
    1. 2-point flip: Reverse a segment of length 2 (swap i, i+1) - rare but possible.
    2. 3-point flip: Reverse a segment of length 3 (i, i+1, i+2 -> i+2, i+1, i) 
       or rotate a kink.
    We will primarily use a "pivot" move: pick a segment [i..j] and try to re-attach it 
    in a different orientation or location. 
    For simplicity and speed, we will use a "kink rotation" and "segment reversal" heuristic.
    Actually, a robust and simple local move is: 
    - Pick two indices i < j.
    - Try to reverse the segment chain[i:j+1].
    - Check if the new segment is self-avoiding and connects properly.
    """
    current_energy = _energy(sequence, chain)
    best_energy = current_energy
    best_chain = chain[:]
    
    n = len(chain)
    # Precompute occupied set for fast lookup
    occupied = set(chain)
    
    # We will perform a number of random local moves
    # To be efficient, we limit the length of segments we try to move/reverse.
    # Long segments are hard to re-insert.
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        # Choose a move type
        # 1. Single step rotation (kink)
        # 2. Short segment reversal (length 2 to 5)
        
        move_type = rng.random()
        
        if move_type < 0.5:
            # Try to rotate a kink: pick i, look at chain[i-1], chain[i], chain[i+1]
            # Try to move chain[i] to a different neighbor of chain[i-1] (keeping chain[i+1] attached to new chain[i])
            # This is complex to implement correctly with self-avoidance checks for the whole chain.
            # Simpler: Try to reverse a short segment.
            
            # Pick a random start index for a short segment
            if n < 3:
                continue
            length = rng.randint(2, min(5, n - 1))
            start_idx = rng.randint(0, n - length - 1)
            
            # Segment is chain[start_idx : start_idx + length + 1]
            # Reversing it means chain[start_idx] connects to old chain[start_idx+length+1] (which is fixed)
            # and old chain[start_idx+length] connects to old chain[start_idx-1] (which is fixed)
            # Wait, reversing a segment in a linear chain:
            # Original: ... A - S1 - S2 - ... - Sk - B ...
            # Reversed: ... A - Sk - ... - S2 - S1 - B ...
            # This is valid if the reversed segment doesn't collide with the rest of the chain.
            
            seg = chain[start_idx : start_idx + length + 1]
            rev_seg = seg[::-1]
            
            # Check self-avoidance for the reversed segment against the rest of the chain
            # The rest of the chain is chain[:start_idx] and chain[start_idx+length+1:]
            # We need to ensure rev_seg elements are not in the rest, 
            # AND that the connection points are valid.
            
            # The "rest" set excludes the segment itself.
            # However, we must be careful: the endpoints of the segment connect to the rest.
            # In the reversed version:
            # - rev_seg[0] (old end) connects to chain[start_idx-1] (if exists)
            # - rev_seg[-1] (old start) connects to chain[start_idx+length+1] (if exists)
            
            # Let's build the candidate chain
            # To avoid O(N) copying every time, we can check validity first.
            
            # Validity Check:
            # 1. No internal collisions in rev_seg (already self-avoiding, so rev is too)
            # 2. No collision with rest of chain.
            
            # Create a set of the rest of the chain for O(1) lookup
            # Doing this every step is O(N). 
            # Optimization: Maintain a global occupied set? 
            # Since we are modifying the chain, maintaining it is tricky.
            # For N <= 136, O(N) check is fast enough if we limit steps.
            
            rest_set = set(chain[:start_idx]) | set(chain[start_idx + length + 1:])
            
            # Check if any node in rev_seg is in rest_set
            # Note: rev_seg[0] connects to chain[start_idx-1]. 
            # If start_idx > 0, rev_seg[0] must not be chain[start_idx-1] (which is in rest_set).
            # Actually, if rev_seg[0] == chain[start_idx-1], it's a collision.
            # Similarly for the other end.
            
            valid = True
            for p in rev_seg:
                if p in rest_set:
                    valid = False
                    break
            
            if valid:
                # Construct new chain
                new_chain = chain[:start_idx] + rev_seg + chain[start_idx + length + 1:]
                
                # Calculate energy
                # We can just call _energy, it's O(N).
                new_energy = _energy(sequence, new_chain)
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = new_chain
                    chain = new_chain # Update current state for next move
                    # Update occupied set if we were maintaining it, but we aren't.
                    # We recompute rest_set in next iteration.

        else:
            # Try a "slide" move? 
            # Pick i, move chain[i] to a different neighbor of chain[i-1]?
            # This requires checking if chain[i+1] can still attach to new chain[i].
            # This is effectively a 2-residue move.
            if n < 2:
                continue
            i = rng.randint(1, n - 2) # 1 to n-2 to have both neighbors
            
            prev_p = chain[i-1]
            curr_p = chain[i]
            next_p = chain[i+1]
            
            # Find a new position for curr_p that is a neighbor of prev_p
            # and allows next_p to be a neighbor of new_curr_p
            # AND is not occupied by any other part of the chain.
            
            candidates = []
            for np in _neighbours(prev_p):
                if np == curr_p:
                    continue
                # Check if next_p is a neighbor of np
                dx = next_p[0] - np[0]
                dy = next_p[1] - np[1]
                dz = next_p[2] - np[2]
                if abs(dx) + abs(dy) + abs(dz) == 1:
                    # Check if np is occupied by anything else
                    # We need to know if np is in chain, excluding i
                    # O(N) check
                    occupied_elsewhere = False
                    for k, p in enumerate(chain):
                        if k == i:
                            continue
                        if p == np:
                            occupied_elsewhere = True
                            break
                    if not occupied_elsewhere:
                        candidates.append(np)
            
            if candidates:
                new_p = rng.choice(candidates)
                # Construct new chain
                new_chain = chain[:]
                new_chain[i] = new_p
                
                new_energy = _energy(sequence, new_chain)
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = new_chain
                    chain = new_chain

    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Best conformation using greedy init + local search refinement.
    """
    if dimension != 3:
        raise ValueError("Only dimension 3 is supported")
    
    # Time allocation strategy:
    # We have ~315 seconds.
    # We will run multiple independent trials.
    # Each trial:
    # 1. Greedy grow (fast)
    # 2. Local search (slower, adaptive time)
    
    # Estimate time per trial based on sequence length.
    # Local search steps: proportional to N^2?
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0 # Energy is negative, so 0 is worst (unfolded)
    
    # If we can't even make a straight line (shouldn't happen for N < 1000 in 3D), return straight line
    # But greedy usually works.
    
    # We will run trials until deadline.
    # To ensure we make progress, we limit the local search steps per trial.
    # Let's say 5000 steps per trial.
    
    steps_per_trial = 5000
    
    while time.time() < _DEADLINE:
        # 1. Grow
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            # Retry with different rng seed? Or break?
            # Greedy rarely fails for these lengths.
            continue
            
        # 2. Refine
        # Calculate remaining time for this trial to be fair
        # If we have very little time, skip refinement and just take the greedy result
        time_left = _DEADLINE - time.time()
        
        # Rough heuristic: if time_left is small, do fewer steps
        # Let's just do a fixed number of steps, but check deadline inside _local_search
        refined_chain, refined_energy = _local_search(sequence, chain, rng, steps_per_trial, _DEADLINE)
        
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain
            
        # If we found a very good solution (e.g. energy close to theoretical min), we could stop early
        # But we don't know the theoretical min. 
        # We can stop if we've been running for a long time and improvements are rare.
        # For now, just keep trying.
        
        # Optional: If best_energy is very low (e.g. -N/2), maybe break?
        # No, keep going to beat others.

    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        best_energy = _energy(sequence, best_chain)

    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
