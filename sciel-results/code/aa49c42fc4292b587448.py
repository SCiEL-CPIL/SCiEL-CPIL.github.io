import random
import time

# EVOLVE-BLOCK-START

_T0 = time.time()
_DEADLINE = 200.0  # Safety margin before 210s kill

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _energy(seq, chain):
    """
    Compute energy: -number of H-H contacts.
    A contact is between two H residues that are on neighboring lattice sites
    but NOT adjacent in the sequence (|i - j| > 1).
    """
    n = len(chain)
    if n == 0:
        return 0
    
    # Map position to index
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
        
    contacts = 0
    for i, p in enumerate(chain):
        if seq[i] == 'H':
            x, y = p
            # Check 4 neighbors
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                q = (x + dx, y + dy)
                j = pos_to_idx.get(q)
                if j is not None:
                    if abs(i - j) > 1 and seq[j] == 'H':
                        contacts += 1
    return -contacts

def _grow(seq):
    """
    Generate a random self-avoiding walk (SAW) using greedy growth.
    Returns None if stuck (should be rare for N=100 with backtracking or careful choice, 
    but simple greedy can get stuck. We'll use a simple random walk with immediate backtracking 
    avoidance. If stuck, return None to trigger a restart.
    """
    n = len(seq)
    if n == 0:
        return []
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    # Directions: East, West, North, South
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for i in range(1, n):
        last_x, last_y = chain[-1]
        valid_moves = []
        for dx, dy in directions:
            nx, ny = last_x + dx, last_y + dy
            if (nx, ny) not in occupied:
                valid_moves.append((nx, ny))
        
        if not valid_moves:
            return None  # Stuck
        
        # Pick a random valid move
        nx, ny = random.choice(valid_moves)
        chain.append((nx, ny))
        occupied[(nx, ny)] = i
        
    return chain

def _local_search(seq, chain, max_steps):
    """
    Local search using pivot moves.
    A pivot move rotates the tail of the chain (from index i to end) around the pivot point (index i-1)
    by 90, 180, or 270 degrees.
    We check for collisions with the head (indices 0 to i-1).
    """
    n = len(chain)
    if n < 3:
        return chain
    
    current_chain = list(chain)
    current_energy = _energy(seq, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    rng = random.Random()
    steps = 0
    
    while steps < max_steps and _remaining() > 5:
        steps += 1
        # Choose a pivot index. Must be at least 2 to have a tail of length >= 1 to rotate?
        # If i=1, tail is [1..n-1]. Pivot is 0.
        # If i=n-2, tail is [n-2, n-1].
        i = rng.randint(1, n - 2)
        
        pivot_x, pivot_y = current_chain[i-1]
        tail = current_chain[i:]
        
        # Try 3 rotations
        # We need to rotate the vector from pivot to tail[0], and propagate the rest of the tail's shape.
        # Since the tail is a rigid body in terms of relative coordinates, we can just rotate the first step 
        # and then add the original deltas.
        
        dx0 = tail[0][0] - pivot_x
        dy0 = tail[0][1] - pivot_y
        
        rotations = [
            (dy0, -dx0),    # 90 deg CCW
            (-dx0, -dy0),   # 180 deg
            (-dy0, dx0)     # 90 deg CW
        ]
        
        rng.shuffle(rotations)
        
        moved = False
        for ndx, ndy in rotations:
            # Construct new tail
            new_tail = []
            cur_x = pivot_x + ndx
            cur_y = pivot_y + ndy
            new_tail.append((cur_x, cur_y))
            
            valid = True
            # Check collision of first point with head?
            # Head is current_chain[0:i]
            # But we also need to check against the rest of the head and the pivot?
            # The pivot is part of the head in the context of collision check for the new tail.
            # Actually, the new tail starts at i. The head is 0..i-1.
            # The new tail points must not collide with 0..i-1.
            
            # Optimization: Use a set for head positions
            # For N=100, creating a set every time is O(N). 100 steps * 100 = 10k ops. Fine.
            head_set = set(current_chain[:i])
            
            if (cur_x, cur_y) in head_set:
                valid = False
            
            if valid:
                # Propagate rest of tail
                for k in range(1, len(tail)):
                    # Delta from tail[k-1] to tail[k]
                    ddx = tail[k][0] - tail[k-1][0]
                    ddy = tail[k][1] - tail[k-1][1]
                    cur_x += ddx
                    cur_y += ddy
                    new_tail.append((cur_x, cur_y))
                    
                    if (cur_x, cur_y) in head_set:
                        valid = False
                        break
            
            if valid:
                # Construct new chain
                new_chain = current_chain[:i] + new_tail
                new_energy = _energy(seq, new_chain)
                
                if new_energy <= current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    moved = True
                    break
        
        if not moved:
            # Optional: Small probability to accept worse move to escape local minima?
            # For HP, hill climbing is often good. Let's stick to hill climbing for speed.
            pass
            
    return best_chain

def fold(sequence, dimension):
    """
    Best conformation using randomized growth and local search with time guard.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    best_chain = None
    best_energy = 0 # Energy is negative
    
    # If sequence is very short, return a valid straight line
    if n <= 2:
        return [[i, 0] for i in range(n)]
        
    # Main search loop
    while _remaining() > 10:
        # 1. Generate initial chain
        chain = _grow(sequence)
        if chain is None:
            # Rare, but if we get stuck growing, just try again
            # To avoid infinite loop if _grow always fails (unlikely), we could add a fallback
            continue
            
        # 2. Local search
        # Allocate a fixed number of steps for local search.
        # 100 steps is fast. 500 steps might be better.
        # Let's use 200 steps.
        improved_chain = _local_search(sequence, chain, 200)
        
        e = _energy(sequence, improved_chain)
        
        if best_chain is None or e < best_energy:
            best_chain = improved_chain
            best_energy = e
            
        # Optional: Early exit if we found a very good solution?
        # Max possible contacts for N residues is roughly N-2 (if fully packed).
        # If best_energy <= -(n-2), we can't do better (actually we can, but it's rare to hit the absolute min).
        # We'll rely on the time limit.
        
    if best_chain is None:
        # Fallback: straight line
        best_chain = [[i, 0] for i in range(n)]
        
    return best_chain

# EVOLVE-BLOCK-END
