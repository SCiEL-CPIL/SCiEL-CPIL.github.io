"""
HP Lattice Protein Folding
Uses Simulated Annealing with Pivot Moves and Local Search.
"""
# EVOLVE-BLOCK-START
import time
import random
import math

# Global start time for deadline management
_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    # Leave 15s safety margin before 210s wall-clock limit
    return 195.0 - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

def _energy(sequence, chain):
    """Calculate the energy (negative number of H-H contacts)."""
    # Map position to index
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _make_initial_chain(sequence):
    """
    Generate a compact initial chain using a spiral-like greedy approach.
    """
    n = len(sequence)
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0)}
    
    # Directions: Right, Down, Left, Up (Spiral)
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dir_idx = 0
    step_count = 0
    steps_in_dir = 1
    
    for i in range(1, n):
        moved = False
        # Try to continue in current direction or turn
        # Simple spiral logic:
        # Step 1: Right
        # Step 1: Down
        # Step 2: Left
        # Step 2: Up
        # Step 3: Right
        # ...
        
        # A simpler robust initializer: Random Greedy with preference for H-H
        # Let's stick to a deterministic spiral first, fallback to random if stuck.
        
        # Actually, let's use a "Greedy Random" initializer that prefers compactness.
        break # Skip spiral logic for now, use the one below for simplicity and robustness.
        
    # Fallback: Random Greedy Initializer
    chain = [(0, 0)]
    occupied = {(0, 0)}
    rng = random.Random(42)
    
    for i in range(1, n):
        candidates = []
        last = chain[-1]
        for nb in _neighbours(last):
            if nb not in occupied:
                # Score: prefer H-H contact if current is H
                score = rng.random()
                if sequence[i] == 'H':
                    for prev_nb in _neighbours(nb):
                        if prev_nb in occupied:
                            idx = chain.index(prev_nb) # O(N) but N is small in init
                            if idx < i - 1 and sequence[idx] == 'H':
                                score -= 1.0 # Prefer contact
                candidates.append((score, nb))
        
        if not candidates:
            # Stuck, try to backtrack or just fail (should not happen often for small N)
            # For robustness, if stuck, return what we have? No, must be full length.
            # Let's try a simple straight line if greedy fails?
            # But greedy rarely fails for N<100 on 2D lattice if we pick random.
            # If it fails, we just extend in any free direction, even if it breaks "greedy".
            pass
            
        # Pick best candidate
        if candidates:
            # Sort by score
            candidates.sort()
            best_site = candidates[0][1]
            chain.append(best_site)
            occupied.add(best_site)
        else:
            # Emergency: try all neighbors again without scoring?
            # If truly stuck, this initializer fails.
            # Let's try a different seed or strategy.
            # For now, break and let SA fix it? No, chain must be length N.
            # Let's just pick a random valid neighbor if any.
            valids = [nb for nb in _neighbours(last) if nb not in occupied]
            if valids:
                best_site = random.choice(valids)
                chain.append(best_site)
                occupied.add(best_site)
            else:
                # Completely stuck. Return a straight line as last resort.
                # This will have high energy (low contacts) but is valid.
                return [(i, 0) for i in range(n)]
                
    return chain

def _pivot_move(chain, occupied, k):
    """
    Apply a pivot move at index k.
    Returns (new_chain, new_occupied, success)
    """
    n = len(chain)
    if k <= 0 or k >= n - 1:
        return None, None, False
        
    pivot = chain[k]
    head = chain[:k+1]
    tail = chain[k+1:]
    
    # Rotation matrices for 2D
    # 90 deg: (x, y) -> (-y, x)
    # 180 deg: (x, y) -> (-x, -y)
    # 270 deg: (x, y) -> (y, -x)
    
    rotations = [
        lambda p: (-p[1], p[0]),
        lambda p: (-p[0], -p[1]),
        lambda p: (p[1], -p[0])
    ]
    
    rng = random.Random()
    rot_func = rng.choice(rotations)
    
    # Generate new tail
    new_tail = []
    last_point = pivot
    for p in tail:
        # Translate relative to pivot, rotate, translate back
        dx = p[0] - pivot[0]
        dy = p[1] - pivot[1]
        rdx, rdy = rot_func((dx, dy))
        new_point = (pivot[0] + rdx, pivot[1] + rdy)
        new_tail.append(new_point)
        last_point = new_point
        
    new_chain = head + new_tail
    new_occupied = set(new_chain)
    
    # Check for self-intersection
    # We know head is valid. We know tail relative distances are preserved.
    # We only need to check if new_tail intersects with head[:-1] (excluding pivot which is shared)
    # and if new_tail has internal collisions (it shouldn't, but let's be safe? No, rotation is isometry).
    # So just check intersection with head[0:-1]
    
    head_set = set(head[:-1])
    for p in new_tail:
        if p in head_set:
            return None, None, False
            
    return new_chain, new_occupied, True

def _local_search(sequence, chain, occupied, max_steps=1000):
    """
    Simple local search: try all single-step moves and small pivots to improve energy.
    """
    current_energy = _energy(sequence, chain)
    improved = True
    steps = 0
    
    while improved and steps < max_steps:
        improved = False
        steps += 1
        n = len(chain)
        
        # Try pivot moves at every position
        for k in range(1, n - 1):
            # Try all 3 rotations
            for rot_idx in range(3):
                # We need a helper to apply specific rotation
                # Re-using _pivot_move logic but forcing rotation
                pivot = chain[k]
                head = chain[:k+1]
                tail = chain[k+1:]
                
                # Apply rotation
                dxs = [p[0] - pivot[0] for p in tail]
                dys = [p[1] - pivot[1] for p in tail]
                
                if rot_idx == 0: # 90
                    new_dx = [-d for d in dys]
                    new_dy = [d for d in dxs]
                elif rot_idx == 1: # 180
                    new_dx = [-d for d in dxs]
                    new_dy = [-d for d in dys]
                else: # 270
                    new_dx = [d for d in dys]
                    new_dy = [-d for d in dxs]
                    
                new_tail = [(pivot[0] + nx, pivot[1] + ny) for nx, ny in zip(new_dx, new_dy)]
                new_chain = head + new_tail
                
                # Check validity
                valid = True
                head_set = set(head[:-1])
                for p in new_tail:
                    if p in head_set:
                        valid = False
                        break
                        
                if valid:
                    new_energy = _energy(sequence, new_chain)
                    if new_energy < current_energy:
                        chain = new_chain
                        occupied = set(chain)
                        current_energy = new_energy
                        improved = True
                        break
            if improved:
                break
                
    return chain, current_energy

def fold(sequence, dimension):
    """
    Best conformation using Simulated Annealing.
    """
    _init_time()
    n = len(sequence)
    
    # Allocate time based on length
    # Longer sequences are harder.
    # Total budget ~ 100s for 10 sequences.
    # Let's give 5s + 0.05 * N seconds?
    # N=20 -> 6s. N=100 -> 10s.
    # Sum approx 10 * 8 = 80s. Safe.
    time_limit = 5.0 + 0.05 * n
    if time_limit > 15.0:
        time_limit = 15.0
        
    # Check global deadline
    if _time_remaining() < time_limit + 5.0:
        # If we are running out of time, just do a quick greedy
        chain = _make_initial_chain(sequence)
        return [list(p) for p in chain]
        
    # 1. Initialize
    chain = _make_initial_chain(sequence)
    if not chain or len(chain) != n:
        # Fallback straight line
        chain = [(i, 0) for i in range(n)]
        
    occupied = set(chain)
    best_chain = chain
    best_energy = _energy(sequence, chain)
    
    # 2. Simulated Annealing
    start_temp = 1.0
    end_temp = 0.01
    # Number of steps
    # We have time_limit. Each step is ~ O(N) for energy calc + O(N) for collision.
    # N=100, ~200 ops. Python ~ 100k-1M ops/sec?
    # Let's estimate 10,000 steps per second is safe.
    target_steps = int(time_limit * 10000)
    
    current_temp = start_temp
    cooling_rate = (end_temp / start_temp) ** (1.0 / target_steps) if target_steps > 0 else 1.0
    
    rng = random.Random()
    
    steps_done = 0
    start_time = time.time()
    
    while steps_done < target_steps:
        # Check time
        if time.time() - start_time > time_limit:
            break
            
        # Pick pivot
        k = rng.randint(1, n - 2)
        
        new_chain, new_occupied, success = _pivot_move(chain, occupied, k)
        
        if not success:
            # Move rejected due to collision
            # Still cool? Yes, usually.
            current_temp *= cooling_rate
            steps_done += 1
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - best_energy # Wait, we should compare to current energy, not best
        # We need to track current_energy separately
        
        # Let's restructure to track current_energy
        # Actually, let's just compare to current_energy
        # I missed current_energy variable in the loop above.
        
        # Let's rewrite the SA loop properly.
        pass
        
    # The previous loop was pseudocode. Let's implement the real one.
    
    current_chain = chain
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain
    best_energy = current_energy
    
    for _ in range(target_steps):
        if time.time() - start_time > time_limit:
            break
            
        k = rng.randint(1, n - 2)
        new_chain, new_occupied, success = _pivot_move(current_chain, None, k)
        
        if not success:
            current_temp *= cooling_rate
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / current_temp):
            current_chain = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = new_chain
                
        current_temp *= cooling_rate
        
    # 3. Local Search on best found
    best_chain, best_energy = _local_search(sequence, best_chain, set(best_chain))
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
