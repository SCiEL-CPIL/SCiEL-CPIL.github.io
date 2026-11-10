"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
_DEADLINE = time.time() + 175.0

def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2

def _greedy_grow(sequence, dimension, rng, temperature=0.0):
    """
    Randomized greedy chain growth.
    Prefer sites that create new H-H contacts, then sites with more free space.
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        res = sequence[i]
        last = chain[-1]
        options = []
        
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            # Gain: number of new H-H contacts
            gain = 0
            if res == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            
            # Free space heuristic
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # Score: prioritize gain, then free space, then randomness
            # If temperature > 0, add noise
            score = gain * 10.0 + free * 0.1
            if temperature > 0:
                score += rng.random() * temperature
            
            options.append((score, site))
        
        if not options:
            # Stuck. This shouldn't happen often for small n if we use randomization.
            # Return None to signal failure.
            return None
        
        # Pick best option
        # Sort by score descending
        options.sort(key=lambda x: x[0], reverse=True)
        best_score, best_site = options[0]
        
        # With small probability, pick a non-best option to escape local minima
        if temperature > 0 and rng.random() < 0.1:
            # Pick from top 3 if available
            k = min(3, len(options))
            _, best_site = options[rng.randint(0, k - 1)]
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain

def _greedy_grow_biased(sequence, dimension, rng):
    """
    Grow with stronger bias towards compactness.
    """
    return _greedy_grow(sequence, dimension, rng, temperature=0.5)

def _local_improve(sequence, chain, rng, max_iters=5000):
    """
    Local improvement using 2-opt style moves (reversing a segment).
    This is often more effective than regrow for HP because it preserves
    the overall shape better while allowing local rearrangements.
    """
    n = len(chain)
    if n < 5:
        return chain, _compute_energy(sequence, chain)
    
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
            
        # Pick two indices i and j to reverse the segment [i, j]
        # We must ensure the move is valid (self-avoiding and connected).
        # Reversing [i, j] changes the chain to:
        # P_0 ... P_{i-1}, P_j, P_{j-1}, ..., P_i, P_{j+1} ... P_{n-1}
        
        i = rng.randint(1, n - 2)
        j = rng.randint(i + 1, n - 1)
        
        # Check if the reversed segment collides with the rest of the chain
        # The new segment is chain[j], chain[j-1], ..., chain[i]
        # The fixed parts are chain[0..i-1] and chain[j+1..n-1]
        
        # We need to check if any point in the reversed segment (except the endpoints which connect to fixed parts)
        # is in the fixed set.
        
        # Fixed set: {chain[0], ..., chain[i-1]} U {chain[j+1], ..., chain[n-1]}
        # Note: chain[i-1] connects to new chain[i] (which is old chain[j])
        # Note: chain[j+1] connects to new chain[j] (which is old chain[i])
        
        # Let's construct the new chain and check validity.
        # For efficiency, we can check collisions first.
        
        # Points to check: chain[i+1] ... chain[j-1] (the internal points of the reversed segment)
        # They must not be in the fixed set.
        # Also, the new endpoints must not collide with the fixed set (excluding their immediate neighbors).
        
        # Actually, a simpler check:
        # Build the new chain.
        new_chain = chain[:i] + list(reversed(chain[i:j+1])) + chain[j+1:]
        
        # Check self-avoidance
        # Since n is small (<=100), we can just check if all points are unique.
        if len(set(new_chain)) != n:
            continue
        
        # Check connectivity (unit steps)
        # The internal steps of the reversed segment are unit steps (since they were unit steps before).
        # We only need to check the two new connections:
        # 1. chain[i-1] to new_chain[i] (which is chain[j])
        # 2. new_chain[j] (which is chain[i]) to chain[j+1]
        
        p_prev = new_chain[i-1]
        p_new_start = new_chain[i]
        if abs(p_prev[0] - p_new_start[0]) + abs(p_prev[1] - p_new_start[1]) != 1:
            continue
            
        p_new_end = new_chain[j]
        p_next = new_chain[j+1]
        if abs(p_new_end[0] - p_next[0]) + abs(p_new_end[1] - p_next[1]) != 1:
            continue
            
        # If valid, compute energy
        new_energy = _compute_energy(sequence, new_chain)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_chain = new_chain
            current_energy = new_energy
        else:
            # Accept with some probability (simulated annealing style)
            # But for simplicity and speed, let's just do hill climbing or very mild SA.
            # Given the 120s budget, pure hill climbing might get stuck.
            # Let's use a simple SA.
            delta = new_energy - current_energy
            if delta < 0 or (rng.random() < (10 ** (-delta / 1.0))):
                current_energy = new_energy
                chain = new_chain
    
    return best_chain, best_energy

def _regrow_sa(sequence, chain, rng, max_iters=2000):
    """
    Regrow-based simulated annealing.
    """
    n = len(chain)
    if n < 5:
        return chain, _compute_energy(sequence, chain)
    
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    temp = 1.0
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        
        if temp < 0.05:
            break
        
        # Pick a random split point
        split = rng.randint(2, n - 2)
        
        # Regrow suffix from split
        prefix = chain[:split]
        occupied = {p: i for i, p in enumerate(prefix)}
        last = prefix[-1]
        
        new_suffix = []
        success = True
        
        for i in range(split, n):
            res = sequence[i]
            options = []
            for site in _neighbours(last):
                if site in occupied:
                    continue
                gain = 0
                if res == 'H':
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                score = gain * 5.0 + free * 0.5 + rng.random() * temp
                options.append((score, site))
            
            if not options:
                success = False
                break
            
            options.sort(key=lambda x: x[0], reverse=True)
            best_score, best_site = options[0]
            
            # With some probability, pick a random valid option
            if rng.random() < 0.2:
                _, best_site = options[rng.randint(0, min(3, len(options)-1))]
            
            occupied[best_site] = i
            new_suffix.append(best_site)
            last = best_site
        
        if not success:
            temp *= 0.95
            continue
        
        new_chain = prefix + new_suffix
        new_energy = _compute_energy(sequence, new_chain)
        
        delta = new_energy - current_energy
        if delta < 0 or (rng.random() < (10 ** (-delta / max(temp, 0.1)))):
            current_energy = new_energy
            chain = new_chain
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                temp = min(temp * 1.05, 2.0)
            else:
                temp *= 0.95
        else:
            temp *= 0.95
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold the sequence using a combination of Greedy Growth and Local Improvement.
    1. Generate diverse initial conformations using randomized greedy growth.
    2. Optimize each using Regrow-SA or 2-opt.
    3. Keep the best.
    """
    rng = random.Random(42)
    n = len(sequence)
    
    best_chain = None
    best_energy = 0
    
    # Time allocation
    # We have ~170s.
    # Let's allocate time to multiple independent runs.
    
    num_runs = 8
    time_per_run = 15.0 # 120s total
    
    for run_idx in range(num_runs):
        if time.time() > _DEADLINE - 10:
            break
        
        # Generate initial conformation
        # Alternate between different greedy biases
        if run_idx % 2 == 0:
            init_chain = _greedy_grow(sequence, dimension, rng, temperature=0.5)
        else:
            init_chain = _greedy_grow_biased(sequence, dimension, rng)
        
        if init_chain is None:
            # Fallback: try a random SAW
            # Simple random walk with backtracking
            init_chain = _generate_random_saw(n, rng)
        
        if init_chain is None or len(init_chain) != n:
            # Last resort: straight line
            init_chain = [(i, 0) for i in range(n)]
        
        # Run optimization
        # Alternate between Regrow-SA and 2-opt
        if run_idx % 2 == 0:
            sa_chain, sa_energy = _regrow_sa(sequence, init_chain, rng, max_iters=5000)
        else:
            sa_chain, sa_energy = _local_improve(sequence, init_chain, rng, max_iters=5000)
        
        if sa_energy < best_energy:
            best_chain = sa_chain
            best_energy = sa_energy
    
    # Fallback
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
        
    return [list(p) for p in best_chain]

def _generate_random_saw(n, rng):
    """
    Generate a random self-avoiding walk of length n.
    """
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    def gen(length, current, occupied):
        if length == 0:
            return [current]
        
        dirs = list(range(4))
        rng.shuffle(dirs)
        
        for d in dirs:
            nx, ny = current[0] + [(1,0),(0,-1),(-1,0),(0,1)][d]
            new_pos = (nx, ny)
            if new_pos not in occupied:
                occupied.add(new_pos)
                suffix = gen(length - 1, new_pos, occupied)
                if suffix:
                    return [current] + suffix
                occupied.remove(new_pos)
        
        return None

    occupied = {(0, 0)}
    result = gen(n - 1, (0, 0), occupied)
    if result is None:
        return [(i, 0) for i in range(n)]
    return result
# EVOLVE-BLOCK-END
