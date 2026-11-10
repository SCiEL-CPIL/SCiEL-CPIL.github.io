"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
_DEADLINE = time.time() + 180.0


def _compute_energy(sequence, chain):
    """Compute the HP contact energy: -1 for each H-H non-bonded contact."""
    n = len(chain)
    if n < 3:
        return 0
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i

    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j_pos = (x + dx, y + dy)
            j = pos_to_idx.get(j_pos)
            if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts // 2  # Each contact counted twice


def _random_saw(n, rng):
    """Generate a random self-avoiding walk of length n."""
    if n == 0:
        return []
    
    # Start at origin
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        x, y = last
        # Shuffle directions for randomness
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        rng.shuffle(directions)
        
        placed = False
        for dx, dy in directions:
            new_pos = (x + dx, y + dy)
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied[new_pos] = i
                placed = True
                break
        
        if not placed:
            # Stuck - return None
            return None
    
    return chain


def _pivot_move(chain, rng):
    """
    Apply a pivot move: choose a random internal point and reverse the tail.
    This is a global rearrangement that can escape local minima.
    Returns the new chain or None if invalid.
    """
    n = len(chain)
    if n < 3:
        return chain[:]
    
    # Choose pivot point (not the ends)
    pivot_idx = rng.randint(1, n - 2)
    
    # Reverse the tail from pivot_idx to end
    new_chain = chain[:pivot_idx] + chain[pivot_idx:][::-1]
    
    # Check self-avoidance
    seen = set()
    for p in new_chain:
        if p in seen:
            return None
        seen.add(p)
    
    # Check unit steps
    for i in range(1, n):
        x1, y1 = new_chain[i-1]
        x2, y2 = new_chain[i]
        if abs(x1 - x2) + abs(y1 - y2) != 1:
            return None
    
    return new_chain


def _local_improve(sequence, chain, rng, max_iters=2000, time_budget=5.0):
    """
    Improve chain using pivot moves and local search.
    Uses simulated annealing to accept worse moves sometimes.
    """
    if not chain or len(chain) < 3:
        return chain[:]
    
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    start_time = time.time()
    temp = 1.0
    cooling = 0.995
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Try a pivot move
        new_chain = _pivot_move(chain, rng)
        if new_chain is None:
            # Try a simple local move: try to shift one residue if possible
            # This is complex; skip for now and just continue
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta/temp) if worse
        if delta < 0 or (delta > 0 and rng.random() < pow(2.718281828, -delta / max(temp, 0.01))):
            chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_chain = chain[:]
                best_energy = current_energy
        
        # Cool down
        temp *= cooling
    
    return best_chain


def _min_conflicts_fold(sequence, n, rng, time_budget=3.0):
    """
    Min-conflicts approach: start with a random valid SAW, then use pivot moves
    to minimize energy (maximize H-H contacts).
    """
    if n == 0:
        return []
    
    # Generate initial random SAW
    chain = None
    for _ in range(100):
        chain = _random_saw(n, rng)
        if chain is not None:
            break
    
    if chain is None:
        # Fallback: try a spiral or simple pattern
        chain = _fallback_saw(n)
        if chain is None:
            return [[i, 0] for i in range(n)]
    
    # Improve with local search
    improved = _local_improve(sequence, chain, rng, max_iters=1500, time_budget=time_budget)
    return improved


def _fallback_saw(n):
    """Generate a simple spiral-like SAW as fallback."""
    if n == 0:
        return []
    
    chain = [(0, 0)]
    if n == 1:
        return chain
    
    # Simple pattern: go right, up, left, down in expanding loops
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dir_idx = 0
    steps_in_dir = 1
    dx, dy = directions[0]
    
    for i in range(1, n):
        x, y = chain[-1]
        new_x, new_y = x + dx, y + dy
        
        # Check if we can move
        if (new_x, new_y) not in set(chain):
            chain.append((new_x, new_y))
        else:
            # Try next direction
            dir_idx = (dir_idx + 1) % 4
            dx, dy = directions[dir_idx]
            new_x, new_y = x + dx, y + dy
            if (new_x, new_y) not in set(chain):
                chain.append((new_x, new_y))
            else:
                # Stuck - return what we have
                break
        
        # After 2 moves in same direction, turn
        if i % 2 == 1:
            dir_idx = (dir_idx + 1) % 4
            dx, dy = directions[dir_idx]
    
    # If we didn't get all residues, pad with a simple extension
    while len(chain) < n:
        x, y = chain[-1]
        # Try to extend in a direction that doesn't collide
        found = False
        for dx2, dy2 in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if (x + dx2, y + dy2) not in set(chain):
                chain.append((x + dx2, y + dy2))
                found = True
                break
        if not found:
            break
    
    return chain if len(chain) == n else None


def fold(sequence, dimension):
    """
    Find the best conformation for the given HP sequence.
    Returns a list of [x, y] coordinates.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Fallback: straight line
    straight = [[i, 0] for i in range(n)]
    
    best_chain = None
    best_energy = 0  # Energy is negative, so we want the most negative (lowest)
    
    rng = random.Random(42)
    
    # Time budget per sequence
    per_seq_budget = 15.0
    
    # Multiple restarts with min-conflicts
    num_restarts = 8
    
    for r in range(num_restarts):
        if time.time() > _DEADLINE - 5:
            break
        
        # Vary the random seed for diversity
        rng_seq = random.Random(42 + r * 1000)
        
        chain = _min_conflicts_fold(sequence, n, rng_seq, time_budget=per_seq_budget / num_restarts)
        
        if chain is None or len(chain) != n:
            continue
        
        # Verify validity
        valid = True
        seen = set()
        for i, p in enumerate(chain):
            if p in seen:
                valid = False
                break
            seen.add(p)
            if i > 0:
                x1, y1 = chain[i-1]
                x2, y2 = p
                if abs(x1 - x2) + abs(y1 - y2) != 1:
                    valid = False
                    break
        
        if not valid:
            continue
        
        energy = _compute_energy(sequence, chain)
        
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        return straight
        
    # Convert tuples to lists
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
