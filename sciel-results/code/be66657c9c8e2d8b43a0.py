"""Rewritten HP lattice protein folding solver with simulated annealing."""
# EVOLVE-BLOCK-START
import time
import random
import math

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # Safety margin before 120s limit


def _remaining_time():
    return max(0.0, _DEADLINE - time.time())


def _neighbours_2d(x, y):
    """Yield the 4 neighbours of (x, y)."""
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _compute_energy(sequence, chain):
    """Compute energy (negative count of H-H contacts)."""
    n = len(chain)
    where = {}
    for i, (x, y) in enumerate(chain):
        where[(x, y)] = i
    
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            x, y = chain[i]
            for nx, ny in _neighbours_2d(x, y):
                j = where.get((nx, ny))
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _initial_chain(sequence, dimension=2):
    """Create a simple spiral-like initial conformation."""
    n = len(sequence)
    chain = [(0, 0)]
    dx, dy = 1, 0  # start moving right
    # Use a simple pattern: go right, down, left, up, then expand
    # Actually, let's just do a zigzag or simple path
    # For simplicity, start with a straight line then it will be improved
    for i in range(1, n):
        x, y = chain[-1]
        # Simple alternating pattern to avoid immediate backtracking
        if i % 4 == 1:
            x += 1
        elif i % 4 == 2:
            y += 1
        elif i % 4 == 3:
            x -= 1
        else:
            y -= 1
        chain.append((x, y))
    
    # Check for self-avoidance; if violated, fall back to straight line
    seen = set()
    for p in chain:
        if p in seen:
            # Fallback: straight line
            return [(i, 0) for i in range(n)]
        seen.add(p)
    return chain


def _is_valid_chain(chain):
    """Check if chain is self-avoiding with unit steps."""
    n = len(chain)
    if n == 0:
        return True
    seen = set()
    for i, (x, y) in enumerate(chain):
        if (x, y) in seen:
            return False
        seen.add((x, y))
        if i > 0:
            px, py = chain[i-1]
            if abs(x - px) + abs(y - py) != 1:
                return False
    return True


def _try_move(chain, move_idx, sequence):
    """
    Try to apply a move to the chain. move_idx selects the move type.
    Returns (new_chain, delta_energy) or (None, None) if move invalid.
    """
    n = len(chain)
    if n < 2:
        return None, None
    
    # Choose a random interior residue (not endpoints for simplicity, but we can include)
    i = random.randint(1, n - 2)
    
    # Current position
    x, y = chain[i]
    
    # The two neighbours in the chain
    prev = chain[i - 1]
    next_pos = chain[i + 1]
    
    # The two possible new positions: the other side of prev or next
    # If we move residue i, it must stay adjacent to both i-1 and i+1
    # The only valid moves are to the two positions that maintain adjacency
    # Actually, for a 2D lattice, moving a single residue is constrained.
    # Let's use a different approach: pick two residues and try to swap or rotate.
    
    # Simpler approach: pick a random interior segment and try to flip it
    # Or use a "pivot" move: pick a random residue i (1 <= i <= n-1) and
    # rotate the tail (i+1 to end) around residue i.
    
    # Let's implement a pivot move: pick i from 1 to n-1, rotate tail around i
    i = random.randint(1, n - 1)
    pivot = chain[i]
    px, py = pivot
    
    # The tail is chain[i+1:]
    tail = chain[i+1:]
    if not tail:
        return None, None
    
    # The direction from pivot to the first tail element
    tx, ty = tail[0]
    dx, dy = tx - px, ty - py
    
    # Rotate the entire tail by 90, 180, or 270 degrees around the pivot
    angle = random.choice([1, 2, 3])  # 90, 180, 270 degrees
    
    # Rotation matrix for 2D
    if angle == 1:  # 90 degrees CCW: (x,y) -> (-y, x)
        def rotate(x, y):
            return (-y, x)
    elif angle == 2:  # 180 degrees: (x,y) -> (-x, -y)
        def rotate(x, y):
            return (-x, -y)
    else:  # 270 degrees CCW: (x,y) -> (y, -x)
        def rotate(x, y):
            return (y, -x)
    
    # Rotate the tail
    new_tail = []
    for x, y in tail:
        # Translate to origin, rotate, translate back
        rx, ry = rotate(x - px, y - py)
        new_tail.append((rx + px, ry + py))
    
    # Check if the new tail is valid:
    # 1. The first element of new_tail must be adjacent to pivot
    nx, ny = new_tail[0]
    if abs(nx - px) + abs(ny - py) != 1:
        return None, None
    
    # 2. No self-intersections in the new tail
    seen = set()
    for x, y in new_tail:
        if (x, y) in seen:
            return None, None
        seen.add((x, y))
    
    # 3. New tail must not intersect the head (chain[0:i+1])
    head_set = set(chain[:i+1])
    for x, y in new_tail:
        if (x, y) in head_set:
            return None, None
    
    # Construct new chain
    new_chain = chain[:i+1] + new_tail
    
    # Compute delta energy
    old_energy = _compute_energy(sequence, chain)
    new_energy = _compute_energy(sequence, new_chain)
    
    return new_chain, new_energy - old_energy


def _anneal(sequence, dimension=2, time_budget=10.0):
    """Simulated annealing to find low-energy conformation."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0)]
    
    # Initialize
    chain = _initial_chain(sequence, dimension)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    start_temp = 2.0
    end_temp = 0.01
    t = start_temp
    
    # Time-based annealing
    start_time = time.time()
    max_time = min(time_budget, _remaining_time() - 1.0)
    if max_time <= 0:
        return best_chain
    
    steps = 0
    while time.time() - start_time < max_time:
        steps += 1
        
        # Try a move
        new_chain, delta = _try_move(chain, 0, sequence)
        if new_chain is None:
            continue
        
        # Acceptance criterion
        if delta < 0 or random.random() < math.exp(-delta / t):
            chain = new_chain
            current_energy += delta
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        # Cool down
        progress = (time.time() - start_time) / max_time
        t = start_temp * (end_temp / start_temp) ** progress
    
    return best_chain


def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    Returns a list of [x, y] lattice points.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    # Ensure dimension is 2
    if dimension != 2:
        dimension = 2
    
    # Multiple restarts with simulated annealing
    rng = random.Random(42 + n)
    random.seed(rng)
    
    best_chain = None
    best_energy = float('inf')
    
    # Determine number of restarts based on time budget
    # Start with a few restarts, each with a time budget
    restarts = 0
    max_restarts = 50
    
    while _remaining_time() > 5.0 and restarts < max_restarts:
        # Give each restart a portion of the remaining time
        time_per_restart = min(8.0, max(0.5, (_remaining_time() - 5.0) / max(1, max_restarts - restarts)))
        
        chain = _anneal(sequence, dimension, time_budget=time_per_restart)
        energy = _compute_energy(sequence, chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
        
        restarts += 1
        
        # If we found a very good solution, reduce restarts
        if best_energy <= -n // 2:
            break
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    # Convert to list of lists
    return [[x, y] for x, y in best_chain]
# EVOLVE-BLOCK-END
