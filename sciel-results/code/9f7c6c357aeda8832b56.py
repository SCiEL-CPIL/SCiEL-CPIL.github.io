"""
HP lattice protein folding using Hamiltonian path skeleton + pivot-move local search.

Strategy:
1. Generate a deterministic space-filling grid path (snake pattern) that visits
   every cell in a bounding box exactly once. This gives a compact conformation
   with many potential H-H contacts.
2. Map the input sequence onto this path.
3. Use pivot-move local search to refine the conformation, allowing the chain
   to "breathe" and optimize H-H contacts while maintaining self-avoidance.
4. Multiple restarts with different starting configurations and time-bounded search.
"""

import random
import time
import math

# EVOLVE-BLOCK-START

def _neighbours_2d(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(sequence, chain):
    """Compute the HP energy: -number of H-H non-bonded contacts."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = where.get((x + dx, y + dy))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _generate_snake_path(length):
    """
    Generate a snake-like Hamiltonian path on a 2D grid that visits exactly
    'length' cells. The grid dimensions are chosen to be as square as possible.
    Returns a list of (x, y) tuples forming the path.
    """
    # Find grid dimensions w x h such that w * h >= length, w and h close
    w = int(math.ceil(math.sqrt(length)))
    h = int(math.ceil(length / w))
    
    # Build the snake path
    path = []
    for row in range(h):
        if row % 2 == 0:
            for col in range(w):
                path.append((col, row))
        else:
            for col in range(w - 1, -1, -1):
                path.append((col, row))
    
    # Trim to exactly 'length' cells
    path = path[:length]
    return path


def _generate_compact_path(length, seed_offset=0):
    """
    Generate a compact path by filling a grid in a space-filling curve pattern.
    Uses a Hilbert-curve-like approach for better locality.
    """
    # Try multiple grid shapes and pick the one that gives the most contacts
    best_path = None
    best_score = -10**9
    
    for w in range(2, int(math.sqrt(length)) + 3):
        h = int(math.ceil(length / w))
        if w * h < length:
            continue
        
        # Generate snake path for this grid shape
        path = []
        for row in range(h):
            if row % 2 == 0:
                for col in range(w):
                    if len(path) < length:
                        path.append((col, row))
            else:
                for col in range(w - 1, -1, -1):
                    if len(path) < length:
                        path.append((col, row))
        
        if len(path) != length:
            continue
        
        # Score this path by counting potential H-H contacts
        # (using a dummy all-H sequence for scoring)
        where = {p: i for i, p in enumerate(path)}
        contacts = 0
        for i in range(length):
            x, y = path[i]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                j = where.get((x + dx, y + dy))
                if j is not None and j > i + 1:
                    contacts += 1
        
        if contacts > best_score:
            best_score = contacts
            best_path = path
    
    if best_path is None:
        best_path = _generate_snake_path(length)
    
    return best_path


def _pivot_move(sequence, chain, pivot_idx, rng, deadline):
    """
    Attempt a pivot move at pivot_idx. A pivot move rotates the tail of the chain
    (from pivot_idx to end) around the pivot point.
    Returns the new chain if the move is valid and improves or doesn't worsen energy,
    otherwise returns None.
    """
    n = len(chain)
    if pivot_idx < 1 or pivot_idx >= n - 1:
        return None
    
    # The tail is chain[pivot_idx:]
    tail = chain[pivot_idx:]
    head = chain[:pivot_idx]
    pivot_point = chain[pivot_idx]
    
    # Try 3 rotations (90, 180, 270 degrees)
    for angle in (1, 2, 3):  # 90, 180, 270 degrees
        # Rotate tail around pivot_point
        rotated_tail = []
        for (x, y) in tail:
            dx, dy = x - pivot_point[0], y - pivot_point[1]
            if angle == 1:  # 90 degrees CCW
                nx, ny = -dy, dx
            elif angle == 2:  # 180 degrees
                nx, ny = -dx, -dy
            else:  # 270 degrees CCW (or 90 CW)
                nx, ny = dy, -dx
            rotated_tail.append((pivot_point[0] + nx, pivot_point[1] + ny))
        
        new_chain = head + rotated_tail
        
        # Check self-avoidance
        occupied = set()
        valid = True
        for i, p in enumerate(new_chain):
            if p in occupied:
                valid = False
                break
            occupied.add(p)
        
        if not valid:
            continue
        
        # Check that consecutive points are adjacent
        for i in range(1, len(new_chain)):
            dx = abs(new_chain[i][0] - new_chain[i-1][0])
            dy = abs(new_chain[i][1] - new_chain[i-1][1])
            if dx + dy != 1:
                valid = False
                break
        
        if not valid:
            continue
        
        return new_chain
    
    return None


def _local_search(sequence, chain, rng, deadline, max_iters=5000):
    """
    Perform local search using pivot moves. Accepts moves that don't worsen energy
    with a small probability (simulated annealing-like).
    """
    n = len(chain)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    no_improve_count = 0
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        # Choose a random pivot point (not at the ends)
        pivot_idx = rng.randint(1, n - 2)
        
        new_chain = _pivot_move(sequence, chain, pivot_idx, rng, deadline)
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        
        # Accept if better, or with small probability if worse
        if new_energy <= current_energy:
            chain = new_chain
            current_energy = new_energy
            no_improve_count = 0
        else:
            # Simulated annealing: accept worse moves with decreasing probability
            delta = current_energy - new_energy  # positive means worse
            if delta < 5 and rng.random() < 0.1:
                chain = new_chain
                current_energy = new_energy
                no_improve_count += 1
            else:
                no_improve_count += 1
        
        # Track best
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(chain)
        
        # Early termination if stuck
        if no_improve_count > 200:
            break
    
    return best_chain, best_energy


def _greedy_growth(sequence, dimension, rng):
    """
    One randomized greedy chain growth: prefer sites with new H-H contacts, then free space.
    """
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours_2d(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours_2d(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours_2d(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random() * 0.5, site))
        
        if not options:
            return None
        
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    
    return chain


def fold(sequence, dimension):
    """
    Find the best conformation using a combination of:
    1. Hamiltonian path skeleton initialization
    2. Greedy growth initialization
    3. Pivot-move local search refinement
    """
    n = len(sequence)
    if n <= 1:
        return [[0, 0]]
    
    # Set deadline: use ~15 seconds per sequence (10 sequences * 15s = 150s < 180s limit)
    # But be conservative: 12s per sequence to leave margin
    per_seq_time = 12.0
    start_time = time.time()
    deadline = start_time + per_seq_time
    
    best_chain = None
    best_energy = 0  # Energy is negative, so 0 is the worst (no contacts)
    
    # Strategy 1: Hamiltonian path skeleton + local search
    if time.time() < deadline:
        path = _generate_compact_path(n)
        if path is not None and len(path) == n:
            chain = list(path)
            # Run local search
            refined_chain, energy = _local_search(sequence, chain, random.Random(42), deadline, max_iters=3000)
            if energy < best_energy:
                best_energy = energy
                best_chain = refined_chain
    
    # Strategy 2: Multiple greedy growth + local search
    rng = random.Random(12345 + n)
    growth_attempts = 30
    for attempt in range(growth_attempts):
        if time.time() > deadline:
            break
        
        chain = _greedy_growth(sequence, dimension, rng)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
        
        # If we have time, refine with local search
        if time.time() < deadline - 2.0:
            refined_chain, refined_energy = _local_search(sequence, chain, rng, deadline, max_iters=1500)
            if refined_energy < best_energy:
                best_energy = refined_energy
                best_chain = refined_chain
    
    # Strategy 3: Try different grid shapes for the skeleton
    if time.time() < deadline:
        for w in range(3, int(math.sqrt(n)) + 4):
            if time.time() > deadline:
                break
            h = int(math.ceil(n / w))
            if w * h < n:
                continue
            
            path = []
            for row in range(h):
                if row % 2 == 0:
                    for col in range(w):
                        if len(path) < n:
                            path.append((col, row))
                else:
                    for col in range(w - 1, -1, -1):
                        if len(path) < n:
                            path.append((col, row))
            
            if len(path) != n:
                continue
            
            chain = list(path)
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
            
            if time.time() < deadline - 1.0:
                refined_chain, refined_energy = _local_search(sequence, chain, random.Random(999), deadline, max_iters=1000)
                if refined_energy < best_energy:
                    best_energy = refined_energy
                    best_chain = refined_chain
    
    # Fallback: straight line (always valid)
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
