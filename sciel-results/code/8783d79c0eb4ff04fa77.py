"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
import time


def _energy(sequence, chain):
    """Compute the HP energy: -number of non-adjacent H-H contacts."""
    if chain is None:
        return 0
    where = {tuple(p): i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i in range(n):
        if sequence[i] == "H":
            px, py = chain[i]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (px + dx, py + dy)
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == "H":
                    if i < j:
                        contacts += 1
    return -contacts


def _is_valid_walk(chain, dimension=2):
    """Check if chain is a valid self-avoiding walk with unit steps."""
    if chain is None:
        return False
    n = len(chain)
    if n == 0:
        return True
    seen = set()
    for i, p in enumerate(chain):
        pt = tuple(p)
        if pt in seen:
            return False
        seen.add(pt)
        if i > 0:
            prev = tuple(chain[i - 1])
            dist = sum(abs(a - b) for a, b in zip(pt, prev))
            if dist != 1:
                return False
    return True


def _min_conflicts_fold(sequence, dimension, rng, max_iter=10000, time_limit=8.0):
    """
    Min-conflicts local search on a grid representation.
    
    We represent the chain as a list of grid positions. The search state is
    a partial or complete assignment of residues to grid cells. We use a
    constraint-satisfaction view:
    
    Constraints:
    1. Self-avoidance: no two residues occupy the same cell.
    2. Unit-step: consecutive residues must be on adjacent lattice sites.
    3. Objective: maximize H-H contacts (minimize energy).
    
    Strategy:
    - Start with a random valid self-avoiding walk (using a simple random walk).
    - Use local moves that preserve validity:
      * Pivot moves: rotate a suffix around a pivot point.
      * Endpoint moves: extend/retract the last residue to a nearby empty cell.
      * Segment moves: try to move a small segment to a nearby location.
    - Score = energy (we minimize).
    - Use simulated annealing to escape local minima.
    """
    n = len(sequence)
    if n <= 1:
        return [(0, 0)] * n

    # Generate initial random self-avoiding walk
    def _random_walk(n, rng):
        path = [(0, 0)]
        visited = {(0, 0)}
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        def dfs(depth):
            if depth == n:
                return True
            rng.shuffle(directions)
            for dx, dy in directions:
                new_pos = (path[-1][0] + dx, path[-1][1] + dy)
                if new_pos not in visited:
                    visited.add(new_pos)
                    path.append(new_pos)
                    if dfs(depth + 1):
                        return True
                    path.pop()
                    visited.remove(new_pos)
            return False
        
        if not dfs(1):
            return [(i, 0) for i in range(n)]
        return path

    current_chain = _random_walk(n, rng)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    T = 2.0
    cooling = 0.9995
    
    # Precompute H positions for faster energy delta calculation
    h_indices = [i for i, c in enumerate(sequence) if c == "H"]
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Choose move type
        move_type = rng.random()
        
        if n <= 2:
            break
        
        if move_type < 0.6:
            # Pivot move: rotate suffix around a pivot
            k = rng.randint(1, n - 2)
            pivot = current_chain[k]
            tail = current_chain[k:]
            
            rot = rng.choice([1, 2, 3])
            new_tail = []
            for x, y in tail:
                rx, ry = x - pivot[0], y - pivot[1]
                if rot == 1:
                    nx, ny = ry, -rx
                elif rot == 2:
                    nx, ny = -rx, -ry
                else:
                    nx, ny = -ry, rx
                new_tail.append((pivot[0] + nx, pivot[1] + ny))
            
            # Check validity
            head_set = set(tuple(p) for p in current_chain[:k])
            new_tail_set = set(tuple(p) for p in new_tail)
            
            valid = True
            if new_tail_set & head_set:
                valid = False
            if len(new_tail_set) != len(new_tail):
                valid = False
            
            if valid:
                new_chain = current_chain[:k] + new_tail
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - current_energy
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_chain = new_chain
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
        
        elif move_type < 0.85:
            # Endpoint move: try to move the last residue to a different neighbor
            if n < 2:
                continue
            last = current_chain[-1]
            prev = current_chain[-2]
            
            # Get all valid neighbors of prev that are not in the chain (except last)
            chain_set = set(tuple(p) for p in current_chain[:-1])
            neighbors = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                new_pos = (prev[0] + dx, prev[1] + dy)
                if new_pos not in chain_set and new_pos != tuple(last):
                    neighbors.append(new_pos)
            
            if neighbors:
                new_last = rng.choice(neighbors)
                new_chain = current_chain[:-1] + [new_last]
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - current_energy
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_chain = new_chain
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
        
        else:
            # Segment move: try to move a small segment (2-3 residues) to a nearby location
            seg_len = rng.choice([2, 3])
            if n <= seg_len:
                continue
            
            # Pick a segment starting at index k (not at the ends to keep connections)
            k = rng.randint(1, n - seg_len - 1)
            seg = current_chain[k:k+seg_len]
            before = current_chain[k-1]
            after = current_chain[k+seg_len] if k+seg_len < n else None
            
            # Try to find a new location for the segment
            # The segment must connect: before -> new_seg[0] and new_seg[-1] -> after
            # For simplicity, try shifting the segment by one step in a random direction
            
            shift_dx, shift_dy = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            
            new_seg = [(x + shift_dx, y + shift_dy) for x, y in seg]
            
            # Check if the new segment is valid
            # 1. No collision with the rest of the chain
            rest_indices = set(range(n)) - set(range(k, k+seg_len))
            rest_set = set(tuple(current_chain[i]) for i in rest_indices)
            new_seg_set = set(tuple(p) for p in new_seg)
            
            valid = True
            if new_seg_set & rest_set:
                valid = False
            if len(new_seg_set) != len(new_seg):
                valid = False
            
            # 2. Check connections
            if valid:
                # before -> new_seg[0] must be unit step
                if abs(before[0] - new_seg[0][0]) + abs(before[1] - new_seg[0][1]) != 1:
                    valid = False
            
            if valid and after is not None:
                # new_seg[-1] -> after must be unit step
                if abs(new_seg[-1][0] - after[0]) + abs(new_seg[-1][1] - after[1]) != 1:
                    valid = False
            
            if valid:
                new_chain = current_chain[:k] + new_seg + current_chain[k+seg_len:]
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - current_energy
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_chain = new_chain
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
        
        T *= cooling
    
    return best_chain


def fold(sequence, dimension):
    """
    Find the best conformation using min-conflicts local search.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    best_chain = None
    best_energy = 0
    
    start_time = time.time()
    time_limit = 10.0
    
    restarts = 0
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        seed = 12345 + len(sequence) + restarts * 1000
        rng = random.Random(seed)
        
        chain = _min_conflicts_fold(sequence, dimension, rng, max_iter=5000, time_limit=8.0)
        
        if chain is not None and _is_valid_walk(chain):
            energy = _energy(sequence, chain)
            if best_chain is None or energy < best_energy:
                best_chain = chain
                best_energy = energy
        
        restarts += 1
        if restarts >= 4:
            break
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
