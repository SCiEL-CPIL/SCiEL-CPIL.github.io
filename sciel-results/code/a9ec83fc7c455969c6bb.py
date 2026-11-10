"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
import time


def _compute_energy(sequence, chain):
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


def _is_valid(chain):
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
            dist = abs(pt[0] - prev[0]) + abs(pt[1] - prev[1])
            if dist != 1:
                return False
    return True


def _count_violations(chain):
    """Count constraint violations: self-intersections and non-unit steps."""
    if chain is None:
        return 0
    n = len(chain)
    if n == 0:
        return 0
    seen = {}
    violations = 0
    for i, p in enumerate(chain):
        pt = tuple(p)
        if pt in seen:
            violations += 1  # self-intersection
        else:
            seen[pt] = i
        if i > 0:
            prev = tuple(chain[i - 1])
            dist = abs(pt[0] - prev[0]) + abs(pt[1] - prev[1])
            if dist != 1:
                violations += 1  # non-unit step
    return violations


def _min_conflicts_fold(sequence, dimension, rng, max_iter=5000):
    """
    Min-conflicts local search on grid representation.
    Treats the entire walk as a global assignment problem.
    """
    n = len(sequence)
    if n <= 1:
        return [(0, 0)] * n

    # Initialize with a random self-avoiding walk
    def _init_walk():
        path = [(0, 0)]
        visited = {(0, 0)}
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        def dfs(depth):
            if depth == n:
                return True
            rng.shuffle(dirs)
            for dx, dy in dirs:
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

    current = _init_walk()
    best = list(current)
    best_energy = _compute_energy(sequence, best)

    for it in range(max_iter):
        # Pick a random position to move
        idx = rng.randint(1, n - 2)  # Avoid endpoints for simplicity
        
        # Get current position
        cx, cy = current[idx]
        
        # Try all 4 possible moves
        rng.shuffle([(1, 0), (-1, 0), (0, 1), (0, -1)])
        moved = False
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_pos = (cx + dx, cy + dy)
            
            # Check if this move would create immediate violations
            # We need to check:
            # 1. new_pos not occupied by another part of chain (except neighbors idx-1, idx+1)
            # 2. Step from idx-1 to new_pos is unit
            # 3. Step from new_pos to idx+1 is unit
            
            # Build occupancy map excluding idx
            occ = {}
            for i, p in enumerate(current):
                if i != idx:
                    occ[tuple(p)] = i
            
            if new_pos in occ:
                # Check if it's one of the adjacent positions (allowed)
                occ_idx = occ[new_pos]
                if abs(occ_idx - idx) > 1:
                    continue  # Collision with non-adjacent part
            
            # Check step validity
            if idx > 0:
                px, py = current[idx - 1]
                if abs(new_pos[0] - px) + abs(new_pos[1] - py) != 1:
                    continue
            if idx < n - 1:
                px, py = current[idx + 1]
                if abs(new_pos[0] - px) + abs(new_pos[1] - py) != 1:
                    continue
            
            # Apply move
            old_pos = current[idx]
            current[idx] = new_pos
            
            # Check full validity
            if _is_valid(current):
                energy = _compute_energy(sequence, current)
                if energy < best_energy:
                    best_energy = energy
                    best = list(current)
                moved = True
                break
            else:
                # Revert
                current[idx] = old_pos
        
        if not moved:
            # Try pivot move as fallback
            if n > 3:
                k = rng.randint(1, n - 2)
                pivot = current[k]
                tail = current[k:]
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
                
                new_chain = current[:k] + new_tail
                if _is_valid(new_chain):
                    energy = _compute_energy(sequence, new_chain)
                    if energy < best_energy:
                        best_energy = energy
                        best = list(new_chain)
                    current = new_chain

    return best


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
        
        max_iter = 3000
        seed = 12345 + len(sequence) + restarts * 1000
        rng = random.Random(seed)
        
        chain = _min_conflicts_fold(sequence, dimension, rng, max_iter=max_iter)
        
        if chain is not None and _is_valid(chain):
            energy = _compute_energy(sequence, chain)
            if best_chain is None or energy < best_energy:
                best_chain = chain
                best_energy = energy
        
        restarts += 1
        if restarts >= 5:
            break
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
