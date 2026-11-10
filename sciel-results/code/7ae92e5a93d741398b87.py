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


def _generate_random_walk(n, rng):
    """Generate a random self-avoiding walk of length n."""
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


def _min_conflicts_global(sequence, rng, max_iters=20000, max_restarts=5):
    """
    Constraint-satisfaction formulation using min-conflicts local search.
    Treats the entire self-avoiding walk as a global assignment problem.
    Uses pivot moves to maintain validity while optimizing H-H contacts.
    """
    n = len(sequence)
    if n <= 1:
        return [(0, 0)] * n

    best_chain = None
    best_energy = float('inf')
    
    for restart in range(max_restarts):
        # Start from a random valid self-avoiding walk
        current_chain = _generate_random_walk(n, rng)
        current_energy = _energy(sequence, current_chain)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(current_chain)
        
        # Min-conflicts: iteratively improve by making moves that reduce energy
        no_improve = 0
        for it in range(max_iters):
            if n <= 2:
                break
            
            # Choose a pivot point (index k from 1 to n-2)
            k = rng.randint(1, n - 2)
            pivot = current_chain[k]
            
            # Rotate the tail (from k to n-1) around the pivot
            rot = rng.choice([1, 2, 3])
            tail = current_chain[k:]
            new_tail = []
            for (x, y) in tail:
                rx, ry = x - pivot[0], y - pivot[1]
                if rot == 1:  # 90 deg clockwise
                    nx, ny = ry, -rx
                elif rot == 2:  # 180 deg
                    nx, ny = -rx, -ry
                else:  # 270 deg clockwise
                    nx, ny = -ry, rx
                new_tail.append((pivot[0] + nx, pivot[1] + ny))
            
            # Check validity: no self-intersection
            head_set = set(tuple(p) for p in current_chain[:k])
            new_tail_set = set(tuple(p) for p in new_tail)
            
            if new_tail_set & head_set or len(new_tail_set) != len(new_tail):
                no_improve += 1
                if no_improve > 500:
                    break
                continue
            
            new_chain = current_chain[:k] + new_tail
            new_energy = _energy(sequence, new_chain)
            
            # Accept if better, or with probability if equal
            if new_energy < current_energy:
                current_chain = new_chain
                current_energy = new_energy
                no_improve = 0
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
            elif new_energy == current_energy and rng.random() < 0.2:
                current_chain = new_chain
                no_improve += 1
            else:
                no_improve += 1
            
            if no_improve > 500:
                break
    
    return best_chain


def fold(sequence, dimension):
    """
    Find the best conformation using constraint-satisfaction min-conflicts.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    # Use time-based allocation for multiple restarts
    start_time = time.time()
    time_limit = 11.0  # seconds per sequence, leaving buffer for 10 sequences in 120s
    
    best_chain = None
    best_energy = float('inf')
    
    # Multiple restarts with different seeds
    restarts_done = 0
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        seed = 42 + len(sequence) * 1000 + restarts_done * 7
        rng = random.Random(seed)
        
        # Adaptive max_iters based on remaining time
        remaining = time_limit - elapsed
        max_iters = min(20000, max(1000, int(remaining * 2000)))
        
        chain = _min_conflicts_global(sequence, rng, max_iters=max_iters, max_restarts=1)
        
        if chain is not None:
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
        
        restarts_done += 1
        if restarts_done >= 8:
            break
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
