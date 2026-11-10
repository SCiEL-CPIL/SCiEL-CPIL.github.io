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


def _generate_compact_initial(sequence, rng):
    """Generate a compact initial self-avoiding walk biased toward dense packing."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0)] * n
    
    # Start with a small spiral or compact pattern
    path = [(0, 0)]
    visited = {(0, 0)}
    
    # Use a biased random walk that prefers to fill in compact regions
    # Try to keep the walk within a bounding box of size ~sqrt(n)
    max_coord = int(np.sqrt(n)) + 2
    
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    def dfs(depth, max_depth=50):
        if depth == n:
            return True
        if depth > max_depth * 10:  # Safety check
            return False
        
        # Shuffle directions but bias toward inward moves
        rng.shuffle(directions)
        
        # Compute current bounding box to bias moves
        if path:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
        
        for dx, dy in directions:
            new_pos = (path[-1][0] + dx, path[-1][1] + dy)
            # Keep within reasonable bounds
            if abs(new_pos[0]) > max_coord or abs(new_pos[1]) > max_coord:
                continue
            if new_pos not in visited:
                visited.add(new_pos)
                path.append(new_pos)
                if dfs(depth + 1, max_depth):
                    return True
                path.pop()
                visited.remove(new_pos)
        return False
    
    if not dfs(1):
        # Fallback: simple random walk
        path = [(0, 0)]
        visited = {(0, 0)}
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        def dfs_simple(depth):
            if depth == n:
                return True
            rng.shuffle(directions)
            for dx, dy in directions:
                new_pos = (path[-1][0] + dx, path[-1][1] + dy)
                if new_pos not in visited:
                    visited.add(new_pos)
                    path.append(new_pos)
                    if dfs_simple(depth + 1):
                        return True
                    path.pop()
                    visited.remove(new_pos)
            return False
        
        if not dfs_simple(1):
            return [(i, 0) for i in range(n)]
    
    return path


def _pivot_move(chain, k, rot, rng):
    """Perform a pivot move: rotate the tail from index k by rot*90 degrees."""
    n = len(chain)
    if k < 1 or k >= n - 1:
        return None
    
    pivot = chain[k]
    tail = chain[k:]
    
    new_tail = []
    for (x, y) in tail:
        rx, ry = x - pivot[0], y - pivot[1]
        if rot == 1:  # 90 deg clockwise
            nx, ny = ry, -rx
        elif rot == 2:  # 180 deg
            nx, ny = -rx, -ry
        else:  # 270 deg clockwise (90 ccw)
            nx, ny = -ry, rx
        new_tail.append((pivot[0] + nx, pivot[1] + ny))
    
    # Check validity
    head_set = set(tuple(p) for p in chain[:k])
    new_tail_set = set(tuple(p) for p in new_tail)
    
    if new_tail_set & head_set or len(new_tail_set) != len(new_tail):
        return None
    
    return chain[:k] + new_tail


def _endpoint_move(chain, rng):
    """Move the endpoint to a different valid position."""
    n = len(chain)
    if n < 2:
        return None
    
    last = chain[-1]
    prev = chain[-2]
    chain_set = set(tuple(p) for p in chain[:-1])
    
    candidates = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        np_ = (prev[0] + dx, prev[1] + dy)
        if np_ != last and tuple(np_) not in chain_set:
            candidates.append(np_)
    
    if not candidates:
        return None
    
    new_last = rng.choice(candidates)
    return chain[:-1] + [new_last]


def _local_search(sequence, chain, rng, max_iter=3000, time_limit=None, start_time=None):
    """
    Local search with pivot and endpoint moves.
    Uses simulated annealing acceptance criterion.
    """
    n = len(sequence)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    T = 1.0
    T_min = 0.01
    cooling = 0.998
    
    no_improve = 0
    max_no_improve = 500
    
    for it in range(max_iter):
        if time_limit is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.85 and n > 2:
            # Pivot move
            k = rng.randint(1, n - 2)
            rot = rng.choice([1, 2, 3])
            new_chain = _pivot_move(current_chain, k, rot, rng)
        else:
            # Endpoint move
            new_chain = _endpoint_move(current_chain, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Acceptance criterion
        if delta < 0:
            current_chain = new_chain
            current_energy = new_energy
            no_improve = 0
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        else:
            # Simulated annealing acceptance
            if T > T_min:
                accept_prob = np.exp(-delta / T)
                if rng.random() < accept_prob:
                    current_chain = new_chain
                    current_energy = new_energy
            no_improve += 1
        
        T = max(T_min, T * cooling)
        
        if no_improve > max_no_improve:
            break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Find the best conformation using local search with multiple restarts.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    start_time = time.time()
    time_limit = 11.0  # seconds per sequence, leaving buffer for 10 sequences in 120s
    
    best_chain = None
    best_energy = float('inf')
    
    # Multiple restarts with different initial conditions
    num_restarts = 8
    
    for restart in range(num_restarts):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        seed = 42 + len(sequence) * 100 + restart * 7919
        rng = random.Random(seed)
        
        # Generate initial chain
        initial_chain = _generate_compact_initial(sequence, rng)
        
        # Run local search
        chain, energy = _local_search(
            sequence, initial_chain, rng,
            max_iter=2000,
            time_limit=time_limit - (time.time() - start_time) - 0.5,
            start_time=start_time
        )
        
        if chain is not None and energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
