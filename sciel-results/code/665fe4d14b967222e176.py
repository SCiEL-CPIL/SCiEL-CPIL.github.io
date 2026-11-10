"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys

# Deadline guard: 330s wall-clock kill, use 320s as safety margin
_DEADLINE = 320.0
_start_time = None


def _get_remaining_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _DEADLINE - (time.time() - _start_time)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


class _SpatialGrid:
    """3D spatial hash grid for O(1) neighbor lookups."""
    def __init__(self, chain):
        self.pos_to_idx = {}
        for i, p in enumerate(chain):
            self.pos_to_idx[tuple(p)] = i
    
    def get(self, pos):
        return self.pos_to_idx.get(tuple(pos))
    
    def contains(self, pos):
        return tuple(pos) in self.pos_to_idx
    
    def update(self, old_pos, new_pos, idx):
        old_t = tuple(old_pos)
        new_t = tuple(new_pos)
        del self.pos_to_idx[old_t]
        self.pos_to_idx[new_t] = idx


def _energy(sequence, chain):
    """Compute energy: -number of H-H contacts on non-adjacent lattice neighbours."""
    grid = _SpatialGrid(chain)
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = grid.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _energy_delta_move(sequence, chain, i, new_pos):
    """
    Compute energy change when moving residue i from chain[i] to new_pos.
    Returns (delta_energy, is_valid) where delta = new_energy - old_energy.
    """
    n = len(chain)
    old_pos = chain[i]
    if new_pos == old_pos:
        return 0, True
    
    # Check if new_pos is occupied by another residue
    for k, p in enumerate(chain):
        if k != i and tuple(p) == tuple(new_pos):
            return 0, False
    
    # Check unit steps with neighbors
    if i > 0:
        d = sum(abs(new_pos[d] - chain[i-1][d]) for d in range(len(new_pos)))
        if d != 1:
            return 0, False
    if i < n - 1:
        d = sum(abs(new_pos[d] - chain[i+1][d]) for d in range(len(new_pos)))
        if d != 1:
            return 0, False
    
    # Compute energy delta
    # Remove old contacts involving residue i
    delta = 0
    is_h_i = sequence[i] == 'H'
    
    if is_h_i:
        # Contacts lost: H-H pairs where i is involved
        for q in _neighbours(old_pos):
            for k, p in enumerate(chain):
                if k == i:
                    continue
                if tuple(p) == tuple(q):
                    if abs(k - i) > 1 and sequence[k] == 'H':
                        delta += 1  # losing a contact increases energy (less negative)
                    break
    
    # Add new contacts involving residue i at new_pos
    if is_h_i:
        for q in _neighbours(new_pos):
            for k, p in enumerate(chain):
                if k == i:
                    continue
                if tuple(p) == tuple(q):
                    if abs(k - i) > 1 and sequence[k] == 'H':
                        delta -= 1  # gaining a contact decreases energy (more negative)
                    break
    
    return delta, True


def _greedy_grow(sequence, dimension, rng, temp=0.0):
    """
    One chain growth. Uses a scoring function that balances:
    - New H-H contacts (primary)
    - Free space (to avoid trapping)
    - Random noise (for diversity)
    """
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Count new H-H contacts this site would make
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            
            # Count free neighbours (to avoid dead ends)
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            
            # Score: contacts dominate, free space breaks ties, noise for diversity
            score = gain * 10.0 + free * 0.5 + rng.random() * (1.0 + temp)
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick best (with some randomness if temp > 0)
        if temp > 0 and rng.random() < 0.3:
            # Sometimes pick randomly from top options
            options.sort(key=lambda x: -x[0])
            k = min(len(options), 1 + int(rng.expovariate(1.0 / max(1, temp))))
            site = options[rng.randint(0, k - 1)][1]
        else:
            site = max(options)[1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain


def _grow_2d_projection(sequence, rng, temp=0.0):
    """
    Grow a 2D projection of the chain, then lift to 3D.
    This leverages known 2D optimal structures as a strong starting point.
    """
    n = len(sequence)
    # Start at origin in 2D
    chain_2d = [(0, 0)]
    occupied_2d = {(0, 0): 0}
    
    for i in range(1, n):
        options = []
        last = chain_2d[-1]
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            site = (last[0] + dx, last[1] + dy)
            if site in occupied_2d:
                continue
            # Count new H-H contacts
            gain = 0
            if sequence[i] == "H":
                for qx, qy in [(site[0]-1,site[1]),(site[0]+1,site[1]),(site[0],site[1]-1),(site[0],site[1]+1)]:
                    j = occupied_2d.get((qx, qy))
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            
            # Count free neighbours
            free = 0
            for qx, qy in [(site[0]-1,site[1]),(site[0]+1,site[1]),(site[0],site[1]-1),(site[0],site[1]+1)]:
                if (qx, qy) not in occupied_2d:
                    free += 1
            
            score = gain * 10.0 + free * 0.5 + rng.random() * (1.0 + temp)
            options.append((score, site))
        
        if not options:
            return None
        
        if temp > 0 and rng.random() < 0.3:
            options.sort(key=lambda x: -x[0])
            k = min(len(options), 1 + int(rng.expovariate(1.0 / max(1, temp))))
            site = options[rng.randint(0, k - 1)][1]
        else:
            site = max(options)[1]
        
        occupied_2d[site] = i
        chain_2d.append(site)
    
    # Lift to 3D by adding z-coordinate
    # Use a simple heuristic: assign z based on parity to avoid conflicts
    chain_3d = []
    for i, (x, y) in enumerate(chain_2d):
        z = 0
        # Try to find a z that doesn't conflict
        for z_try in range(0, 4):
            conflict = False
            for j in range(i):
                if chain_3d[j][0] == x and chain_3d[j][1] == y and chain_3d[j][2] == z_try:
                    conflict = True
                    break
            if not conflict:
                z = z_try
                break
        chain_3d.append((x, y, z))
    
    # Verify self-avoidance
    seen = set()
    for p in chain_3d:
        if p in seen:
            return None
        seen.add(p)
    
    # Verify unit steps
    for k in range(n - 1):
        d = abs(chain_3d[k][0] - chain_3d[k+1][0]) + abs(chain_3d[k][1] - chain_3d[k+1][1]) + abs(chain_3d[k][2] - chain_3d[k+1][2])
        if d != 1:
            return None
    
    return chain_3d


def _local_search_grid(sequence, chain, dimension, rng, max_moves=500):
    """
    Local search using spatial grid for fast validity checks.
    Uses single-residue moves and 2-opt reversals.
    """
    best_chain = list(chain)
    n = len(best_chain)
    best_energy = _energy(sequence, best_chain)
    grid = _SpatialGrid(best_chain)
    
    for _ in range(max_moves):
        if _get_remaining_time() < 1.0:
            break
        
        improved = False
        
        # Try single-residue moves
        for _attempt in range(5):
            i = rng.randint(1, n - 2)
            current = best_chain[i]
            
            for site in _neighbours(current):
                if grid.contains(site):
                    continue
                
                prev = best_chain[i-1]
                next_ = best_chain[i+1]
                
                d1 = sum(abs(site[d] - prev[d]) for d in range(dimension))
                d2 = sum(abs(site[d] - next_[d]) for d in range(dimension))
                
                if d1 != 1 or d2 != 1:
                    continue
                
                # Valid move! Compute new energy
                new_chain = list(best_chain)
                new_chain[i] = site
                new_energy = _energy(sequence, new_chain)
                
                if new_energy < best_energy:
                    best_chain = new_chain
                    best_energy = new_energy
                    grid = _SpatialGrid(best_chain)
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            # Try 2-opt reversal
            a = rng.randint(0, n - 2)
            b = rng.randint(a + 1, n - 1)
            
            if b - a == 1:
                continue
            
            new_chain = best_chain[:a] + best_chain[a:b+1][::-1] + best_chain[b+1:]
            
            # Check self-avoidance
            seen = set()
            valid = True
            for p in new_chain:
                if p in seen:
                    valid = False
                    break
                seen.add(p)
            
            if not valid:
                continue
            
            # Check unit steps
            for k in range(n - 1):
                diff = sum(abs(new_chain[k][d] - new_chain[k+1][d]) for d in range(dimension))
                if diff != 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            new_energy = _energy(sequence, new_chain)
            if new_energy < best_energy:
                best_chain = new_chain
                best_energy = new_energy
                grid = _SpatialGrid(best_chain)
    
    return best_chain, best_energy


def _polish_grid(sequence, chain, dimension, rng, max_moves=2000):
    """
    Polish pass using spatial grid: try moving individual residues to adjacent free sites.
    """
    best_chain = list(chain)
    n = len(best_chain)
    best_energy = _energy(sequence, best_chain)
    grid = _SpatialGrid(best_chain)
    
    for _ in range(max_moves):
        if _get_remaining_time() < 1.0:
            break
        
        improved = False
        
        # Pick a random interior residue to try moving
        i = rng.randint(1, n - 2)
        current = best_chain[i]
        
        for site in _neighbours(current):
            if grid.contains(site):
                continue
            
            prev = best_chain[i-1]
            next_ = best_chain[i+1]
            
            d1 = sum(abs(site[d] - prev[d]) for d in range(dimension))
            d2 = sum(abs(site[d] - next_[d]) for d in range(dimension))
            
            if d1 != 1 or d2 != 1:
                continue
            
            # Valid move! Compute new energy
            new_chain = list(best_chain)
            new_chain[i] = site
            new_energy = _energy(sequence, new_chain)
            
            if new_energy < best_energy:
                best_chain = new_chain
                best_energy = new_energy
                grid = _SpatialGrid(best_chain)
                improved = True
                break
        
        if not improved:
            # Try a few more random positions
            for _ in range(3):
                i = rng.randint(1, n - 2)
                current = best_chain[i]
                
                for site in _neighbours(current):
                    if grid.contains(site):
                        continue
                    
                    prev = best_chain[i-1]
                    next_ = best_chain[i+1]
                    
                    d1 = sum(abs(site[d] - prev[d]) for d in range(dimension))
                    d2 = sum(abs(site[d] - next_[d]) for d in range(dimension))
                    
                    if d1 != 1 or d2 != 1:
                        continue
                    
                    new_chain = list(best_chain)
                    new_chain[i] = site
                    new_energy = _energy(sequence, new_chain)
                    
                    if new_energy < best_energy:
                        best_chain = new_chain
                        best_energy = new_energy
                        grid = _SpatialGrid(best_chain)
                        improved = True
                        break
                
                if improved:
                    break
    
    return best_chain, best_energy


def _final_polish_grid(sequence, chain, dimension, rng, max_moves=5000):
    """
    Final high-precision polish using spatial grid: sweep all interior residues in order.
    """
    best_chain = list(chain)
    n = len(best_chain)
    best_energy = _energy(sequence, best_chain)
    grid = _SpatialGrid(best_chain)
    
    for _ in range(max_moves):
        if _get_remaining_time() < 2.0:
            break
        
        improved = False
        
        # Sweep through all interior residues in sequence
        for i in range(1, n - 1):
            current = best_chain[i]
            prev = best_chain[i-1]
            next_ = best_chain[i+1]
            
            for site in _neighbours(current):
                if grid.contains(site):
                    continue
                
                d1 = sum(abs(site[d] - prev[d]) for d in range(dimension))
                d2 = sum(abs(site[d] - next_[d]) for d in range(dimension))
                
                if d1 != 1 or d2 != 1:
                    continue
                
                new_chain = list(best_chain)
                new_chain[i] = site
                new_energy = _energy(sequence, new_chain)
                
                if new_energy < best_energy:
                    best_chain = new_chain
                    best_energy = new_energy
                    grid = _SpatialGrid(best_chain)
                    improved = True
                    break  # Restart sweep with updated chain
            
            if improved:
                break
        
        if not improved:
            break  # No improvement found in full sweep
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Best conformation using multiple strategies:
    1. 2D projection growths (fast, strong starting points)
    2. 3D greedy growths with varying temperatures
    3. Local search with spatial grid
    4. Polish passes with spatial grid
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0  # start with 0 (no contacts)
    
    # Phase 1: 2D projection growths (fast, good starting points)
    n_2d_restarts = 50 if n <= 48 else 30 if n <= 80 else 20
    
    for restart in range(n_2d_restarts):
        if _get_remaining_time() < 10.0:
            break
        
        temp = 0.0 if restart < n_2d_restarts // 2 else rng.uniform(0.1, 1.0)
        
        chain = _grow_2d_projection(sequence, rng, temp)
        if chain is None:
            continue
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best_chain = chain
            best_energy = value
    
    # Phase 2: 3D greedy growths
    n_3d_restarts = 100 if n <= 48 else 50 if n <= 80 else 30
    
    for restart in range(n_3d_restarts):
        if _get_remaining_time() < 15.0:
            break
        
        temp = 0.0 if restart < n_3d_restarts // 2 else rng.uniform(0.1, 1.0)
        
        chain = _greedy_grow(sequence, dimension, rng, temp)
        if chain is None:
            continue
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best_chain = chain
            best_energy = value
    
    # Phase 3: Local search on best chain using spatial grid
    if best_chain is not None and _get_remaining_time() > 20.0:
        best_chain, best_energy = _local_search_grid(sequence, best_chain, dimension, rng, max_moves=5000)
    
    # Phase 4: Polish pass with spatial grid
    if best_chain is not None and _get_remaining_time() > 10.0:
        best_chain, best_energy = _polish_grid(sequence, best_chain, dimension, rng)
    
    # Phase 5: Final high-precision polish with spatial grid
    if best_chain is not None and _get_remaining_time() > 5.0:
        best_chain, best_energy = _final_polish_grid(sequence, best_chain, dimension, rng)
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
