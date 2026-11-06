"""Improved HP lattice protein folding with simulated annealing and time management."""
# EVOLVE-BLOCK-START
import random
import time
import sys

# Global time tracking
_START_TIME = None
_WALL_CLOCK_LIMIT = 200.0  # Safety margin before 210s limit


def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_remaining():
    _init_time()
    return _WALL_CLOCK_LIMIT - (time.time() - _START_TIME)


def _neighbours(point):
    """Generate all 4 neighbours of a point on square lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _is_adjacent(i, j):
    """Check if positions i and j in chain are adjacent (|i-j| == 1)."""
    return abs(i - j) == 1


def _compute_energy(sequence, chain):
    """Compute the energy of a conformation. Energy = -number of H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and not _is_adjacent(i, j) and sequence[j] == 'H':
                    # Count each pair once (only when i < j)
                    if i < j:
                        contacts += 1
    return -contacts


def _grow_random(sequence, dimension, rng):
    """Grow a random self-avoiding walk."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start}
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site not in occupied:
                options.append(site)
        
        if not options:
            return None
        
        site = rng.choice(options)
        chain.append(site)
        occupied.add(site)
    
    return chain


def _grow_greedy(sequence, dimension, rng, alpha=0.7):
    """Grow a chain with a mix of greedy (contact-maximizing) and random moves."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start}
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site not in occupied:
                # Score: prefer sites that create H-H contacts
                gain = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        if q in occupied:
                            # Check if the occupied position has an H residue
                            # We need to find which index is at q
                            # This is O(n) but acceptable for small sequences
                            pass
                
                # Count free neighbours for flexibility
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                
                # Calculate contact gain more efficiently
                contact_gain = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        if q in occupied:
                            # Find index at q
                            for idx, pos in enumerate(chain):
                                if pos == q and idx < i - 1 and sequence[idx] == 'H':
                                    contact_gain += 1
                                    break
                
                score = contact_gain * 2.0 + free * 0.1 + rng.random() * alpha
                options.append((score, site))
        
        if not options:
            return None
        
        # Pick best with some randomness
        options.sort(key=lambda x: x[0], reverse=True)
        # Pick from top few with probability
        k = min(3, len(options))
        site = options[rng.randint(0, k - 1)][1]
        
        chain.append(site)
        occupied.add(site)
    
    return chain


def _local_search(sequence, chain, max_iters=500, rng=None):
    """
    Local search via random moves (pivot, flip, shift).
    Returns improved chain and energy.
    """
    if rng is None:
        rng = random.Random()
    
    n = len(chain)
    current_chain = list(chain)
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Build position lookup
    def build_lookup(chain):
        return {p: i for i, p in enumerate(chain)}
    
    def is_self_avoiding(chain):
        return len(set(chain)) == len(chain)
    
    def apply_move(chain, move_type, params):
        """Apply a move and return new chain if valid, else None."""
        new_chain = list(chain)
        n = len(new_chain)
        
        if move_type == 'pivot':
            # Pick a pivot point and rotate the tail
            if n < 3:
                return None
            pivot_idx = params[0]
            if pivot_idx < 1 or pivot_idx >= n - 1:
                return None
            
            # The tail is from pivot_idx+1 to end
            tail = new_chain[pivot_idx + 1:]
            head = new_chain[:pivot_idx + 1]
            
            # Try rotating tail by 90, 180, 270 degrees around pivot
            pivot = new_chain[pivot_idx]
            for angle in range(4):
                rotated = []
                for p in tail:
                    # Rotate around pivot
                    dx = p[0] - pivot[0]
                    dy = p[1] - pivot[1]
                    if angle == 0:
                        new_p = (pivot[0] + dx, pivot[1] + dy)
                    elif angle == 1:
                        new_p = (pivot[0] - dy, pivot[1] + dx)
                    elif angle == 2:
                        new_p = (pivot[0] - dx, pivot[1] - dy)
                    else:
                        new_p = (pivot[0] + dy, pivot[1] - dx)
                    rotated.append(new_p)
                
                candidate = head + rotated
                if is_self_avoiding(candidate):
                    return candidate
        
        elif move_type == 'flip':
            # Reverse a segment
            if n < 3:
                return None
            i, j = params
            if i < 0 or j >= n or i >= j:
                return None
            
            # Check if reversing segment i..j creates self-intersection
            segment = new_chain[i:j+1]
            rest = new_chain[:i] + new_chain[j+1:]
            candidate = rest + list(reversed(segment))
            if is_self_avoiding(candidate):
                return candidate
        
        elif move_type == 'shift':
            # Move a single residue to an adjacent free site
            idx = params[0]
            if idx < 1 or idx >= n - 1:
                return None
            
            p = new_chain[idx]
            for q in _neighbours(p):
                # Check if q is free (not occupied by other residues)
                if q in new_chain and new_chain.index(q) != idx:
                    continue
                # Also check that moving doesn't break connectivity
                # The residue at idx connects to idx-1 and idx+1
                # After move, it must still be adjacent to both
                prev_p = new_chain[idx - 1]
                next_p = new_chain[idx + 1]
                
                # Check distance
                if abs(q[0] - prev_p[0]) + abs(q[1] - prev_p[1]) != 1:
                    continue
                if abs(q[0] - next_p[0]) + abs(q[1] - next_p[1]) != 1:
                    continue
                
                new_chain[idx] = q
                if is_self_avoiding(new_chain):
                    return list(new_chain)
        
        return None
    
    for it in range(max_iters):
        # Choose move type
        move_type = rng.choice(['pivot', 'flip', 'shift'])
        
        if move_type == 'pivot':
            params = [rng.randint(1, n - 2)]
        elif move_type == 'flip':
            i = rng.randint(0, n - 2)
            j = rng.randint(i, n - 1)
            params = [i, j]
        else:
            params = [rng.randint(1, n - 2)]
        
        new_chain = apply_move(current_chain, move_type, params)
        if new_chain is None:
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        
        # Accept if better or with some probability (simple acceptance for now)
        if new_energy < current_energy:
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold a protein sequence into a low-energy conformation.
    Uses multiple strategies: greedy growth, random growth, and local search.
    """
    _init_time()
    rng = random.Random(12345 + len(sequence))
    
    n = len(sequence)
    
    # Track best solution found
    best_chain = None
    best_energy = 0  # Start with 0 (unfolded)
    
    # Generate initial solutions using different methods
    initial_chains = []
    
    # Method 1: Greedy growth with varying alpha
    for alpha in [0.3, 0.5, 0.7, 1.0]:
        if _time_remaining() < 5:
            break
        chain = _grow_greedy(sequence, dimension, rng, alpha=alpha)
        if chain is not None:
            initial_chains.append(chain)
    
    # Method 2: Random growth
    for _ in range(5):
        if _time_remaining() < 5:
            break
        chain = _grow_random(sequence, dimension, rng)
        if chain is not None:
            initial_chains.append(chain)
    
    # Evaluate all initial chains
    for chain in initial_chains:
        if chain is None:
            continue
        energy = _compute_energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    # If no valid chain found, use a straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = 0
    
    # Local search improvement
    if _time_remaining() > 10:
        # Run local search on best chain
        improved_chain, improved_energy = _local_search(
            sequence, best_chain, max_iters=200, rng=rng
        )
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    # If time remains, try more iterations with different seeds
    while _time_remaining() > 15:
        # Generate a new random chain and improve it
        chain = _grow_random(sequence, dimension, rng)
        if chain is None:
            continue
        
        energy = _compute_energy(sequence, chain)
        
        # Quick local search
        if _time_remaining() > 10:
            chain, energy = _local_search(sequence, chain, max_iters=100, rng=rng)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    return [list(p) for p in best_chain]


# EVOLVE-BLOCK-END
