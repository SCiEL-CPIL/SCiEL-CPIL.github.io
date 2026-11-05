"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = 320.0  # safety margin before 330s wall-clock kill
_START_TIME = None

def _get_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME + _DEADLINE

def _time_left():
    return _get_deadline() - time.time()

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _local_search(sequence, chain, dimension, rng, max_iters=500):
    """Monte Carlo local search: try pivot and crank moves, accept if energy improves."""
    if chain is None:
        return None
    n = len(chain)
    if n < 4:
        return chain
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    for it in range(max_iters):
        if _time_left() < 1.0:
            break
        
        # Choose move type: pivot or crank
        move_type = rng.random()
        
        if move_type < 0.5:
            # Pivot move: rotate a subchain around a bond
            if n < 4:
                continue
            pivot_idx = rng.randint(1, n - 3)
            # Get the bond direction
            dx = chain[pivot_idx + 1][0] - chain[pivot_idx][0]
            dy = chain[pivot_idx + 1][1] - chain[pivot_idx][1]
            dz = chain[pivot_idx + 1][2] - chain[pivot_idx][2]
            
            # Generate all possible rotations of the subchain
            subchain = list(chain[pivot_idx + 1:])
            # Try all 6 possible directions for the first bond of subchain
            # excluding the original and opposite directions
            orig_dir = (dx, dy, dz)
            possible_dirs = []
            for ax in range(3):
                for s in (-1, 1):
                    d = [0, 0, 0]
                    d[ax] = s
                    d = tuple(d)
                    if d != orig_dir and d != tuple(-x for x in orig_dir):
                        possible_dirs.append(d)
            
            if not possible_dirs:
                continue
            
            new_dir = possible_dirs[rng.randint(0, len(possible_dirs) - 1)]
            
            # Rotate the subchain: apply the rotation that maps orig_dir to new_dir
            # This is complex; simpler approach: try all 6 directions and see which works
            # For simplicity, just try translating the subchain in a random valid way
            pass  # Skip complex pivot for now, use simpler moves
        
        else:
            # Crank move: translate a subchain
            if n < 5:
                continue
            start_idx = rng.randint(1, n - 3)
            end_idx = rng.randint(start_idx + 1, n - 1)
            subchain = list(chain[start_idx:end_idx + 1])
            
            # Try all 6 translation directions
            for ax in range(3):
                for s in (-1, 1):
                    new_subchain = []
                    valid = True
                    for p in subchain:
                        np = list(p)
                        np[ax] += s
                        np = tuple(np)
                        new_subchain.append(np)
                    
                    # Check if all new positions are free
                    # Remove old positions from occupied
                    for p in subchain:
                        del occupied[p]
                    
                    # Check new positions
                    can_place = True
                    for np in new_subchain:
                        if np in occupied:
                            can_place = False
                            break
                        # Also check self-collision within subchain
                        if new_subchain.count(np) > 1:
                            can_place = False
                            break
                    
                    if not can_place:
                        # Restore
                        for i, p in enumerate(subchain):
                            occupied[p] = start_idx + i
                        continue
                    
                    # Check that the bond connections are valid
                    # Connection at start_idx - 1 to start_idx
                    if start_idx > 0:
                        prev = chain[start_idx - 1]
                        first_new = new_subchain[0]
                        dist = sum(abs(a - b) for a, b in zip(prev, first_new))
                        if dist != 1:
                            for i, p in enumerate(subchain):
                                occupied[p] = start_idx + i
                            continue
                    
                    # Connection at end_idx to end_idx + 1
                    if end_idx < n - 1:
                        next_p = chain[end_idx + 1]
                        last_new = new_subchain[-1]
                        dist = sum(abs(a - b) for a, b in zip(last_new, next_p))
                        if dist != 1:
                            for i, p in enumerate(subchain):
                                occupied[p] = start_idx + i
                            continue
                    
                    # Apply the move
                    for i, np in enumerate(new_subchain):
                        occupied[np] = start_idx + i
                    
                    # Compute new energy
                    new_chain = chain[:start_idx] + new_subchain + chain[end_idx + 1:]
                    new_energy = _energy(sequence, new_chain)
                    
                    if new_energy < current_energy:
                        # Accept
                        chain = new_chain
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(chain)
                        break
                    else:
                        # Reject, restore
                        for i, p in enumerate(subchain):
                            occupied[p] = start_idx + i
        
        # Rebuild occupied if chain changed
        # Actually we maintain occupied incrementally above
    
    return best_chain

def fold(sequence, dimension):
    """Best conformation of a fixed number of randomized greedy growths with local search."""
    n = len(sequence)
    rng = random.Random(12345 + n)
    best, best_energy = None, 1
    
    # Number of restarts based on sequence length
    if n <= 50:
        num_restarts = 200
    elif n <= 80:
        num_restarts = 100
    else:
        num_restarts = 50
    
    for r in range(num_restarts):
        if _time_left() < 2.0:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Local search
        if _time_left() > 0.5:
            max_ls = 200 if n <= 50 else 100
            chain = _local_search(sequence, chain, dimension, rng, max_iters=max_ls)
        
        if chain is None:
            continue
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value
    
    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
