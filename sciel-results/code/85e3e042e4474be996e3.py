"""HP lattice protein folding using E2T Monte Carlo moves with simulated annealing."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = time.time() + 175.0


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    n = len(chain)
    if n < 2:
        return 0
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _is_valid_chain(chain):
    """Check if chain is self-avoiding with unit steps."""
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            prev = chain[i-1]
            if sum(abs(a-b) for a, b in zip(p, prev)) != 1:
                return False
    return True


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


def _e2t_move(chain, sequence, rng):
    """
    Perform an E2T (End-to-Tail) move: pick a random internal segment,
    and try to reattach the tail to the head side.
    Returns (new_chain, delta_energy) or (None, 0) if move not possible.
    """
    n = len(chain)
    if n < 4:
        return None, 0
    
    # Pick a random cut point
    cut = rng.randint(2, n - 2)
    
    head = chain[:cut]
    tail = chain[cut:]
    
    # Try to attach tail to the free end of head
    # The tail must start adjacent to the last position of head
    # and not collide with head
    
    # We can try rotating the tail
    # For simplicity, try a few random rotations
    best_new_chain = None
    best_delta = 0
    
    # Compute energy of current chain for delta
    # Actually, let's just try to find a valid attachment
    
    # The tail's first residue is at tail[0]. We need to place tail such that
    # tail[0] is adjacent to head[-1], and the rest of tail doesn't collide with head.
    
    # Try all 4 rotations of the tail
    for rotation in range(4):
        # Rotate tail by 90 degrees clockwise
        if rotation == 0:
            rotated_tail = list(tail)
        else:
            # Rotate 90 degrees clockwise: (x, y) -> (y, -x)
            rotated_tail = [(p[1], -p[0]) for p in tail]
        
        # Now we need to translate rotated_tail so that rotated_tail[0] is adjacent to head[-1]
        # and the whole rotated_tail doesn't collide with head.
        
        # The relative positions in rotated_tail are fixed. We need to find a translation
        # such that rotated_tail[0] + translation is adjacent to head[-1].
        
        # There are 4 possible adjacent positions to head[-1]
        head_end = head[-1]
        for adj in _neighbours(head_end):
            # translation = adj - rotated_tail[0]
            tx = adj[0] - rotated_tail[0][0]
            ty = adj[1] - rotated_tail[0][1]
            
            new_tail = [(p[0] + tx, p[1] + ty) for p in rotated_tail]
            
            # Check if new_tail collides with head
            head_set = set(head)
            valid = True
            for p in new_tail:
                if p in head_set:
                    valid = False
                    break
            
            if valid:
                new_chain = head + new_tail
                # Check self-avoidance within new_tail (should be preserved by rotation/translation)
                # and between new_tail and head (already checked)
                # Also check that the connection between head[-1] and new_tail[0] is valid
                # (it is, by construction)
                
                # Check unit steps in new_tail
                for i in range(1, len(new_tail)):
                    if sum(abs(a-b) for a, b in zip(new_tail[i], new_tail[i-1])) != 1:
                        valid = False
                        break
                
                if valid:
                    # Check self-avoidance of entire new_chain
                    if len(set(new_chain)) == len(new_chain):
                        best_new_chain = new_chain
                        break
        
        if best_new_chain is not None:
            break
    
    if best_new_chain is None:
        return None, 0
    
    # Compute delta energy
    old_energy = _compute_energy(sequence, chain)
    new_energy = _compute_energy(sequence, best_new_chain)
    delta = new_energy - old_energy
    
    return best_new_chain, delta


def _pivot_move(chain, sequence, rng):
    """
    Perform a pivot move: pick a random internal point, rotate the tail around that point.
    Returns (new_chain, delta_energy) or (None, 0) if move not possible.
    """
    n = len(chain)
    if n < 4:
        return None, 0
    
    # Pick a random pivot point
    pivot_idx = rng.randint(1, n - 2)
    pivot = chain[pivot_idx]
    
    head = chain[:pivot_idx + 1]
    tail = chain[pivot_idx + 1:]
    
    # Try all 4 rotations of the tail around the pivot
    for rotation in range(4):
        if rotation == 0:
            rotated_tail = list(tail)
        else:
            # Rotate 90 degrees clockwise around origin, then translate back
            rotated_tail = [(p[1] - pivot[1], -(p[0] - pivot[0])) for p in tail]
            rotated_tail = [(p[0] + pivot[0], p[1] + pivot[1]) for p in rotated_tail]
        
        # Check if rotated_tail collides with head
        head_set = set(head)
        valid = True
        for p in rotated_tail:
            if p in head_set:
                valid = False
                break
        
        if valid:
            new_chain = head + rotated_tail
            
            # Check self-avoidance
            if len(set(new_chain)) == len(new_chain):
                # Check unit steps
                ok = True
                for i in range(1, len(new_chain)):
                    if sum(abs(a-b) for a, b in zip(new_chain[i], new_chain[i-1])) != 1:
                        ok = False
                        break
                
                if ok:
                    old_energy = _compute_energy(sequence, chain)
                    new_energy = _compute_energy(sequence, new_chain)
                    delta = new_energy - old_energy
                    return new_chain, delta
    
    return None, 0


def _sa_fold(sequence, dimension, rng, time_budget):
    """Simulated annealing with E2T and pivot moves."""
    n = len(sequence)
    
    # Initialize with greedy growth
    chain = _grow(sequence, dimension, rng)
    if chain is None:
        # Fallback to a simple spiral or line
        chain = []
        x, y = 0, 0
        dx, dy = 1, 0
        for i in range(n):
            chain.append((x, y))
            x += dx
            y += dy
            # Simple zigzag
            if i % 4 == 3:
                dx, dy = -dy, dx
    
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    start_temp = 1.0
    end_temp = 0.01
    steps = 5000
    
    start_time = time.time()
    
    for step in range(steps):
        if time.time() > _DEADLINE - 5:
            break
        
        # Choose move type
        if rng.random() < 0.5:
            new_chain, delta = _e2t_move(chain, sequence, rng)
        else:
            new_chain, delta = _pivot_move(chain, sequence, rng)
        
        if new_chain is None:
            continue
        
        # Acceptance criterion
        if delta < 0:
            # Improvement
            chain = new_chain
            current_energy += delta
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        else:
            # Deterioration: accept with probability exp(-delta / T)
            T = start_temp * (end_temp / start_temp) ** (step / steps)
            if rng.random() < pow(2.718281828, -delta / T):
                chain = new_chain
                current_energy += delta
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation found by simulated annealing with E2T and pivot moves."""
    n = len(sequence)
    
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(12345 + len(sequence))
    
    # Run multiple SA restarts
    best_chain = None
    best_energy = 1
    
    # Time budget per sequence
    time_per_seq = 120.0 / 10  # 12 seconds per sequence
    
    start_time = time.time()
    
    # First, try greedy growth for a good initial solution
    greedy_chain = _grow(sequence, dimension, rng)
    if greedy_chain is not None:
        greedy_energy = _compute_energy(sequence, greedy_chain)
        if greedy_energy < best_energy:
            best_energy = greedy_energy
            best_chain = greedy_chain
    
    # Then run SA
    while time.time() - start_time < time_per_seq and time.time() < _DEADLINE - 10:
        sa_chain, sa_energy = _sa_fold(sequence, dimension, rng, time_per_seq)
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_chain = sa_chain
    
    if best_chain is None:
        # Fallback to a straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
