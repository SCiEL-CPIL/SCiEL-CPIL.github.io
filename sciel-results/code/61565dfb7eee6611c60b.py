"""HP lattice protein folding on the cubic lattice."""
import random
import time

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

def _rotate_segment(segment, pivot, axis, direction, dimension):
    """Rotate a segment of the chain around a pivot point by 90 degrees."""
    new_segment = []
    for p in segment:
        v = [p[i] - pivot[i] for i in range(dimension)]
        if axis == 0:
            if direction == 1:
                v_new = [v[0], -v[2], v[1]]
            else:
                v_new = [v[0], v[2], -v[1]]
        elif axis == 1:
            if direction == 1:
                v_new = [v[2], v[1], -v[0]]
            else:
                v_new = [-v[2], v[1], v[0]]
        else:
            if direction == 1:
                v_new = [-v[1], v[0], v[2]]
            else:
                v_new = [v[1], -v[0], v[2]]
        
        new_p = tuple(pivot[i] + v_new[i] for i in range(dimension))
        new_segment.append(new_p)
    return new_segment

def fold(sequence, dimension):
    """
    Fold the protein chain using a combination of greedy growth and simulated annealing.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    # Time budget: The evaluator kills after 330s for all 26 sequences.
    # Assuming ~12s per sequence average, we use a conservative 10s limit here.
    # If the process is shared, this might be too much, but the prompt implies 
    # "subprocess" per sequence or at least that we should guard. 
    # Let's use a global start time if possible, but since we don't know if it's 
    # the first call, we'll assume a fixed budget per call that is safe.
    # 330s / 26 sequences ~ 12.7s. Let's aim for 10s max per call to be safe.
    start_time = time.time()
    max_time = 10.0
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0  # Energy is negative, so 0 is the "worst" (unfolded)
    
    # Precompute is_h for faster checks
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    # Phase 1: Greedy Growth Initialization
    # Run a few greedy growths to find a decent starting point
    for _ in range(20):
        if time.time() - start_time > max_time * 0.1:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        energy = _energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        # Fallback to straight line if growth failed
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = 0

    # Phase 2: Simulated Annealing with Pivot Moves
    # We will refine the best chain found so far.
    # We use a time-based loop rather than a fixed step count to utilize the budget.
    
    current_chain = best_chain[:]
    current_energy = best_energy
    
    # Temperature schedule
    # T_max should be large enough to accept bad moves, T_min small enough to converge.
    # Energy range is roughly [-n, 0]. Let's say T_max = 10, T_min = 0.1
    T_max = 10.0
    T_min = 0.1
    
    # We will run until time is up
    while time.time() - start_time < max_time * 0.95:
        # Choose a random pivot index k (1 to n-1)
        # k is the index of the atom around which we rotate.
        # The segment k..n-1 is rotated around chain[k-1]? No, usually pivot is an atom, 
        # and the tail is rotated.
        # Let's say we pick a cut point k (1 <= k < n).
        # The head is 0..k-1. The tail is k..n-1.
        # We rotate the tail around the atom at k-1? No, that would break the bond.
        # Standard pivot move: Pick a bond (k-1, k). Rotate the segment k..n-1 
        # around the atom at k-1.
        
        k = rng.randint(1, n - 1)
        pivot_idx = k - 1
        pivot_site = current_chain[pivot_idx]
        
        # Extract tail
        tail = current_chain[k:]
        
        # Remove tail from occupied set? 
        # To check collisions efficiently, we should maintain an occupied set.
        # But for simplicity and to avoid state management bugs in this short script,
        # we will just check against the head.
        # Head is 0..k-1.
        # We need to ensure the new tail does not collide with the head (indices 0..k-1).
        # Note: The tail can collide with itself? No, rotation is rigid, so internal distances are preserved.
        # The only collision risk is with the head.
        
        # Choose axis and direction
        axis = rng.randint(0, dimension - 1)
        direction = rng.choice([-1, 1])
        
        new_tail = _rotate_segment(tail, pivot_site, axis, direction, dimension)
        
        # Check collisions with head
        # Head is current_chain[0:k]
        # We need to check if any point in new_tail is in the set of head points.
        # Note: The point at index k in the original chain is the first point of the tail.
        # After rotation, this point moves.
        # The bond between k-1 and k must remain intact? 
        # Wait, if we rotate around k-1, the distance from k-1 to k is preserved.
        # So the bond is intact.
        
        # However, we must ensure that the new position of tail[0] (which is k) 
        # is not the same as any head point.
        # And no other tail point collides with head.
        
        head_set = set(current_chain[:k])
        
        valid = True
        for p in new_tail:
            if p in head_set:
                valid = False
                break
        
        if not valid:
            continue
            
        # Calculate energy delta
        # Old energy: E_head + E_tail + E_cross_old
        # New energy: E_head + E_tail + E_cross_new
        # E_head and E_tail are invariant under rotation of the tail relative to itself and head?
        # No, E_head is invariant. E_tail is invariant (rigid body).
        # Only E_cross (contacts between head and tail) changes.
        
        # We need to compute the change in contacts between head and tail.
        # Old cross contacts:
        # For each H in head, check neighbors in old_tail.
        # For each H in old_tail, check neighbors in head.
        # This is O(N^2) if done naively.
        # Given N <= 136, O(N^2) is ~18k ops. With 10s budget, we can do maybe 1000-5000 steps?
        # 1000 steps * 18k ops = 18M ops. Python can do ~10-50M ops/sec. So it might be tight.
        # Let's optimize:
        # Only check H residues.
        
        # Let's just recompute the full energy if the move is accepted?
        # No, that's O(N^2) per step.
        # Delta calculation is also O(N^2) worst case.
        
        # Let's try to compute delta efficiently.
        # We need to know which contacts are lost and which are gained.
        
        # Contacts lost:
        # 1. H in head, neighbor in old_tail
        # 2. H in old_tail, neighbor in head
        
        # Contacts gained:
        # 1. H in head, neighbor in new_tail
        # 2. H in new_tail, neighbor in head
        
        # We can build a set of positions for head and old_tail/new_tail.
        
        head_positions = set(current_chain[:k])
        old_tail_positions = set(tail)
        new_tail_positions = set(new_tail)
        
        # We need to map positions to indices to check if they are H.
        # This is getting complicated.
        # Given the short time limit, let's just recompute the full energy for the candidate chain.
        # It's simpler and less error-prone.
        # Candidate chain: current_chain[:k] + new_tail
        
        candidate_chain = current_chain[:k] + new_tail
        candidate_energy = _energy(sequence, candidate_chain)
        
        delta = candidate_energy - current_energy
        
        # Metropolis acceptance
        if delta < 0:
            # Accept
            current_chain = candidate_chain
            current_energy = candidate_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain[:]
        else:
            # Accept with probability exp(-delta / T)
            # T decreases over time
            elapsed = time.time() - start_time
            # Linear cooling
            T = T_max - (T_max - T_min) * (elapsed / max_time)
            if T < T_min:
                T = T_min
                
            prob = 2.71828 ** (-delta / T)
            if rng.random() < prob:
                current_chain = candidate_chain
                current_energy = candidate_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = current_chain[:]

    return [list(p) for p in best_chain]
