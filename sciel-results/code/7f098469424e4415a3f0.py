import random
import time

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

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

def _init_chain(sequence, dimension, rng):
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            # Prefer sites that might lead to contacts? Just random for now
            options.append(site)
        if not options:
            return None
        site = rng.choice(options)
        occupied[site] = i
        chain.append(site)
    return chain

def _try_kink_flip(chain, sequence, occupied, i, rng):
    # i is the index of the middle point p[i]
    # Neighbors in chain are i-1 and i+1
    if i == 0 or i == len(chain) - 1:
        return None, 0, None, None
    
    p_prev = chain[i-1]
    p_next = chain[i+1]
    p_curr = chain[i]
    
    # Find common neighbors of p_prev and p_next
    candidates = []
    for q in _neighbours(p_prev):
        if q == p_curr:
            continue
        # Check if q is neighbor of p_next
        dx = abs(q[0] - p_next[0])
        dy = abs(q[1] - p_next[1])
        if dx + dy == 1:
            candidates.append(q)
            
    if not candidates:
        return None, 0, None, None
        
    # Pick a random candidate
    q = rng.choice(candidates)
    
    # Check if q is occupied
    if q in occupied:
        return None, 0, None, None
        
    # Calculate energy delta
    # Remove p_curr, add q
    # We need to check contacts involving p_curr and q
    
    # Helper to get contact count for a point at index i
    def get_contacts(point, idx):
        count = 0
        if sequence[idx] != 'H':
            return 0
        for n in _neighbours(point):
            j = occupied.get(n)
            if j is not None and abs(j - idx) > 1 and sequence[j] == 'H':
                count += 1
        return count
        
    old_contacts = get_contacts(p_curr, i)
    
    # Temporarily remove p_curr from occupied to check q's contacts against rest
    # But careful: occupied is a dict {point: index}
    # We can't easily "remove" and "add" without modifying state if we want to rollback
    # Let's just calculate new contacts for q assuming p_curr is gone
    
    # For q, contacts are with neighbors in occupied, EXCEPT p_curr (which is being removed)
    # And EXCEPT i-1, i+1 (which are neighbors in chain, but distance 1, so not contacts)
    # Wait, occupied contains all points.
    
    new_contacts = 0
    if sequence[i] == 'H':
        for n in _neighbours(q):
            j = occupied.get(n)
            if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                new_contacts += 1
                
    delta = -(new_contacts - old_contacts)
    
    # If accepted, update chain and occupied
    # We return the new state to let caller decide
    new_chain = chain[:]
    new_chain[i] = q
    new_occupied = occupied.copy()
    del new_occupied[p_curr]
    new_occupied[q] = i
    
    return new_chain, delta, new_occupied, q

def fold(sequence, dimension):
    start_time = time.time()
    deadline = start_time + 11.0 # 11s budget per sequence
    
    best_chain = None
    best_energy = 0 # Energy is negative, so 0 is "worse" than -1
    
    # If sequence is very short, just return a valid walk
    if len(sequence) < 5:
        return [list(p) for p in _init_chain(sequence, dimension, random.Random(1)) or [tuple([i]+[0]*(dimension-1)) for i in range(len(sequence))]]
        
    for restart in range(10): # Try up to 10 restarts
        if time.time() > deadline:
            break
            
        rng = random.Random(restart * 1000 + len(sequence))
        chain = _init_chain(sequence, dimension, rng)
        if chain is None:
            continue
            
        occupied = {p: i for i, p in enumerate(chain)}
        current_energy = _energy(sequence, chain)
        
        if best_chain is None or current_energy < best_energy:
            best_chain = chain[:]
            best_energy = current_energy
            
        # Simulated Annealing
        T = 1.0
        T_min = 0.01
        alpha = 0.995
        steps = 0
        max_steps = 5000
        
        while steps < max_steps:
            if time.time() > deadline:
                break
                
            # Pick move
            # 80% Kink, 20% Small Reverse
            if rng.random() < 0.8:
                i = rng.randint(1, len(chain) - 2)
                new_chain, delta, new_occupied, _ = _try_kink_flip(chain, sequence, occupied, i, rng)
                if new_chain is None:
                    continue
            else:
                # Segment Reverse
                length = rng.randint(2, 4)
                a = rng.randint(0, len(chain) - length - 1)
                b = a + length
                
                # Check if reversal is valid
                # Segment is chain[a:b+1]
                # Reversed: chain[b], ..., chain[a]
                # New endpoints: chain[b] connects to chain[a-1] (if a>0) and chain[a] connects to chain[b+1] (if b<N-1)
                
                valid = True
                # Check connectivity
                if a > 0:
                    p_a = chain[a]
                    p_b = chain[b]
                    p_prev = chain[a-1]
                    if abs(p_b[0]-p_prev[0]) + abs(p_b[1]-p_prev[1]) != 1:
                        valid = False
                if valid and b < len(chain) - 1:
                    p_a = chain[a]
                    p_b = chain[b]
                    p_next = chain[b+1]
                    if abs(p_a[0]-p_next[0]) + abs(p_a[1]-p_next[1]) != 1:
                        valid = False
                        
                if not valid:
                    continue
                    
                # Check self-avoidance of reversed segment
                # The segment points are the same, just order changed.
                # So self-avoidance within segment is preserved.
                # But we must check if the reversed segment intersects the rest of the chain?
                # No, the set of points is the same. So no new collisions with rest of chain.
                # Wait, the points are the same! So self-avoidance is automatically preserved.
                # We only need to check connectivity at the boundaries.
                
                # So the move is always valid if connectivity holds.
                # Calculate energy delta
                # Since the set of points is the same, and the sequence is the same,
                # the contacts are the same?
                # NO. Contacts depend on indices.
                # If we reverse the segment, the indices of the points in the segment change relative to the rest?
                # No, the indices $i$ are fixed positions in the chain.
                # The points $p_i$ change for $i \in [a, b]$.
                # $p_i$ becomes $p_{a+b-i}$.
                # So the set of points is the same, but the mapping from index to point is permuted.
                # This changes which pairs of indices are adjacent in space.
                # So energy changes.
                
                # We must recalculate energy or do incremental.
                # Given complexity, let's just recalculate energy for this move.
                new_chain = chain[:]
                new_chain[a:b+1] = reversed(new_chain[a:b+1])
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - current_energy
                new_occupied = occupied # Same points, so same occupied set
                # But we need to update chain in state if accepted
                
            # Metropolis
            if delta < 0 or rng.random() < pow(2.71828, -delta / T):
                chain = new_chain
                occupied = new_occupied
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
            else:
                # If rejected, we need to keep old chain/occupied
                # The variables new_chain etc were local
                pass
                
            T *= alpha
            if T < T_min:
                T = T_min
            steps += 1
            
    if best_chain is None:
        # Fallback
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        
    return [list(p) for p in best_chain]
