import random
import math

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _calc_energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _grow_greedy(sequence, dimension, rng):
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

def fold(sequence, dimension):
    n = len(sequence)
    if n < 2:
        return [[0] * dimension] * n

    rng = random.Random(12345 + len(sequence))
    
    # Initial Conformation
    best_init = None
    best_init_energy = 0
    for _ in range(5):
        c = _grow_greedy(sequence, dimension, rng)
        if c:
            e = _calc_energy(sequence, c)
            if best_init is None or e < best_init_energy:
                best_init = c
                best_init_energy = e
                
    if best_init is None:
        best_init = [(i, 0) if dimension == 2 else (i, 0, 0) for i in range(n)]
        best_init_energy = _calc_energy(sequence, best_init)

    current_chain = list(best_init)
    current_energy = best_init_energy
    best_chain = list(current_chain)
    best_energy = current_energy
    
    occupied = {p: i for i, p in enumerate(current_chain)}
    is_h = [c == 'H' for c in sequence]
    
    max_steps = 100000
    T = 1.0
    cooling = 0.9999
    
    for step in range(max_steps):
        # Randomly choose between Single Residue Move and Pivot Move
        if rng.random() < 0.5:
            # Single Residue Move
            i = rng.randint(0, n - 1)
            axis = rng.randint(0, dimension - 1)
            step_dir = rng.choice([-1, 1])
            
            old_pos = current_chain[i]
            new_pos = list(old_pos)
            new_pos[axis] += step_dir
            new_pos = tuple(new_pos)
            
            if new_pos in occupied:
                continue
                
            delta_e = 0
            if is_h[i]:
                # Remove old contacts
                for q in _neighbours(old_pos):
                    j = occupied.get(q)
                    if j is not None and abs(j - i) > 1 and is_h[j]:
                        delta_e += 1
                # Add new contacts
                for q in _neighbours(new_pos):
                    j = occupied.get(q)
                    if j is not None and abs(j - i) > 1 and is_h[j]:
                        delta_e -= 1
                        
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                del occupied[old_pos]
                occupied[new_pos] = i
                current_chain[i] = new_pos
                current_energy += delta_e
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
        else:
            # Pivot Move
            if n > 2:
                i = rng.randint(1, n - 2)
                head = current_chain[:i+1]
                tail = current_chain[i+1:]
                
                # Remove tail from occupied
                for p in tail:
                    del occupied[p]
                    
                pivot_point = current_chain[i]
                
                # Rotate tail
                # Vectors relative to pivot
                tail_vecs = [(p[0]-pivot_point[0], p[1]-pivot_point[1]) for p in tail]
                
                # Random rotation: 90, 270, 180
                rot_type = rng.choice([1, -1, 2])
                
                new_tail = []
                collision = False
                for v in tail_vecs:
                    if rot_type == 1: # 90 CW
                        nv = (v[1], -v[0])
                    elif rot_type == -1: # 90 CCW
                        nv = (-v[1], v[0])
                    else: # 180
                        nv = (-v[0], -v[0])
                    
                    np_ = (pivot_point[0] + nv[0], pivot_point[1] + nv[1])
                    if np_ in occupied:
                        collision = True
                        break
                    new_tail.append(np_)
                    
                if collision:
                    # Restore tail
                    for p in tail:
                        occupied[p] = current_chain.index(p) # This is O(N), but rare
                    continue
                
                # Calculate Delta E
                # Old contacts: Tail with Head
                old_contacts = 0
                if is_h:
                    for k, p in enumerate(tail):
                        idx = i + 1 + k
                        if is_h[idx]:
                            for q in _neighbours(p):
                                j = occupied.get(q)
                                if j is not None and j <= i and is_h[j]:
                                    old_contacts += 1
                                    
                # New contacts: New Tail with Head
                new_contacts = 0
                for k, p in enumerate(new_tail):
                    idx = i + 1 + k
                    if is_h[idx]:
                        for q in _neighbours(p):
                            j = occupied.get(q)
                            if j is not None and j <= i and is_h[j]:
                                new_contacts += 1
                                
                delta_e = old_contacts - new_contacts # Energy = -Contacts, so Delta E = -(New - Old) = Old - New
                
                if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                    # Accept
                    # Update occupied
                    for p in new_tail:
                        occupied[p] = i + 1 + new_tail.index(p) # This is O(N^2) in worst case!
                    
                    # Update chain
                    current_chain[i+1:] = new_tail
                    current_energy += delta_e
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                else:
                    # Restore tail
                    for p in tail:
                        occupied[p] = i + 1 + tail.index(p)
                        
        T *= cooling

    return [list(p) for p in best_chain]
