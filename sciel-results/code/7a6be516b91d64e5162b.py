"""Improved HP lattice folding: Fast growth + Local Search with Time Guard."""
# EVOLVE-BLOCK-START
import random
import time

_T0 = time.time()
_DEADLINE = 195.0  # 210s limit, 15s safety margin

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _energy(seq, chain):
    """Calculate energy of a conformation. Energy = - (number of H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if seq[i] == 'H':
            x, y = p
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                j = where.get((x + dx, y + dy))
                if j is not None and j > i + 1 and seq[j] == 'H':
                    contacts += 1
    return -contacts

def _grow(seq, dim, rng):
    """Randomized greedy growth. Returns a list of tuples or None if stuck."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(seq)):
        last = chain[-1]
        x, y = last
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        rng.shuffle(neighbors)
        
        valid_sites = []
        for site in neighbors:
            if site not in occupied:
                valid_sites.append(site)
        
        if not valid_sites:
            return None
            
        best_site = None
        best_score = -1
        for site in valid_sites:
            score = 0
            if seq[i] == 'H':
                sx, sy = site
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    j = occupied.get((sx+dx, sy+dy))
                    if j is not None and j < i-1 and seq[j] == 'H':
                        score += 1
            score += rng.random() * 0.1
            if score > best_score:
                best_score = score
                best_site = site
                
        if best_site is None:
            return None
            
        occupied[best_site] = i
        chain.append(best_site)
        
    return chain

def _local_search(seq, chain, time_limit):
    """Metropolis local search to improve energy."""
    n = len(seq)
    if n < 5:
        return chain
    
    best_chain = list(chain)
    best_energy = _energy(seq, best_chain)
    current_chain = list(best_chain)
    current_energy = best_energy
    
    # Precompute occupied set for collision checks
    occupied = set(current_chain)
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.99
    
    rng = random.Random()
    start_time = time.time()
    
    while (time.time() - start_time) < time_limit and _remaining() > 5.0:
        if T < T_min:
            break
            
        move_type = rng.random()
        
        if move_type < 0.6:
            # Single Residue Move
            if n < 3: break
            k = rng.randint(1, n - 2)
            dx, dy = [(1,0), (-1,0), (0,1), (0,-1)][rng.randint(0, 3)]
            
            old_p = current_chain[k]
            new_p = (old_p[0] + dx, old_p[1] + dy)
            
            # Check connectivity
            prev_p = current_chain[k-1]
            next_p = current_chain[k+1]
            
            if abs(prev_p[0] - new_p[0]) + abs(prev_p[1] - new_p[1]) != 1:
                continue
            if abs(next_p[0] - new_p[0]) + abs(next_p[1] - new_p[1]) != 1:
                continue
            
            # Check collision
            # new_p must not be in occupied, except old_p
            if new_p != old_p and new_p in occupied:
                continue
            
            # Apply move
            new_chain = current_chain.copy()
            new_chain[k] = new_p
            new_energy = _energy(seq, new_chain)
            
            if new_energy < current_energy:
                current_chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_chain = list(current_chain)
                    best_energy = current_energy
                # Update occupied
                occupied.remove(old_p)
                occupied.add(new_p)
            else:
                delta = new_energy - current_energy
                if delta < 0:
                    current_chain = new_chain
                    current_energy = new_energy
                    occupied.remove(old_p)
                    occupied.add(new_p)
                else:
                    if rng.random() < pow(2.71828, -delta / T):
                        current_chain = new_chain
                        current_energy = new_energy
                        occupied.remove(old_p)
                        occupied.add(new_p)
                        
        elif move_type < 0.9:
            # Sliding Move
            if n < 4: break
            i = rng.randint(0, n - 3)
            j = rng.randint(i + 2, n - 1)
            dx, dy = [(1,0), (-1,0), (0,1), (0,-1)][rng.randint(0, 3)]
            
            # Check connectivity
            if i > 0:
                prev_p = current_chain[i-1]
                new_start = (current_chain[i][0] + dx, current_chain[i][1] + dy)
                if abs(prev_p[0] - new_start[0]) + abs(prev_p[1] - new_start[1]) != 1:
                    continue
            
            if j < n - 1:
                next_p = current_chain[j+1]
                new_end = (current_chain[j][0] + dx, current_chain[j][1] + dy)
                if abs(next_p[0] - new_end[0]) + abs(next_p[1] - new_end[1]) != 1:
                    continue
            
            # Check collisions
            # Create a set of points to exclude (the segment itself)
            segment_points = set(current_chain[i:j+1])
            
            valid = True
            for k in range(i, j+1):
                px, py = current_chain[k]
                new_p = (px + dx, py + dy)
                # If new_p is in occupied AND not in segment_points, it's a collision
                if new_p in occupied and new_p not in segment_points:
                    valid = False
                    break
            
            if not valid:
                continue
            
            new_chain = current_chain[:i] + [(x+dx, y+dy) for x, y in current_chain[i:j+1]] + current_chain[j+1:]
            new_energy = _energy(seq, new_chain)
            
            if new_energy < current_energy:
                current_chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_chain = list(current_chain)
                    best_energy = current_energy
                # Update occupied
                for p in segment_points:
                    occupied.remove(p)
                for p in new_chain[i:j+1]:
                    occupied.add(p)
            else:
                delta = new_energy - current_energy
                if delta < 0:
                    current_chain = new_chain
                    current_energy = new_energy
                    for p in segment_points:
                        occupied.remove(p)
                    for p in new_chain[i:j+1]:
                        occupied.add(p)
                else:
                    if rng.random() < pow(2.71828, -delta / T):
                        current_chain = new_chain
                        current_energy = new_energy
                        for p in segment_points:
                            occupied.remove(p)
                        for p in new_chain[i:j+1]:
                            occupied.add(p)
                            
        else:
            # Reversal (only if ends are neighbors)
            if n < 4: break
            i = rng.randint(0, n - 3)
            j = rng.randint(i + 2, n - 1)
            
            p_i = current_chain[i]
            p_j = current_chain[j]
            
            # Check if p_i and p_j are neighbors
            if abs(p_i[0] - p_j[0]) + abs(p_i[1] - p_j[1]) != 1:
                continue
            
            # Reversal is always self-avoiding (same set of points)
            new_chain = current_chain[:i] + list(reversed(current_chain[i:j+1])) + current_chain[j+1:]
            new_energy = _energy(seq, new_chain)
            
            if new_energy < current_energy:
                current_chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_chain = list(current_chain)
                    best_energy = current_energy
            else:
                delta = new_energy - current_energy
                if delta < 0:
                    current_chain = new_chain
                    current_energy = new_energy
                else:
                    if rng.random() < pow(2.71828, -delta / T):
                        current_chain = new_chain
                        current_energy = new_energy

        T *= cooling_rate
        
    return best_chain

def fold(sequence, dimension):
    """Best conformation using growth + local search within time budget."""
    n = len(sequence)
    if n == 0:
        return []
    
    best_chain = None
    best_energy = 0
    
    rng = random.Random(12345 + len(sequence))
    
    baseline_chain = _grow(sequence, dimension, rng)
    if baseline_chain:
        best_chain = baseline_chain
        best_energy = _energy(sequence, best_chain)
    else:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
        
    if _remaining() > 1.0:
        t_limit = min(5.0, _remaining() * 0.5)
        if t_limit > 0.1:
            improved = _local_search(sequence, best_chain, t_limit)
            e = _energy(sequence, improved)
            if e < best_energy:
                best_chain = improved
                best_energy = e
        
        while _remaining() > 2.0:
            new_chain = _grow(sequence, dimension, rng)
            if not new_chain:
                continue
            
            e_new = _energy(sequence, new_chain)
            if e_new < best_energy:
                best_chain = new_chain
                best_energy = e_new
                
            if _remaining() > 2.0:
                improved = _local_search(sequence, new_chain, 1.0)
                e_imp = _energy(sequence, improved)
                if e_imp < best_energy:
                    best_chain = improved
                    best_energy = e_imp

    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
