"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline guard: must finish well before 330s wall-clock.
_TOTAL_START = time.time()
_SAFETY_MARGIN = 30.0  # seconds to leave before hard kill

def _remaining_time():
    """Return remaining seconds before our safety deadline."""
    return _TOTAL_START + 300.0 - _SAFETY_MARGIN - time.time()

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
            # Heuristic: Encourage compactness by penalizing distance from origin slightly?
            # Or just keep the free space bonus.
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Add a small random factor to break ties
            score = gain + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        # Pick the best option
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _try_single_move(sequence, chain, occupied, i, rng):
    """Try to move residue i to a neighbor site. Returns (new_site, delta) or None."""
    n = len(chain)
    old_site = chain[i]
    best_site = None
    best_delta = 0
    
    for new_site in _neighbours(old_site):
        # Check if new_site is free
        if new_site in occupied:
            continue
        
        # Check adjacency with prev
        if i > 0:
            prev = chain[i-1]
            if abs(new_site[0]-prev[0]) + abs(new_site[1]-prev[1]) + abs(new_site[2]-prev[2]) != 1:
                continue
        # Check adjacency with next
        if i < n - 1:
            nxt = chain[i+1]
            if abs(new_site[0]-nxt[0]) + abs(new_site[1]-nxt[1]) + abs(new_site[2]-nxt[2]) != 1:
                continue
        
        # Calculate delta energy
        delta = 0
        if sequence[i] == "H":
            # Lose contacts at old_site
            for q in _neighbours(old_site):
                j = occupied.get(q)
                if j is not None and j != i and sequence[j] == "H" and abs(i-j) > 1:
                    delta += 1
            # Gain contacts at new_site
            for q in _neighbours(new_site):
                j = occupied.get(q)
                if j is not None and j != i and sequence[j] == "H" and abs(i-j) > 1:
                    delta -= 1
        
        if best_site is None or delta < best_delta:
            best_site = new_site
            best_delta = delta
    
    if best_site is not None:
        return best_site, best_delta
    return None

def _try_two_move(sequence, chain, occupied, i, rng):
    """Try to move residues i and i+1. Returns (new_i, new_ip1, delta) or None."""
    n = len(chain)
    if i + 1 >= n:
        return None
        
    old_i = chain[i]
    old_ip1 = chain[i+1]
    
    for new_i in _neighbours(old_i):
        if new_i in occupied:
            occ_idx = occupied[new_i]
            if occ_idx not in [i, i+1]:
                continue
            if new_i == old_i:
                continue
        
        if i > 0:
            prev = chain[i-1]
            if abs(new_i[0]-prev[0]) + abs(new_i[1]-prev[1]) + abs(new_i[2]-prev[2]) != 1:
                continue
        
        for new_ip1 in _neighbours(new_i):
            if new_ip1 in occupied:
                occ_idx = occupied[new_ip1]
                if occ_idx not in [i, i+1]:
                    continue
                if new_ip1 == old_ip1 and new_i == old_i:
                    continue
            
            if i + 2 < n:
                nxt = chain[i+2]
                if abs(new_ip1[0]-nxt[0]) + abs(new_ip1[1]-nxt[1]) + abs(new_ip1[2]-nxt[2]) != 1:
                    continue
            
            delta = 0
            if sequence[i] == "H":
                for q in _neighbours(old_i):
                    j = occupied.get(q)
                    if j is not None and j not in [i, i+1] and sequence[j] == "H" and abs(i-j) > 1:
                        delta += 1
            if sequence[i+1] == "H":
                for q in _neighbours(old_ip1):
                    j = occupied.get(q)
                    if j is not None and j not in [i, i+1] and sequence[j] == "H" and abs(i+1-j) > 1:
                        delta += 1
            
            if sequence[i] == "H":
                for q in _neighbours(new_i):
                    j = occupied.get(q)
                    if j is not None and j not in [i, i+1] and sequence[j] == "H" and abs(i-j) > 1:
                        delta -= 1
            if sequence[i+1] == "H":
                for q in _neighbours(new_ip1):
                    j = occupied.get(q)
                    if j is not None and j not in [i, i+1] and sequence[j] == "H" and abs(i+1-j) > 1:
                        delta -= 1
            
            return new_i, new_ip1, delta
    return None

def _try_three_move(sequence, chain, occupied, i, rng):
    """Try to move residues i, i+1, i+2. Returns (new_i, new_ip1, new_ip2, delta) or None."""
    n = len(sequence)
    if i + 2 >= n:
        return None
        
    old_i = chain[i]
    old_ip1 = chain[i+1]
    old_ip2 = chain[i+2]
    
    for new_i in _neighbours(old_i):
        if new_i in occupied:
            occ_idx = occupied[new_i]
            if occ_idx not in [i, i+1, i+2]:
                continue
            if new_i == old_i:
                continue
        
        if i > 0:
            prev = chain[i-1]
            if abs(new_i[0]-prev[0]) + abs(new_i[1]-prev[1]) + abs(new_i[2]-prev[2]) != 1:
                continue
        
        for new_ip1 in _neighbours(new_i):
            if new_ip1 in occupied:
                occ_idx = occupied[new_ip1]
                if occ_idx not in [i, i+1, i+2]:
                    continue
                if new_ip1 == old_ip1 and new_i == old_i:
                    continue
            
            for new_ip2 in _neighbours(new_ip1):
                if new_ip2 in occupied:
                    occ_idx = occupied[new_ip2]
                    if occ_idx not in [i, i+1, i+2]:
                        continue
                    if new_ip2 == old_ip2 and new_ip1 == old_ip1 and new_i == old_i:
                        continue
                
                if i + 2 < n - 1:
                    nxt = chain[i+3]
                    if abs(new_ip2[0]-nxt[0]) + abs(new_ip2[1]-nxt[1]) + abs(new_ip2[2]-nxt[2]) != 1:
                        continue
                
                delta = 0
                if sequence[i] == "H":
                    for q in _neighbours(old_i):
                        j = occupied.get(q)
                        if j is not None and j not in [i, i+1, i+2] and sequence[j] == "H" and abs(i-j) > 1:
                            delta += 1
                if sequence[i+1] == "H":
                    for q in _neighbours(old_ip1):
                        j = occupied.get(q)
                        if j is not None and j not in [i, i+1, i+2] and sequence[i+1-j] != "H" if False else abs(i+1-j) > 1:
                            delta += 1
                if sequence[i+2] == "H":
                    for q in _neighbours(old_ip2):
                        j = occupied.get(q)
                        if j is not None and j not in [i, i+1, i+2] and sequence[j] == "H" and abs(i+2-j) > 1:
                            delta += 1
                
                if sequence[i] == "H":
                    for q in _neighbours(new_i):
                        j = occupied.get(q)
                        if j is not None and j not in [i, i+1, i+2] and sequence[j] == "H" and abs(i-j) > 1:
                            delta -= 1
                if sequence[i+1] == "H":
                    for q in _neighbours(new_ip1):
                        j = occupied.get(q)
                        if j is not None and j not in [i, i+1, i+2] and sequence[j] == "H" and abs(i+1-j) > 1:
                            delta -= 1
                if sequence[i+2] == "H":
                    for q in _neighbours(new_ip2):
                        j = occupied.get(q)
                        if j is not None and j not in [i, i+1, i+2] and sequence[j] == "H" and abs(i+2-j) > 1:
                            delta -= 1
                
                return new_i, new_ip1, new_ip2, delta
    return None

def _local_search(sequence, chain, max_iters, rng, time_limit):
    """
    Simulated annealing with single, two, and three-residue moves.
    """
    n = len(sequence)
    if n < 3:
        return chain
    
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    occupied = {p: i for i, p in enumerate(chain)}
    
    iters = 0
    no_improve_count = 0
    # Improved annealing schedule
    t0 = 2.0
    t1 = 0.001
    cool_rate = (t1 / t0) ** (1.0 / max(1, max_iters))
    temperature = t0
    
    while iters < max_iters:
        if time.time() > time_limit:
            break
        iters += 1
        
        # Choose move type: 50% single, 30% double, 20% triple
        move_type = rng.random()
        
        if move_type < 0.5:
            # Single residue move
            i = rng.randrange(1, n - 1)
            result = _try_single_move(sequence, chain, occupied, i, rng)
            if result:
                new_site, delta = result
                new_energy = current_energy + delta
                accept = False
                if delta < 0:
                    accept = True
                elif temperature > 0.0001:
                    if rng.random() < math.exp(-delta / temperature):
                        accept = True
                
                if accept:
                    old_site = chain[i]
                    occupied.pop(old_site)
                    occupied[new_site] = i
                    chain[i] = new_site
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
        elif move_type < 0.8:
            # Two residue move
            i = rng.randrange(1, n - 2)
            result = _try_two_move(sequence, chain, occupied, i, rng)
            if result:
                new_i, new_ip1, delta = result
                new_energy = current_energy + delta
                accept = False
                if delta < 0:
                    accept = True
                elif temperature > 0.0001:
                    if rng.random() < math.exp(-delta / temperature):
                        accept = True
                
                if accept:
                    old_i = chain[i]
                    old_ip1 = chain[i+1]
                    occupied.pop(old_i)
                    occupied.pop(old_ip1)
                    occupied[new_i] = i
                    occupied[new_ip1] = i + 1
                    chain[i] = new_i
                    chain[i+1] = new_ip1
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
        else:
            # Three residue move
            i = rng.randrange(1, n - 3)
            result = _try_three_move(sequence, chain, occupied, i, rng)
            if result:
                new_i, new_ip1, new_ip2, delta = result
                new_energy = current_energy + delta
                accept = False
                if delta < 0:
                    accept = True
                elif temperature > 0.0001:
                    if rng.random() < math.exp(-delta / temperature):
                        accept = True
                
                if accept:
                    old_i = chain[i]
                    old_ip1 = chain[i+1]
                    old_ip2 = chain[i+2]
                    occupied.pop(old_i)
                    occupied.pop(old_ip1)
                    occupied.pop(old_ip2)
                    occupied[new_i] = i
                    occupied[new_ip1] = i + 1
                    occupied[new_ip2] = i + 2
                    chain[i] = new_i
                    chain[i+1] = new_ip1
                    chain[i+2] = new_ip2
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
        
        # Cool down
        temperature *= cool_rate
        
        # If stuck for a long time, restart temperature to escape local minima
        if no_improve_count > 2000:
            temperature = max(temperature, 0.3)
            no_improve_count = 0
            if time.time() > time_limit:
                break

    return best_chain

def fold(sequence, dimension):
    """Best conformation of a fixed number of randomized greedy growths, with local search."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0
    
    n = len(sequence)
    
    # Run growths with dynamic budget based on remaining time
    if n <= 48:
        num_growths = 300
    elif n <= 80:
        num_growths = 200
    else:
        num_growths = 150
    
    for attempt in range(num_growths):
        if _remaining_time() < 5.0:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if best is None or value < best_energy:
            best, best_energy = chain, value
    
    # Try local search on the best chain if time permits
    if best is not None and _remaining_time() > 10.0:
        remaining = _remaining_time()
        ls_time_limit = min(time.time() + remaining - 5.0, _TOTAL_START + 300.0 - _SAFETY_MARGIN)
        # Scale iterations with length, but cap it
        max_ls_iters = max(2000, min(80000, 300 * len(sequence)))
        improved_chain = _local_search(sequence, best, max_ls_iters, rng, ls_time_limit)
        improved_energy = _energy(sequence, improved_chain)
        if improved_energy < best_energy:
            best, best_energy = improved_chain, improved_energy
            
        # Second pass with more aggressive cooling if time remains
        if _remaining_time() > 5.0:
            ls_time_limit2 = min(time.time() + _remaining_time() - 2.0, _TOTAL_START + 300.0 - _SAFETY_MARGIN)
            max_ls_iters2 = max(1000, min(40000, 150 * len(sequence)))
            improved_chain2 = _local_search(sequence, best, max_ls_iters2, rng, ls_time_limit2)
            improved_energy2 = _energy(sequence, improved_chain2)
            if improved_energy2 < best_energy:
                best, best_energy = improved_chain2, improved_energy2
        
        # Final polish pass with very low temperature for precise local optimization
        if _remaining_time() > 3.0:
            ls_time_limit3 = min(time.time() + _remaining_time() - 1.0, _TOTAL_START + 300.0 - _SAFETY_MARGIN)
            max_ls_iters3 = max(500, min(20000, 100 * len(sequence)))
            improved_chain3 = _local_search(sequence, best, max_ls_iters3, rng, ls_time_limit3)
            improved_energy3 = _energy(sequence, improved_chain3)
            if improved_energy3 < best_energy:
                best, best_energy = improved_chain3, improved_energy3
    
    if best is None:  # a straight line is always self-avoiding
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
