"""HP lattice protein folding: greedy growth + simulated annealing refinement."""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = time.time()
_DEADLINE = 200.0  # Safety margin before 210s kill


def _remaining_time():
    return max(0.0, _DEADLINE - (time.time() - _START_TIME))


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute HP energy: -number of H-H contacts (non-adjacent in chain)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _energy_delta(sequence, chain, idx, old_site, new_site, where):
    """
    Compute energy delta for moving residue idx from old_site to new_site.
    Returns delta_e (new_energy - old_energy).
    """
    delta = 0
    # Remove contacts involving idx at old_site
    if sequence[idx] == 'H':
        for q in _neighbours(old_site):
            j = where.get(q)
            if j is not None and j != idx:
                if abs(j - idx) > 1 and sequence[j] == 'H':
                    delta += 1  # removing a contact increases energy by 1
    # Add contacts involving idx at new_site
    if sequence[idx] == 'H':
        for q in _neighbours(new_site):
            j = where.get(q)
            if j is not None and j != idx:
                if abs(j - idx) > 1 and sequence[j] == 'H':
                    delta -= 1  # adding a contact decreases energy by 1
    return delta


def _grow(sequence, dimension, rng, bias=1.0):
    """
    One randomized greedy chain growth.
    bias > 1 favors contacts, bias < 1 favors free space.
    """
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
            score = gain * bias + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _sa_refine(sequence, chain, time_budget, rng):
    """
    Refine a conformation using simulated annealing with incremental energy.
    Returns (best_chain, best_energy).
    """
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    where = {p: i for i, p in enumerate(chain)}
    
    # Temperature schedule
    T = 2.0
    T_min = 0.01
    cooling_rate = 0.999
    
    start_time = time.time()
    max_time = time_budget
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_time:
            break
        if T < T_min:
            break
        
        # Choose move type
        move_type = rng.random()
        
        accepted = False
        
        if move_type < 0.6:
            # Single-residue pivot move
            idx = rng.randint(1, n - 2)
            old_site = chain[idx]
            prev_site = chain[idx - 1]
            next_site = chain[idx + 1]
            
            candidates = []
            for site in _neighbours(old_site):
                if abs(site[0] - prev_site[0]) + abs(site[1] - prev_site[1]) != 1:
                    continue
                if abs(site[0] - next_site[0]) + abs(site[1] - next_site[1]) != 1:
                    continue
                if site in where:
                    occ_idx = where[site]
                    if occ_idx != idx:
                        continue
                candidates.append(site)
            
            if candidates:
                new_site = rng.choice(candidates)
                if new_site != old_site:
                    delta_e = _energy_delta(sequence, chain, idx, old_site, new_site, where)
                    if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
                        del where[old_site]
                        where[new_site] = idx
                        chain[idx] = new_site
                        current_energy += delta_e
                        accepted = True
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(chain)
        
        elif move_type < 0.8:
            # Segment reversal
            if n >= 4:
                i = rng.randint(0, n - 3)
                j = rng.randint(i + 2, n - 1)
                
                # Check boundary connectivity
                if i > 0:
                    if abs(chain[i - 1][0] - chain[j][0]) + abs(chain[i - 1][1] - chain[j][1]) != 1:
                        pass  # skip this move
                    else:
                        # Check self-avoidance
                        old_segment = list(chain[i:j + 1])
                        new_segment = old_segment[::-1]
                        
                        temp_where = dict(where)
                        for pt in old_segment:
                            if pt in temp_where:
                                del temp_where[pt]
                        
                        valid = True
                        for pt in new_segment:
                            if pt in temp_where:
                                valid = False
                                break
                        
                        if valid:
                            chain[i:j + 1] = new_segment
                            for k in range(i, j + 1):
                                where[chain[k]] = k
                            
                            new_energy = _energy(sequence, chain)
                            delta_e = new_energy - current_energy
                            
                            if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
                                current_energy = new_energy
                                accepted = True
                                if current_energy < best_energy:
                                    best_energy = current_energy
                                    best_chain = list(chain)
                            else:
                                # Revert
                                chain[i:j + 1] = old_segment
                                for k in range(i, j + 1):
                                    where[chain[k]] = k
                else:
                    # i == 0, check j < n-1
                    if j < n - 1:
                        if abs(chain[i][0] - chain[j + 1][0]) + abs(chain[i][1] - chain[j + 1][1]) != 1:
                            pass
                        else:
                            old_segment = list(chain[i:j + 1])
                            new_segment = old_segment[::-1]
                            
                            temp_where = dict(where)
                            for pt in old_segment:
                                if pt in temp_where:
                                    del temp_where[pt]
                            
                            valid = True
                            for pt in new_segment:
                                if pt in temp_where:
                                    valid = False
                                    break
                            
                            if valid:
                                chain[i:j + 1] = new_segment
                                for k in range(i, j + 1):
                                    where[chain[k]] = k
                                
                                new_energy = _energy(sequence, chain)
                                delta_e = new_energy - current_energy
                                
                                if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
                                    current_energy = new_energy
                                    accepted = True
                                    if current_energy < best_energy:
                                        best_energy = current_energy
                                        best_chain = list(chain)
                                else:
                                    chain[i:j + 1] = old_segment
                                    for k in range(i, j + 1):
                                        where[chain[k]] = k
        
        else:
            # Segment rotation (90 degrees)
            if n >= 4:
                i = rng.randint(0, n - 3)
                j = rng.randint(i + 2, n - 1)
                
                pivot = chain[i]
                seg = chain[i:j + 1]
                
                # Rotate 90 degrees clockwise around pivot
                rotated_seg = [pivot]
                for pt in seg[1:]:
                    dx = pt[0] - pivot[0]
                    dy = pt[1] - pivot[1]
                    new_pt = (pivot[0] + dy, pivot[1] - dx)
                    rotated_seg.append(new_pt)
                
                # Check self-avoidance
                temp_where = dict(where)
                for pt in seg[1:]:
                    if pt in temp_where:
                        del temp_where[pt]
                
                valid = True
                for pt in rotated_seg[1:]:
                    if pt in temp_where:
                        valid = False
                        break
                
                if valid:
                    # Check boundary connectivity
                    if i > 0:
                        if abs(chain[i - 1][0] - rotated_seg[0][0]) + abs(chain[i - 1][1] - rotated_seg[0][1]) != 1:
                            pass
                        elif j < n - 1:
                            if abs(rotated_seg[-1][0] - chain[j + 1][0]) + abs(rotated_seg[-1][1] - chain[j + 1][1]) != 1:
                                pass
                            else:
                                chain[i:j + 1] = rotated_seg
                                for k in range(i, j + 1):
                                    where[chain[k]] = k
                                
                                new_energy = _energy(sequence, chain)
                                delta_e = new_energy - current_energy
                                
                                if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
                                    current_energy = new_energy
                                    accepted = True
                                    if current_energy < best_energy:
                                        best_energy = current_energy
                                        best_chain = list(chain)
                                else:
                                    chain[i:j + 1] = seg
                                    for k in range(i, j + 1):
                                        where[chain[k]] = k
                    else:
                        if j < n - 1:
                            if abs(rotated_seg[-1][0] - chain[j + 1][0]) + abs(rotated_seg[-1][1] - chain[j + 1][1]) != 1:
                                pass
                            else:
                                chain[i:j + 1] = rotated_seg
                                for k in range(i, j + 1):
                                    where[chain[k]] = k
                                
                                new_energy = _energy(sequence, chain)
                                delta_e = new_energy - current_energy
                                
                                if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
                                    current_energy = new_energy
                                    accepted = True
                                    if current_energy < best_energy:
                                        best_energy = current_energy
                                        best_chain = list(chain)
                                else:
                                    chain[i:j + 1] = seg
                                    for k in range(i, j + 1):
                                        where[chain[k]] = k
        
        # Cool down
        T *= cooling_rate
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using greedy growth + simulated annealing refinement."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    # Determine time budget for this sequence
    # Reserve some time for all sequences, but give more to longer ones
    time_per_seq = _remaining_time() / 10.0
    if time_per_seq < 1.0:
        time_per_seq = 1.0
    
    best_chain = None
    best_energy = 0  # Start with 0 (unfolded) as baseline
    
    # Phase 1: Generate diverse initial conformations via greedy growth
    num_growths = min(50, max(10, n // 2))
    initial_chains = []
    
    for i in range(num_growths):
        if _remaining_time() < 5.0:
            break
        # Vary bias to get diversity
        bias = 0.5 + rng.random() * 1.5
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is not None:
            initial_chains.append(chain)
    
    if not initial_chains:
        # Fallback: straight line
        return [list([i] + [0] * (dimension - 1)) for i in range(n)]
    
    # Phase 2: Refine the best initial chains with SA
    # Sort by energy and refine the top few
    scored_chains = []
    for chain in initial_chains:
        e = _energy(sequence, chain)
        scored_chains.append((e, chain))
    
    scored_chains.sort(key=lambda x: x[0])
    
    # Refine top 3 chains (or fewer if time is tight)
    num_refine = min(3, len(scored_chains))
    
    for e, chain in scored_chains[:num_refine]:
        if _remaining_time() < 3.0:
            break
        
        # Allocate time based on how much time we have
        refine_time = min(15.0, _remaining_time() / (num_refine + 1))
        
        refined_chain, refined_energy = _sa_refine(sequence, chain, refine_time, rng)
        
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain
    
    # Phase 3: If we have time, do more greedy growths and refine
    while _remaining_time() > 5.0:
        bias = 0.5 + rng.random() * 1.5
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue
        
        e = _energy(sequence, chain)
        if e >= best_energy:
            continue
        
        # Refine this promising chain
        refine_time = min(10.0, _remaining_time() / 2)
        refined_chain, refined_energy = _sa_refine(sequence, chain, refine_time, rng)
        
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain
    
    if best_chain is None:
        # Fallback: use the best initial chain
        best_chain = scored_chains[0][1]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
