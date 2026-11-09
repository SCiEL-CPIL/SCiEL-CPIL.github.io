import time
import random
import math

_START_TIME = time.time()
_TOTAL_BUDGET = 240.0  # Total CPU budget for all 26 sequences
_DEADLINE_MARGIN = 15.0  # Safety margin before 330s wall-clock kill

def _neighbours(point):
    x, y, z = point
    return [(x+1, y, z), (x-1, y, z), (x, y+1, z), (x, y-1, z), (x, y+1, z), (x, y, z-1)]

def _energy(sequence, chain):
    """
    Calculate the energy of a chain.
    Energy = - (number of H-H contacts)
    """
    if not chain:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, rng):
    """
    Greedy growth to create an initial self-avoiding walk.
    """
    N = len(sequence)
    if N == 0:
        return []
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, N):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Heuristic: prefer sites that close loops or are close to other H's
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        idx = occupied[q]
                        if idx < i - 1 and sequence[idx] == "H":
                            gain += 1
            # Add small random noise to avoid deterministic traps
            score = gain * 10.0 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
            
        best_score, best_site = max(options)
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain

def _delta_energy_on_regrow(sequence, head, new_tail, head_map, tail_map):
    """
    Compute the change in energy when replacing old tail with new_tail.
    We only need to count H-H contacts that involve the new tail.
    Contacts entirely within head are unchanged.
    Contacts between head and new_tail + within new_tail are new.
    """
    # Count H-H contacts in new_tail with head and within new_tail
    new_contacts = 0
    N_head = len(head)
    for i, pos in enumerate(new_tail):
        if sequence[N_head + i] == 'H':
            for q in _neighbours(pos):
                # Check if q is in head
                if q in head_map:
                    idx_h = head_map[q]
                    if sequence[idx_h] == 'H':
                        new_contacts += 1
                # Check if q is in new_tail (with higher index to avoid double-counting)
                elif q in tail_map:
                    idx_t = tail_map[q]
                    if idx_t > i and sequence[N_head + idx_t] == 'H':
                        new_contacts += 1
    return -new_contacts

def fold(sequence, dimension):
    N = len(sequence)
    if N == 0:
        return []
    
    # Compute deadline based on global budget
    elapsed = time.time() - _START_TIME
    remaining = _TOTAL_BUDGET - elapsed
    if remaining < 1.0:
        # Not enough time, return a quick answer
        chain = _greedy_grow(sequence, random.Random(42))
        if chain is None:
            chain = [(i, 0, 0) for i in range(N)]
        return [list(p) for p in chain]
    
    # Use up to 90% of remaining time, with a cap of 12s per sequence
    # and a floor of 1s
    seq_time = min(12.0, max(1.0, remaining * 0.9 / max(1, 26)))
    deadline = time.time() + seq_time
    
    rng = random.Random(42)
    
    # Initialize with best of a few greedy chains
    best_chain = None
    best_energy = 0
    
    for _ in range(10):
        chain = _greedy_grow(sequence, rng)
        if chain:
            e = _energy(sequence, chain)
            if best_chain is None or e < best_energy:
                best_chain = chain
                best_energy = e
                
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(N)]
        best_energy = _energy(sequence, best_chain)
        
    current_chain = best_chain[:]
    current_energy = best_energy
    
    # Tail Regrowth Search with stochastic regrowth and adaptive temperature
    T = 3.0
    T_min = 0.05
    accept_count = 0
    total_count = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        # Pick a cut point - bias toward middle for longer chains
        # For short chains, uniform is fine
        if N > 30:
            # Beta-like distribution centered at N/2
            i = int(rng.betavariate(2.0, 2.0) * (N - 2)) + 1
            i = max(1, min(N - 2, i))
        else:
            i = rng.randint(1, N - 2)
        
        head = current_chain[:i]
        
        # Prepare data structures for regrowth
        head_map = {p: idx for idx, p in enumerate(head)}
        occupied = set(head_map.keys())
        
        new_tail = []
        tail_map = {} # pos -> local index in new_tail
        current_pos = head[-1]
        
        valid = True
        tail_energy = 0  # Track energy contribution of new tail
        
        for k in range(i, N):
            local_k = k - i  # local index in new_tail
            options = []
            for site in _neighbours(current_pos):
                if site in occupied:
                    continue
                
                gain = 0
                if sequence[k] == "H":
                    for q in _neighbours(site):
                        # Check if q is in head
                        if q in head_map:
                            idx_h = head_map[q]
                            if sequence[idx_h] == "H":
                                gain += 1
                        # Check if q is in new_tail (only earlier positions to avoid double-count)
                        elif q in tail_map:
                            idx_t = tail_map[q]
                            if idx_t < local_k and sequence[i + idx_t] == "H":
                                gain += 1
                                
                # Stochastic scoring: gain dominates, but add noise
                score = gain * 8.0 + rng.gauss(0, 1.0)
                options.append((score, site))
                
            if not options:
                valid = False
                break
                
            # Use softmax-like selection for stochasticity
            if len(options) == 1:
                best_site = options[0][1]
            else:
                # Find max score for numerical stability
                max_score = max(s for s, _ in options)
                weights = [math.exp((s - max_score) / max(T, 0.1)) for s, _ in options]
                total_w = sum(weights)
                r = rng.random() * total_w
                cum = 0
                chosen = options[-1][1]
                for w, (_, site) in zip(weights, options):
                    cum += w
                    if r <= cum:
                        chosen = site
                        break
                best_site = chosen
            
            new_tail.append(best_site)
            tail_map[best_site] = local_k
            occupied.add(best_site)
            current_pos = best_site
            
        if not valid:
            no_improve_count += 1
            continue
            
        new_chain = head + new_tail
        
        # Compute delta energy: new tail contacts minus old tail contacts
        # Old tail contacts: we need to compute them
        old_tail = current_chain[i:]
        old_tail_map = {p: idx for idx, p in enumerate(old_tail)}
        
        # Count H-H contacts in old tail with head and within old tail
        old_contacts = 0
        N_head = len(head)
        for i_t, pos in enumerate(old_tail):
            if sequence[N_head + i_t] == 'H':
                for q in _neighbours(pos):
                    if q in head_map:
                        idx_h = head_map[q]
                        if sequence[idx_h] == 'H':
                            old_contacts += 1
                    elif q in old_tail_map:
                        idx_t = old_tail_map[q]
                        if idx_t > i_t and sequence[N_head + idx_t] == 'H':
                            old_contacts += 1
        
        # Count H-H contacts in new tail with head and within new tail
        new_contacts = 0
        for i_t, pos in enumerate(new_tail):
            if sequence[N_head + i_t] == 'H':
                for q in _neighbours(pos):
                    if q in head_map:
                        idx_h = head_map[q]
                        if sequence[idx_h] == 'H':
                            new_contacts += 1
                    elif q in tail_map:
                        idx_t = tail_map[q]
                        if idx_t > i_t and sequence[N_head + idx_t] == 'H':
                            new_contacts += 1
        
        delta_e = (-new_contacts) - (-old_contacts)
        new_energy = current_energy + delta_e
        
        # Metropolis Acceptance
        total_count += 1
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            accept_count += 1
            current_chain = new_chain
            current_energy = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = new_chain[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Adaptive cooling
            if T > T_min:
                T *= 0.998
        else:
            no_improve_count += 1
        
        # Periodic temperature reset if stuck
        if no_improve_count > 500 and T > T_min:
            T = min(T * 1.5, 2.0)
            no_improve_count = 0
        
        # Check if we should break early
        if time.time() > deadline - 0.5:
            break
        
    return [list(p) for p in best_chain]
