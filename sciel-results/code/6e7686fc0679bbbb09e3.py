"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 190.0  # Safety margin


def _remaining_time():
    return _DEADLINE - time.time()


def _neighbours(point):
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


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
    """Randomized greedy chain growth."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    is_H = [c == "H" for c in sequence]
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if is_H[i]:
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and is_H[occupied[q]])
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Bias towards gaining contacts, but also keep options open
            score = gain * 2.0 + free * 0.1 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick best, or random among top few?
        # Let's pick the best to be greedy, but randomness in score helps
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _rotate_point(p, center, angle):
    x, y = p
    cx, cy = center
    dx, dy = x - cx, y - cy
    if angle == 90:
        return (cx - dy, cy + dx)
    elif angle == 180:
        return (cx - dx, cy - dy)
    elif angle == 270:
        return (cx + dy, cy - dx)
    return p


def _delta_energy(sequence, chain, is_H, i, angle):
    """
    Calculate energy change for a pivot move at index i with given angle.
    Returns None if the move is invalid (collision).
    """
    N = len(chain)
    if i <= 0 or i >= N - 1:
        return None

    head = chain[:i+1]
    tail = chain[i+1:]
    
    # Rotate tail
    new_tail = [_rotate_point(p, chain[i], angle) for p in tail]
    
    # Check collision: new_tail must not intersect head
    head_set = set(head)
    for p in new_tail:
        if p in head_set:
            return None

    # Calculate Delta E
    # Old contacts between head and tail
    # New contacts between head and new_tail
    
    # We need to know which positions in tail/new_tail are H.
    # tail indices are i+1 to N-1.
    
    old_contacts = 0
    new_contacts = 0
    
    # To speed up, we can precompute H positions in tail?
    # But we need to check adjacency to head.
    
    # Iterate through head residues
    # Only head residues that are H can form contacts
    # And we only care about their neighbors.
    
    # Optimization: 
    # Old tail set for fast lookup? 
    # We can just check if a neighbor of a head residue is in the old_tail list.
    # Since N is small, O(N^2) might be too slow if we do it naively for every step.
    # But we only do it for valid moves.
    
    # Let's use sets for tail positions for O(1) lookup.
    old_tail_set = set(tail)
    new_tail_set = set(new_tail)
    
    # We also need to know if the residue at that position is H.
    # pos_to_idx for the whole chain?
    # Let's build a local map for the tail part to check H status quickly?
    # Or just iterate through head and check neighbors.
    
    # For each h in head:
    #   if is_H[index_of_h]:
    #     for q in neighbours(h):
    #       if q in old_tail_set:
    #         if is_H[index_of_q_in_chain]: old_contacts += 1
    #       if q in new_tail_set:
    #         if is_H[index_of_q_in_new_chain]: new_contacts += 1
    # 
    # Problem: index_of_q.
    # We know q is in tail. The index is i + 1 + pos_in_tail.
    # We can build a dict pos_to_relative_index for old_tail and new_tail.
    
    old_tail_map = {p: idx for idx, p in enumerate(tail)}
    new_tail_map = {p: idx for idx, p in enumerate(new_tail)}
    
    for idx_h, h in enumerate(head):
        if not is_H[idx_h]:
            continue
        for q in _neighbours(h):
            if q in old_tail_map:
                idx_t = i + 1 + old_tail_map[q]
                if is_H[idx_t]:
                    old_contacts += 1
            if q in new_tail_map:
                idx_t = i + 1 + new_tail_map[q]
                if is_H[idx_t]:
                    new_contacts += 1
                    
    # Energy is -contacts.
    # Delta E = New_E - Old_E = -New_Contacts - (-Old_Contacts) = Old_Contacts - New_Contacts
    return old_contacts - new_contacts


def fold(sequence, dimension):
    """Best conformation using Greedy Init + Simulated Annealing."""
    rng = random.Random(12345 + len(sequence))
    
    # 1. Generate initial conformations
    best_chain = None
    best_energy = 0
    candidates = []
    
    for _ in range(50):
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        e = _energy(sequence, chain)
        if e < best_energy:
            best_energy = e
            best_chain = chain
        candidates.append((e, chain))
        
    if best_chain is None:
        # Fallback: straight line
        return [[i, 0] for i in range(len(sequence))]
        
    # Sort candidates by energy
    candidates.sort(key=lambda x: x[0])
    
    is_H = [c == "H" for c in sequence]
    N = len(sequence)
    
    # 2. Simulated Annealing on the best candidates
    # We will process the top 5 candidates if time permits.
    
    current_best_chain = best_chain
    current_best_energy = best_energy
    
    for start_idx in range(min(5, len(candidates))):
        if _remaining_time() < 5.0:
            break
            
        chain = candidates[start_idx][1][:]
        e = candidates[start_idx][0]
        
        T = 1.0
        cooling = 0.99
        min_T = 0.01
        
        # Run SA until time limit or T too low
        while _remaining_time() > 2.0:
            if T < min_T:
                # Reheat
                T = 1.0
                cooling = 0.99
                
            # Pick random pivot
            i = rng.randint(1, N - 2)
            angle = rng.choice([90, 180, 270])
            
            delta_e = _delta_energy(sequence, chain, is_H, i, angle)
            
            if delta_e is not None:
                if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                    # Accept move
                    # Apply rotation to chain
                    head = chain[:i+1]
                    tail = chain[i+1:]
                    new_tail = [_rotate_point(p, chain[i], angle) for p in tail]
                    chain = head + new_tail
                    e += delta_e
                    
                    if e < current_best_energy:
                        current_best_energy = e
                        current_best_chain = chain[:]
                        
            T *= cooling
            
    return [list(p) for p in current_best_chain]
# EVOLVE-BLOCK-END
