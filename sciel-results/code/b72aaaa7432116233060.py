# EVOLVE-BLOCK-START
import random
import time
import math


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


def _sa_fold(sequence, dimension, rng, time_budget):
    """Simulated annealing with kink flip moves and incremental energy."""
    n = len(sequence)
    if n <= 2:
        return [(i, 0, 0) for i in range(n)] if dimension == 3 else [(i, 0) for i in range(n)]
    
    # Initialize: straight line along x-axis
    chain = [(i, 0, 0) for i in range(n)] if dimension == 3 else [(i, 0) for i in range(n)]
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    # Precompute which residues are H
    is_h = [c == 'H' for c in sequence]
    
    # Count initial contacts
    def _count_contacts():
        contacts = 0
        for i in range(n):
            if is_h[i]:
                for q in _neighbours(chain[i]):
                    j = pos_to_idx.get(q)
                    if j is not None and j > i + 1 and is_h[j]:
                        contacts += 1
        return contacts
    
    current_contacts = _count_contacts()
    current_energy = -current_contacts
    best_energy = current_energy
    best_chain = list(chain)
    
    # Temperature schedule
    t0 = 1.0
    t_end = 0.01
    total_steps = max(2000, n * 30)
    cooling_rate = (t_end / t0) ** (1.0 / total_steps)
    
    current_t = t0
    start_time = time.time()
    step = 0
    steps_per_sec = 5000  # rough estimate
    
    while step < total_steps:
        if step % 1000 == 0:
            elapsed = time.time() - start_time
            if elapsed >= time_budget:
                break
        
        # Kink flip: pick residue i (1 to n-2), move to a neighbor
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        
        # Get candidates: neighbors of old_pos not in occupied
        candidates = []
        for q in _neighbours(old_pos):
            if q not in pos_to_idx:
                candidates.append(q)
        
        if not candidates:
            step += 1
            current_t *= cooling_rate
            continue
        
        new_pos = rng.choice(candidates)
        
        # Compute delta energy
        # Only contacts involving residue i change
        delta_contacts = 0
        
        if is_h[i]:
            # Contacts lost: H neighbors of old_pos that are not chain-adjacent to i
            for q in _neighbours(old_pos):
                j = pos_to_idx.get(q)
                if j is not None and abs(j - i) > 1 and is_h[j]:
                    delta_contacts -= 1
            # Contacts gained: H neighbors of new_pos that are not chain-adjacent to i
            for q in _neighbours(new_pos):
                j = pos_to_idx.get(q)
                if j is not None and abs(j - i) > 1 and is_h[j]:
                    delta_contacts += 1
        
        # Also check if moving i breaks or creates contacts between other pairs
        # Actually, only pairs involving i can change since only i moves
        # But we need to be careful: the neighbor j at position q might have
        # its contact count change if i was providing a contact
        
        # The above correctly handles: for each H neighbor j of old_pos (non-adjacent), 
        # we lose a contact. For each H neighbor j of new_pos (non-adjacent), we gain one.
        # This is correct because only residue i's position changes.
        
        delta_e = -delta_contacts  # energy = -contacts
        
        if delta_e <= 0 or (current_t > 1e-9 and rng.random() < math.exp(-delta_e / current_t)):
            # Accept move
            pos_to_idx.pop(old_pos)
            chain[i] = new_pos
            pos_to_idx[new_pos] = i
            current_contacts += delta_contacts
            current_energy = -current_contacts
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        current_t *= cooling_rate
        step += 1
    
    if dimension == 3:
        return [list(p) for p in best_chain]
    else:
        return [list(p) for p in best_chain]


def fold(sequence, dimension):
    """Fold a sequence using simulated annealing with multiple restarts."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    # Time budget: ~330s total for 26 sequences, use ~11s per sequence with margin
    per_seq_budget = 10.0
    
    # Number of restarts based on sequence length
    if n <= 48:
        n_restarts = 5
    elif n <= 80:
        n_restarts = 4
    else:
        n_restarts = 3
    
    restart_budget = per_seq_budget / n_restarts
    
    best_chain = None
    best_energy = 0  # Worst possible
    
    for r in range(n_restarts):
        rng = random.Random(12345 + r * 7919 + len(sequence) * 131)
        chain = _sa_fold(sequence, dimension, rng, restart_budget)
        if chain is None:
            continue
        e = _energy(sequence, chain)
        if e < best_energy:
            best_energy = e
            best_chain = chain
    
    if best_chain is None:
        best_chain = [[i] + [0] * (dimension - 1) for i in range(n)]
    
    return best_chain
# EVOLVE-BLOCK-END
