import time
import random
import math

_T0 = time.time()
_DEADLINE = 195.0

def _neighbours(point):
    x, y = point
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

def _energy(seq, chain):
    # O(N) or O(N log N)
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if seq[i] == 'H':
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None and abs(i - j) > 1 and seq[j] == 'H':
                    contacts += 1
    return -contacts

def _is_valid(chain):
    return len(set(chain)) == len(chain)

def _generate_initial(seq):
    n = len(seq)
    if n == 0: return []
    if n == 1: return [(0,0)]
    
    chain = [(0,0)]
    occupied = {(0,0)}
    for i in range(1, n):
        last = chain[-1]
        neighbors = _neighbours(last)
        # Filter valid
        valid = [p for p in neighbors if p not in occupied]
        if not valid:
            return None
        
        # Score neighbors
        scored = []
        for p in valid:
            # Count H neighbors in current chain
            h_count = 0
            if seq[i] == 'H':
                for q in _neighbours(p):
                    if q in occupied:
                        idx = chain.index(q) # O(N) - bad. Use dict?
                        # We don't have pos_to_idx easily here without building it.
                        # For N=100, index is fine.
                        if seq[idx] == 'H':
                            h_count += 1
            # Add some randomness
            score = h_count + random.random() * 0.5
            scored.append((score, p))
        
        scored.sort(reverse=True)
        best_p = scored[0][1]
        chain.append(best_p)
        occupied.add(best_p)
        
    return chain

def _optimize(seq, chain):
    n = len(chain)
    if n < 2: return chain, _energy(seq, chain)
    
    current_energy = _energy(seq, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    T = 5.0
    T_min = 0.1
    cooling = 0.995
    
    # Time check inside loop?
    # We will call this from the main loop which checks time.
    # But we should also have an internal step limit to avoid infinite loops if T doesn't drop fast enough.
    steps = 0
    max_steps = 5000 # per restart
    
    while T > T_min and steps < max_steps:
        steps += 1
        move_type = random.choice(['single', 'tail'])
        
        if move_type == 'single':
            if n < 3: continue
            i = random.randint(1, n-2)
            p = chain[i]
            neighbors = _neighbours(p)
            new_p = random.choice(neighbors)
            
            # Check collision
            # new_p must not be in chain, except possibly i-1 or i+1?
            # Actually, new_p must be distinct from all other positions.
            # It can be the position of i-1 or i+1? No, that would mean the chain crosses itself or doubles back immediately.
            # Unit step constraint: dist(chain[i-1], new_p) must be 1? No, chain[i-1] is fixed.
            # The move is valid if:
            # 1. new_p is not in {chain[j] for j != i}
            # 2. dist(chain[i-1], new_p) == 1 (always true if new_p is neighbor of old p? No.)
            #    Wait. chain[i-1] is fixed. chain[i] moves to new_p.
            #    The bond (i-1, i) must remain unit length.
            #    So new_p must be a neighbor of chain[i-1].
            #    Also bond (i, i+1) must remain unit length.
            #    So new_p must be a neighbor of chain[i+1].
            #    So new_p must be a common neighbor of chain[i-1] and chain[i+1].
            #    And new_p must not be occupied by any other residue.
            
            # Let's refine Single Move:
            # Find common neighbors of p_prev and p_next.
            # p_prev = chain[i-1], p_next = chain[i+1]
            # candidates = set(_neighbours(p_prev)) & set(_neighbours(p_next))
            # Remove p (current pos)
            # Remove any pos in chain (except i)
            
            p_prev = chain[i-1]
            p_next = chain[i+1]
            candidates = set(_neighbours(p_prev)) & set(_neighbours(p_next))
            candidates.discard(p)
            
            # Check occupancy
            occupied_set = set(chain)
            # We need to check if candidate is in occupied_set
            # But we are moving i, so p is "free" in the new config?
            # Actually, we check if candidate is in {chain[j] for j != i}.
            # So:
            valid_moves = []
            for c in candidates:
                if c not in occupied_set or c == p:
                    # c == p is the "no move" case, skip
                    if c != p:
                        # Check if c is occupied by someone else
                        # occupied_set contains p.
                        # If c is in occupied_set and c != p, it's blocked.
                        if c in occupied_set:
                            continue
                        valid_moves.append(c)
            
            if not valid_moves:
                continue
                
            new_p = random.choice(valid_moves)
            
            # Calculate Delta E
            # This is tricky to do incrementally.
            # For N=100, full re-calc is fast.
            new_chain = list(chain)
            new_chain[i] = new_p
            new_energy = _energy(seq, new_chain)
            
            delta = new_energy - current_energy
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                    
        else: # tail rotation
            if n < 4: continue
            k = random.randint(2, n-1) # Pivot at k-1?
            # Rotate tail k..n-1 around k-1
            pivot = chain[k-1]
            
            # Get tail
            tail = chain[k:]
            # Rotate each point in tail 90 deg around pivot
            # (x, y) -> (x + (y - py), y - (x - px)) ?
            # Vector v = p - pivot. Rotated v' = (-v_y, v_x) or (v_y, -v_x)
            # Let's try both directions? Or just one?
            # Random direction.
            direction = random.choice([1, -1])
            
            new_tail = []
            for p in tail:
                vx = p[0] - pivot[0]
                vy = p[1] - pivot[1]
                if direction == 1:
                    nvx, nvy = -vy, vx
                else:
                    nvx, nvy = vy, -vx
                new_tail.append((pivot[0] + nvx, pivot[1] + nvy))
            
            new_chain = chain[:k] + new_tail
            
            # Check validity
            if len(set(new_chain)) != n:
                continue
            
            new_energy = _energy(seq, new_chain)
            delta = new_energy - current_energy
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                    
        T *= cooling
        
    return best_chain, best_energy

def fold(sequence, dimension):
    start_time = time.time()
    best_chain = None
    best_energy = 0 # 0 is worst (unfolded)
    
    # Fallback: Straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(len(sequence))]
        best_energy = _energy(sequence, best_chain)
        
    while time.time() - start_time < _DEADLINE - 5: # 5s margin
        # Generate initial
        init_chain = _generate_initial(sequence)
        if init_chain is None:
            continue
            
        # Optimize
        opt_chain, opt_energy = _optimize(sequence, init_chain)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_chain = opt_chain
            
        # If we found a very good solution, maybe stop?
        # Theoretical max contacts is roughly N/2.
        # If best_energy is close to -N/2, we can stop.
        # But we don't know the max exactly.
        
    return [list(p) for p in best_chain]
