# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 118.0


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbours(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    n = len(chain)
    if n < 2:
        return 0
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts // 2


def _grow_compact(sequence, rng, compactness):
    """
    Greedy chain growth with a compactness parameter.
    Higher compactness biases toward filling space densely.
    Returns a valid self-avoiding chain or None.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        current = chain[-1]
        options = []
        for site in _neighbours(current):
            if site in occupied:
                continue
            
            # Compute score: prefer sites with more H neighbors (for H residues)
            # and fewer exposed neighbors (for compactness)
            score = 0.0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            score += 2.0
                        elif j < i - 1:
                            score += 0.5
            # Compactness: prefer sites with fewer free neighbors
            free_count = sum(1 for q in _neighbours(site) if q not in occupied)
            score -= compactness * free_count * 0.3
            
            # Add small noise for diversity
            score += rng.random() * 0.5
            
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick the best option
        options.sort(key=lambda x: x[0], reverse=True)
        best_site = options[0][1]
        occupied[best_site] = i
        chain.append(best_site)
    
    return chain


def _is_valid_chain(chain):
    """Check self-avoidance and unit steps."""
    if not chain:
        return True
    seen = set()
    for p in chain:
        if p in seen:
            return False
        seen.add(p)
    for i in range(1, len(chain)):
        dx = abs(chain[i][0] - chain[i-1][0])
        dy = abs(chain[i][1] - chain[i-1][1])
        if dx + dy != 1:
            return False
    return True


def _apply_reversal(chain, i, j):
    """
    Attempt a reversal move. Returns new chain if valid, else None.
    0 <= i < j < len(chain).
    """
    n = len(chain)
    if i >= j:
        return None
    
    # Check boundary conditions
    if i > 0:
        p_prev = chain[i - 1]
        p_end = chain[j]
        if abs(p_prev[0] - p_end[0]) + abs(p_prev[1] - p_end[1]) != 1:
            return None
            
    if j < n - 1:
        p_start = chain[i]
        p_next = chain[j + 1]
        if abs(p_start[0] - p_next[0]) + abs(p_start[1] - p_next[1]) != 1:
            return None
            
    new_chain = chain[:i] + chain[i:j+1][::-1] + chain[j+1:]
    
    if not _is_valid_chain(new_chain):
        return None
        
    return new_chain


def _apply_pivot(chain, i, angle):
    """
    Attempt a pivot move. Rotate tail chain[i+1:] around chain[i].
    angle is 1, 2, or 3 (multiples of 90 degrees).
    Returns new chain if valid, else None.
    """
    n = len(chain)
    if i <= 0 or i >= n - 1:
        return None
        
    pivot = chain[i]
    tail = chain[i+1:]
    prefix = chain[:i+1]
    
    rotated_tail = []
    for p in tail:
        x, y = p[0] - pivot[0], p[1] - pivot[1]
        for _ in range(angle):
            x, y = -y, x
        rotated_tail.append((pivot[0] + x, pivot[1] + y))
        
    new_chain = prefix + rotated_tail
    
    if not _is_valid_chain(new_chain):
        return None
        
    return new_chain


def _calc_delta_energy_tail(sequence, chain, pos_map, i, new_tail):
    """
    Calculate energy difference for a tail move.
    i is the pivot index. The tail is chain[i+1:].
    new_tail is the list of new points for the tail.
    Returns delta_energy (new - old).
    """
    n = len(chain)
    # Indices in tail: i+1 to n-1
    # Body: 0 to i
    
    # Helper to check if two indices are H-H contacts
    def is_contact(idx1, idx2, p1, p2):
        if sequence[idx1] != 'H' or sequence[idx2] != 'H':
            return 0
        if abs(idx1 - idx2) <= 1:
            return 0
        # Check lattice distance
        if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1:
            return 1
        return 0

    old_energy = 0
    new_energy = 0
    
    # 1. Contacts within the tail (old vs new)
    # Old tail points: chain[i+1] ... chain[n-1]
    # New tail points: new_tail[0] ... new_tail[len-1]
    # Indices are i+1 ... n-1 in both cases.
    
    # Old internal tail contacts
    for k in range(i+1, n):
        p_k = chain[k]
        for q in _neighbours(p_k):
            # Find index of q in the tail part only?
            # We can use pos_map but we must ensure we only count pairs once and within the tail.
            # Simpler: iterate over all pairs in tail? O(K^2). K is small (<=5).
            pass
            
    # Efficient way for small K:
    # Remove old tail from pos_map temporarily? No, we can just check.
    
    # Let's just calculate full energy of the tail segment and its interaction with body.
    # Since K is small, we can do this:
    
    # Old Energy Contribution from Tail:
    # - Internal tail contacts
    # - Contacts between tail and body
    
    # New Energy Contribution from Tail:
    # - Internal new tail contacts
    # - Contacts between new tail and body
    
    # Body is fixed.
    
    # 1. Internal Old Tail
    for a in range(i+1, n):
        for b in range(a+1, n):
            if abs(chain[a][0] - chain[b][0]) + abs(chain[a][1] - chain[b][1]) == 1:
                if sequence[a] == 'H' and sequence[b] == 'H':
                    old_energy += 1
                    
    # 2. Internal New Tail
    # new_tail[0] corresponds to index i+1
    # new_tail[k] corresponds to index i+1+k
    for a_idx in range(len(new_tail)):
        for b_idx in range(a_idx+1, len(new_tail)):
            idx_a = i + 1 + a_idx
            idx_b = i + 1 + b_idx
            if abs(new_tail[a_idx][0] - new_tail[b_idx][0]) + abs(new_tail[a_idx][1] - new_tail[b_idx][1]) == 1:
                if sequence[idx_a] == 'H' and sequence[idx_b] == 'H':
                    new_energy += 1

    # 3. Old Tail-Body Contacts
    for a in range(i+1, n):
        p_a = chain[a]
        for q in _neighbours(p_a):
            # Is q in the body?
            # q is in pos_map. If pos_map[q] <= i, it is in the body.
            # But we must ensure q is not in the tail (which it isn't, as we are checking neighbours of a tail node, and if it's in the tail, it would be a tail-tail contact, already handled? 
            # No, tail-tail contacts are handled by the double loop above.
            # So if q is in pos_map and pos_map[q] <= i, it is a body contact.
            idx_q = pos_map.get(q)
            if idx_q is not None and idx_q <= i:
                if sequence[a] == 'H' and sequence[idx_q] == 'H':
                    old_energy += 1

    # 4. New Tail-Body Contacts
    for a_idx in range(len(new_tail)):
        idx_a = i + 1 + a_idx
        p_a = new_tail[a_idx]
        for q in _neighbours(p_a):
            idx_q = pos_map.get(q)
            # q must be in the body (index <= i)
            # Note: q might be in the old tail positions, but those are now "free" or "occupied by new tail"?
            # pos_map currently holds the OLD chain.
            # So if idx_q <= i, it is a body node.
            # If idx_q > i, it is an old tail node. We ignore it for body contact.
            if idx_q is not None and idx_q <= i:
                if sequence[idx_a] == 'H' and sequence[idx_q] == 'H':
                    new_energy += 1

    return new_energy - old_energy


def _anneal(sequence, initial_chain, time_limit, rng):
    """
    Simulated Annealing with incremental tail moves.
    """
    n = len(initial_chain)
    if n < 3:
        return initial_chain[:], _energy(sequence, initial_chain)
    
    current_chain = initial_chain[:]
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain[:]
    best_energy = current_energy
    
    # Build pos_map
    pos_map = {}
    for i, p in enumerate(current_chain):
        pos_map[p] = i
        
    T = 2.0
    cooling_rate = 0.999
    
    deadline = time.time() + time_limit
    
    stall_counter = 0
    max_stall = 100
    
    while time.time() < deadline:
        if _remaining_time() < 1.0:
            break
        
        # Choose move type
        # 80% Tail Moves, 20% Reversals (for large jumps)
        if rng.random() < 0.8:
            # Tail Move
            # Pick tail length 1 to min(5, n-1)
            max_tail = min(5, n - 1)
            tail_len = rng.randint(1, max_tail)
            i = n - 1 - tail_len # Pivot index
            if i <= 0: continue
            
            # Try 3 rotations
            moved = False
            for angle in [1, 2, 3]:
                # Generate new tail
                pivot = current_chain[i]
                tail = current_chain[i+1:]
                
                new_tail = []
                for p in tail:
                    x, y = p[0] - pivot[0], p[1] - pivot[1]
                    for _ in range(angle):
                        x, y = -y, x
                    new_tail.append((pivot[0] + x, pivot[1] + y))
                
                # Check validity:
                # 1. Self-avoidance: new_tail points must not be in pos_map (except for the tail itself which we are removing/replacing)
                #    And new_tail points must not collide with each other.
                valid = True
                new_tail_set = set()
                for p in new_tail:
                    if p in new_tail_set:
                        valid = False
                        break
                    new_tail_set.add(p)
                    # Check if p is occupied by a body node (index <= i)
                    # If it is occupied by a tail node, that's fine (we are replacing it)
                    # But we must ensure the new tail doesn't hit the body.
                    idx = pos_map.get(p)
                    if idx is not None and idx <= i:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check connectivity at pivot
                # new_tail[0] must be neighbor of pivot
                if abs(new_tail[0][0] - pivot[0]) + abs(new_tail[0][1] - pivot[1]) != 1:
                    continue
                
                # Calculate Delta E
                delta_e = _calc_delta_energy_tail(sequence, current_chain, pos_map, i, new_tail)
                
                # Metropolis
                if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 0.01)):
                    # Accept
                    # Update pos_map
                    for p in current_chain[i+1:]:
                        del pos_map[p]
                    for k, p in enumerate(new_tail):
                        pos_map[p] = i + 1 + k
                    
                    # Update chain
                    current_chain[i+1:] = new_tail
                    current_energy += delta_e
                    stall_counter = 0
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = current_chain[:]
                    moved = True
                    break # Try next move
                
        else:
            # Reversal (Slower, but good for escaping)
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            new_chain = _apply_reversal(current_chain, i, j)
            if new_chain is None:
                continue
            
            new_energy = _energy(sequence, new_chain)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 0.01)):
                # Accept
                # Rebuild pos_map (expensive, but rare)
                pos_map = {}
                for k, p in enumerate(new_chain):
                    pos_map[p] = k
                
                current_chain = new_chain
                current_energy = new_energy
                stall_counter = 0
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = current_chain[:]
        
        T *= cooling_rate
        if T < 0.05:
            if rng.random() < 0.3:
                T = 1.5
            else:
                break
        
        if stall_counter > max_stall:
            # Reheat
            T = 1.0
            stall_counter = 0
        
        if _remaining_time() < 0.5:
            break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation using model-and-optimize.
    """
    rng = random.Random(42 + len(sequence) * 7)
    n = len(sequence)
    
    if n < 2:
        return [[0, 0]] * n
        
    best_chain = None
    best_energy = 0  # Energy is negative, so 0 is worse than any contact
    
    # Phase 1: Generate diverse initial chains
    compactness_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    time_for_growth = 3.0
    growths_per_value = max(2, int(time_for_growth / len(compactness_values) / 0.02))
    
    initial_chains = []
    
    for compactness in compactness_values:
        if _remaining_time() < 5.0:
            break
        for _ in range(growths_per_value):
            if _remaining_time() < 5.0:
                break
            chain = _grow_compact(sequence, rng, compactness)
            if chain is not None and _is_valid_chain(chain):
                initial_chains.append(chain)
                
    if not initial_chains:
        initial_chains.append([(i, 0) for i in range(n)])
        
    scored_chains = []
    for chain in initial_chains:
        e = _energy(sequence, chain)
        scored_chains.append((e, chain))
        
    scored_chains.sort(key=lambda x: x[0])
    
    top_k = min(6, len(scored_chains))
    candidates = [c[1] for c in scored_chains[:top_k]]
    
    # Phase 2: Anneal
    total_time_remaining = _remaining_time()
    time_per_candidate = total_time_remaining / len(candidates)
    
    for idx, chain in enumerate(candidates):
        if _remaining_time() < 1.5:
            break
            
        alloc_time = min(time_per_candidate, _remaining_time() - 1.0)
        if alloc_time < 0.2:
            continue
            
        best_c, best_e = _anneal(sequence, chain, alloc_time, rng)
        
        if best_e < best_energy:
            best_energy = best_e
            best_chain = best_c
    
    if best_chain is None:
        if initial_chains:
            best_chain = initial_chains[0]
        else:
            best_chain = [(i, 0) for i in range(n)]
            
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
