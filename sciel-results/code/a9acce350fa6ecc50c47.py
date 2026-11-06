"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = time.time() + 310  # safety margin before 330s kill


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _is_neighbour(p1, p2):
    return sum(abs(a - b) for a, b in zip(p1, p2)) == 1


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth."""
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


def _pivot_move(chain, i, rng, dimension):
    """Try a pivot move at position i."""
    if i <= 1 or i >= len(chain) - 1:
        return None
    
    n = len(chain)
    head = chain[:i+1]
    tail = chain[i+1:]
    
    axis = rng.randint(0, dimension - 1)
    sign = rng.choice([-1, 1])
    
    pivot = chain[i]
    new_tail = []
    for pos in tail:
        vec = [pos[k] - pivot[k] for k in range(dimension)]
        if axis == 0:
            new_vec = [vec[0], -sign * vec[2], sign * vec[1]]
        elif axis == 1:
            new_vec = [sign * vec[2], vec[1], -sign * vec[0]]
        else:
            new_vec = [-sign * vec[1], sign * vec[0], vec[2]]
        new_pos = tuple(pivot[k] + new_vec[k] for k in range(dimension))
        new_tail.append(new_pos)
    
    new_chain = head + new_tail
    
    # Check self-avoidance
    if len(set(new_chain)) != n:
        return None
    
    # Check unit steps
    for k in range(n - 1):
        if not _is_neighbour(new_chain[k], new_chain[k+1]):
            return None
            
    return new_chain


def _reversal_move(chain, i, j):
    """Reverse segment [i, j]. Returns new chain."""
    if i >= j:
        return chain
    return chain[:i] + chain[i:j+1][::-1] + chain[j+1:]


def _delta_energy_reversal(sequence, chain, i, j):
    """Calculate delta energy for reversal [i, j]."""
    # Pairs that change adjacency status:
    # Old: (i-1, i) and (j, j+1)
    # New: (i-1, j) and (i, i+1)
    
    delta = 0
    
    # Helper to check if a pair (p1, p2) contributes to energy loss/gain
    # If they are H-H and spatial neighbors:
    # If they were adjacent (not counted) and become non-adjacent (counted): delta -= 1
    # If they were non-adjacent (counted) and become adjacent (not counted): delta += 1
    
    def check_pair(p1_idx, p2_idx, was_adjacent):
        nonlocal delta
        if sequence[p1_idx] == 'H' and sequence[p2_idx] == 'H':
            if _is_neighbour(chain[p1_idx], chain[p2_idx]):
                if was_adjacent:
                    # Was not counted, now counted
                    delta -= 1
                else:
                    # Was counted, now not counted
                    delta += 1
    # Note: The "was_adjacent" flag depends on the move.
    # For the pairs we are checking, we know their old and new status.
    
    # 1. Pair (i-1, i)
    # Old: Adjacent. New: Non-adjacent (unless i-1 == j, which is impossible for i<j)
    if i > 0:
        check_pair(i-1, i, True)
        
    # 2. Pair (j, j+1)
    # Old: Adjacent. New: Non-adjacent
    if j < len(chain) - 1:
        check_pair(j, j+1, True)
        
    # 3. Pair (i-1, j)
    # Old: Non-adjacent. New: Adjacent
    if i > 0:
        check_pair(i-1, j, False)
        
    # 4. Pair (i, i+1)
    # Old: Non-adjacent (if i+1 <= j, they are internal, but wait.
    # If i+1 <= j, (i, i+1) was adjacent in the old chain? Yes.
    # But in the new chain, (i, i+1) is adjacent?
    # Let's re-verify the new adjacencies.
    # New chain: ... P_{i-1}, P_j, P_{j-1}, ..., P_i, P_{i+1}, ...
    # The pair (i, i+1) in the new chain corresponds to indices i and i+1.
    # Position i is P_i. Position i+1 is P_{j-1} (if j > i).
    # So the pair at indices (i, i+1) is (P_i, P_{j-1}).
    # The pair (P_i, P_{i+1}) is NOT adjacent in the new chain unless j = i+1.
    # If j = i+1, new chain is ... P_{i-1}, P_{i+1}, P_i, P_{i+2} ...
    # Then (P_i, P_{i+1}) is adjacent? No, P_i is at index i+1, P_{i+1} is at index i.
    # So (P_{i+1}, P_i) is adjacent.
    # So the pair (P_i, P_{i+1}) becomes adjacent in the new chain.
    # Was it adjacent in the old chain? Yes.
    # So if it was adjacent and becomes adjacent, no change?
    # Wait.
    # Old: (P_i, P_{i+1}) is adjacent.
    # New: (P_i, P_{i+1}) is adjacent?
    # If j = i+1:
    # Old: ... P_{i-1}, P_i, P_{i+1}, P_{i+2} ...
    # New: ... P_{i-1}, P_{i+1}, P_i, P_{i+2} ...
    # Adjacencies in New: (P_{i-1}, P_{i+1}), (P_{i+1}, P_i), (P_i, P_{i+2}).
    # So (P_i, P_{i+1}) is adjacent (as P_{i+1}, P_i).
    # So status doesn't change.
    
    # If j > i+1:
    # New: ... P_{i-1}, P_j, ..., P_i, P_{i+1}, ...
    # Is (P_i, P_{i+1}) adjacent?
    # P_i is at index j. P_{i+1} is at index j+1? No.
    # Let's map indices.
    # Old index k -> New index:
    # k < i: k
    # i <= k <= j: i + j - k
    # k > j: k
    # So P_i is at new index j.
    # P_{i+1} is at new index i + j - (i+1) = j - 1.
    # So P_{i+1} is at index j-1.
    # P_i is at index j.
    # So (P_{i+1}, P_i) are adjacent in the new chain.
    # So the pair (P_i, P_{i+1}) is always adjacent in both old and new chains (for i < j).
    # So no change in adjacency status for (P_i, P_{i+1}).
    
    # So the only pairs that change status are:
    # (i-1, i): Adj -> Non-Adj
    # (j, j+1): Adj -> Non-Adj
    # (i-1, j): Non-Adj -> Adj
    # (i+1, j): Non-Adj -> Adj  (Wait, is this pair involved?)
    # Let's check (i+1, j).
    # Old: P_{i+1} at i+1, P_j at j. Not adjacent (unless j=i+1).
    # New: P_{i+1} at j-1, P_j at i.
    # Are indices j-1 and i adjacent?
    # If j = i+1, indices are i and i. Same index? No.
    # If j = i+1: P_{i+1} at i, P_j (=P_{i+1}) at i. Wait.
    # If j = i+1:
    # P_i at j = i+1.
    # P_{i+1} at j-1 = i.
    # So P_{i+1} is at i, P_i is at i+1.
    # Pair (P_{i+1}, P_i) is adjacent.
    # Pair (i+1, j) is (P_{i+1}, P_{i+1})? No, j=i+1.
    # The pair is (P_{i+1}, P_j) = (P_{i+1}, P_{i+1}). Same residue. Not a pair.
    
    # So we only need to check:
    # 1. (i-1, i)
    # 2. (j, j+1)
    # 3. (i-1, j)
    # 4. (i+1, j) ?
    # Let's check (i+1, j).
    # Old: Non-adjacent.
    # New: P_{i+1} at j-1. P_j at i.
    # Are j-1 and i adjacent?
    # j-1 - i = j - i - 1.
    # If j = i+1, diff is 0. Not adjacent (same pos? No, different residues).
    # If j = i+2, diff is 1. Adjacent indices.
    # So if j = i+2, (P_{i+1}, P_j) become adjacent.
    # So yes, (i+1, j) can change status.
    
    if i + 1 <= j:
        check_pair(i+1, j, False)
        
    return delta


def _anneal(sequence, chain, dimension, rng, max_iters=1000):
    """Simulated annealing refinement."""
    current = list(chain)
    current_energy = _energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    T_start = 2.0
    T_end = 0.01
    alpha = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    n = len(current)
    
    for _ in range(max_iters):
        if time.time() > _DEADLINE:
            break
            
        # Choose move type
        if rng.random() < 0.8:
            # Reversal
            i = rng.randint(1, n - 2)
            j = rng.randint(i + 1, n - 1)
            new_chain = _reversal_move(current, i, j)
            delta = _delta_energy_reversal(sequence, current, i, j)
            new_energy = current_energy + delta
        else:
            # Pivot
            i = rng.randint(2, n - 2)
            new_chain = _pivot_move(current, i, rng, dimension)
            if new_chain is None:
                continue
            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy
            
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best = list(current)
                best_energy = current_energy
        T *= alpha
        
    return best, best_energy


def fold(sequence, dimension):
    """Best conformation using greedy growth followed by simulated annealing."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0
    
    n = len(sequence)
    if n <= 48:
        n_restarts = 200
    elif n <= 80:
        n_restarts = 150
    else:
        n_restarts = 100
        
    for _ in range(n_restarts):
        if time.time() > _DEADLINE:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        refined, refined_energy = _anneal(sequence, chain, dimension, rng, max_iters=500)
        
        if best is None or refined_energy < best_energy:
            best, best_energy = refined, refined_energy
            
    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        best_energy = _energy(sequence, best)
        
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
