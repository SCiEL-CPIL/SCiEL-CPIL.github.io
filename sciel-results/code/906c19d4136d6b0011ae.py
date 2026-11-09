"""HP lattice protein folding: contact-map-guided construction + efficient local refinement."""
# EVOLVE-BLOCK-START
import random
import time
from math import exp
from heapq import heappush, heappop

_START_TIME = time.time()
_DEADLINE = 205.0  # Safety margin before 210s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute the energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _compute_contact_map(sequence):
    """
    Identify high-value H-H contact pairs based on sequence distance.
    Returns a list of (i, j, priority) tuples where i < j, both are H,
    and the distance |i-j| is small enough to be geometrically feasible.
    Priority is higher for shorter distances (more likely to form contacts).
    """
    n = len(sequence)
    h_indices = [i for i in range(n) if sequence[i] == "H"]
    
    contacts = []
    for a in range(len(h_indices)):
        for b in range(a + 1, len(h_indices)):
            i, j = h_indices[a], h_indices[b]
            dist = j - i - 1  # Number of residues between them
            # Minimum path length on lattice is dist+1 (need to go around)
            # For a contact, we need them to be adjacent, so dist+1 must be >= 2
            # Actually, for two residues to be adjacent (not chain-adjacent), 
            # they need at least 2 residues between them in the sequence
            # (i.e., j - i >= 3)
            if j - i >= 3:
                # Priority: shorter distances are more valuable and feasible
                priority = max(0, 10 - dist)
                if priority > 0:
                    contacts.append((i, j, priority))
    
    # Sort by priority descending, then by distance ascending
    contacts.sort(key=lambda x: (-x[2], x[1] - x[0]))
    return contacts


def _guided_growth(sequence, dimension, contact_map, rng, max_time=5.0):
    """
    Grow the chain using a priority queue that favors positions
    that can form high-priority H-H contacts.
    """
    n = len(sequence)
    if n <= 2:
        return [(0, 0)] + [(i, 0) for i in range(1, n)]
    
    start_time = time.time()
    
    # Build a lookup: for each residue index, what contacts can it form?
    contact_by_residue = {}
    for i, j, priority in contact_map:
        if i not in contact_by_residue:
            contact_by_residue[i] = []
        contact_by_residue[i].append((j, priority))
        if j not in contact_by_residue:
            contact_by_residue[j] = []
        contact_by_residue[j].append((i, priority))
    
    best_chain = None
    best_energy = 0
    
    # Try multiple growth strategies
    num_attempts = 8
    
    for attempt in range(num_attempts):
        if time.time() - start_time > max_time or _remaining_time() < 1.0:
            break
        
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        
        # Use a priority queue for candidate positions
        # Each entry: (negative_score, position, residue_index)
        # We'll grow step by step, choosing the best next position
        
        def score_position(pos, idx, chain, occupied):
            """Score a candidate position for placing residue idx."""
            score = 0.0
            
            if sequence[idx] != "H":
                return 0.0
            
            # Check if placing here forms any H-H contacts
            for q in _neighbours(pos):
                if q in occupied:
                    j = occupied[q]
                    if j < idx - 1 and sequence[j] == "H":
                        # This is a potential contact
                        # Boost score if it's a high-priority contact
                        priority = 1
                        if idx in contact_by_residue:
                            for (other_idx, pri) in contact_by_residue[idx]:
                                if other_idx == j:
                                    priority = pri
                                    break
                        score += priority * 2.0
            
            # Small bonus for being close to other H residues (for future contacts)
            if sequence[idx] == "H":
                for q in _neighbours(pos):
                    if q in occupied:
                        j = occupied[q]
                        if sequence[j] == "H" and j < idx - 1:
                            score += 0.5
            
            return score
        
        # Greedy growth with occasional randomness
        for idx in range(1, n):
            if time.time() - start_time > max_time or _remaining_time() < 1.0:
                break
            
            last_pos = chain[-1]
            candidates = []
            for pos in _neighbours(last_pos):
                if pos not in occupied:
                    s = score_position(pos, idx, chain, occupied)
                    candidates.append((s, pos))
            
            if not candidates:
                break
            
            # Sort by score descending, with some randomness
            candidates.sort(key=lambda x: -x[0])
            
            # Choose from top candidates with some probability
            if rng.random() < 0.3 and len(candidates) > 1:
                # Pick randomly from top 3
                top_k = min(3, len(candidates))
                chosen = rng.choice(candidates[:top_k])[1]
            else:
                chosen = candidates[0][1]
            
            chain.append(chosen)
            occupied[chosen] = idx
        
        if len(chain) == n:
            e = _energy(sequence, chain)
            if e < best_energy:
                best_energy = e
                best_chain = chain[:]
    
    if best_chain is not None:
        return best_chain
    else:
        # Fallback: straight line
        return [(i, 0) for i in range(n)]


def _local_move_fast(sequence, chain, rng):
    """
    Fast local move: try reversal or pivot on a random segment.
    Returns (new_chain, energy_delta) or (None, 0) if no valid move.
    """
    n = len(chain)
    if n < 4:
        return None, 0
    
    move_type = rng.random()
    
    if move_type < 0.5:
        # Reversal move: reverse a segment of the chain
        # Pick two cut points
        i = rng.randint(1, n - 2)
        j = rng.randint(i + 1, n - 1)
        
        # Check if reversal is valid
        # The segment chain[i:j+1] is reversed
        # Need to check:
        # 1. chain[i-1] is adjacent to chain[j] (new first of reversed segment)
        # 2. chain[i] is adjacent to chain[j+1] (new last of reversed segment, if j < n-1)
        # 3. No self-avoidance violations within the reversed segment
        
        # Check adjacency constraints
        if abs(chain[i-1][0] - chain[j][0]) + abs(chain[i-1][1] - chain[j][1]) != 1:
            return None, 0
        
        if j < n - 1:
            if abs(chain[i][0] - chain[j+1][0]) + abs(chain[i][1] - chain[j+1][1]) != 1:
                return None, 0
        
        # Build new chain
        new_chain = chain[:i] + chain[i:j+1][::-1] + chain[j+1:]
        
        # Check self-avoidance
        if len(set(new_chain)) != n:
            return None, 0
        
        # Compute energy delta
        old_energy = _energy(sequence, chain)
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - old_energy
        
        return new_chain, delta
    
    else:
        # Pivot move: rotate the tail at a pivot point
        pivot = rng.randint(1, n - 3)
        tail = chain[pivot + 1:]
        
        # Get the direction from pivot to tail[0]
        dx = tail[0][0] - chain[pivot][0]
        dy = tail[0][1] - chain[pivot][1]
        
        # Try rotating by 90, 180, 270 degrees
        rotations = [
            (-dy, dx),    # 90 deg
            (-dx, -dy),   # 180 deg
            (dy, -dx),    # 270 deg
        ]
        
        for rd, rd2 in rotations:
            if abs(rd) + abs(rd2) != 1:
                continue
            
            new_tail = [None] * len(tail)
            new_tail[0] = (chain[pivot][0] + rd, chain[pivot][1] + rd2)
            
            # Check if new_tail[0] collides with head
            head_set = set(chain[:pivot+1])
            if new_tail[0] in head_set:
                continue
            
            # Check if the rest of the tail can be placed
            valid = True
            for k in range(1, len(tail)):
                odx = tail[k][0] - tail[k-1][0]
                ody = tail[k][1] - tail[k-1][1]
                new_tail[k] = (new_tail[k-1][0] + odx, new_tail[k-1][1] + ody)
                
                if new_tail[k] in head_set or new_tail[k] in new_tail[:k]:
                    valid = False
                    break
            
            if valid:
                new_chain = chain[:pivot+1] + new_tail
                
                # Check self-avoidance
                if len(set(new_chain)) != n:
                    continue
                
                old_energy = _energy(sequence, chain)
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - old_energy
                
                return new_chain, delta
    
    return None, 0


def _anneal_fast(sequence, chain, max_steps, rng):
    """
    Fast simulated annealing with efficient local moves.
    """
    n = len(chain)
    if n < 4:
        return chain
    
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    # Adaptive temperature schedule
    T_start = 3.0
    T_end = 0.05
    cooling = (T_end / T_start) ** (1.0 / max_steps)
    
    no_improve = 0
    patience = 200
    
    for step in range(max_steps):
        if _remaining_time() < 0.5:
            break
        
        T = T_start * (cooling ** step)
        
        new_chain, delta = _local_move_fast(sequence, chain, rng)
        if new_chain is None:
            continue
        
        if delta <= 0 or rng.random() < exp(-delta / T):
            chain = new_chain
            current_energy += delta
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
                no_improve = 0
            else:
                no_improve += 1
            
            # Restart if stuck for too long
            if no_improve > patience:
                no_improve = 0
                # Occasionally restart from best
                if rng.random() < 0.3:
                    chain = best_chain[:]
                    current_energy = best_energy
    
    return best_chain


def fold(sequence, dimension):
    """
    Best conformation using contact-map-guided growth + efficient local refinement.
    """
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    # Compute contact map to guide growth
    contact_map = _compute_contact_map(sequence)
    
    # Phase 1: Guided growth to get initial solutions
    growth_time = min(4.0, _remaining_time() * 0.3)
    initial_chain = _guided_growth(sequence, dimension, contact_map, rng, max_time=growth_time)
    
    if initial_chain is None:
        # Fallback: straight line
        return [list(p) for p in [(i, 0) for i in range(n)]]
    
    best_chain = initial_chain
    best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Local refinement with simulated annealing
    time_budget = _remaining_time() - 2.0
    
    # Estimate steps based on chain length
    # Shorter chains can afford more steps per unit time
    base_steps = 3000
    steps_per_run = max(500, int(base_steps * (80 / max(n, 10))))
    
    # Multiple annealing runs, starting from best and sometimes from fresh growth
    num_runs = 5
    for run in range(num_runs):
        if _remaining_time() < 1.0:
            break
        
        if run == 0:
            # First run: refine the initial solution
            start_chain = best_chain[:]
        elif run == 1:
            # Second run: fresh guided growth
            fresh_time = min(2.0, _remaining_time() * 0.2)
            fresh_chain = _guided_growth(sequence, dimension, contact_map, rng, max_time=fresh_time)
            if fresh_chain is not None:
                start_chain = fresh_chain
            else:
                start_chain = best_chain[:]
        else:
            # Subsequent runs: start from best, with some perturbation
            start_chain = best_chain[:]
            # Perturb the chain
            for _ in range(3):
                new_chain, _ = _local_move_fast(sequence, start_chain, rng)
                if new_chain is not None:
                    start_chain = new_chain
        
        # Run annealing
        improved = _anneal_fast(sequence, start_chain, steps_per_run, rng)
        improved_energy = _energy(sequence, improved)
        
        if improved_energy < best_energy:
            best_chain = improved
            best_energy = improved_energy
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
