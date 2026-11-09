"""HP lattice protein folding: beam search over SAWs with contact-aware scoring + polish SA."""
# EVOLVE-BLOCK-START
import random
import time
from math import exp
from heapq import heappush, heappop, heapreplace

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


def _partial_contact_score(sequence, chain, occupied):
    """
    Score a partial chain by counting H-H contacts formed so far.
    Returns the number of contacts (higher is better for scoring, 
    but we minimize energy = -contacts).
    """
    contacts = 0
    n = len(chain)
    for i in range(n):
        if sequence[i] != "H":
            continue
        p = chain[i]
        for q in _neighbours(p):
            j = occupied.get(q)
            if j is not None and j > i + 1 and sequence[j] == "H":
                contacts += 1
    return contacts


def _beam_search(sequence, dimension, beam_width=50, max_time=8.0):
    """
    Beam search over self-avoiding walks with contact-aware scoring.
    Maintains a beam of the best partial conformations and expands them.
    """
    n = len(sequence)
    if n <= 2:
        return [(0, 0)] + [(i, 0) for i in range(1, n)]
    
    start_time = time.time()
    
    # Each beam entry: (score, chain, occupied_dict)
    # score is negative contacts (so lower is better, matching energy)
    # Start with residue 0 at origin
    initial_chain = [(0, 0)]
    initial_occupied = {(0, 0): 0}
    initial_score = 0  # No contacts possible with 1 residue
    
    # Use a max-heap (negate scores for min-heap behavior)
    beam = [(-initial_score, 0, initial_chain, initial_occupied)]
    
    best_chain = None
    best_energy = 0
    
    for step in range(1, n):
        if time.time() - start_time > max_time or _remaining_time() < 1.0:
            break
        
        new_beam = []
        
        # Expand each state in the beam
        for neg_score, _, chain, occupied in beam:
            last_pos = chain[-1]
            candidates = []
            
            for pos in _neighbours(last_pos):
                if pos in occupied:
                    continue
                
                # Try placing residue `step` at `pos`
                new_occupied = dict(occupied)
                new_occupied[pos] = step
                new_chain = chain + [pos]
                
                # Compute new contacts formed by this placement
                delta_contacts = 0
                if sequence[step] == "H":
                    for q in _neighbours(pos):
                        j = new_occupied.get(q)
                        if j is not None and j < step - 1 and sequence[j] == "H":
                            delta_contacts += 1
                        elif j is not None and j == step - 1:
                            # Chain-adjacent, not a contact
                            pass
                        elif j is not None and j > step:
                            # Future residue, can't count yet
                            pass
                
                new_score = -neg_score - delta_contacts  # Lower is better
                candidates.append((new_score, new_chain, new_occupied))
            
            # Keep only the best candidates from this parent
            if candidates:
                candidates.sort(key=lambda x: x[0])
                for c in candidates[:beam_width // 2 + 1]:
                    new_beam.append((-c[0], step, c[1], c[2]))
        
        # Keep only the top beam_width states
        new_beam.sort(key=lambda x: -x[0])  # Sort by score (lower is better)
        beam = new_beam[:beam_width]
        
        if not beam:
            break
        
        # Track best complete solution if we reached the end
        if step == n - 1:
            for neg_score, _, chain, occupied in beam:
                e = _energy(sequence, chain)
                if e < best_energy:
                    best_energy = e
                    best_chain = chain[:]
    
    if best_chain is not None:
        return best_chain
    
    # If beam search didn't complete, return the best partial extended to full
    if beam:
        # Take the best state and try to complete it greedily
        neg_score, _, chain, occupied = beam[0]
        return _greedy_complete(sequence, chain, occupied)
    
    # Fallback: straight line
    return [(i, 0) for i in range(n)]


def _greedy_complete(sequence, chain, occupied):
    """
    Greedily complete a partial chain by choosing the best next position
    at each step based on contact formation.
    """
    n = len(sequence)
    current_len = len(chain)
    
    for step in range(current_len, n):
        last_pos = chain[-1]
        best_pos = None
        best_score = float('inf')
        
        for pos in _neighbours(last_pos):
            if pos in occupied:
                continue
            
            # Score: prefer positions that form H-H contacts
            score = 0
            if sequence[step] == "H":
                for q in _neighbours(pos):
                    j = occupied.get(q)
                    if j is not None and j < step - 1 and sequence[j] == "H":
                        score -= 1  # Lower is better
            
            if score < best_score:
                best_score = score
                best_pos = pos
        
        if best_pos is None:
            # No valid position; break and return what we have
            break
        
        chain.append(best_pos)
        occupied[best_pos] = step
    
    if len(chain) == n:
        return chain
    
    # If we couldn't complete, return straight line fallback
    return [(i, 0) for i in range(n)]


def _apply_reversal(chain, i, j):
    """
    Apply a reversal move: reverse chain[i:j+1].
    Returns new chain if valid, else None.
    """
    n = len(chain)
    if i < 1 or j >= n or j <= i:
        return None
    
    # Check adjacency constraints
    if abs(chain[i-1][0] - chain[j][0]) + abs(chain[i-1][1] - chain[j][1]) != 1:
        return None
    
    if j < n - 1:
        if abs(chain[i][0] - chain[j+1][0]) + abs(chain[i][1] - chain[j+1][1]) != 1:
            return None
    
    new_chain = chain[:i] + chain[i:j+1][::-1] + chain[j+1:]
    
    # Check self-avoidance
    if len(set(new_chain)) != n:
        return None
    
    return new_chain


def _apply_pivot(chain, pivot):
    """
    Apply a pivot move: rotate the tail at pivot by 90, 180, or 270 degrees.
    Returns list of valid new chains.
    """
    n = len(chain)
    if pivot < 1 or pivot >= n - 2:
        return []
    
    tail = chain[pivot + 1:]
    dx = tail[0][0] - chain[pivot][0]
    dy = tail[0][1] - chain[pivot][1]
    
    rotations = [
        (-dy, dx),    # 90 deg
        (-dx, -dy),   # 180 deg
        (dy, -dx),    # 270 deg
    ]
    
    valid_chains = []
    head_set = set(chain[:pivot+1])
    
    for rd, rd2 in rotations:
        if abs(rd) + abs(rd2) != 1:
            continue
        
        new_tail = [None] * len(tail)
        new_tail[0] = (chain[pivot][0] + rd, chain[pivot][1] + rd2)
        
        if new_tail[0] in head_set:
            continue
        
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
            if len(set(new_chain)) == n:
                valid_chains.append(new_chain)
    
    return valid_chains


def _anneal(sequence, chain, max_steps, rng):
    """
    Simulated annealing with reversal and pivot moves.
    """
    n = len(chain)
    if n < 4:
        return chain
    
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    T_start = 2.5
    T_end = 0.05
    cooling = (T_end / T_start) ** (1.0 / max_steps)
    
    no_improve = 0
    patience = 300
    
    for step in range(max_steps):
        if _remaining_time() < 0.5:
            break
        
        T = T_start * (cooling ** step)
        
        # Choose move type
        if rng.random() < 0.6:
            # Reversal
            i = rng.randint(1, n - 2)
            j = rng.randint(i + 1, n - 1)
            new_chain = _apply_reversal(chain, i, j)
        else:
            # Pivot
            pivot = rng.randint(1, n - 3)
            candidates = _apply_pivot(chain, pivot)
            new_chain = rng.choice(candidates) if candidates else None
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta <= 0 or rng.random() < exp(-delta / T):
            chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > patience:
                no_improve = 0
                if rng.random() < 0.3:
                    chain = best_chain[:]
                    current_energy = best_energy
    
    return best_chain


def _local_polish(sequence, chain, rng, max_iters=200):
    """
    Local polish: try all short reversals and pivots, accept improvements.
    """
    n = len(chain)
    if n < 4:
        return chain
    
    current_chain = chain[:]
    current_energy = _energy(sequence, current_chain)
    
    for _ in range(max_iters):
        if _remaining_time() < 0.3:
            break
        
        improved = False
        
        # Try short reversals
        for length in range(2, min(8, n - 1)):
            for i in range(1, n - length):
                j = i + length - 1
                new_chain = _apply_reversal(current_chain, i, j)
                if new_chain is not None:
                    new_energy = _energy(sequence, new_chain)
                    if new_energy < current_energy:
                        current_chain = new_chain
                        current_energy = new_energy
                        improved = True
                        break
            if improved:
                break
        
        if improved:
            continue
        
        # Try pivots
        for pivot in range(1, n - 2):
            candidates = _apply_pivot(current_chain, pivot)
            for new_chain in candidates:
                new_energy = _energy(sequence, new_chain)
                if new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return current_chain


def fold(sequence, dimension):
    """
    Best conformation using beam search + SA refinement + local polish.
    """
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    # Phase 1: Beam search for good initial conformations
    beam_time = min(6.0, _remaining_time() * 0.4)
    beam_width = 40 if n <= 40 else 25
    
    initial_chain = _beam_search(sequence, dimension, beam_width=beam_width, max_time=beam_time)
    
    if initial_chain is None:
        return [list(p) for p in [(i, 0) for i in range(n)]]
    
    best_chain = initial_chain
    best_energy = _energy(sequence, best_chain)
    
    # Phase 2: SA refinement with multiple starts
    time_budget = _remaining_time() - 2.0
    
    # Estimate steps based on chain length
    base_steps = 5000
    steps_per_run = max(1000, int(base_steps * (60 / max(n, 10))))
    
    # Multiple runs: first from beam search result, then fresh beam searches
    num_runs = 4
    for run in range(num_runs):
        if _remaining_time() < 1.5:
            break
        
        if run == 0:
            start_chain = best_chain[:]
        elif run == 1:
            # Fresh beam search with different beam width
            fresh_time = min(3.0, _remaining_time() * 0.3)
            fresh_bw = 30 if n <= 40 else 20
            fresh_chain = _beam_search(sequence, dimension, beam_width=fresh_bw, max_time=fresh_time)
            if fresh_chain is not None:
                start_chain = fresh_chain
            else:
                start_chain = best_chain[:]
        else:
            # Perturb best chain
            start_chain = best_chain[:]
            for _ in range(5):
                if rng.random() < 0.5:
                    i = rng.randint(1, n - 2)
                    j = rng.randint(i + 1, n - 1)
                    new_chain = _apply_reversal(start_chain, i, j)
                else:
                    pivot = rng.randint(1, n - 3)
                    candidates = _apply_pivot(start_chain, pivot)
                    new_chain = rng.choice(candidates) if candidates else None
                
                if new_chain is not None:
                    start_chain = new_chain
        
        # Run annealing
        improved = _anneal(sequence, start_chain, steps_per_run, rng)
        
        # Local polish
        polished = _local_polish(sequence, improved, rng, max_iters=100)
        
        polished_energy = _energy(sequence, polished)
        if polished_energy < best_energy:
            best_chain = polished
            best_energy = polished_energy
    
    # Final polish on the best chain
    final = _local_polish(sequence, best_chain, rng, max_iters=300)
    final_energy = _energy(sequence, final)
    if final_energy < best_energy:
        best_chain = final
        best_energy = final_energy
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
