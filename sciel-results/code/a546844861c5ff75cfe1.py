import random
import time
import math

# EVOLVE-BLOCK-START

_DEADLINE = 320.0  # safety margin before 330s kill
_start_time = None

def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute HP energy: -1 per H-H contact (non-bonded neighbours)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _grow_greedy(sequence, dimension, rng):
    """Fast greedy growth: prefer sites with new H-H contacts, then free space."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        best_site = None
        best_score = -1
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            score = gain * 10 + free + rng.random()
            if score > best_score:
                best_score = score
                best_site = site
        if best_site is None:
            return None
        occupied[best_site] = i
        chain.append(best_site)
    return chain

def _simulated_annealing(sequence, dimension, chain, rng, time_budget):
    """
    Simulated annealing to improve a chain.
    Moves: pick a random non-terminal residue, try to move it to a neighbouring free site.
    Returns improved chain.
    """
    n = len(sequence)
    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    current_chain = list(chain)
    current_energy = best_energy
    
    # Build occupied set
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    T_start = 1.0
    T_end = 0.01
    iterations = max(1, int(time_budget * 50000))  # ~50k iterations per second
    
    for it in range(iterations):
        # Check time
        if time.time() - _start_time > _DEADLINE - 5:
            break
        
        # Pick a random internal residue (not ends)
        i = rng.randint(1, n - 2)
        old_pos = current_chain[i]
        
        # Try random neighbour
        neighbours = list(_neighbours(old_pos))
        rng.shuffle(neighbours)
        moved = False
        for new_pos in neighbours:
            if new_pos in occupied:
                continue
            # Check that moving i to new_pos doesn't create a bond conflict
            # i must still be adjacent to i-1 and i+1
            # Actually, we're moving residue i, so bonds (i-1,i) and (i,i+1) change
            # This is a "pivot" or "local move" - but standard HP SA uses k-exchanges or local moves
            # Let's do a simpler move: just move the residue if it stays connected
            # For simplicity, let's use a different move: swap two non-adjacent residues
            break
        
        # Actually, let's use a different approach: random local move
        # Pick a residue, try to move it to an adjacent free site
        # But we need to maintain connectivity. Let's use a simpler heuristic.
        
        # Alternative: try moving residue i to a random free site adjacent to old_pos
        # and check if chain remains valid (self-avoiding, unit steps)
        # For unit steps to be maintained, residue i must still be adjacent to i-1 and i+1
        # So new_pos must be adjacent to current_chain[i-1] AND current_chain[i+1]
        
        # This is restrictive. Let's use a different move: k=3 local moves or just accept
        # a simpler approach: try all neighbours of old_pos that are free, and check validity.
        
        for new_pos in neighbours:
            if new_pos in occupied:
                continue
            # Check if i-1 is adjacent to new_pos
            d1 = sum(abs(a - b) for a, b in zip(current_chain[i-1], new_pos))
            d2 = sum(abs(a - b) for a, b in zip(current_chain[i+1], new_pos))
            if d1 != 1 or d2 != 1:
                continue
            # Also check new_pos is not equal to i-1 or i+1 positions
            if new_pos == current_chain[i-1] or new_pos == current_chain[i+1]:
                continue
            
            # Valid move: update
            old_pos_copy = old_pos
            current_chain[i] = new_pos
            del occupied[old_pos]
            occupied[new_pos] = i
            
            # Compute energy change
            # Only contacts involving residue i or its neighbours change
            # For simplicity, recompute energy (expensive but correct)
            new_energy = _energy(sequence, current_chain)
            
            delta_e = new_energy - current_energy
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T_start):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(current_chain)
                moved = True
                break
            else:
                # Revert
                current_chain[i] = old_pos
                del occupied[new_pos]
                occupied[old_pos] = i
        
        if not moved:
            # No valid move found, try a different strategy
            pass
    
    return best_chain

def _local_search(sequence, dimension, chain, rng, time_budget):
    """
    Local search: iteratively try all single-residue moves and take the best improving move.
    """
    n = len(sequence)
    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    current_chain = list(chain)
    current_energy = best_energy
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    while time.time() - _start_time < _DEADLINE - 10:
        improved = False
        best_move = None
        best_move_energy = current_energy
        
        for i in range(1, n - 1):
            old_pos = current_chain[i]
            for new_pos in _neighbours(old_pos):
                if new_pos in occupied:
                    continue
                d1 = sum(abs(a - b) for a, b in zip(current_chain[i-1], new_pos))
                d2 = sum(abs(a - b) for a, b in zip(current_chain[i+1], new_pos))
                if d1 != 1 or d2 != 1:
                    continue
                if new_pos == current_chain[i-1] or new_pos == current_chain[i+1]:
                    continue
                
                # Try the move
                current_chain[i] = new_pos
                del occupied[old_pos]
                occupied[new_pos] = i
                
                new_energy = _energy(sequence, current_chain)
                
                # Revert
                current_chain[i] = old_pos
                del occupied[new_pos]
                occupied[old_pos] = i
                
                if new_energy < best_move_energy:
                    best_move_energy = new_energy
                    best_move = (i, old_pos, new_pos)
        
        if best_move is None:
            break
        
        i, old_pos, new_pos = best_move
        current_chain[i] = new_pos
        del occupied[old_pos]
        occupied[new_pos] = i
        current_energy = best_move_energy
        best_chain = list(current_chain)
        best_energy = current_energy
        improved = True
    
    return best_chain

def fold(sequence, dimension):
    """
    Fold a protein sequence using a combination of greedy growth, simulated annealing,
    and local search.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = len(sequence)
    if n <= 2:
        return [[i] + [0] * (dimension - 1) for i in range(n)]
    
    deadline = _DEADLINE - 10  # leave margin
    
    # Phase 1: Generate multiple diverse initial chains using greedy growth
    rng = random.Random(12345 + len(sequence))
    candidates = []
    
    # Try greedy growth with different random seeds
    num_greedy = min(200, max(50, int((deadline - (time.time() - _start_time)) * 100)))
    for _ in range(num_greedy):
        if time.time() > deadline:
            break
        chain = _grow_greedy(sequence, dimension, rng)
        if chain is not None:
            energy = _energy(sequence, chain)
            candidates.append((energy, chain))
    
    if not candidates:
        # Fallback: straight line
        return [[i] + [0] * (dimension - 1) for i in range(n)]
    
    # Sort by energy (most negative first)
    candidates.sort(key=lambda x: x[0])
    
    # Phase 2: Improve top candidates with local search
    best_chain = candidates[0][1]
    best_energy = candidates[0][0]
    
    # Allocate time for local search on top candidates
    top_k = min(10, len(candidates))
    per_candidate_time = max(0.5, (deadline - (time.time() - _start_time)) / top_k)
    
    for idx in range(top_k):
        if time.time() > deadline:
            break
        cand_energy, cand_chain = candidates[idx]
        if cand_energy > best_energy + 2:
            break  # not worth trying
        
        # Local search
        improved_chain = _local_search(sequence, dimension, cand_chain, rng, per_candidate_time)
        improved_energy = _energy(sequence, improved_chain)
        
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain
    
    # Phase 3: Simulated annealing on the best chain if time remains
    if time.time() < deadline - 5:
        sa_budget = deadline - 5 - time.time()
        if sa_budget > 1:
            sa_chain = _simulated_annealing(sequence, dimension, best_chain, rng, sa_budget)
            sa_energy = _energy(sequence, sa_chain)
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_chain = sa_chain
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
