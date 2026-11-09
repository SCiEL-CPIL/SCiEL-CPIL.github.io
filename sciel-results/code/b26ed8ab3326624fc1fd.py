"""HP lattice protein folding: model-and-optimize with compact growth and multi-start SA."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 200.0


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
    h_positions = set()
    for i, c in enumerate(sequence):
        if c == "H":
            h_positions.add(i)
    contacts = 0
    for i in h_positions:
        p = chain[i]
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and abs(i - j) > 1 and j in h_positions:
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


def _anneal(sequence, initial_chain, time_limit, rng, polish=False):
    """
    Simulated Annealing starting from initial_chain.
    If polish=True, uses a tighter cooling schedule and shorter moves for fine-tuning.
    Returns best_chain, best_energy.
    """
    n = len(initial_chain)
    if n < 3:
        return initial_chain[:], _energy(sequence, initial_chain)
    
    current_chain = initial_chain[:]
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain[:]
    best_energy = current_energy
    
    if polish:
        T = 0.8
        cooling_rate = 0.995
        max_stall = 100
    else:
        T = 2.0
        cooling_rate = 0.999
        max_stall = 200
    
    deadline = time.time() + time_limit
    
    stall_counter = 0
    
    while time.time() < deadline:
        if _remaining_time() < 1.0:
            break
        
        # Choose move type
        if polish:
            # For polish: prefer short reversals and small pivots
            r = rng.random()
            if r < 0.6:
                # Short reversal: length 2-6
                seg_len = rng.randint(2, min(6, n - 1))
                i = rng.randint(0, n - 1 - seg_len)
                j = i + seg_len
                new_chain = _apply_reversal(current_chain, i, j)
                if new_chain is None:
                    continue
            else:
                # Small pivot near the end
                i = rng.randint(max(1, n - 10), n - 2)
                angle = rng.choice([1, 3])
                new_chain = _apply_pivot(current_chain, i, angle)
                if new_chain is None:
                    continue
        else:
            if rng.random() < 0.5:
                # Reversal
                i = rng.randint(0, n - 2)
                j = rng.randint(i + 1, n - 1)
                new_chain = _apply_reversal(current_chain, i, j)
                if new_chain is None:
                    continue
            else:
                # Pivot
                i = rng.randint(1, n - 2)
                angle = rng.choice([1, 2, 3])
                new_chain = _apply_pivot(current_chain, i, angle)
                if new_chain is None:
                    continue
        
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Metropolis criterion
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 0.01)):
            current_chain = new_chain
            current_energy = new_energy
            stall_counter = 0
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain[:]
        else:
            stall_counter += 1
        
        T *= cooling_rate
        if T < 0.03:
            # Reheat occasionally to escape local minima (only in non-polish mode)
            if not polish and rng.random() < 0.3:
                T = 1.5
            else:
                break
        
        if stall_counter > max_stall:
            # Try a larger perturbation
            if not polish:
                if rng.random() < 0.5:
                    i = rng.randint(0, n - 2)
                    j = rng.randint(i + 1, n - 1)
                    new_chain = _apply_reversal(current_chain, i, j)
                    if new_chain is not None:
                        current_chain = new_chain
                        current_energy = _energy(sequence, current_chain)
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = current_chain[:]
            else:
                # In polish mode, just reset stall
                pass
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
    
    # Phase 1: Generate diverse initial chains with different compactness levels
    compactness_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    time_for_growth = 2.5
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
                
    # If no valid chains, fallback
    if not initial_chains:
        initial_chains.append([(i, 0) for i in range(n)])
        
    # Evaluate and sort initial chains by energy
    scored_chains = []
    for chain in initial_chains:
        e = _energy(sequence, chain)
        scored_chains.append((e, chain))
        
    scored_chains.sort(key=lambda x: x[0])
    
    # Keep top K candidates for annealing
    top_k = min(4, len(scored_chains))
    candidates = [c[1] for c in scored_chains[:top_k]]
    
    # Phase 2: Anneal each candidate with allocated time
    # Allocate more time to the best candidate
    total_time_remaining = _remaining_time()
    
    # Give 60% of time to the best candidate, split rest among others
    if len(candidates) == 1:
        times = [total_time_remaining - 2.0]
    else:
        best_time = (total_time_remaining - 2.0) * 0.55
        rest_time = (total_time_remaining - 2.0) * 0.45 / (len(candidates) - 1)
        times = [best_time] + [rest_time] * (len(candidates) - 1)
    
    for idx, chain in enumerate(candidates):
        if _remaining_time() < 1.5:
            break
            
        alloc_time = min(times[idx], _remaining_time() - 1.0)
        if alloc_time < 0.2:
            continue
            
        best_c, best_e = _anneal(sequence, chain, alloc_time, rng, polish=False)
        
        if best_e < best_energy:
            best_energy = best_e
            best_chain = best_c
    
    # Phase 3: Final polish pass on the best chain
    if best_chain is not None and _remaining_time() > 3.0:
        polish_time = min(5.0, _remaining_time() - 1.0)
        if polish_time > 1.0:
            polished_chain, polished_e = _anneal(sequence, best_chain, polish_time, rng, polish=True)
            if polished_e < best_energy:
                best_energy = polished_e
                best_chain = polished_chain
    
    # Fallback
    if best_chain is None:
        if initial_chains:
            best_chain = initial_chains[0]
        else:
            best_chain = [(i, 0) for i in range(n)]
            
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
