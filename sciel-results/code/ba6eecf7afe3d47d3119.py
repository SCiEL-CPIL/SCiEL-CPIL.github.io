"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 200.0  # Safety margin for 210s kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _neighbours_tuple(point):
    """Return the 4 neighbours of a 2D lattice point as a tuple."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy(sequence, chain):
    """Compute the HP contact energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y = p
            # Unroll for speed
            j1 = where.get((x + 1, y))
            if j1 is not None and j1 > i + 1 and sequence[j1] == "H":
                contacts += 1
            j2 = where.get((x - 1, y))
            if j2 is not None and j2 > i + 1 and sequence[j2] == "H":
                contacts += 1
            j3 = where.get((x, y + 1))
            if j3 is not None and j3 > i + 1 and sequence[j3] == "H":
                contacts += 1
            j4 = where.get((x, y - 1))
            if j4 is not None and j4 > i + 1 and sequence[j4] == "H":
                contacts += 1
    return -contacts


def _grow(sequence, dimension, rng, bias=1.0):
    """One randomized greedy chain growth."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    
    # Precompute H indices for faster check? No, sequence[i] is fast.
    
    for i in range(1, n):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            gain = 0
            if sequence[i] == "H":
                x, y = site
                # Check neighbors for H
                if sequence[occupied.get((x + 1, y), -1)] == "H" if (x+1,y) in occupied else False: pass # This is slow
                # Better:
                for nb in _neighbours_tuple(site):
                    if nb in occupied:
                        idx = occupied[nb]
                        if idx < i - 1 and sequence[idx] == "H":
                            gain += 1
            
            # Heuristic: Compactness
            # Use distance to last H or centroid
            # For speed, use distance to last point (which is 1) -> not useful
            # Use distance to centroid of last 10 points?
            # Let's just use gain and some randomness, plus a slight preference for staying in a box
            # Box constraint: keep x, y within [-n/2, n/2]
            x, y = site
            box_penalty = 0
            limit = n // 2 + 2
            if abs(x) > limit or abs(y) > limit:
                box_penalty = 5.0
            
            score = bias * gain - box_penalty + rng.random() * 2.0
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick best with some randomness (softmax-like or just max with tie-break)
        # To allow diversity, pick from top 3 if scores are close
        options.sort(key=lambda x: x[0], reverse=True)
        if len(options) > 1 and rng.random() < 0.2:
             # Pick randomly from top half
             k = max(1, len(options) // 2)
             site = options[rng.randint(0, k-1)][1]
        else:
             site = options[0][1]
        
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move(chain, rng):
    """Apply a pivot move. Returns new chain or None."""
    n = len(chain)
    if n < 4:
        return None
    pivot_idx = rng.randint(1, n - 2)
    pivot = chain[pivot_idx]
    segment_start = pivot_idx + 1
    segment = chain[segment_start:]
    fixed_set = set(chain[:pivot_idx + 1])
    
    # Try 3 rotations
    for angle in (90, -90, 180):
        rotated = []
        valid = True
        for p in segment:
            dx, dy = p[0] - pivot[0], p[1] - pivot[1]
            if angle == 90:
                ndx, ndy = -dy, dx
            elif angle == -90:
                ndx, ndy = dy, -dx
            else:
                ndx, ndy = -dx, -dy
            new_p = (pivot[0] + ndx, pivot[1] + ndy)
            if new_p in fixed_set:
                valid = False
                break
            rotated.append(new_p)
        
        if valid:
            return chain[:segment_start] + rotated
    return None


def _crankshaft_move(chain, rng):
    """Apply a crankshaft move (rotate a short segment)."""
    n = len(chain)
    if n < 5:
        return None
    # Segment length 3 or 4
    seg_len = rng.choice([3, 4])
    if n < seg_len + 2:
        return None
    start_idx = rng.randint(1, n - seg_len - 1)
    pivot = chain[start_idx]
    segment = chain[start_idx + 1 : start_idx + 1 + seg_len]
    fixed_set = set(chain[:start_idx + 1] + chain[start_idx + 1 + seg_len:])
    
    for angle in (90, -90, 180):
        rotated = []
        valid = True
        for p in segment:
            dx, dy = p[0] - pivot[0], p[1] - pivot[1]
            if angle == 90:
                ndx, ndy = -dy, dx
            elif angle == -90:
                ndx, ndy = dy, -dx
            else:
                ndx, ndy = -dx, -dy
            new_p = (pivot[0] + ndx, pivot[1] + ndy)
            if new_p in fixed_set:
                valid = False
                break
            rotated.append(new_p)
        
        if valid:
            return chain[:start_idx + 1] + rotated + chain[start_idx + 1 + seg_len:]
    return None


def _slither_move(chain, rng):
    """Try to move the end of the chain."""
    n = len(chain)
    if n < 4:
        return None
    
    last = chain[-1]
    prev = chain[-2]
    occupied = set(chain[:-1])
    
    for site in _neighbours(prev):
        if site in occupied:
            continue
        return chain[:-1] + [site]
    return None


def _local_search(sequence, chain, max_iterations=5000, time_limit=5.0, rng=None):
    """Simulated Annealing local search."""
    if chain is None:
        return None

    if rng is None:
        rng = random.Random(42)

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    temperature = 5.0
    cooling = 0.9995
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 10:
            break

        # Choose move
        r = rng.random()
        if r < 0.5:
            new_chain = _pivot_move(current_chain, rng)
        elif r < 0.8:
            new_chain = _crankshaft_move(current_chain, rng)
        else:
            new_chain = _slither_move(current_chain, rng)

        if new_chain is None:
            continue

        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy

        # Metropolis criterion
        if delta < 0 or rng.random() < math.exp(-delta / max(temperature, 0.1)):
            current_chain = new_chain
            current_energy = new_energy

            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)

        temperature *= cooling

    return best_chain


def _iterated_local_search(sequence, initial_chain, time_limit, rng):
    """ILS: Run SA, perturb, run SA again."""
    current_chain = list(initial_chain)
    best_chain = list(initial_chain)
    best_energy = _energy(sequence, best_chain)
    
    start_time = time.time()
    
    while _time_left() > 15 and time.time() - start_time < time_limit:
        # Run SA
        improved = _local_search(sequence, current_chain, max_iterations=1000, time_limit=2.0, rng=rng)
        if improved is not None:
            e = _energy(sequence, improved)
            if e < best_energy:
                best_energy = e
                best_chain = list(improved)
        
        # Perturb the current solution (or best if we want to refine best)
        # Perturb current to explore new areas
        perturbations = rng.randint(3, 8)
        for _ in range(perturbations):
            move_type = rng.random()
            if move_type < 0.6:
                new_c = _pivot_move(current_chain, rng)
            elif move_type < 0.8:
                new_c = _crankshaft_move(current_chain, rng)
            else:
                new_c = _slither_move(current_chain, rng)
            
            if new_c is not None:
                current_chain = new_c
        
        # If we got stuck (no improvement in SA), maybe restart from best?
        # For now, keep exploring from perturbed current.

    return best_chain


def fold(sequence, dimension):
    """Best conformation using multiple strategies."""
    seq_len = len(sequence)
    rng = random.Random(12345 + seq_len)
    
    best_chain = None
    best_energy = 0

    # Determine number of starts and time per start
    if seq_len <= 30:
        num_starts = 20
        time_per_start = 5.0
    elif seq_len <= 50:
        num_starts = 15
        time_per_start = 4.0
    elif seq_len <= 70:
        num_starts = 10
        time_per_start = 3.0
    else:
        num_starts = 8
        time_per_start = 2.5

    # Pass 1: Generate diverse starts and optimize the best few
    candidates = []
    for i in range(num_starts):
        if _time_left() < 20:
            break
        bias = 1.0 + rng.random() * 2.0
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is not None:
            e = _energy(sequence, chain)
            candidates.append((e, chain))
    
    if not candidates:
        return [list(p) for p in [(i, 0) for i in range(seq_len)]]
    
    # Sort by energy and take top few for ILS
    candidates.sort(key=lambda x: x[0])
    
    # Allocate time to top candidates
    time_left = _time_left() - 25
    if time_left < 0:
        time_left = 5.0
        
    # Give more time to better candidates
    # Simple strategy: iterate top 3 candidates, spend proportional time
    top_k = min(3, len(candidates))
    time_per_candidate = time_left / top_k
    
    for i in range(top_k):
        if _time_left() < 20:
            break
        e, chain = candidates[i]
        if e < best_energy:
            best_energy = e
            best_chain = chain
            
        # Run ILS on this chain
        improved = _iterated_local_search(sequence, chain, time_per_candidate, rng)
        if improved is not None:
            e_imp = _energy(sequence, improved)
            if e_imp < best_energy:
                best_energy = e_imp
                best_chain = improved

    # Final Polish
    if best_chain is not None and _time_left() > 10:
        polished = _local_search(sequence, best_chain, max_iterations=5000, time_limit=min(5.0, _time_left() - 5), rng=rng)
        if polished is not None:
            e_pol = _energy(sequence, polished)
            if e_pol < best_energy:
                best_energy = e_pol
                best_chain = polished

    if best_chain is None:
        best_chain = [(i, 0) for i in range(seq_len)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
