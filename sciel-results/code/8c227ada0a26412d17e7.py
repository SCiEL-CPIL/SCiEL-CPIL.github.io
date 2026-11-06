"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math
import sys

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 195.0  # Safety margin for 210s kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _energy(sequence, chain):
    """Compute the HP contact energy of a conformation.
    Energy = - (number of H-H contacts)
    """
    n = len(chain)
    where = [None] * (2 * n + 10)
    offset = n + 5
    for i, p in enumerate(chain):
        key = (p[0] + offset) * (2 * n + 10) + (p[1] + offset)
        where[key] = i
    contacts = 0
    for i in range(n):
        if sequence[i] == "H":
            x, y = chain[i]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                key = (x + dx + offset) * (2 * n + 10) + (y + dy + offset)
                j = where[key]
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _apply_pivot_fast(chain, pivot_idx, angle):
    """
    Apply a pivot move at pivot_idx with given angle (90, -90, 180).
    Returns None if invalid (collision), else new chain list.
    Optimized with precomputed rotation.
    """
    n = len(chain)
    if pivot_idx < 1 or pivot_idx >= n - 1:
        return None
    
    segment_len = n - pivot_idx - 1
    if segment_len == 0:
        return None
    
    px, py = chain[pivot_idx]
    
    # Precompute fixed set (only need to check moving segment against fixed)
    fixed_set = set(chain[:pivot_idx + 1])
    
    # Determine rotation function
    if angle == 90:
        rot = lambda dx, dy: (-dy, dx)
    elif angle == -90:
        rot = lambda dx, dy: (dy, -dx)
    else:  # 180
        rot = lambda dx, dy: (-dx, -dy)
    
    # Build new chain
    new_chain = chain[:pivot_idx + 1]
    
    # Track occupied for internal collision check of moving segment
    # Since rotation is rigid, internal collisions are preserved,
    # but we need to check against fixed part
    seen_in_segment = set()
    
    for i in range(pivot_idx + 1, n):
        dx, dy = chain[i][0] - px, chain[i][1] - py
        ndx, ndy = rot(dx, dy)
        new_p = (px + ndx, py + ndy)
        
        if new_p in fixed_set:
            return None
        if new_p in seen_in_segment:
            return None
        seen_in_segment.add(new_p)
        new_chain.append(new_p)
    
    return new_chain


def _apply_slither_fast(chain, rng):
    """
    Slither move: Move the last residue to a different valid neighbor.
    """
    n = len(chain)
    if n < 2:
        return None
    
    prev = chain[-2]
    last = chain[-1]
    occupied = set(chain[:-1])
    
    candidates = []
    px, py = prev
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        site = (px + dx, py + dy)
        if site != last and site not in occupied:
            candidates.append(site)
    
    if not candidates:
        return None
        
    new_last = rng.choice(candidates)
    new_chain = chain[:-1] + [new_last]
    return new_chain


def _apply_2step_pivot(chain, pivot_idx, angle1, angle2, rng):
    """
    Two-step pivot: rotate segment at pivot_idx by angle1, then at pivot_idx+1 by angle2.
    This can escape local minima that single pivots cannot.
    """
    n = len(chain)
    if pivot_idx < 1 or pivot_idx >= n - 2:
        return None
    
    # First pivot
    intermediate = _apply_pivot_fast(chain, pivot_idx, angle1)
    if intermediate is None:
        return None
    
    # Second pivot on the intermediate chain
    if pivot_idx + 1 < len(intermediate) - 1:
        result = _apply_pivot_fast(intermediate, pivot_idx + 1, angle2)
        return result
    return None


def _grow_greedy(sequence, bias, rng, seed_direction=None):
    """
    Greedy growth with bias towards contacts and compactness.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    # Start with a random initial direction
    if seed_direction is not None:
        dirs = [seed_direction, (0, 1), (0, -1), (1, 0), (-1, 0)]
        # Remove duplicates and ensure first is preferred
        unique_dirs = []
        for d in dirs:
            if d not in unique_dirs:
                unique_dirs.append(d)
        rng.shuffle(unique_dirs)
        # Put seed_direction first
        unique_dirs.remove(seed_direction)
        unique_dirs.insert(0, seed_direction)
    else:
        unique_dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        rng.shuffle(unique_dirs)
    
    # Try each initial direction
    for start_dir in unique_dirs:
        chain = [[0, 0]]
        occupied = {(0, 0): 0}
        success = True
        
        for i in range(1, n):
            curr = chain[-1]
            options = []
            
            # Compute centroid for compactness
            cx = sum(p[0] for p in chain) / len(chain)
            cy = sum(p[1] for p in chain) / len(chain)
            
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                site = (curr[0] + dx, curr[1] + dy)
                if site in occupied:
                    continue
                
                gain = 0
                if sequence[i] == "H":
                    for dx2, dy2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        q = (site[0] + dx2, site[1] + dy2)
                        if q in occupied:
                            j = occupied[q]
                            if j < i - 1 and sequence[j] == "H":
                                gain += 1
                
                # Compactness: distance to centroid
                dist = math.sqrt((site[0] - cx)**2 + (site[1] - cy)**2)
                
                # Free space heuristic
                free = 0
                for dx3, dy3 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if (site[0] + dx3, site[1] + dy3) not in occupied:
                        free += 1
                
                # Score
                score = bias * gain - 0.3 * dist + 0.15 * free + rng.random() * 0.5
                options.append((score, site))
            
            if not options:
                success = False
                break
            
            # Pick best option
            best_site = max(options, key=lambda x: x[0])[1]
            chain.append(list(best_site))
            occupied[best_site] = i
        
        if success:
            return chain
    
    return None


def _local_search(sequence, chain, time_limit, rng):
    """
    Simulated Annealing local search with multiple move types.
    """
    current_chain = [list(p) for p in chain]
    current_energy = _energy(sequence, current_chain)
    best_chain = [list(p) for p in current_chain]
    best_energy = current_energy
    
    n = len(current_chain)
    
    # Temperature schedule
    T = max(2.0, n * 0.1)
    T_min = 0.05
    
    start_time = time.time()
    
    # Adaptive cooling
    if n > 80:
        cooling = 0.99995
    elif n > 50:
        cooling = 0.9998
    else:
        cooling = 0.9995
    
    iter_count = 0
    no_improve = 0
    stall_limit = 500
    
    while time.time() - start_time < time_limit:
        if _time_left() < 15:
            break
        
        iter_count += 1
        
        # Choose move type
        r = rng.random()
        new_chain = None
        
        if r < 0.6:
            # Single pivot
            pivot_idx = rng.randint(1, n - 2)
            angle = rng.choice([90, -90, 180])
            new_chain = _apply_pivot_fast(current_chain, pivot_idx, angle)
        elif r < 0.8:
            # Slither
            new_chain = _apply_slither_fast(current_chain, rng)
        elif r < 0.9:
            # Two-step pivot
            pivot_idx = rng.randint(1, n - 3)
            angle1 = rng.choice([90, -90, 180])
            angle2 = rng.choice([90, -90, 180])
            new_chain = _apply_2step_pivot(current_chain, pivot_idx, angle1, angle2, rng)
        else:
            # Random pivot with bias towards middle
            pivot_idx = rng.randint(max(1, n // 4), min(n - 2, 3 * n // 4))
            angle = rng.choice([90, -90, 180])
            new_chain = _apply_pivot_fast(current_chain, pivot_idx, angle)
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0:
            # Improving move
            current_chain = new_chain
            current_energy = new_energy
            no_improve = 0
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = [list(p) for p in current_chain]
        else:
            # Worsening move: Metropolis criterion
            if T > T_min:
                # Avoid overflow in exp
                if delta / T < 50:
                    exp_val = math.exp(-delta / T)
                    if rng.random() < exp_val:
                        current_chain = new_chain
                        current_energy = new_energy
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                no_improve += 1
        
        T *= cooling
        
        # Restart if stalled too long
        if no_improve > stall_limit:
            no_improve = 0
            T = max(T, 1.0)
    
    return best_chain


def _polish(sequence, chain, time_limit=2.0):
    """
    Deterministic hill-climbing polish: accept only strictly improving moves.
    Multiple passes to ensure convergence.
    """
    if chain is None:
        return None
    
    current = [list(p) for p in chain]
    curr_e = _energy(sequence, current)
    n = len(current)
    
    start_time = time.time()
    improved_any = True
    
    while improved_any and time.time() - start_time < time_limit:
        if _time_left() < 15:
            break
        improved_any = False
        
        for pivot_idx in range(1, n - 1):
            for angle in (90, -90, 180):
                new_c = _apply_pivot_fast(current, pivot_idx, angle)
                if new_c:
                    new_e = _energy(sequence, new_c)
                    if new_e < curr_e:
                        current = new_c
                        curr_e = new_e
                        improved_any = True
                        break
            if improved_any:
                break
        
        # Also try slither moves in polish
        if not improved_any:
            for _ in range(n):
                new_c = _apply_slither_fast(current, random.Random())
                if new_c:
                    new_e = _energy(sequence, new_c)
                    if new_e < curr_e:
                        current = new_c
                        curr_e = new_e
                        improved_any = True
                        break
    
    return current


def fold(sequence, dimension):
    """
    Main entry point.
    1. Generate multiple diverse initial conformations.
    2. Run local search on the top-K best initial conformations.
    3. Polish the best result.
    4. Return the best result found.
    """
    seq_len = len(sequence)
    
    if seq_len <= 1:
        return [[0, 0]]
    
    remaining = _time_left()
    
    # Emergency: just return a valid self-avoiding walk
    if remaining < 3:
        # Try a simple zigzag
        chain = [[0, 0]]
        occupied = {(0, 0): 0}
        for i in range(1, seq_len):
            placed = False
            for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                site = (chain[-1][0] + dx, chain[-1][1] + dy)
                if site not in occupied:
                    chain.append([site[0], site[1]])
                    occupied[site] = i
                    placed = True
                    break
            if not placed:
                # Fallback: straight line
                return [[i, 0] for i in range(seq_len)]
        return chain
    
    # Time allocation based on remaining time
    if remaining < 10:
        time_for_init = 1.0
        time_for_search = remaining - 3.0
        num_greedy = 10
        top_k = 2
    elif remaining < 30:
        time_for_init = 2.0
        time_for_search = remaining - 4.0
        num_greedy = 20
        top_k = 3
    else:
        time_for_init = 3.0
        time_for_search = remaining - 5.0
        num_greedy = 30
        top_k = 4
    
    # Ensure minimum search time
    time_for_search = max(time_for_search, 2.0)
    
    rng = random.Random(12345 + seq_len)
    
    best_chain = None
    best_energy = float('inf')
    
    # Phase 1: Initialization - Generate diverse candidates
    init_start = time.time()
    candidates = []
    
    # Generate greedy chains with varying biases
    for i in range(num_greedy):
        if time.time() - init_start > time_for_init:
            break
        bias = 0.5 + rng.random() * 2.5
        chain = _grow_greedy(sequence, bias, rng)
        if chain:
            e = _energy(sequence, chain)
            candidates.append((e, chain))
    
    # Also try some random walks
    for i in range(max(5, num_greedy // 3)):
        if time.time() - init_start > time_for_init:
            break
        chain = [[0, 0]]
        occupied = {(0, 0): 0}
        success = True
        for j in range(1, seq_len):
            options = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                site = (chain[-1][0] + dx, chain[-1][1] + dy)
                if site not in occupied:
                    options.append(site)
            if not options:
                success = False
                break
            site = rng.choice(options)
            chain.append([site[0], site[1]])
            occupied[site] = j
        if success:
            e = _energy(sequence, chain)
            candidates.append((e, chain))
    
    # Sort by energy
    candidates.sort(key=lambda x: x[0])
    
    if not candidates:
        # Fallback
        best_chain = [[i, 0] for i in range(seq_len)]
        return best_chain
    
    # Phase 2: Local Search on top candidates
    search_start = time.time()
    
    for idx, (e, chain) in enumerate(candidates[:top_k]):
        if time.time() - search_start > time_for_search:
            break
        if _time_left() < 15:
            break
        
        # Allocate remaining search time evenly among remaining candidates
        remaining_candidates = top_k - idx
        time_per_candidate = (time_for_search - (time.time() - search_start)) / remaining_candidates
        time_per_candidate = max(time_per_candidate, 1.0)
        
        improved = _local_search(sequence, chain, time_per_candidate * 0.95, rng)
        
        if improved:
            e_imp = _energy(sequence, improved)
            if e_imp < best_energy:
                best_energy = e_imp
                best_chain = improved
    
    # Phase 3: Final Polish (Hill Climb)
    if best_chain is not None and _time_left() > 15:
        polish_time = min(3.0, _time_left() - 15)
        polished = _polish(sequence, best_chain, polish_time)
        if polished:
            e_pol = _energy(sequence, polished)
            if e_pol < best_energy:
                best_energy = e_pol
                best_chain = polished
    
    if best_chain is None:
        best_chain = candidates[0][1]
        
    return best_chain
# EVOLVE-BLOCK-END
