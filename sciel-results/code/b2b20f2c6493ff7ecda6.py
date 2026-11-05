"""HP lattice protein folding on the square lattice using CSP-style backtracking with contact-aware heuristics."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_TOTAL_BUDGET = 195.0  # Safety margin before 210s kill


def _energy(sequence, chain):
    """Compute HP contact energy: -1 for each H-H pair on neighbouring lattice sites not adjacent in chain."""
    where = {p: i for i, p in enumerate(chain)}
    h_indices = [i for i, s in enumerate(sequence) if s == "H"]
    contacts = 0
    for i in h_indices:
        x, y = chain[i]
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            j = where.get((nx, ny))
            if j is not None and j > i + 1 and sequence[j] == "H":
                contacts += 1
    return -contacts


def _backtrack_fold(sequence, rng, time_limit, max_nodes=200000):
    """
    CSP-style backtracking search with forward checking and contact-aware heuristic.
    Each residue's position is a variable; domain is the 4 lattice neighbors.
    Prune domains based on self-avoidance. Heuristic prioritizes placements
    that maximize potential H-H contacts with already-placed H residues.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    current_contacts = 0
    
    # Precompute H positions for quick lookup
    h_set = set(i for i, s in enumerate(sequence) if s == "H")
    
    best_chain = list(chain)
    best_energy = 0  # Will be negative or zero
    
    # We'll do a depth-first search with backtracking
    # State: chain, occupied, current_contacts
    
    # For efficiency, track contacts incrementally
    # When we place residue i at position p, new contacts are:
    #   for each neighbor q of p that is occupied by residue j < i-1:
    #     if sequence[i]=='H' and sequence[j]=='H': +1 contact
    
    nodes_visited = 0
    start_time = time.time()
    
    # We use a recursive approach with explicit stack for large n
    # But for simplicity, let's use recursion with a node limit
    
    def _search(depth, chain, occupied, current_contacts):
        nonlocal nodes_visited, best_chain, best_energy
        
        if time.time() - start_time > time_limit:
            return False
        if nodes_visited > max_nodes:
            return False
        
        nodes_visited += 1
        
        if depth == n:
            # Complete chain found
            e = -current_contacts
            if e < best_energy:
                best_energy = e
                best_chain = list(chain)
            return True
        
        # Get possible positions for residue at depth
        last_pos = chain[-1]
        lx, ly = last_pos
        candidates = [(lx + 1, ly), (lx - 1, ly), (lx, ly + 1), (lx, ly - 1)]
        
        # Filter: must not be occupied
        valid = []
        for pos in candidates:
            if pos not in occupied:
                valid.append(pos)
        
        if not valid:
            return False
        
        # Score each candidate by potential new H-H contacts
        scored = []
        for pos in valid:
            px, py = pos
            gain = 0
            if sequence[depth] == "H":
                # Check neighbors of pos for already-placed H residues
                for nx, ny in ((px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)):
                    j = occupied.get((nx, ny))
                    if j is not None and j < depth - 1 and sequence[j] == "H":
                        gain += 1
            # Also consider future potential: how many H residues are yet to be placed
            # that could potentially contact this position
            # This is a soft heuristic
            scored.append((-gain, rng.random(), pos))
        
        # Sort by score (most negative gain first = most contacts)
        scored.sort()
        
        # Try each candidate in order
        for _, _, pos in scored:
            # Forward check: ensure that placing this residue doesn't
            # completely block the next residue (simple check)
            chain.append(pos)
            occupied[pos] = depth
            
            # Compute new contacts
            new_contacts = current_contacts
            if sequence[depth] == "H":
                px, py = pos
                for nx, ny in ((px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)):
                    j = occupied.get((nx, ny))
                    if j is not None and j < depth - 1 and sequence[j] == "H":
                        new_contacts += 1
            
            # Pruning: if even with all remaining H's making contacts, 
            # we can't beat best_energy, skip
            remaining_h = sum(1 for i in range(depth + 1, n) if sequence[i] == "H")
            # Max possible additional contacts is bounded by remaining_h * 2 (each H can have up to 2 new contacts)
            # This is very loose; use a tighter bound
            if -new_contacts - remaining_h * 2 >= best_energy and best_energy < 0:
                # Can't beat best, but this is very loose so don't prune aggressively
                pass
            
            if depth + 1 < n:
                # Check if next residue has at least one valid position
                nx_, ny_ = pos
                has_valid_next = False
                for nnx, nny in ((nx_ + 1, ny_), (nx_ - 1, ny_), (nx_, nny + 1), (nx_, nny - 1)):
                    if (nnx, nny) not in occupied:
                        has_valid_next = True
                        break
                if not has_valid_next:
                    # Dead end
                    chain.pop()
                    del occupied[pos]
                    continue
            
            if _search(depth + 1, chain, occupied, new_contacts):
                # Continue searching for better solutions
                chain.pop()
                del occupied[pos]
                # Check time
                if time.time() - start_time > time_limit:
                    return True
            else:
                chain.pop()
                del occupied[pos]
        
        return True
    
    # For small sequences, do full backtracking
    if n <= 25:
        max_nodes = 500000
    elif n <= 40:
        max_nodes = 300000
    else:
        max_nodes = 200000
    
    _search(0, chain, occupied, current_contacts)
    
    return best_chain


def _greedy_grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        lx, ly = chain[-1]
        for site in ((lx + 1, ly), (lx - 1, ly), (lx, ly + 1), (lx, ly - 1)):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for qx, qy in ((site[0] + 1, site[1]), (site[0] - 1, site[1]),
                                             (site[0], site[1] + 1), (site[0], site[1] - 1))
                           if (qx, qy) in occupied and occupied[(qx, qy)] < i - 1 and sequence[occupied[(qx, qy)]] == "H")
            free = sum(1 for qx, qy in ((site[0] + 1, site[1]), (site[0] - 1, site[1]),
                                        (site[0], site[1] + 1), (site[0], site[1] - 1))
                       if (qx, qy) not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _try_crankshaft(sequence, chain, k1, k2, rot_choice):
    """
    Attempt a crankshaft move: rotate the segment chain[k1+1..k2] around the pivot at chain[k1].
    """
    n = len(chain)
    if k2 - k1 < 2:
        return None, None
    
    px, py = chain[k1]
    
    if rot_choice == 0:
        def rot(x, y): return (-y, x)
    elif rot_choice == 1:
        def rot(x, y): return (-x, -y)
    else:
        def rot(x, y): return (y, -x)
    
    prefix = chain[:k1 + 1]
    suffix = chain[k2 + 1:]
    seg = chain[k1 + 1:k2 + 1]
    
    fixed_occupied = set(prefix) | set(suffix)
    
    new_seg = []
    for j in range(len(seg)):
        ox = seg[j][0] - px
        oy = seg[j][1] - py
        nx, ny = rot(ox, oy)
        new_pos = (px + nx, py + ny)
        new_seg.append(new_pos)
    
    for pos in new_seg:
        if pos in fixed_occupied:
            return None, None
    if len(set(new_seg)) != len(new_seg):
        return None, None
    
    new_chain = prefix + new_seg + suffix
    
    seen = set()
    for i, p in enumerate(new_chain):
        if p in seen:
            return None, None
        seen.add(p)
        if i > 0:
            dx = abs(new_chain[i][0] - new_chain[i - 1][0])
            dy = abs(new_chain[i][1] - new_chain[i - 1][1])
            if dx + dy != 1:
                return None, None
    
    new_energy = _energy(sequence, new_chain)
    return new_chain, new_energy


def _try_pivot_move(sequence, chain, k, rot_choice):
    """
    Attempt a pivot move at index k with a specific rotation.
    """
    n = len(chain)
    if k >= n - 1:
        return None, None
    
    px, py = chain[k]
    
    if rot_choice == 0:
        def rot(x, y): return (-y, x)
    elif rot_choice == 1:
        def rot(x, y): return (-x, -y)
    else:
        def rot(x, y): return (y, -x)
    
    prefix_occupied = set(chain[:k + 1])
    
    new_chain = list(chain[:k + 1])
    
    for j in range(k + 1, n):
        ox = chain[j][0] - px
        oy = chain[j][1] - py
        nx, ny = rot(ox, oy)
        new_pos = (px + nx, py + ny)
        
        if new_pos in prefix_occupied:
            return None, None
        
        new_chain.append(new_pos)
    
    seen = set()
    for i, p in enumerate(new_chain):
        if p in seen:
            return None, None
        seen.add(p)
        if i > 0:
            dx = abs(new_chain[i][0] - new_chain[i - 1][0])
            dy = abs(new_chain[i][1] - new_chain[i - 1][1])
            if dx + dy != 1:
                return None, None
    
    new_energy = _energy(sequence, new_chain)
    return new_chain, new_energy


def _pivot_search(sequence, chain, time_limit, rng):
    """
    Monte Carlo search using pivot and crankshaft moves with simulated annealing.
    """
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    T_start = 2.0
    T_end = 0.01
    start_time = time.time()
    
    max_iters = 300000
    cooling_factor = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        move_type = rng.random()
        
        if move_type < 0.4 and n > 6:
            k1 = rng.randint(0, n - 3)
            k2 = rng.randint(k1 + 2, n - 1)
            rot_choice = rng.randint(0, 2)
            new_chain, new_energy = _try_crankshaft(sequence, current_chain, k1, k2, rot_choice)
        else:
            k = rng.randint(0, n - 2)
            rot_choice = rng.randint(0, 2)
            new_chain, new_energy = _try_pivot_move(sequence, current_chain, k, rot_choice)
        
        if new_chain is None:
            continue
        
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        T *= cooling_factor
    
    return best_chain, best_energy


def _greedy_polish(sequence, chain, max_rounds=100):
    """
    Greedy polish: repeatedly scan pivot and crankshaft moves, accept improvements.
    """
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    for _ in range(max_rounds):
        improved = False
        
        for k in range(n - 1):
            for rot_choice in range(3):
                new_chain, new_energy = _try_pivot_move(sequence, current_chain, k, rot_choice)
                if new_chain is not None and new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    improved = True
                    break
            if improved:
                break
        
        if n > 6:
            for k1 in range(n - 2):
                for k2 in range(k1 + 2, n):
                    for rot_choice in range(3):
                        new_chain, new_energy = _try_crankshaft(sequence, current_chain, k1, k2, rot_choice)
                        if new_chain is not None and new_energy < current_energy:
                            current_chain = new_chain
                            current_energy = new_energy
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return current_chain, current_energy


def fold(sequence, dimension):
    """Fold the sequence into a low-energy conformation."""
    n = len(sequence)
    if n == 0:
        return []
    
    elapsed = time.time() - _GLOBAL_START
    if elapsed > _TOTAL_BUDGET:
        return [[i, 0] for i in range(n)]
    
    remaining = _TOTAL_BUDGET - elapsed
    
    if remaining < 2.0:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0
    
    # Allocate time between backtracking, greedy+SA, and polish
    # For short sequences, backtracking is powerful
    if n <= 25:
        bt_time = remaining * 0.40
        sa_time = remaining * 0.40
        polish_time = remaining * 0.15
    elif n <= 40:
        bt_time = remaining * 0.25
        sa_time = remaining * 0.55
        polish_time = remaining * 0.15
    else:
        bt_time = remaining * 0.10
        sa_time = remaining * 0.70
        polish_time = remaining * 0.15
    
    # Phase 1: CSP-style backtracking (most effective for short sequences)
    if bt_time > 0.5:
        bt_chain = _backtrack_fold(sequence, rng, bt_time)
        if bt_chain is not None and len(bt_chain) == n:
            bt_energy = _energy(sequence, bt_chain)
            best_chain = bt_chain
            best_energy = bt_energy
    
    # Phase 2: Multiple greedy restarts + SA
    if sa_time > 0.5 and time.time() - _GLOBAL_START < _TOTAL_BUDGET - 5.0:
        if n <= 30:
            num_restarts = 4
        elif n <= 50:
            num_restarts = 3
        else:
            num_restarts = 2
        time_per_restart = sa_time / num_restarts
        
        for k in range(num_restarts):
            if time.time() - _GLOBAL_START > _TOTAL_BUDGET - 10.0:
                break
            
            init_rng = random.Random(12345 + n + k * 100)
            initial = _greedy_grow(sequence, dimension, init_rng)
            
            if initial is None:
                initial = [(i, 0) for i in range(n)]
            
            refined_chain, refined_energy = _pivot_search(sequence, initial, time_per_restart, rng)
            
            if refined_chain is not None:
                if best_chain is None or refined_energy < best_energy:
                    best_chain = refined_chain
                    best_energy = refined_energy
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 3: Polish
    if time.time() - _GLOBAL_START < _TOTAL_BUDGET - 3.0:
        polished_chain, polished_energy = _greedy_polish(sequence, best_chain, max_rounds=150)
        if polished_energy < best_energy:
            best_chain = polished_chain
            best_energy = polished_energy
    
    return [list(p) for p in best_chain]


# EVOLVE-BLOCK-END
