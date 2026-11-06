"""HP lattice protein folding: fragment re-insertion + reversal moves with fast energy."""
# EVOLVE-BLOCK-START
import time
import random
import math

_T0 = time.time()
_DEADLINE = 195.0  # 210s wall-clock limit, 15s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _energy_fast(seq, chain):
    """Fast energy using a position->index map and precomputed H indices."""
    where = {p: i for i, p in enumerate(chain)}
    h_set = set(i for i, c in enumerate(seq) if c == "H")
    contacts = 0
    for i in h_set:
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = where.get((x + dx, y + dy))
            if j is not None and j > i + 1 and j in h_set:
                contacts += 1
    return -contacts


def _is_valid_chain(chain):
    """Check self-avoidance and unit steps."""
    n = len(chain)
    if n < 2:
        return True
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            dx = abs(chain[i][0] - chain[i - 1][0])
            dy = abs(chain[i][1] - chain[i - 1][1])
            if dx + dy != 1:
                return False
    return True


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    h_set = set(i for i, c in enumerate(sequence) if c == "H")
    for i in range(1, n):
        options = []
        x, y = chain[-1]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            site = (x + dx, y + dy)
            if site in occupied:
                continue
            gain = 0
            if i in h_set:
                sx, sy = site
                for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q = (sx + adx, sy + ady)
                    if q in occupied:
                        idx = occupied[q]
                        if idx < i - 1 and idx in h_set:
                            gain += 1
            free = 0
            sx, sy = site
            for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (sx + adx, sy + ady) not in occupied:
                    free += 1
            # Weight H-H contacts much more heavily
            score = 2.0 * gain + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _try_reinsert_fragments(chain, sequence, rng, max_frag_len=6):
    """
    Attempt to re-insert a fragment: cut out chain[a:b], then try to regrow it
    from chain[a-1] (if a>0) connecting to chain[b] (if b<n).
    Returns (new_chain, energy_delta) or (None, 0).
    Uses DFS with heuristic ordering for reliability.
    """
    n = len(chain)
    if n < 5:
        return None, 0

    # Pick fragment length and position
    L = rng.randint(2, min(max_frag_len, n - 3))
    a = rng.randint(1, n - L - 1)  # a >= 1 so there's a left neighbor
    b = a + L  # b <= n-1 so there's a right neighbor

    # The fragment to remove: chain[a:b]
    # Left neighbor: chain[a-1], Right neighbor: chain[b]
    left_neighbor = chain[a - 1]
    right_neighbor = chain[b]

    # The rest of the chain without the fragment
    rest = chain[:a] + chain[b:]
    rest_occupied = set(rest)

    # Check if right_neighbor is at Manhattan distance L from left_neighbor
    lx, ly = left_neighbor
    rx, ry = right_neighbor
    manhattan = abs(rx - lx) + abs(ry - ly)
    if manhattan > L or (L - manhattan) % 2 != 0:
        return None, 0

    # DFS to find a self-avoiding path of exactly L steps from left to right
    # avoiding rest_occupied
    def _dfs(current, remaining, path, occupied):
        if remaining == 0:
            if current == right_neighbor:
                return list(path)
            return None

        x, y = current
        # Order neighbors by proximity to target for better heuristic
        neighbors = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            site = (x + dx, y + dy)
            if site in occupied:
                continue
            # Heuristic pruning: can we still reach end in remaining steps?
            dist_to_end = abs(right_neighbor[0] - site[0]) + abs(right_neighbor[1] - site[1])
            if dist_to_end > remaining or (remaining - dist_to_end) % 2 != 0:
                continue
            # Prefer sites closer to target
            neighbors.append((dist_to_end, site))
        
        neighbors.sort()
        
        for _, site in neighbors:
            path.append(site)
            occupied.add(site)
            result = _dfs(site, remaining - 1, path, occupied)
            if result is not None:
                return result
            path.pop()
            occupied.discard(site)
        return None

    path = _dfs(left_neighbor, L, [left_neighbor], {left_neighbor} | rest_occupied)
    if path is None:
        return None, 0

    # path[0] is left_neighbor, path[-1] should be right_neighbor
    if path[-1] != right_neighbor:
        return None, 0

    # The new fragment positions are path[1:] (excluding the start which is already in chain)
    # Build new chain: chain[:a] + path[1:] + chain[b:]
    new_frag = path[1:]  # L elements
    if len(new_frag) != L:
        return None, 0

    new_chain = chain[:a] + new_frag + chain[b:]

    # Validate
    if not _is_valid_chain(new_chain):
        return None, 0

    return new_chain, 0


def _reversal_move(chain, rng, max_len=None):
    """Random reversal move."""
    n = len(chain)
    if n < 4:
        return None
    if max_len is None:
        max_len = min(8, n - 2)
    L = rng.randint(2, max_len)
    a = rng.randint(0, n - L - 1)
    b = a + L
    new_chain = chain[:a] + chain[a:b + 1][::-1] + chain[b + 1:]
    if _is_valid_chain(new_chain):
        return new_chain
    return None


def _kink_move(chain, rng):
    """
    Kink move: pick a residue i (1..n-2) and try to move it to an adjacent
    free position, if the result is a valid self-avoiding walk.
    """
    n = len(chain)
    if n < 3:
        return None
    i = rng.randint(1, n - 2)
    x, y = chain[i]
    options = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        site = (x + dx, y + dy)
        # Must not be occupied by any other residue
        if site in set(chain) and site != chain[i]:
            continue
        # Check if it's adjacent to both neighbors
        if i > 0:
            prev = chain[i - 1]
            if abs(site[0] - prev[0]) + abs(site[1] - prev[1]) != 1:
                continue
        if i < n - 1:
            nxt = chain[i + 1]
            if abs(site[0] - nxt[0]) + abs(site[1] - nxt[1]) != 1:
                continue
        options.append(site)
    
    if not options:
        return None
    
    site = rng.choice(options)
    new_chain = list(chain)
    new_chain[i] = site
    if _is_valid_chain(new_chain):
        return new_chain
    return None


def _pivot_move(chain, rng):
    """Perform a pivot move: rotate a suffix of the chain around a pivot point."""
    n = len(chain)
    if n < 4:
        return None
    
    # Choose pivot index (not at ends)
    pivot = rng.randint(1, n - 2)
    pivot_pos = chain[pivot]
    suffix = chain[pivot + 1:]
    
    # Generate all 4 rotations of the suffix relative to pivot
    rotations = []
    prefix_set = set(chain[:pivot+1])
    
    for rot in range(1, 4):  # 90, 180, 270 degrees
        rotated = []
        valid = True
        for j, (x, y) in enumerate(suffix):
            # Rotate relative to pivot
            rx, ry = x - pivot_pos[0], y - pivot_pos[1]
            # Apply rotation
            if rot == 1:  # 90 deg CCW: (x,y) -> (-y,x)
                new_x, new_y = -ry, rx
            elif rot == 2:  # 180 deg: (x,y) -> (-x,-y)
                new_x, new_y = -rx, -ry
            else:  # 270 deg CCW: (x,y) -> (y,-x)
                new_x, new_y = ry, -rx
            
            pos = (pivot_pos[0] + new_x, pivot_pos[1] + new_y)
            
            # Check if this position is in the prefix or already in rotated
            if pos in prefix_set:
                valid = False
                break
            if pos in rotated:
                valid = False
                break
            
            # Check connectivity with previous rotated point
            if j > 0:
                prev = rotated[-1]
                if abs(pos[0] - prev[0]) + abs(pos[1] - prev[1]) != 1:
                    valid = False
                    break
            
            rotated.append(pos)
        
        if valid and len(rotated) == len(suffix):
            new_chain = chain[:pivot + 1] + rotated
            if len(set(new_chain)) == n:
                rotations.append(new_chain)
    
    if not rotations:
        return None
    return rng.choice(rotations)


def _try_local_search(sequence, chain, rng, max_iters=2000):
    """Perform local search with simulated annealing."""
    n = len(chain)
    current_energy = _energy_fast(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    T = 5.0
    cooling = 0.9995
    
    for it in range(max_iters):
        if _remaining() < 5.0:
            break
        
        # Choose a move
        r = rng.random()
        if r < 0.4:
            # Fragment re-insertion
            new_chain, _ = _try_reinsert_fragments(chain, sequence, rng, max_frag_len=6)
        elif r < 0.6:
            # Reversal
            new_chain = _reversal_move(chain, rng)
        elif r < 0.75:
            # Kink
            new_chain = _kink_move(chain, rng)
        elif r < 0.9:
            # Pivot
            new_chain = _pivot_move(chain, rng)
        else:
            # Random walk fragment
            new_chain, _ = _try_reinsert_fragments(chain, sequence, rng, max_frag_len=4)
        
        if new_chain is None:
            continue
        
        # Verify self-avoidance
        if len(set(new_chain)) != n:
            continue
        
        # Verify connectivity
        valid = True
        for i in range(n - 1):
            if abs(new_chain[i][0] - new_chain[i+1][0]) + abs(new_chain[i][1] - new_chain[i+1][1]) != 1:
                valid = False
                break
        if not valid:
            continue
        
        new_energy = _energy_fast(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Simulated annealing acceptance
        if delta < 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(new_chain)
        
        T *= cooling
        if T < 0.1:
            T = 0.1
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold sequence into low-energy conformation using simulated annealing."""
    if dimension != 2:
        # Fallback for non-2D (shouldn't happen per task)
        return [[i] + [0] * (dimension - 1) for i in range(len(sequence))]
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0  # Start with unfolded (energy 0)
    
    # Time budget: use most of the available time
    start_time = time.time()
    time_budget = min(_remaining() - 5.0, 180.0)  # Leave 5s buffer
    
    # Number of restarts based on sequence length
    if n <= 30:
        num_restarts = 40
    elif n <= 50:
        num_restarts = 25
    elif n <= 70:
        num_restarts = 18
    else:
        num_restarts = 12
    
    iters_per_restart = max(1000, int(800000 / n))
    
    for restart in range(num_restarts):
        if _remaining() < 10.0:
            break
        
        # Generate initial conformation
        chain = _grow(sequence, dimension, rng)
        if chain is None or len(chain) != n:
            continue
        
        # Verify it's valid
        if len(set(chain)) != n:
            continue
        valid = True
        for i in range(n - 1):
            if abs(chain[i][0] - chain[i+1][0]) + abs(chain[i][1] - chain[i+1][1]) != 1:
                valid = False
                break
        if not valid:
            continue
        
        # Run local search
        chain, energy = _try_local_search(sequence, chain, rng, max_iters=iters_per_restart)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
        
        # Check time
        if time.time() - start_time > time_budget:
            break
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [[x, y] for x, y in best_chain]
# EVOLVE-BLOCK-END
