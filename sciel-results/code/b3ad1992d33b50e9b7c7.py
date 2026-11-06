"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 200.0  # Safety margin for 210s kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _energy(sequence, chain):
    """Compute the HP contact energy of a conformation."""
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y = p
            for q in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow_greedy(sequence, rng, bias=1.0):
    """Randomized greedy chain growth."""
    n = len(sequence)
    if n == 0:
        return []
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    for i in range(1, n):
        x, y = chain[-1]
        options = []
        for site in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                sx, sy = site
                for q in ((sx+1, sy), (sx-1, sy), (sx, sy+1), (sx, sy-1)):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = 0
            sx, sy = site
            for q in ((sx+1, sy), (sx-1, sy), (sx, sy+1), (sx, sy-1)):
                if q not in occupied:
                    free += 1
            score = bias * gain + 0.1 * free + rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_move(chain, rng):
    """Try to change one residue's position by moving it to an adjacent free site."""
    n = len(chain)
    if n < 3:
        return None
    idx = rng.randint(1, n - 2)  # Not first or last
    x, y = chain[idx]
    # Try moving this residue to a neighboring position
    for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
        new_pos = (x + dx, y + dy)
        # Check it's not occupied by another residue
        occupied = set(chain)
        if new_pos in occupied:
            continue
        # Check connectivity: chain[idx-1] must be adjacent to new_pos,
        # and new_pos must be adjacent to chain[idx+1]
        px, py = chain[idx - 1]
        if abs(px - new_pos[0]) + abs(py - new_pos[1]) != 1:
            continue
        nx, ny = chain[idx + 1]
        if abs(nx - new_pos[0]) + abs(ny - new_pos[1]) != 1:
            continue
        # Valid move
        new_chain = list(chain)
        new_chain[idx] = new_pos
        return new_chain
    return None


def _crankshaft_move(chain, rng):
    """Rotate a short subchain (2 or 3 residues) around the preceding residue."""
    n = len(chain)
    if n < 5:
        return None
    sub_len = rng.choice([2, 3])
    if n < sub_len + 3:
        return None
    start = rng.randint(1, n - sub_len - 1)
    pivot = chain[start - 1]
    segment = chain[start:start + sub_len]
    for angle in (90, -90, 180):
        rotated = []
        for p in segment:
            dx, dy = p[0] - pivot[0], p[1] - pivot[1]
            if angle == 90:
                ndx, ndy = -dy, dx
            elif angle == -90:
                ndx, ndy = dy, -dx
            else:
                ndx, ndy = -dx, -dy
            rotated.append((pivot[0] + ndx, pivot[1] + ndy))
        new_chain = chain[:start] + rotated + chain[start + sub_len:]
        # Check self-avoidance
        if len(set(new_chain)) != len(new_chain):
            continue
        # Check unit steps
        valid = True
        for i in range(1, len(new_chain)):
            if abs(new_chain[i][0] - new_chain[i-1][0]) + abs(new_chain[i][1] - new_chain[i-1][1]) != 1:
                valid = False
                break
        if valid:
            return new_chain
    return None


def _reversal_move(chain, rng):
    """Reverse a subchain segment."""
    n = len(chain)
    if n < 5:
        return None
    length = rng.randint(2, min(8, n - 2))
    start = rng.randint(1, n - length - 1)
    end = start + length
    # The segment chain[start:end+1] is reversed
    # But we need to maintain connectivity at boundaries
    # chain[start-1] must be adjacent to chain[end] (new first of reversed)
    # chain[end+1] must be adjacent to chain[start] (new last of reversed)
    px, py = chain[start - 1]
    ex, ey = chain[end]
    if abs(px - ex) + abs(py - ey) != 1:
        return None
    if end + 1 < n:
        nx, ny = chain[end + 1]
        sx, sy = chain[start]
        if abs(nx - sx) + abs(ny - sy) != 1:
            return None
    # Perform reversal
    new_chain = list(chain)
    new_chain[start:end+1] = chain[start:end+1][::-1]
    # Check self-avoidance
    if len(set(new_chain)) != len(new_chain):
        return None
    # Check unit steps
    valid = True
    for i in range(1, len(new_chain)):
        if abs(new_chain[i][0] - new_chain[i-1][0]) + abs(new_chain[i][1] - new_chain[i-1][1]) != 1:
            valid = False
            break
    if valid:
        return new_chain
    return None


def _simulated_annealing(sequence, chain, time_limit=5.0, rng=None):
    """Simulated annealing with multiple move types."""
    if rng is None:
        rng = random.Random(hash(tuple(chain)) % (2**32))
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    temperature = 2.0
    cooling = 0.9995
    min_temp = 0.1
    
    while time.time() - start_time < time_limit and _time_left() > 15:
        # Choose move type
        r = rng.random()
        if r < 0.3:
            new_chain = _local_move(current_chain, rng)
        elif r < 0.7:
            new_chain = _crankshaft_move(current_chain, rng)
        else:
            new_chain = _reversal_move(current_chain, rng)
        
        if new_chain is None:
            temperature = max(min_temp, temperature * cooling)
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < pow(2.718281828, -delta / max(temperature, 0.05)):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        temperature = max(min_temp, temperature * cooling)
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using multiple strategies."""
    seq_len = len(sequence)
    
    # Time budget allocation based on sequence length
    if seq_len <= 25:
        num_grow = 600
        sa_time = 12.0
    elif seq_len <= 40:
        num_grow = 400
        sa_time = 10.0
    elif seq_len <= 60:
        num_grow = 250
        sa_time = 8.0
    elif seq_len <= 80:
        num_grow = 150
        sa_time = 6.0
    else:
        num_grow = 100
        sa_time = 5.0
    
    rng = random.Random(42 + seq_len)
    best_chain = None
    best_energy = 0
    
    # Phase 1: Generate diverse initial conformations via greedy growth
    for i in range(num_grow):
        if _time_left() < 25:
            break
        bias = 0.3 + rng.random() * 1.7
        chain = _grow_greedy(sequence, rng, bias=bias)
        if chain is None:
            continue
        e = _energy(sequence, chain)
        if best_chain is None or e < best_energy:
            best_energy = e
            best_chain = chain
    
    if best_chain is None:
        # Fallback: simple spiral
        best_chain = [(0, 0)]
        for i in range(1, seq_len):
            if i % 4 == 1:
                best_chain.append((best_chain[-1][0] + 1, best_chain[-1][1]))
            elif i % 4 == 2:
                best_chain.append((best_chain[-1][0], best_chain[-1][1] + 1))
            elif i % 4 == 3:
                best_chain.append((best_chain[-1][0] - 1, best_chain[-1][1]))
            else:
                best_chain.append((best_chain[-1][0], best_chain[-1][1] - 1))
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Simulated annealing on the best chain
    if _time_left() > 30:
        improved = _simulated_annealing(sequence, best_chain, time_limit=sa_time, rng=rng)
        if improved is not None:
            e = _energy(sequence, improved)
            if e < best_energy:
                best_energy = e
                best_chain = improved
    
    # Phase 3: If time remains, do more SA restarts from good chains
    if _time_left() > 20:
        # Generate a few more initial chains and improve them
        extra_starts = min(20, max(5, int(_time_left() / 3)))
        for i in range(extra_starts):
            if _time_left() < 15:
                break
            bias = 0.3 + rng.random() * 1.7
            chain = _grow_greedy(sequence, rng, bias=bias)
            if chain is None:
                continue
            e = _energy(sequence, chain)
            if e < best_energy + 4:  # Close enough to be worth improving
                t = min(3.0, max(1.0, _time_left() / 10.0))
                improved = _simulated_annealing(sequence, chain, time_limit=t, rng=rng)
                if improved is not None:
                    e2 = _energy(sequence, improved)
                    if e2 < best_energy:
                        best_energy = e2
                        best_chain = improved
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
