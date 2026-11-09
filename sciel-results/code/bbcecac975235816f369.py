"""HP lattice protein folding with optimized local search."""
# EVOLVE-BLOCK-START
import random
import math


def _compute_energy(sequence, chain):
    """Compute the HP contact energy: -number of non-consecutive H-H contacts."""
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y = p
            for q in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _local_search(sequence, chain, max_iter=50000):
    """Local search with simulated annealing and proper incremental energy updates."""
    n = len(chain)
    if n <= 1:
        return chain

    # Build initial structures
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Build occupied set for fast lookup
    occupied_set = {}
    for i, p in enumerate(chain):
        occupied_set[p] = i

    # Precompute H positions for faster contact checks
    h_positions = [i for i in range(n) if sequence[i] == 'H']

    # Simulated annealing
    T = 3.0
    T_min = 0.05
    cooling = 0.9995
    rng = random.Random(42 + n)

    for iteration in range(max_iter):
        if T < T_min:
            break

        # Pick a random residue to move (not endpoints)
        i = rng.randint(1, n - 2)

        current_pos = chain[i]
        moves = []
        x, y = current_pos
        
        # Try all 4 possible moves
        for new_pos in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            if new_pos in occupied_set:
                continue
            # Check connectivity with neighbors
            prev = chain[i - 1]
            if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) != 1:
                continue
            nxt = chain[i + 1]
            if abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) != 1:
                continue
            moves.append(new_pos)

        if not moves:
            continue

        new_pos = rng.choice(moves)

        # Compute energy delta incrementally
        # Remove old position from occupied_set temporarily
        del occupied_set[current_pos]
        
        # Count contacts at old_pos (excluding i itself)
        old_contacts = 0
        if sequence[i] == 'H':
            ox, oy = current_pos
            for q in ((ox+1,oy), (ox-1,oy), (ox,oy+1), (ox,oy-1)):
                j = occupied_set.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    old_contacts += 1

        # Count contacts at new_pos
        new_contacts = 0
        if sequence[i] == 'H':
            nx, ny = new_pos
            for q in ((nx+1,ny), (nx-1,ny), (nx,ny+1), (nx,ny-1)):
                j = occupied_set.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    new_contacts += 1

        delta_contacts = new_contacts - old_contacts
        delta_energy = -delta_contacts

        # Update position if accepted
        if delta_energy <= 0 or rng.random() < math.exp(-delta_energy / T):
            # Update occupied set
            occupied_set[new_pos] = i
            chain[i] = new_pos
            current_energy += delta_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        else:
            # Revert
            occupied_set[current_pos] = i
            chain[i] = current_pos

        T *= cooling

    return best_chain


def _initial_chain_spiral(sequence, dimension):
    """Create a compact spiral initial conformation."""
    n = len(sequence)
    chain = [(0, 0)]
    occupied = {(0, 0)}

    # Spiral pattern: go right, down, left, up, right longer, etc.
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dir_idx = 0
    step_count = 1
    steps_in_direction = 0

    for i in range(1, n):
        placed = False
        d = dirs[dir_idx]
        new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1])
        if new_pos not in occupied:
            chain.append(new_pos)
            occupied.add(new_pos)
            steps_in_direction += 1
            if steps_in_direction >= step_count:
                steps_in_direction = 0
                dir_idx = (dir_idx + 1) % 4
                if dir_idx == 0 or dir_idx == 2:
                    step_count += 1
            placed = True
        else:
            # Try next direction
            for nd in dirs:
                np = (chain[-1][0] + nd[0], chain[-1][1] + nd[1])
                if np not in occupied:
                    chain.append(np)
                    occupied.add(np)
                    dir_idx = dirs.index(nd)
                    steps_in_direction = 1
                    placed = True
                    break
        if not placed:
            # Fallback
            for d2 in [(1,0), (-1,0), (0,1), (0,-1)]:
                np = (chain[-1][0] + d2[0], chain[-1][1] + d2[1])
                if np not in occupied:
                    chain.append(np)
                    occupied.add(np)
                    placed = True
                    break
            if not placed:
                np = (chain[-1][0] + 1, chain[-1][1])
                while np in occupied:
                    np = (np[0] + 1, np[1])
                chain.append(np)
                occupied.add(np)

    return chain


def _initial_chain_compact(sequence, dimension):
    """Create a compact initial conformation by filling a square in boustrophedon."""
    n = len(sequence)
    side = int(math.ceil(math.sqrt(n)))
    chain = []
    occupied = set()
    
    # Fill a grid in a boustrophedon pattern
    for row in range(side):
        if row % 2 == 0:
            for col in range(side):
                if len(chain) >= n:
                    break
                pos = (col, row)
                chain.append(pos)
                occupied.add(pos)
        else:
            for col in range(side-1, -1, -1):
                if len(chain) >= n:
                    break
                pos = (col, row)
                chain.append(pos)
                occupied.add(pos)
    
    # Ensure it's a valid self-avoiding walk with unit steps
    valid = True
    for i in range(1, len(chain)):
        if abs(chain[i][0] - chain[i-1][0]) + abs(chain[i][1] - chain[i-1][1]) != 1:
            valid = False
            break
    
    if not valid:
        return _initial_chain_spiral(sequence, dimension)
    
    return chain


def fold(sequence, dimension):
    """Find a low-energy conformation for the given HP sequence."""
    if dimension != 2:
        raise ValueError("Only 2D is supported")

    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]

    rng = random.Random(12345 + n)

    # Try multiple initial conformations and local searches
    best_chain = None
    best_energy = float('inf')

    # Strategy 1: Compact spiral initialization
    chain1 = _initial_chain_spiral(sequence, dimension)
    chain1 = _local_search(sequence, chain1, max_iter=50000)
    e1 = _compute_energy(sequence, chain1)
    if e1 < best_energy:
        best_energy = e1
        best_chain = chain1

    # Strategy 2: Compact grid initialization
    chain_compact = _initial_chain_compact(sequence, dimension)
    chain_compact = _local_search(sequence, chain_compact, max_iter=40000)
    e_compact = _compute_energy(sequence, chain_compact)
    if e_compact < best_energy:
        best_energy = e_compact
        best_chain = chain_compact

    # Strategy 3: Random walk initializations with different seeds
    for attempt in range(10):
        seed = 12345 + n + attempt * 1000
        rng_local = random.Random(seed)
        # Generate a random self-avoiding walk with bias toward compactness
        chain2 = [(0, 0)]
        occupied = {(0, 0)}
        success = True
        for i in range(1, n):
            options = []
            # Prefer moves that stay near the origin for compactness
            for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_pos = (chain2[-1][0] + d[0], chain2[-1][1] + d[1])
                if new_pos not in occupied:
                    # Weight by distance from origin to prefer compactness
                    options.append((abs(new_pos[0]) + abs(new_pos[1]), new_pos))
            if not options:
                success = False
                break
            # Choose with preference for closer positions
            options.sort(key=lambda x: x[0])
            # Take from top 3 with some randomness
            k = min(3, len(options))
            chosen = rng_local.choice(options[:k])
            new_pos = chosen[1]
            chain2.append(new_pos)
            occupied.add(new_pos)

        if not success:
            continue

        chain2 = _local_search(sequence, chain2, max_iter=30000)
        e2 = _compute_energy(sequence, chain2)
        if e2 < best_energy:
            best_energy = e2
            best_chain = chain2

    if best_chain is None:
        # Fallback: straight line
        best_chain = [[i, 0] for i in range(n)]
    else:
        best_chain = [[p[0], p[1]] for p in best_chain]

    return best_chain
# EVOLVE-BLOCK-END
