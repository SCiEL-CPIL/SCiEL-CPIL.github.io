"""HP lattice protein folding on the cubic lattice - structural local-search with contact-map tracking."""
# EVOLVE-BLOCK-START
import random
import time
import math

NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _energy(sequence, chain):
    """Calculate energy of a chain."""
    if not chain:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y, z = p
            for nx, ny, nz in [(x+1,y,z), (x-1,y,z), (x,y+1,z), (x,y-1,z), (x,y,z+1), (x,y,z-1)]:
                j = where.get((nx, ny, nz))
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _grow(sequence, dimension, rng, T=0.0):
    """Grow a chain using heuristic with optional noise."""
    n = len(sequence)
    chain = []
    occupied = {}
    
    for i in range(n):
        if i == 0:
            pos = (0, 0, 0)
        else:
            last_pos = chain[-1]
            x, y, z = last_pos
            candidates = [
                (x+1, y, z), (x-1, y, z),
                (x, y+1, z), (x, y-1, z),
                (x, y, z+1), (x, y, z-1)
            ]
            
            valid = []
            for site in candidates:
                if site in occupied:
                    continue
                # Heuristic: gain in H-H contacts
                gain = 0
                if sequence[i] == "H":
                    sx, sy, sz = site
                    for nx, ny, nz in [(sx+1,sy,sz), (sx-1,sy,sz), (sx,sy+1,sz), (sx,sy-1,sz), (sx,sy,z+1), (sx,sy,z-1)]:
                        j = occupied.get((nx, ny, nz))
                        if j is not None and j < i - 1 and sequence[j] == "H":
                            gain += 1
                free = sum(1 for nx, ny, nz in [(sx+1,sy,sz), (sx-1,sy,sz), (sx,sy+1,sz), (sx,sy-1,sz), (sx,sy,z+1), (sx,sy,z-1)] if (nx,ny,nz) not in occupied)
                score = gain * 10.0 + free * 0.5
                noisy = score + rng.gauss(0, T)
                valid.append((noisy, site))
            
            if not valid:
                return None
            
            valid.sort(key=lambda x: x[0], reverse=True)
            pos = valid[0][1]
        
        chain.append(pos)
        occupied[pos] = i
    
    return chain

def _pivot_move(chain, i, occupied, sequence, rng):
    """
    Perform a pivot move at position i.
    Returns (new_chain, new_energy, success) or (None, None, False)
    """
    n = len(chain)
    if i <= 0 or i >= n - 1:
        return None, None, False
    
    # Get the prefix up to i
    prefix = chain[:i+1]
    # Get the tail from i+1 onwards (relative to position i)
    tail = chain[i+1:]
    
    # Build occupied set for prefix
    occ_prefix = {p: idx for idx, p in enumerate(prefix)}
    
    # Try 12 rotations around the bond (i-1, i)
    # The bond direction is chain[i] - chain[i-1]
    dx = prefix[i][0] - prefix[i-1][0]
    dy = prefix[i][1] - prefix[i-1][1]
    dz = prefix[i][2] - prefix[i-2][2] if i >= 2 else 0
    
    # Generate rotation candidates
    # For a pivot, we rotate the tail around the axis (i-1) -> i
    # The axis direction is (dx, dy, dz)
    
    # Simpler approach: try all 3 possible moves for the first element of the tail
    # that don't go back to prefix[i-1] and are unoccupied
    
    first_tail = tail[0]
    x, y, z = first_tail
    # Current position of tail[0] relative to chain[i]
    rx = x - chain[i][0]
    ry = y - chain[i][1]
    rz = z - chain[i][2]
    
    # Generate all unit vectors except the reverse of (rx, ry, rz)
    candidates = []
    for (dx, dy, dz) in NEIGHBOR_OFFSETS:
        if (dx, dy, dz) == (-rx, -ry, -rz):
            continue
        new_pos = (chain[i][0] + dx, chain[i][1] + dy, chain[i][2] + dz)
        if new_pos not in occ_prefix and new_pos != chain[i]:
            candidates.append((dx, dy, dz))
    
    if not candidates:
        return None, None, False
    
    # Try each candidate
    rng.shuffle(candidates)
    for (dx, dy, dz) in candidates:
        new_tail = []
        valid = True
        new_occ = dict(occ_prefix)
        
        # Rebuild tail
        new_tail.append((chain[i][0] + dx, chain[i][1] + dy, chain[i][2] + dz))
        new_occ[new_tail[0]] = i + 1
        
        for t in range(1, len(tail)):
            last = new_tail[-1]
            lx, ly, lz = last
            # Relative position from previous tail element
            prev_rel = (tail[t][0] - tail[t-1][0], tail[t][1] - tail[t-1][1], tail[t][2] - tail[t-1][2])
            # Apply same relative movement
            candidate_pos = (lx + prev_rel[0], ly + prev_rel[1], lz + prev_rel[2])
            
            # Check if occupied
            if candidate_pos in new_occ:
                # Try alternative moves
                found = False
                for (adx, ady, adz) in NEIGHBOR_OFFSETS:
                    if (adx, ady, adz) == (-prev_rel[0], -prev_rel[1], -prev_rel[2]):
                        continue
                    alt_pos = (lx + adx, ly + ady, lz + adz)
                    if alt_pos not in new_occ:
                        candidate_pos = alt_pos
                        new_tail.append(alt_pos)
                        new_occ[alt_pos] = i + 1 + t
                        found = True
                        break
                if not found:
                    valid = False
                    break
            else:
                new_tail.append(candidate_pos)
                new_occ[candidate_pos] = i + 1 + t
        
        if not valid:
            continue
        
        new_chain = prefix + new_tail
        
        # Check self-avoidance
        if len(set(new_chain)) != len(new_chain):
            continue
        
        new_energy = _energy(sequence, new_chain)
        return new_chain, new_energy, True
    
    return None, None, False

def _local_search(sequence, chain, deadline, rng):
    """
    Local search using pivot moves and simulated annealing.
    """
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    start_time = time.time()
    
    while time.time() < deadline - 2.0:
        T = max(T * cooling_rate, T_min)
        
        # Pick a pivot position
        i = rng.randint(1, n - 2)
        
        new_chain, new_energy, success = _pivot_move(current_chain, i, occupied, sequence, rng)
        
        if not success:
            continue
        
        delta = new_energy - current_energy
        
        # Acceptance criterion
        if delta <= 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
            # Accept move
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
    
    return best_chain

def fold(sequence, dimension):
    """Fold a sequence using growth + local search."""
    start_time = time.time()
    deadline = start_time + 315.0
    
    rng = random.Random(12345 + len(sequence))
    
    n = len(sequence)
    if n == 0:
        return []
    
    # Generate initial chain
    best_chain = None
    best_energy = 0
    
    # Try multiple initializations
    for _ in range(20):
        chain = _grow(sequence, dimension, rng, T=0.5)
        if chain:
            e = _energy(sequence, chain)
            if best_chain is None or e < best_energy:
                best_chain = chain
                best_energy = e
    
    if best_chain is None:
        # Fallback: linear chain
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Local search
    result = _local_search(sequence, best_chain, deadline, rng)
    
    return [list(p) for p in result]
# EVOLVE-BLOCK-END
