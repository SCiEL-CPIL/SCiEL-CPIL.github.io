import random
import time

_DEADLINE = None
_BUDGET_FRACTION = 0.90

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 330 * _BUDGET_FRACTION
    return _DEADLINE

def _is_deadline_exceeded():
    return time.time() > _get_deadline()

_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _calc_energy(chain, sequence):
    """Calculate energy of a chain."""
    if not chain:
        return 0
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    energy = 0
    for i, p in enumerate(chain):
        if sequence[i] != 'H':
            continue
        for dx, dy, dz in _OFFSETS:
            neighbor = (p[0] + dx, p[1] + dy, p[2] + dz)
            if neighbor in pos_to_idx:
                j = pos_to_idx[neighbor]
                if abs(i - j) > 1:
                    if sequence[j] == 'H':
                        energy -= 1
    return energy // 2

def _greedy_construct(sequence):
    """
    Construct a self-avoiding walk greedily, preferring moves that create H-H contacts.
    Returns a list of points or None if stuck.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    chain = [(0, 0, 0)]
    occ = {chain[0]}
    
    # Pre-compute H positions for quick check? Not needed for greedy step.
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        
        for dx, dy, dz in _OFFSETS:
            nxt = (last[0] + dx, last[1] + dy, last[2] + dz)
            if nxt in occ:
                continue
            
            # Score this move: How many new H-H contacts does it create?
            score = 0
            if sequence[i] == 'H':
                # Check neighbors of 'nxt' that are already in the chain
                for dx2, dy2, dz2 in _OFFSETS:
                    nb = (nxt[0] + dx2, nxt[1] + dy2, nxt[2] + dz2)
                    if nb in occ:
                        # Find index of nb
                        # This is O(N) lookup. For N=136, it's okay for construction.
                        # But we need to check if the residue at nb is H.
                        # We need a map from pos to idx or pos to residue.
                        pass
            
            # To make this efficient, we need a pos->idx map updated as we go.
            # Let's refactor to maintain pos_to_idx.
            
            candidates.append((nxt, 0)) # Placeholder
            
        # This simple greedy is hard to score without full map.
        # Let's use a simpler heuristic: Random walk with bias towards H-H if possible.
        # Or just random walk if greedy scoring is too slow.
        
        # Let's try a "Greedy with Lookahead 1" for energy.
        # We need pos_to_idx for the current chain prefix.
        
        # Re-implementation with pos_to_idx
        pass
        
    return None # Fallback

def _fast_local_search(chain, sequence, max_iter=1000):
    """
    Perform fast local search using First Improvement strategy.
    Moves: Pivot moves.
    """
    n = len(chain)
    if n < 3:
        return chain, _calc_energy(chain, sequence)
    
    current_energy = _calc_energy(chain, sequence)
    best_chain = list(chain)
    best_energy = current_energy
    
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    # We need an efficient way to check collisions for pivot moves.
    # For simplicity and speed, we will just try random pivots and check validity.
    
    for _ in range(max_iter):
        if _is_deadline_exceeded():
            break
            
        # Pick a random pivot index
        i = random.randint(0, n - 2)
        
        p0 = chain[i]
        p1 = chain[i+1]
        
        # Vector v = p1 - p0
        v = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        
        # Find perpendicular axes
        axes = []
        for dx, dy, dz in _OFFSETS:
            if v[0]*dx + v[1]*dy + v[2]*dz == 0:
                axes.append((dx, dy, dz))
                
        if not axes:
            continue
            
        axis = random.choice(axes)
        
        def rotate90(p, origin, axis):
            x, y, z = p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]
            ax, ay, az = axis
            if ax == 1: nx, ny, nz = x, -z, y
            elif ax == -1: nx, ny, nz = x, z, -y
            elif ay == 1: nx, ny, nz = -z, y, x
            elif ay == -1: nx, ny, nz = z, y, -x
            elif az == 1: nx, ny, nz = y, -x, z
            elif az == -1: nx, ny, nz = -y, x, z
            else: return None
            return (origin[0] + nx, origin[1] + ny, origin[2] + nz)
            
        new_seg = []
        valid = True
        # The part of the chain that stays fixed is chain[:i+1]
        # The moving part is chain[i+1:]
        # We need to check if the new positions of the moving part collide with the fixed part or themselves.
        
        # Optimization: Instead of rebuilding the whole set, we can just check the new points against the fixed set.
        fixed_set = set(chain[:i+1])
        new_occ = set(fixed_set)
        
        for k in range(i+1, n):
            qk = rotate90(chain[k], p0, axis)
            if qk is None or qk in new_occ:
                valid = False
                break
            new_occ.add(qk)
            new_seg.append(qk)
            
        if not valid:
            continue
            
        new_chain = chain[:i+1] + new_seg
        new_energy = _calc_energy(new_chain, sequence)
        
        if new_energy < current_energy:
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
            # First improvement: restart loop? Or continue?
            # Continuing is fine.
            
    return best_chain, best_energy

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    
    best_chain = None
    best_energy = 0 # Energy is negative, so 0 is worse than -1
    
    # Initial solution: Simple SAW
    # Use a simple random walk generator that is robust
    def _gen_saw(n):
        if n == 0: return []
        chain = [(0, 0, 0)]
        occ = {chain[0]}
        for _ in range(n - 1):
            last = chain[-1]
            moves = []
            for dx, dy, dz in _OFFSETS:
                np_ = (last[0] + dx, last[1] + dy, last[2] + dz)
                if np_ not in occ:
                    moves.append(np_)
            if not moves:
                return None
            np_ = random.choice(moves)
            chain.append(np_)
            occ.add(np_)
        return chain

    # Loop: Generate, Improve, Repeat
    while not _is_deadline_exceeded():
        chain = _gen_saw(n)
        if chain is None:
            # Try again with random restart of the generator logic?
            # If SAW generation fails, it's rare for small N.
            # For large N, it might fail. If it fails, we can't fold.
            # But for N=48-136, SAWs exist.
            # If _gen_saw fails, we should probably just return a flat line or break.
            # Let's assume it succeeds for valid benchmark chains.
            continue
            
        current_energy = _calc_energy(chain, sequence)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(chain)
            
        # Improve
        improved_chain, improved_energy = _fast_local_search(chain, sequence, max_iter=200)
        
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = list(improved_chain)
            
        # If we found a very good solution, we might break early?
        # No, keep searching until deadline.
        
    if best_chain is None:
        # Fallback: Return a straight line (invalid if N>1 on cubic lattice? No, straight line is valid SAW)
        # But energy will be 0.
        return [[i, 0, 0] for i in range(n)]
        
    return [list(p) for p in best_chain]
