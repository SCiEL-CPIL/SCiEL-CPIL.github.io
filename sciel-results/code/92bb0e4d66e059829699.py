"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

# Deadline guard: 330s wall-clock kill, use 320s as safety margin
_DEADLINE = 320.0
_start_time = None


def _get_remaining_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _DEADLINE - (time.time() - _start_time)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy: -number of H-H contacts on non-adjacent lattice neighbours."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng):
    """
    One chain growth using the scoring function from Inspiration 1.
    This function is simple and fast, and reportedly finds ground states for the benchmark set.
    Score: gain + 0.1 * free + noise
    """
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            # Count new H-H contacts this site would make
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            
            # Count free neighbours
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            
            # Inspiration 1 scoring
            score = gain + 0.1 * free + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick best
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    
    return chain


def _local_search_1opt(sequence, chain, dimension, rng, max_moves=1000):
    """
    Fast local search using single-residue moves (1-opt).
    Much faster than 2-opt. Tries to move a single residue to an adjacent free site
    if it maintains connectivity (which is rare for 1-opt in a path, so we use 
    'pivot' moves: rotate a segment around a pivot, or just try to swap/move 
    if we treat it as a graph, but for a path, 1-opt is usually invalid unless 
    we allow the chain to break and rejoin? No, it must be a path.
    
    Actually, for a path, moving a single node i to a new position j is only valid 
    if j is adjacent to i-1 and i+1. This is very restrictive.
    
    A better fast move is 'segment reversal' (2-opt) but with a very small segment 
    size (e.g., length 2 or 3), or 'crankshaft' moves.
    
    However, since Inspiration 1 (pure greedy) scored 1.0, local search might not be 
    necessary for the short chains. For long chains, we will use a simple 2-opt 
    with small segments to fix local minima.
    """
    best_chain = list(chain)
    n = len(best_chain)
    best_energy = _energy(sequence, best_chain)
    
    for _ in range(max_moves):
        if _get_remaining_time() < 1.0:
            break
        
        # Try a small 2-opt reversal (segment length 2 to 5)
        # This is faster than random large 2-opts
        a = rng.randint(0, n - 2)
        seg_len = rng.randint(2, min(5, n - a - 1))
        b = a + seg_len
        
        # Reverse segment [a, b]
        new_chain = best_chain[:a] + best_chain[a:b+1][::-1] + best_chain[b+1:]
        
        # Check self-avoidance and unit steps
        seen = set()
        valid = True
        for k in range(n):
            p = new_chain[k]
            if p in seen:
                valid = False
                break
            seen.add(p)
            if k > 0:
                diff = sum(abs(p[d] - new_chain[k-1][d]) for d in range(dimension))
                if diff != 1:
                    valid = False
                    break
        
        if not valid:
            continue
        
        new_energy = _energy(sequence, new_chain)
        if new_energy < best_energy:
            best_chain = new_chain
            best_energy = new_energy
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Best conformation using:
    1. Many fast greedy growths (Inspiration 1 style)
    2. Fast local search (small 2-opt) for long chains
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0  # Start with 0 (no contacts), we want to minimize (more negative)
    
    # Determine restarts based on chain length and time
    # Short chains: rely on high restart count to find the exact ground state (which is easy)
    # Long chains: fewer restarts, more local search
    if n <= 48:
        n_restarts = 2000
        use_local_search = False
    elif n <= 80:
        n_restarts = 500
        use_local_search = True
    else:
        n_restarts = 200
        use_local_search = True

    for restart in range(n_restarts):
        if _get_remaining_time() < 2.0:
            break
            
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best_chain = chain
            best_energy = value
    
    # Local search for long chains if time permits
    if best_chain is not None and use_local_search and _get_remaining_time() > 5.0:
        # Allocate remaining time to local search
        # Estimate time per move and set max_moves accordingly
        # For simplicity, just run a fixed number of moves that fits in ~50% of remaining time
        # But we must be safe. Let's just run until time runs out or improvement stops.
        best_chain, best_energy = _local_search_1opt(sequence, best_chain, dimension, rng, max_moves=5000)
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
