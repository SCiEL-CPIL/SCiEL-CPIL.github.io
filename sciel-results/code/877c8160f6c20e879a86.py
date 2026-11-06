"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    return 190.0 - (time.time() - _START_TIME)


def _neighbours(point):
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _energy(sequence, chain):
    """Compute HP contact energy: -number of H-H non-adjacent contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _grow_greedy(sequence, rng, alpha=0.5, beta=1.0):
    """Grow chain with greedy heuristic: maximize H-H contacts, then free space."""
    n = len(sequence)
    if n == 0:
        return []
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Count potential H-H contacts if we place residue i here
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            # Count free neighbours (open space)
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            score = alpha * gain + beta * free * 0.1 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        options.sort(reverse=True)
        # Pick from top options with some randomness
        top_k = min(3, len(options))
        chosen = rng.choice(options[:top_k])[1]
        occupied[chosen] = i
        chain.append(chosen)
    
    return chain


def _local_moves(sequence, chain, rng, n_moves=100):
    """Perform local moves: pick a random non-terminal residue and try to move it to an adjacent empty site."""
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    where = {p: i for i, p in enumerate(best_chain)}
    
    current_energy = best_energy
    
    for _ in range(n_moves):
        # Pick a random residue index (not first or last)
        i = rng.randint(1, n - 2)
        pos = best_chain[i]
        
        # Try moving to an adjacent empty site
        candidates = []
        for site in _neighbours(pos):
            if site in where:
                # Check if it's the previous or next residue (can't move there)
                if site == best_chain[i-1] or site == best_chain[i+1]:
                    continue
                # Check if moving would cause collision with other residues
                # We need to temporarily remove i and check
                if where[site] != i:
                    continue
            else:
                # Site is empty, check if it's adjacent to previous/next (would be a unit step)
                # Actually we need to ensure the move keeps unit steps
                # The new position must be adjacent to both chain[i-1] and chain[i+1]
                if site == best_chain[i-1] or site == best_chain[i+1]:
                    continue
                # Check that site is not occupied
                if site not in where:
                    candidates.append(site)
        
        if not candidates:
            continue
        
        new_pos = rng.choice(candidates)
        
        # Validate: new_pos must be adjacent to chain[i-1] and chain[i+1]
        if new_pos not in _neighbours(best_chain[i-1]) or new_pos not in _neighbours(best_chain[i+1]):
            continue
        
        # Check self-avoidance: new_pos should not be occupied by any other residue
        if new_pos in where and where[new_pos] != i:
            continue
        
        # Make the move
        old_pos = best_chain[i]
        del where[old_pos]
        where[new_pos] = i
        best_chain[i] = new_pos
        
        new_energy = _energy(sequence, best_chain)
        delta = new_energy - current_energy
        
        # Accept if better or with some probability (simple acceptance)
        if delta < 0 or rng.random() < 0.3:
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
        
    return best_chain, best_energy


def _simulated_annealing(sequence, chain, rng, n_iter=500, T0=2.0, T1=0.01):
    """Simulated annealing with local moves."""
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current = list(chain)
    where = {p: i for i, p in enumerate(current)}
    current_energy = _energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    for it in range(n_iter):
        T = T0 * (T1 / T0) ** (it / max(1, n_iter - 1))
        
        # Pick random residue to move
        i = rng.randint(1, n - 2)
        pos = current[i]
        
        candidates = []
        for site in _neighbours(pos):
            if site == current[i-1] or site == current[i+1]:
                continue
            if site not in where:
                # Check adjacency constraints
                if site in _neighbours(current[i-1]) and site in _neighbours(current[i+1]):
                    candidates.append(site)
        
        if not candidates:
            continue
        
        new_pos = rng.choice(candidates)
        
        # Make move
        old_pos = current[i]
        del where[old_pos]
        where[new_pos] = i
        current[i] = new_pos
        
        new_energy = _energy(sequence, current)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < __import__('math').exp(-delta / max(T, 1e-10)):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best = list(current)
        else:
            # Revert
            where[old_pos] = i
            del where[new_pos]
            current[i] = old_pos
    
    return best, best_energy


def fold(sequence, dimension):
    """Fold sequence into low-energy self-avoiding conformation on square lattice."""
    _init_time()
    n = len(sequence)
    
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    if n == 2:
        return [[0, 0], [1, 0]]
    if n == 3:
        return [[0, 0], [1, 0], [1, 1]]
    
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Multiple greedy growths with different parameters
    time_budget = _time_remaining()
    if time_budget < 10:
        # Not enough time, just do a quick greedy
        chain = _grow_greedy(sequence, rng)
        if chain:
            return [list(p) for p in chain]
        return [[i, 0] for i in range(n)]
    
    n_grow = min(50, max(10, int(time_budget * 5)))
    
    for trial in range(n_grow):
        if _time_remaining() < 5:
            break
        alpha = 0.3 + rng.random() * 0.7
        beta = 0.5 + rng.random() * 1.0
        chain = _grow_greedy(sequence, rng, alpha=alpha, beta=beta)
        if chain is None:
            continue
        energy = _energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        # Fallback: straight line
        return [[i, 0] for i in range(n)]
    
    # Phase 2: Local optimization with simulated annealing
    time_budget = _time_remaining()
    if time_budget > 20:
        # Do several rounds of SA on the best chain
        n_sa_rounds = min(5, max(1, int(time_budget / 15)))
        for _ in range(n_sa_rounds):
            if _time_remaining() < 5:
                break
            chain, energy = _simulated_annealing(
                sequence, best_chain, rng, 
                n_iter=min(200, max(50, int(_time_remaining() * 2))),
                T0=2.0, T1=0.01
            )
            if energy < best_energy:
                best_chain = chain
                best_energy = energy
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
