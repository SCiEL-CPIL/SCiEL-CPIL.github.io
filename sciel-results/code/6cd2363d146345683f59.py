"""HP lattice protein folding with greedy growth, local search, and deterministic polish."""
# EVOLVE-BLOCK-START
import random
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _build_index(chain):
    return {p: i for i, p in enumerate(chain)}


def _compute_energy(sequence, chain):
    where = _build_index(chain)
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng, temperature=1.0):
    """Randomized greedy chain growth with temperature-based exploration."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H')
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Use temperature to balance greediness vs exploration
            score = gain / temperature + 0.1 * free / temperature + rng.random()
            options.append((score, site))
        if not options:
            return None
        # Pick with some probability based on score (soft-max style)
        if rng.random() < 0.7:  # Mostly greedy
            site = max(options)[1]
        else:
            # Random among top half
            options.sort(reverse=True)
            top_half = options[:max(1, len(options) // 2)]
            site = rng.choice(top_half)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_search(sequence, chain, max_iter=500):
    """Local search using single-residue moves and reversals."""
    n = len(chain)
    if n < 3:
        return chain
    
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    where = _build_index(best_chain)
    
    for _ in range(max_iter):
        improved = False
        # Try single-residue moves
        for i in range(1, n - 1):
            for site in _neighbours(best_chain[i]):
                if site in where and where[site] != i:
                    continue
                # Check if move is valid (doesn't collide)
                if site in where:
                    continue
                # Tentative move
                old_pos = best_chain[i]
                best_chain[i] = site
                new_energy = _compute_energy(sequence, best_chain)
                if new_energy < best_energy:
                    best_energy = new_energy
                    where = _build_index(best_chain)
                    improved = True
                else:
                    best_chain[i] = old_pos
        
        # Try reversals
        for start in range(0, n - 2):
            for end in range(start + 2, n):
                # Check if reversal is valid
                segment = best_chain[start:end+1]
                reversed_segment = list(reversed(segment))
                
                # Check for collisions
                valid = True
                for pos, new_site in enumerate(reversed_segment):
                    idx = start + pos
                    if idx in (start, end):
                        continue  # Endpoints stay or swap
                    if new_site in where and where[new_site] != idx:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Apply reversal
                best_chain[start:end+1] = reversed_segment
                new_energy = _compute_energy(sequence, best_chain)
                if new_energy < best_energy:
                    best_energy = new_energy
                    where = _build_index(best_chain)
                    improved = True
                else:
                    best_chain[start:end+1] = segment
        
        if not improved:
            break
    
    return best_chain


def _deterministic_polish(sequence, chain, max_iter=200):
    """Deterministic polish: only accept strictly improving moves."""
    n = len(chain)
    if n < 3:
        return chain
    
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    where = _build_index(best_chain)
    
    for _ in range(max_iter):
        improved = False
        
        # Try single-residue moves
        for i in range(1, n - 1):
            for site in _neighbours(best_chain[i]):
                if site in where and where[site] != i:
                    continue
                if site in where:
                    continue
                old_pos = best_chain[i]
                best_chain[i] = site
                new_energy = _compute_energy(sequence, best_chain)
                if new_energy < best_energy:
                    best_energy = new_energy
                    where = _build_index(best_chain)
                    improved = True
                else:
                    best_chain[i] = old_pos
        
        # Try reversals
        for start in range(0, n - 2):
            for end in range(start + 2, min(start + 10, n)):  # Limit reversal length for speed
                segment = best_chain[start:end+1]
                reversed_segment = list(reversed(segment))
                
                valid = True
                for pos, new_site in enumerate(reversed_segment):
                    idx = start + pos
                    if idx in (start, end):
                        continue
                    if new_site in where and where[new_site] != idx:
                        valid = False
                        break
                if not valid:
                    continue
                
                best_chain[start:end+1] = reversed_segment
                new_energy = _compute_energy(sequence, best_chain)
                if new_energy < best_energy:
                    best_energy = new_energy
                    where = _build_index(best_chain)
                    improved = True
                else:
                    best_chain[start:end+1] = segment
        
        if not improved:
            break
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using greedy growth, local search, and deterministic polish."""
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin before 210s kill
    
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    
    # Adjust number of attempts based on sequence length
    if n <= 30:
        num_grow = 400
        local_iter = 300
        polish_iter = 150
    elif n <= 50:
        num_grow = 250
        local_iter = 200
        polish_iter = 100
    elif n <= 70:
        num_grow = 150
        local_iter = 150
        polish_iter = 80
    else:
        num_grow = 100
        local_iter = 100
        polish_iter = 60
    
    best_chain = None
    best_energy = 0  # Start with 0 (unfolded) as baseline
    
    for attempt in range(num_grow):
        # Check time
        if time.time() > deadline - 5.0:
            break
        
        # Try different temperatures
        temperature = 0.5 + 0.5 * (attempt % 5) / 5.0
        chain = _grow(sequence, dimension, rng, temperature)
        if chain is None:
            continue
        
        # Local search
        chain = _local_search(sequence, chain, local_iter)
        
        # Deterministic polish
        chain = _deterministic_polish(sequence, chain, polish_iter)
        
        energy = _compute_energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
        
        # Early termination if we hit a very good energy
        if best_energy <= -n * 0.3:  # Heuristic: good enough
            break
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
