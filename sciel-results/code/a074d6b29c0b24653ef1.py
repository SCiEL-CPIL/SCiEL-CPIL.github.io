"""HP lattice protein folding with simulated annealing using reversals and pivots."""
# EVOLVE-BLOCK-START
import time
import random
import math

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 115.0  # Safety margin before 210s kill


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbours(point):
    x, y = point
    for dx, dy in _DELTA:
        yield (x + dx, y + dy)


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H non-bonded contact."""
    n = len(chain)
    if n < 2:
        return 0
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        for dx, dy in _DELTA:
            j = where.get((x + dx, y + dy))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _grow(sequence, dimension, rng, bias=0.5):
    """Greedy chain growth with randomization."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Score: prefer sites that create H-H contacts
            score = rng.random() * bias
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H':
                        score += 1.0
            # Small preference for open space
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += 0.05 * free
            options.append((score, site))
        
        if not options:
            return None
        
        options.sort(key=lambda x: -x[0])
        if bias > 0 and len(options) > 1:
            k = min(len(options), max(1, int(bias * 3)))
            chosen = rng.choice(options[:k])
        else:
            chosen = options[0]
        
        site = chosen[1]
        occupied[site] = i
        chain.append(site)
    
    return chain


def _reverse_move(chain, i, j):
    """Reverse the segment from i to j (inclusive). Returns None if invalid."""
    if i >= j:
        return None
    new_chain = chain[:i] + chain[i:j+1][::-1] + chain[j+1:]
    # Check self-avoidance
    seen = set()
    for p in new_chain:
        if p in seen:
            return None
        seen.add(p)
    # Check unit steps
    for k in range(len(new_chain) - 1):
        x1, y1 = new_chain[k]
        x2, y2 = new_chain[k+1]
        if abs(x1 - x2) + abs(y1 - y2) != 1:
            return None
    return new_chain


def _pivot_move(chain, i):
    """Pivot at position i. Returns list of valid new chains."""
    if i < 1 or i > len(chain) - 2:
        return []
    a = chain[i-1]
    b = chain[i]
    c = chain[i+1]
    
    valid_chains = []
    for site in _neighbours(a):
        if site == b:
            continue
        # Check if site is adjacent to c
        if abs(site[0] - c[0]) + abs(site[1] - c[1]) != 1:
            continue
        # Check if site is not occupied by any other residue
        new_chain = chain[:i] + [site] + chain[i+1:]
        # Check self-avoidance
        seen = set()
        valid = True
        for k, p in enumerate(new_chain):
            if p in seen:
                valid = False
                break
            seen.add(p)
        if valid:
            valid_chains.append(new_chain)
    
    return valid_chains


def _anneal(sequence, chain, rng, max_iters=5000):
    """Simulated annealing with reversals and pivots."""
    if chain is None:
        return None
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    current_chain = list(best_chain)
    current_energy = best_energy
    n = len(current_chain)
    
    T = 5.0
    T_min = 0.01
    cooling = (T_min / T) ** (1.0 / max(1, max_iters))
    
    for it in range(max_iters):
        if _remaining_time() < 0.5:
            break
        
        # Choose move type
        if n >= 4 and rng.random() < 0.7:
            # Reversal move
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            new_chain = _reverse_move(current_chain, i, j)
        else:
            # Pivot move
            i = rng.randint(1, n - 2)
            candidates = _pivot_move(current_chain, i)
            if not candidates:
                T *= cooling
                continue
            new_chain = rng.choice(candidates)
        
        if new_chain is None:
            T *= cooling
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta <= 0 or rng.random() < math.exp(-delta / max(T, 0.001)):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_chain = list(current_chain)
                best_energy = current_energy
        
        T *= cooling
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using greedy growth + simulated annealing."""
    if not sequence or dimension != 2:
        return []
    
    n = len(sequence)
    if n == 0:
        return []
    
    rng = random.Random(12345 + n)
    best_chain = None
    best_energy = 0
    
    # Phase 1: Multiple greedy growths
    num_grows = min(50, max(5, 2000 // n))
    for _ in range(num_grows):
        if _remaining_time() < 5.0:
            break
        bias = 0.3 + rng.random() * 0.7
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is not None:
            energy = _energy(sequence, chain)
            if best_chain is None or energy < best_energy:
                best_chain = chain
                best_energy = energy
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Simulated annealing from best chain
    if _remaining_time() > 3.0:
        max_iters = min(10000, max(1000, int(_remaining_time() * 50)))
        improved = _anneal(sequence, best_chain, rng, max_iters=max_iters)
        if improved is not None:
            energy = _energy(sequence, improved)
            if energy < best_energy:
                best_chain = improved
                best_energy = energy
    
    # Phase 3: Additional annealing restarts if time permits
    if _remaining_time() > 5.0:
        for restart in range(3):
            if _remaining_time() < 2.0:
                break
            bias = 0.3 + rng.random() * 0.7
            chain = _grow(sequence, dimension, rng, bias=bias)
            if chain is not None:
                max_iters = min(5000, max(500, int(_remaining_time() * 20)))
                improved = _anneal(sequence, chain, rng, max_iters=max_iters)
                if improved is not None:
                    energy = _energy(sequence, improved)
                    if energy < best_energy:
                        best_chain = improved
                        best_energy = energy
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
