"""HP lattice protein folding: diverse greedy growth + simulated annealing with reversals and pivots."""
# EVOLVE-BLOCK-START
import time
import random
import math

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 115.0


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbours(point):
    x, y = point
    for dx, dy in _DELTAS:
        yield (x + dx, y + dy)


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H non-bonded contact."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        for dx, dy in _DELTAS:
            j = where.get((x + dx, y + dy))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _grow(sequence, dimension, rng, bias=0.0):
    """Greedy chain growth with randomization bias."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Score: prefer sites with H-H contacts
            score = rng.random() * bias
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H':
                        score += 1.0
            # Small preference for open space to avoid trapping
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += 0.05 * free
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick best with some randomness
        options.sort(key=lambda x: -x[0])
        if bias > 0 and len(options) > 1:
            # Random pick from top few
            k = min(len(options), max(1, int(bias * 3)))
            chosen = rng.choice(options[:k])
        else:
            chosen = options[0]
        
        site = chosen[1]
        occupied[site] = i
        chain.append(site)
    
    return chain


def _apply_reversal(chain, i, j):
    """Apply a reversal move: reverse the segment from i to j inclusive.
    Returns new chain or None if invalid."""
    if i >= j:
        return None
    n = len(chain)
    # Check that the reversed segment maintains connectivity
    if i > 0:
        if abs(chain[i-1][0] - chain[j][0]) + abs(chain[i-1][1] - chain[j][1]) != 1:
            return None
    if j < n - 1:
        if abs(chain[j+1][0] - chain[i][0]) + abs(chain[j+1][1] - chain[i][1]) != 1:
            return None
    
    new_chain = chain[:i] + chain[i:j+1][::-1] + chain[j+1:]
    
    # Check self-avoidance
    seen = set()
    for p in new_chain:
        if p in seen:
            return None
        seen.add(p)
    
    return new_chain


def _apply_pivot(chain, i):
    """Apply a pivot move at position i (move single residue).
    Returns list of possible new chains."""
    n = len(chain)
    if i <= 0 or i >= n - 1:
        return []
    
    results = []
    last = chain[i-1]
    current = chain[i]
    
    for site in _neighbours(last):
        if site == current:
            continue
        # Check if site is occupied by any other residue
        occupied = set(chain)
        occupied.discard(current)
        if site in occupied:
            continue
        
        new_chain = chain[:]
        new_chain[i] = site
        
        # Check self-avoidance
        seen = set()
        valid = True
        for p in new_chain:
            if p in seen:
                valid = False
                break
            seen.add(p)
        
        if valid:
            results.append(new_chain)
    
    return results


def _anneal(sequence, chain, rng, max_iters=500):
    """Simulated annealing with reversal and pivot moves."""
    if chain is None or len(chain) < 3:
        return chain
    
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    T = 1.0
    cooling = 0.995
    
    for it in range(max_iters):
        if _remaining_time() < 0.5:
            break
        
        # Choose move type
        if rng.random() < 0.5 and n >= 4:
            # Reversal move
            i = rng.randint(1, n - 3)
            j = rng.randint(i + 1, n - 1)
            new_chain = _apply_reversal(current_chain, i, j)
            if new_chain is None:
                continue
        else:
            # Pivot move (single residue move)
            i = rng.randint(1, n - 1)
            options = _apply_pivot(current_chain, i)
            if not options:
                continue
            new_chain = rng.choice(options)
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        T *= cooling
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using hybrid search strategy."""
    if not sequence or dimension != 2:
        return []
    
    n = len(sequence)
    if n == 0:
        return []
    
    # For very short sequences, use exhaustive or simple methods
    if n <= 8:
        # Try many greedy growths
        rng = random.Random(12345 + n)
        best_chain = None
        best_energy = 0
        for _ in range(200):
            chain = _grow(sequence, dimension, rng, bias=rng.random())
            if chain is not None:
                energy = _energy(sequence, chain)
                if best_chain is None or energy < best_energy:
                    best_chain = chain
                    best_energy = energy
        if best_chain is None:
            best_chain = [(i, 0) for i in range(n)]
        return [list(p) for p in best_chain]
    
    # For longer sequences, use hybrid approach
    best_chain = None
    best_energy = 0
    
    # Phase 1: Multiple greedy growths with local search
    rng = random.Random(12345 + n)
    num_growths = 50
    for i in range(num_growths):
        if _remaining_time() < 10.0:
            break
        bias = 0.3 + rng.random() * 0.7
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is not None:
            # Apply simulated annealing
            chain = _anneal(sequence, chain, rng, max_iters=100)
            energy = _energy(sequence, chain)
            if best_chain is None or energy < best_energy:
                best_chain = chain
                best_energy = energy
    
    # Phase 2: If we have time, try more sophisticated search
    if _remaining_time() > 15.0 and best_chain is not None:
        # Try to improve with more SA iterations
        improved = _anneal(sequence, best_chain, rng, max_iters=300)
        if improved is not None:
            energy = _energy(sequence, improved)
            if energy < best_energy:
                best_chain = improved
                best_energy = energy
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
