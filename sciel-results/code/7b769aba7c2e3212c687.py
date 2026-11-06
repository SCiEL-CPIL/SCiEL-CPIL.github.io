"""HP lattice protein folding using simulated annealing with reversals and moves."""
# EVOLVE-BLOCK-START
import random
import math
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _build_contact_info(sequence, chain):
    """Build mapping from lattice point to residue index, and track H positions."""
    where = {}
    for i, p in enumerate(chain):
        where[tuple(p)] = i
    h_positions = set()
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            h_positions.add(tuple(p))
    return where, h_positions


def _compute_energy(sequence, chain, where=None, h_positions=None):
    """Compute energy: -number of H-H contacts (non-bonded neighbors)."""
    if where is None or h_positions is None:
        where, h_positions = _build_contact_info(sequence, chain)
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2  # each contact counted twice


def _compute_energy_fast(sequence, chain):
    """Optimized energy computation."""
    where = {}
    for i, p in enumerate(chain):
        where[tuple(p)] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng):
    """Greedy chain growth preferring sites with new H-H contacts."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.05 * free + rng.random() * 0.3, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _reverse(chain, i, j):
    """Reverse segment from index i to j (inclusive)."""
    if i >= j:
        return None
    new_chain = chain[:i] + chain[i:j+1][::-1] + chain[j+1:]
    return new_chain


def _is_self_avoiding(chain):
    """Check if chain is self-avoiding."""
    seen = set()
    for p in chain:
        tp = tuple(p)
        if tp in seen:
            return False
        seen.add(tp)
    return True


def _simulated_annealing(sequence, chain, rng, max_iterations, start_temp, end_temp):
    """Simulated annealing with reversal moves."""
    n = len(sequence)
    current_chain = list(chain)
    current_energy = _compute_energy_fast(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    if max_iterations <= 0:
        return best_chain, best_energy
    
    cooling_factor = (end_temp / start_temp) ** (1.0 / max_iterations)
    
    for it in range(max_iterations):
        temp = start_temp * (cooling_factor ** it)
        if temp < 1e-10:
            temp = 1e-10
        
        # Choose a reversal move
        i = rng.randint(0, n - 2)
        j = rng.randint(i + 1, n - 1)
        
        new_chain = _reverse(current_chain, i, j)
        if new_chain is None:
            continue
        
        if not _is_self_avoiding(new_chain):
            continue
        
        new_energy = _compute_energy_fast(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Find best conformation using greedy growth + simulated annealing."""
    deadline = time.time() + 175  # 5s safety margin before 180s limit
    
    rng = random.Random(42 + len(sequence))
    
    # Generate initial conformations
    initial_chains = []
    for _ in range(10):
        chain = _grow(sequence, dimension, rng)
        if chain is not None:
            initial_chains.append(chain)
    
    if not initial_chains:
        # Fallback: straight line
        return [[i, 0] for i in range(len(sequence))]
    
    best_chain = initial_chains[0]
    best_energy = _compute_energy_fast(sequence, best_chain)
    
    # Run SA on each initial conformation, with time budget
    n = len(sequence)
    # Scale iterations based on sequence length
    base_iters = max(5000, min(50000, 200000 // n))
    
    for init_chain in initial_chains:
        if time.time() > deadline:
            break
        
        # Vary temperature parameters
        start_temp = rng.uniform(0.5, 2.0)
        end_temp = rng.uniform(0.01, 0.1)
        
        # Run SA
        result_chain, result_energy = _simulated_annealing(
            sequence, init_chain, rng, base_iters, start_temp, end_temp
        )
        
        if result_energy < best_energy:
            best_energy = result_energy
            best_chain = result_chain
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    # Final polish: quick local search on best
    if time.time() < deadline - 5:
        result_chain, result_energy = _simulated_annealing(
            sequence, best_chain, rng, base_iters // 2, 0.1, 0.001
        )
        if result_energy < best_energy:
            best_energy = result_energy
            best_chain = result_chain
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
