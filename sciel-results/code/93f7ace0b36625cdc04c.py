"""HP lattice protein folding: fast growth + efficient SA with incremental energy."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 110.0


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbours(point):
    x, y = point
    return ((x+1, y), (x-1, y), (x, y+1), (x, y-1))


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    n = len(chain)
    if n < 2:
        return 0
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts // 2


def _grow(sequence, dimension, rng, bias_hh, bias_free, noise_scale=1.0):
    """One randomized greedy chain growth."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            score = bias_hh * gain + bias_free * free + rng.gauss(0, noise_scale)
            options.append((score, site))
        if not options:
            return None
        best_score, best_site = max(options)
        occupied[best_site] = i
        chain.append(best_site)
    return chain


def _check_self_avoiding(chain):
    """Fast self-avoidance check using a set."""
    seen = set()
    for p in chain:
        if p in seen:
            return False
        seen.add(p)
    return True


def _apply_reversal(chain, i, j):
    """Attempt a reversal move. Returns new chain if valid, else None."""
    n = len(chain)
    if i >= j:
        return None
    
    if i > 0:
        p_prev = chain[i - 1]
        p_end = chain[j]
        if abs(p_prev[0] - p_end[0]) + abs(p_prev[1] - p_end[1]) != 1:
            return None
            
    if j < n - 1:
        p_start = chain[i]
        p_next = chain[j + 1]
        if abs(p_start[0] - p_next[0]) + abs(p_start[1] - p_next[1]) != 1:
            return None
            
    new_chain = chain[:i] + chain[i:j+1][::-1] + chain[j+1:]
    
    if not _check_self_avoiding(new_chain):
        return None
        
    return new_chain


def _apply_pivot(chain, i, angle):
    """Attempt a pivot move. Returns new chain if valid, else None."""
    n = len(chain)
    if i <= 0 or i >= n - 1:
        return None
        
    pivot = chain[i]
    prefix = chain[:i+1]
    tail = chain[i+1:]
    
    rotated_tail = []
    for p in tail:
        x, y = p[0] - pivot[0], p[1] - pivot[1]
        for _ in range(angle):
            x, y = -y, x
        rotated_tail.append((pivot[0] + x, pivot[1] + y))
        
    new_chain = prefix + rotated_tail
    
    if not _check_self_avoiding(new_chain):
        return None
        
    return new_chain


def _anneal(sequence, initial_chain, time_limit, rng):
    """Simulated Annealing with adaptive cooling."""
    n = len(initial_chain)
    current_chain = initial_chain[:]
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain[:]
    best_energy = current_energy
    
    # Adaptive initial temperature
    T = max(0.5, 2.0 * math.log(n + 1) / n)
    # Cooling rate based on expected steps
    expected_steps = max(100, int(time_limit * 500))
    cooling_rate = (0.01 / T) ** (1.0 / expected_steps)
    
    deadline = time.time() + time_limit
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if _remaining_time() < 0.5:
            break
            
        if rng.random() < 0.6:
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            new_chain = _apply_reversal(current_chain, i, j)
        else:
            i = rng.randint(1, n - 2)
            angle = rng.choice([1, 2, 3])
            new_chain = _apply_pivot(current_chain, i, angle)
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 0.01)):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain[:]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        T *= cooling_rate
        
        if T < 0.01 or no_improve > max_no_improve:
            break
            
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold the sequence into a low-energy conformation."""
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    
    if n < 2:
        return [[0, 0]] * n
        
    best_chain = None
    best_energy = 0
    
    # Phase 1: Generate diverse initial chains with more attempts
    bias_configs = [
        (2.0, 0.0),   # Strong HH bias
        (1.0, 0.3),   # Moderate HH
        (0.5, 0.5),   # Balanced
        (0.2, 1.0),   # Free space
        (3.0, 0.0),   # Very strong HH
        (1.5, 0.2),   # HH with some free
    ]
    
    initial_chains = []
    growth_deadline = time.time() + min(3.0, _remaining_time() * 0.15)
    
    for bias_hh, bias_free in bias_configs:
        if time.time() >= growth_deadline:
            break
        for attempt in range(10):
            if time.time() >= growth_deadline or _remaining_time() < 5.0:
                break
            noise = 0.5 + 0.1 * attempt
            chain = _grow(sequence, dimension, rng, bias_hh, bias_free, noise)
            if chain is not None:
                initial_chains.append(chain)
    
    if not initial_chains:
        initial_chains.append([(i, 0) for i in range(n)])
        
    # Deduplicate by energy and keep top candidates
    scored_chains = []
    seen_energies = {}
    for chain in initial_chains:
        e = _energy(sequence, chain)
        key = (e,)
        if key not in seen_energies or e < seen_energies[key]:
            seen_energies[key] = e
            scored_chains.append((e, chain))
    
    scored_chains.sort(key=lambda x: x[0])
    
    # Keep top K unique chains
    top_k = min(6, len(scored_chains))
    candidates = [c[1] for c in scored_chains[:top_k]]
    
    # Phase 2: Anneal each candidate
    remaining = _remaining_time()
    time_per_candidate = max(0.5, remaining / len(candidates))
    
    for idx, chain in enumerate(candidates):
        if _remaining_time() < 1.0:
            break
            
        alloc_time = min(time_per_candidate, _remaining_time() - 0.5)
        if alloc_time < 0.1:
            continue
            
        best_c, best_e = _anneal(sequence, chain, alloc_time, rng)
        
        if best_e < best_energy:
            best_energy = best_e
            best_chain = best_c
            
    if best_chain is None:
        best_chain = initial_chains[0] if initial_chains else [(i, 0) for i in range(n)]
            
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
