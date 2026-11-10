"""Combined HP lattice protein folding: greedy growth with SA segment-reversal polish."""
import random
import time
import math

# Global deadline with safety margin (evaluator kills at 210s wall-clock)
_START_TIME = time.time()
_DEADLINE = _START_TIME + 180.0  # 30s safety margin


def _neighbours2d(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _grow(sequence, dimension, rng, greedy_bias=2.5):
    """Randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    
    for i in range(1, n):
        options = []
        last = chain[-1]
        for site in _neighbours2d(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours2d(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            # Count free neighbours (excluding the current last position to avoid bias)
            free = 0
            for q in _neighbours2d(site):
                if q not in occupied and q != last:
                    free += 1
            score = gain * greedy_bias + 0.15 * free + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    
    return chain


def _energy(sequence, chain):
    """Compute the energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours2d(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _is_valid_walk(chain):
    """Check if the chain is a valid self-avoiding walk with unit steps."""
    if len(chain) <= 1:
        return True
    occupied = set()
    for i, p in enumerate(chain):
        if p in occupied:
            return False
        occupied.add(p)
        if i > 0:
            prev = chain[i - 1]
            dist = abs(p[0] - prev[0]) + abs(p[1] - prev[1])
            if dist != 1:
                return False
    return True


def _reverse_segment(chain, start, end):
    """Reverse a segment of the chain and check validity. Returns (new_chain, valid)."""
    if start >= end:
        return chain, False
    
    new_chain = chain[:start] + chain[start:end + 1][::-1] + chain[end + 1:]
    
    # Check self-avoidance
    occupied = set()
    for p in new_chain:
        if p in occupied:
            return chain, False
        occupied.add(p)
    
    # Check unit steps
    for i in range(1, len(new_chain)):
        dist = abs(new_chain[i][0] - new_chain[i - 1][0]) + abs(new_chain[i][1] - new_chain[i - 1][1])
        if dist != 1:
            return chain, False
    
    return new_chain, True


def _sa_polish(sequence, chain, time_budget, rng):
    """Simulated annealing with segment-reversal moves to polish a conformation."""
    n = len(chain)
    if n < 4 or time_budget <= 0:
        return chain, _energy(sequence, chain)
    
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    # SA parameters
    T = max(1.0, n * 0.08)
    T_min = 0.01
    cooling_rate = 0.9985
    
    start_time = time.time()
    max_iters = int(time_budget * 2000)  # Rough estimate of iterations possible
    
    iters = 0
    while iters < max_iters and time.time() - start_time < time_budget and T > T_min:
        iters += 1
        
        # Choose segment length: prefer shorter for local moves
        if rng.random() < 0.6:
            length = rng.randint(2, min(8, n))
        else:
            length = rng.randint(2, n)
        
        start = rng.randint(0, n - length)
        end = start + length - 1
        
        new_chain, valid = _reverse_segment(chain, start, end)
        if not valid:
            T *= cooling_rate
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Metropolis criterion
        if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
        
        T *= cooling_rate
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using greedy growth with SA segment-reversal polish."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    # Check deadline
    if time.time() >= _DEADLINE - 5:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    
    # Track best solution found
    best_chain = None
    best_energy = 0
    
    # Generate multiple diverse initial conformations via greedy growth
    # Refine top-K candidates to explore multiple basins
    candidates = []
    num_initial = 20  # Generate 20 diverse initial conformations
    
    # Time budget for initial generation
    init_deadline = min(_DEADLINE - 30, time.time() + 5)
    
    for _ in range(num_initial):
        if time.time() >= init_deadline:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        if not _is_valid_walk(chain):
            continue
        energy = _energy(sequence, chain)
        candidates.append((energy, chain))
    
    if not candidates:
        # Fallback to linear
        return [[i, 0] for i in range(n)]
    
    # Sort by energy (best first) and refine top-K
    candidates.sort(key=lambda x: x[0])
    
    # Refine top 5 candidates (or fewer if we have less time)
    top_k = min(5, len(candidates))
    
    # Remaining time for polishing
    remaining_time = _DEADLINE - time.time() - 5  # 5s safety margin
    if remaining_time < 1:
        # Not enough time, return best initial
        best_chain = candidates[0][1]
        best_energy = candidates[0][0]
    else:
        # Allocate time to each candidate: more time for better candidates
        # Give 40% to best, 25% to second, 20% to third, 10% to fourth, 5% to fifth
        time_weights = [0.4, 0.25, 0.2, 0.1, 0.05]
        while len(time_weights) < top_k:
            time_weights.append(0.01)
        
        for k in range(top_k):
            if time.time() >= _DEADLINE - 5:
                break
            
            init_energy, init_chain = candidates[k]
            
            # Allocate time proportional to weight
            alloc_time = remaining_time * time_weights[k]
            if alloc_time < 0.5:
                alloc_time = 0.5
            
            polished_chain, polished_energy = _sa_polish(
                sequence, init_chain, alloc_time, rng
            )
            
            if polished_energy < best_energy:
                best_energy = polished_energy
                best_chain = polished_chain
    
    if best_chain is None:
        best_chain = candidates[0][1]
    
    # Final validity check
    if not _is_valid_walk(best_chain):
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
