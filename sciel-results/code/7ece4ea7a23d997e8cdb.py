"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
_DEADLINE = time.time() + 180.0


def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _compute_energy(sequence, chain):
    """
    Compute the HP contact energy: -1 for each H-H non-bonded contact.
    chain is a list of (x, y) tuples.
    """
    n = len(chain)
    if n < 3:
        return 0
    
    # Map position to index for O(1) lookup
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i

    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        # Check 4 neighbors
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j_pos = (x + dx, y + dy)
            j = pos_to_idx.get(j_pos)
            if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                contacts += 1
    # Each contact is counted twice (once from each H), so divide by 2
    return -contacts // 2


def _greedy_fold(sequence, dimension, rng):
    """
    One randomized greedy chain growth.
    Prioritizes sites with new H-H contacts, then free space, with some randomness.
    Returns a valid self-avoiding chain or None if stuck.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        candidates = []
        last = chain[-1]
        
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            # Calculate immediate gain (new H-H contacts)
            gain = 0
            if sequence[i] == 'H':
                for nb in _neighbours(site):
                    idx = occupied.get(nb)
                    if idx is not None and idx < i - 1 and sequence[idx] == 'H':
                        gain += 1
            
            # Calculate free space (heuristic for avoiding dead ends)
            free = sum(1 for nb in _neighbours(site) if nb not in occupied)
            
            # Score: gain is primary, free space secondary, random noise for exploration
            # Tuned weights: gain is most important, free space helps prevent trapping
            score = gain * 1.0 + free * 0.1 + rng.random()
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        # Pick the best candidate
        # Sorting is O(k log k) where k <= 4, so it's very fast
        candidates.sort(key=lambda x: -x[0])
        best_site = candidates[0][1]
        
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain


def fold(sequence, dimension):
    """
    Find the best conformation for the given HP sequence.
    Returns a list of [x, y] coordinates.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Fallback: straight line (always valid, energy 0)
    straight = [[i, 0] for i in range(n)]
    
    best_chain = None
    best_energy = 0  # Energy is negative, so we want the most negative (lowest)
    
    # Use a fixed seed for reproducibility in the sandbox if needed, 
    # but varying seeds per restart helps exploration.
    # We use a master RNG to generate seeds for individual fold attempts if we wanted,
    # but here we just use one RNG instance for the whole run.
    rng = random.Random(42)
    
    # Number of restarts is dynamic based on time budget.
    # Greedy fold is O(n^2) roughly due to neighbor checks, but n<=100 so it's fast.
    # We can do thousands of restarts.
    
    while time.time() < _DEADLINE - 5:
        chain = _greedy_fold(sequence, dimension, rng)
        
        if chain is None:
            continue
        
        energy = _compute_energy(sequence, chain)
        
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
            
            # Optional: Early stopping if we reach a very good energy?
            # We don't know the theoretical minimum, but if we hit a known best 
            # for that length, we could stop. For now, just keep going.
            
    if best_chain is None:
        return straight
        
    # Convert tuples to lists as required by the interface
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
