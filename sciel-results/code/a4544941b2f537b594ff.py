"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math
import heapq

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
_DEADLINE = time.time() + 170.0

# Directions: 0:R, 1:D, 2:L, 3:U
DIRS = [(1, 0), (0, -1), (-1, 0), (0, 1)]

def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2

def _generate_random_saw(n, rng):
    """
    Generate a random self-avoiding walk of length n.
    """
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    def gen(length, current, occupied):
        if length == 0:
            return [current]
        
        dirs = list(range(4))
        rng.shuffle(dirs)
        
        for d in dirs:
            nx, ny = current[0] + DIRS[d][0], current[1] + DIRS[d][1]
            new_pos = (nx, ny)
            if new_pos not in occupied:
                occupied.add(new_pos)
                suffix = gen(length - 1, new_pos, occupied)
                if suffix:
                    return [current] + suffix
                occupied.remove(new_pos)
        
        return None

    occupied = {(0, 0)}
    result = gen(n - 1, (0, 0), occupied)
    if result is None:
        return [(i, 0) for i in range(n)]
    return result

def _regrow_suffix(chain, sequence, start_idx, rng, temperature=0.0):
    """
    Regrow the suffix of the chain starting from start_idx.
    """
    prefix = chain[:start_idx]
    occupied = {p: i for i, p in enumerate(prefix)}
    new_suffix = []
    last = prefix[-1]
    
    for i in range(start_idx, len(chain)):
        res = sequence[i]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if res == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))
        
        if not options:
            return None
        
        best_score, best_site = max(options, key=lambda x: x[0])
        occupied[best_site] = i
        new_suffix.append(best_site)
        last = best_site
        
    return new_suffix

def _beam_search_fold(sequence, dimension, rng, beam_width=50, max_steps=None):
    """
    Hierarchical Beam Search:
    1. Generate a diverse set of initial random SAWs (the initial beam).
    2. Iteratively expand the best states using Regrow moves.
    3. Prune the beam to keep only the top states by energy.
    
    This focuses computational effort on the most promising conformations,
    unlike parallel independent SA runs which may waste time on local minima.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    if max_steps is None:
        max_steps = 200000 # High cap, controlled by time
    
    # Initialize beam with diverse random walks
    beam = []
    occupied_beams = [] # To avoid duplicates? Maybe not strictly necessary if we dedupe by chain
    
    for _ in range(beam_width):
        chain = _generate_random_saw(n, rng)
        energy = _compute_energy(sequence, chain)
        beam.append((energy, chain))
        
    # Sort beam by energy (best is most negative)
    beam.sort(key=lambda x: x[0])
    
    best_energy = beam[0][0]
    best_chain = beam[0][1]
    
    # Beam Search Loop
    # We will perform a series of "steps" where we take the current beam,
    # generate candidates from each state, and prune.
    
    # To be efficient, we don't expand ALL beam states every time.
    # We expand the top K states (K = beam_width // 2 or similar).
    # And we generate multiple candidates per state.
    
    expansion_width = max(5, beam_width // 2)
    
    for step in range(max_steps):
        if time.time() > _DEADLINE - 10:
            break
            
        new_candidates = []
        
        # Expand the top part of the beam
        for i in range(expansion_width):
            if i >= len(beam):
                break
            _, chain = beam[i]
            
            # Generate a few Regrow moves for this state
            # Number of moves per state
            moves_per_state = 3
            
            for _ in range(moves_per_state):
                # Pick a random split point
                # Bias towards shorter suffixes for finer tuning? Or random?
                # Random split is good for global exploration.
                min_split = 3
                max_split = max(min_split, n - 2)
                if max_split <= min_split:
                    continue
                split = rng.randint(min_split, max_split)
                
                # Regrow with some temperature to allow non-greedy moves
                # Temperature should be low for beam search to be effective
                temp = 0.5
                new_suffix = _regrow_suffix(chain, sequence, split, rng, temperature=temp)
                if new_suffix is None:
                    continue
                
                new_chain = chain[:split] + new_suffix
                new_energy = _compute_energy(sequence, new_chain)
                new_candidates.append((new_energy, new_chain))
        
        if not new_candidates:
            # No candidates generated, maybe stuck?
            # Try to force some moves or break
            if time.time() > _DEADLINE - 20:
                break
            continue
            
        # Merge beam and new candidates
        beam.extend(new_candidates)
        
        # Prune: Keep only the top beam_width states
        beam.sort(key=lambda x: x[0])
        beam = beam[:beam_width]
        
        # Update best
        if beam[0][0] < best_energy:
            best_energy = beam[0][0]
            best_chain = beam[0][1]
            
        # Optional: Early termination if no improvement for many steps?
        # For now, just run until time limit or max_steps.
        
    return best_chain

def fold(sequence, dimension):
    """
    Fold the sequence using Hierarchical Beam Search.
    """
    rng = random.Random(42)
    n = len(sequence)
    
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    # Allocate time for Beam Search
    # We have ~170s.
    # Beam Search is a single continuous process, so we just let it run until deadline.
    
    # For very short sequences, beam search is overkill but safe.
    # For long sequences, it's more efficient than parallel SA.
    
    # Beam width: 
    # Smaller sequences can use larger beams.
    # Larger sequences: smaller beams to allow more steps.
    if n <= 30:
        beam_width = 100
    elif n <= 60:
        beam_width = 50
    else:
        beam_width = 30
        
    best_chain = _beam_search_fold(sequence, dimension, rng, beam_width=beam_width)
    
    # Fallback
    if best_chain is None or len(best_chain) != n:
        best_chain = _generate_random_saw(n, rng)
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
