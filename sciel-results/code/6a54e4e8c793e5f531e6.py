"""
HP lattice protein folding on the cubic lattice.
Strategy: Multi-Start Global Regrowth (Backtracking) with Inlined Computation.
1. Generate diverse initial conformations using randomized greedy growth with H-packing bias.
2. Optimize each conformation via "Regrowth": pick a random split point, keep the prefix, 
   and greedily regrow the suffix to maximize contacts.
3. Use inlined neighbor and contact counting to minimize Python overhead.
"""
import random
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # Safety margin before 330s kill

def _neighbours(point):
    """Yield the 6 face-neighbours of point in 3D."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)
    yield (x, y + 1, z)
    yield (x, y - 1, z)

def _energy(sequence, chain):
    """Compute total energy: negative count of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y, z = p
            # Check 6 neighbors
            for q in ((x+1,y,z), (x-1,y,z), (x,y+1,z), (x,y-1,z), (x,y,z+1), (x,y,z-1)):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, dimension, rng, bias=1.0):
    """
    Fast initial self-avoiding walk with heuristic bias towards H contacts.
    """
    n = len(sequence)
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    # Precompute H positions for quick check? 
    # For growth, we just look at neighbors.
    
    for i in range(1, n):
        last = chain[-1]
        x, y, z = last
        options = []
        
        # Check 6 neighbors
        neighbors = [
            (x + 1, y, z), (x - 1, y, z),
            (x, y + 1, z), (x, y - 1, z),
            (x, y, z + 1), (x, y, z - 1)
        ]
        
        for site in neighbors:
            if site in occupied:
                continue
            
            # Heuristic: 
            # 1. Prefer sites that are adjacent to existing H residues (potential contacts)
            # 2. Prefer sites with more free neighbors (avoid trapping)
            
            sx, sy, sz = site
            h_score = 0
            free_score = 0
            
            # Check neighbors of this site
            for nx, ny, nz in ((sx+1,sy,sz), (sx-1,sy,sz), (sx,sy+1,sz), (sx,sy-1,sz), (sx,sy+1,sz-1) if False else (sx,sy,z+1), (sx,sy,z-1)):
                # Corrected neighbor list for site
                pass 
            
            # Let's do it properly but fast
            site_neighbors = [
                (sx + 1, sy, sz), (sx - 1, sy, sz),
                (sx, sy + 1, sz), (sx, sy - 1, sz),
                (sx, sy, sz + 1), (sx, sy, sz - 1)
            ]
            
            for sn in site_neighbors:
                if sn in occupied:
                    idx = occupied[sn]
                    if sequence[idx] == 'H' and abs(idx - i) > 1:
                        h_score += 1
                else:
                    free_score += 1
            
            # Combine scores
            # Weight H contacts heavily
            score = h_score * bias + free_score * 0.1 + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        # Sort and pick top few
        options.sort(reverse=True)
        top_k = min(3, len(options))
        # Pick randomly from top k to maintain diversity
        site = options[rng.randint(0, top_k - 1)][1]
        
        occupied[site] = i
        chain.append(site)
        
    return chain

def _regrow(sequence, chain, split_idx, rng, bias=1.5):
    """
    Regrow the chain from split_idx to the end.
    Returns new chain or None if stuck.
    """
    n = len(sequence)
    if split_idx >= n:
        return chain
    
    prefix = chain[:split_idx]
    occupied = {p: i for i, p in enumerate(prefix)}
    
    # We need to regrow residues split_idx to n-1
    # The current chain has length n. We are replacing the suffix.
    # Wait, the input chain is a full valid chain.
    # We take prefix [0..split_idx-1].
    # We need to place residues [split_idx..n-1].
    
    # The last point in prefix is chain[split_idx-1]
    last = prefix[-1]
    new_suffix = []
    
    for i in range(split_idx, n):
        x, y, z = last
        neighbors = [
            (x + 1, y, z), (x - 1, y, z),
            (x, y + 1, z), (x, y - 1, z),
            (x, y, z + 1), (x, y, z - 1)
        ]
        
        options = []
        for site in neighbors:
            if site in occupied:
                continue
            
            sx, sy, sz = site
            h_score = 0
            free_score = 0
            
            site_neighbors = [
                (sx + 1, sy, sz), (sx - 1, sy, sz),
                (sx, sy + 1, sz), (sx, sy - 1, sz),
                (sx, sy, sz + 1), (sx, sy, sz - 1)
            ]
            
            for sn in site_neighbors:
                if sn in occupied:
                    idx = occupied[sn]
                    if sequence[idx] == 'H' and abs(idx - i) > 1:
                        h_score += 1
                else:
                    free_score += 1
            
            score = h_score * bias + free_score * 0.1 + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        options.sort(reverse=True)
        top_k = min(3, len(options))
        site = options[rng.randint(0, top_k - 1)][1]
        
        occupied[site] = i
        new_suffix.append(site)
        last = site
        
    return prefix + new_suffix

def _mc_fold(sequence, dimension, time_budget):
    """
    Multi-start Regrowth optimization.
    """
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = float('inf')
    
    start_time = time.time()
    
    # 1. Initial Seed Generation
    # Generate a few diverse seeds
    seeds = []
    num_seeds = min(5, max(2, int(time_budget * 2))) # Scale seeds with time
    
    for s in range(num_seeds):
        chain = _greedy_grow(sequence, dimension, rng, bias=1.0 + s*0.5)
        if chain:
            e = _energy(sequence, chain)
            seeds.append((chain, e))
            if e < best_energy:
                best_energy = e
                best_chain = chain
    
    if not seeds:
        # Fallback
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
        return best_chain
    
    # 2. Iterative Regrowth
    # Alternate between seeds, or just keep the best?
    # Keeping the best is "hill climbing" on the regrowth space.
    # Let's run regrowth on the current best, and occasionally on other seeds to escape local minima.
    
    current_chain = best_chain
    current_energy = best_energy
    
    # Track which seed we are working on
    seed_idx = 0
    
    while time.time() < start_time + time_budget:
        # Pick a split point. 
        # Random split is good for global search.
        # Bias towards earlier splits to allow more freedom?
        # Or later splits for refinement?
        # Let's mix: 50% early (0-1/3), 50% late (1/3-1)
        
        if rng.random() < 0.5:
            split = rng.randint(1, n // 3)
        else:
            split = rng.randint(n // 3, n - 2)
            
        new_chain = _regrow(sequence, current_chain, split, rng, bias=1.5)
        
        if new_chain:
            new_energy = _energy(sequence, new_chain)
            
            if new_energy < current_energy:
                current_chain = new_chain
                current_energy = new_energy
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = new_chain
            else:
                # Accept worse moves with probability?
                # Simple hill climbing is often enough for regrowth.
                # But let's add a small chance to accept worse to escape.
                # If we accept worse, we move current_chain to the worse one.
                delta = new_energy - current_energy
                # Temperature-like acceptance
                if rng.random() < math.exp(-delta / 2.0):
                    current_chain = new_chain
                    current_energy = new_energy
        
        # Occasionally restart from a different seed to diversify
        if len(seeds) > 1 and rng.random() < 0.05:
            seed_idx = (seed_idx + 1) % len(seeds)
            current_chain = seeds[seed_idx][0]
            current_energy = seeds[seed_idx][1]
            # Update best if this seed is better (unlikely if we've been optimizing, but possible if seed was bad initially)
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain

    return best_chain

def fold(sequence, dimension):
    """
    Best conformation using Multi-Start Regrowth.
    """
    n = len(sequence)
    
    # Allocate time per sequence
    # 26 sequences. 325s total.
    # Longer chains need more time? 
    # Let's allocate proportional to length, but with a minimum.
    # Or just fixed? Fixed is safer to ensure all are attempted.
    # However, if we have time left, we should use it.
    
    elapsed = time.time() - _START_TIME
    remaining = _DEADLINE - elapsed
    if remaining < 1.0:
        # Emergency fallback
        return [[i, 0, 0] for i in range(n)]
    
    # Heuristic time allocation: 
    # Short chains (48) can be solved quickly. Long chains (136) are harder.
    # Let's give a base time + extra for length.
    base_time = 2.0
    extra_time = (n - 48) * 0.1
    time_per_seq = base_time + extra_time
    
    # Cap by remaining time
    # If this is the last sequence, we don't know. 
    # Assuming uniform distribution of 26 sequences.
    # If we use 325/26 ~ 12s per sequence, that's plenty for 48, tight for 136.
    # Let's just use a fixed 10s per sequence, capped by remaining.
    budget = 10.0
    if remaining - budget < 5.0:
        budget = remaining - 5.0
        
    if budget < 1.0:
        budget = 0.5

    best_chain = _mc_fold(sequence, dimension, budget)
    
    # Final check: ensure self-avoidance
    if len(set(best_chain)) != n:
        # Fallback
        return [[i, 0, 0] for i in range(n)]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
