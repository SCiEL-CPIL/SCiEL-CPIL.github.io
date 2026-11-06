"""Rewritten program for HP lattice protein folding with local optimization."""
# EVOLVE-BLOCK-START
import random
import time
from collections import defaultdict


def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute energy: -number of H-H contacts (non-consecutive)."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _local_optimize(sequence, chain, max_moves=200, rng=None):
    """Improve a chain using single-residue moves (if valid)."""
    if rng is None:
        rng = random.Random()
    
    n = len(chain)
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    
    # Build position lookup
    pos_map = {p: i for i, p in enumerate(best_chain)}
    
    improved = True
    moves_made = 0
    while improved and moves_made < max_moves:
        improved = False
        # Try moving each residue (except ends) to a neighboring free site
        for i in range(1, n - 1):
            current_site = best_chain[i]
            for new_site in _neighbours(current_site):
                if new_site in pos_map:
                    continue
                # Check if moving residue i to new_site is valid
                # The new site must not be adjacent to any non-consecutive residue
                # Actually, we just need to check self-avoidance after the move
                # But we also need to ensure the chain remains connected
                # For a single residue move, we need new_site to be adjacent to both i-1 and i+1
                if new_site not in pos_map.get(best_chain[i-1], set()) and new_site not in pos_map.get(best_chain[i+1], set()):
                    pass
                # Simpler: just check that new_site is free and not creating issues
                # Actually for HP folding, a valid move must keep the chain self-avoiding
                # and connected. The simplest valid moves are those where the residue
                # moves to a site adjacent to both its neighbors.
                if (new_site == best_chain[i-1] or new_site == best_chain[i+1]):
                    continue
                # Check if new_site is adjacent to i-1 and i+1
                dist_prev = sum(abs(a-b) for a, b in zip(new_site, best_chain[i-1]))
                dist_next = sum(abs(a-b) for a, b in zip(new_site, best_chain[i+1]))
                if dist_prev == 1 and dist_next == 1:
                    # Try the move
                    old_energy = best_energy
                    # Remove from old position
                    del pos_map[current_site]
                    pos_map[new_site] = i
                    best_chain[i] = new_site
                    new_energy = _compute_energy(sequence, best_chain)
                    if new_energy < best_energy:
                        best_energy = new_energy
                        improved = True
                        moves_made += 1
                    else:
                        # Revert
                        del pos_map[new_site]
                        pos_map[current_site] = i
                        best_chain[i] = current_site
                    break  # Only try one move per residue per iteration
    return best_chain, best_energy


def _grow_greedy(sequence, dimension, rng, score_func='contacts'):
    """Grow chain greedily with different scoring functions."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            # Calculate score
            gain = 0
            if sequence[i] == 'H':
                # Count new H-H contacts
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            gain += 1
            
            # Count free neighbors (for flexibility)
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            
            if score_func == 'contacts':
                score = gain + 0.1 * free + rng.random() * 0.5
            elif score_func == 'balanced':
                score = gain * 2 + 0.3 * free + rng.random() * 0.3
            else:
                score = gain + rng.random()
            
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick best option
        best_score, best_site = max(options)
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain


def _hill_climb(sequence, chain, iterations=50, rng=None):
    """Local optimization using single-residue moves."""
    if rng is None:
        rng = random.Random()
    
    n = len(chain)
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    
    for _ in range(iterations):
        improved = False
        # Try moving each interior residue
        for i in range(1, n - 1):
            current = best_chain[i]
            # Find free neighbors that are adjacent to both i-1 and i+1
            candidates = []
            for site in _neighbours(current):
                if site in best_chain:
                    continue
                # Check if site is adjacent to both neighbors
                if (sum(abs(a-b) for a, b in zip(site, best_chain[i-1])) == 1 and
                    sum(abs(a-b) for a, b in zip(site, best_chain[i+1])) == 1):
                    # Check self-avoidance: site must not be in the chain
                    if site not in set(best_chain):
                        candidates.append(site)
            
            if candidates:
                # Try each candidate
                for new_site in candidates:
                    # Temporarily move
                    old_site = best_chain[i]
                    best_chain[i] = new_site
                    new_energy = _compute_energy(sequence, best_chain)
                    if new_energy < best_energy:
                        best_energy = new_energy
                        improved = True
                        break
                    else:
                        best_chain[i] = old_site
        
        if not improved:
            break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold the sequence using greedy growth + local optimization."""
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    best_chain = None
    best_energy = 0  # Energy is negative, so 0 is worst
    
    rng = random.Random(42 + n)
    
    # Phase 1: Multiple greedy growths with different strategies
    strategies = ['contacts', 'balanced', 'random']
    num_growths = min(300, max(50, int(60 * n / 10)))  # Scale with length
    
    for s_idx, strategy in enumerate(strategies):
        for g in range(num_growths // len(strategies)):
            if time.time() > deadline - 30:
                break
            
            chain = _grow_greedy(sequence, dimension, rng, score_func=strategy)
            if chain is None:
                continue
            
            energy = _compute_energy(sequence, chain)
            
            # Phase 2: Local optimization
            if time.time() > deadline - 60:
                if energy < best_energy:
                    best_energy = energy
                    best_chain = chain
                continue
            
            optimized_chain, optimized_energy = _hill_climb(sequence, chain, iterations=30, rng=rng)
            
            if optimized_energy < best_energy:
                best_energy = optimized_energy
                best_chain = optimized_chain
            
            # Check if we've likely found a good solution
            if best_energy <= -n // 2:
                # Good enough, try to improve a bit more
                pass
    
    # Phase 3: If we have time, do more hill climbing on the best chain
    if best_chain is not None and time.time() < deadline - 10:
        final_chain, final_energy = _hill_climb(sequence, best_chain, iterations=100, rng=rng)
        if final_energy < best_energy:
            best_energy = final_energy
            best_chain = final_chain
    
    # Fallback: straight line if no valid chain found
    if best_chain is None:
        best_chain = [[i] + [0] * (dimension - 1) for i in range(n)]
        best_energy = 0
    
    return best_chain
# EVOLVE-BLOCK-END
