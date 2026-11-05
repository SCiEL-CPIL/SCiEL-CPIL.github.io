"""HP lattice protein folding using Local Monte Carlo Simulated Annealing."""
import random
import time
import math

def _compute_energy(sequence, chain):
    n = len(chain)
    if n == 0:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for axis in range(len(p)):
                for step in (-1, 1):
                    q = list(p)
                    q[axis] += step
                    q = tuple(q)
                    j = where.get(q)
                    if j is not None and j > i + 1 and sequence[j] == 'H':
                        contacts += 1
    return -contacts

def _generate_random_saw(length, dimension):
    """Generate a random self-avoiding walk of given length."""
    if length <= 0:
        return []
    chain = [tuple([0]*dimension)]
    visited = {chain[0]}
    
    for _ in range(length - 1):
        last = chain[-1]
        neighbors = []
        for axis in range(dimension):
            for step in (-1, 1):
                nb = list(last)
                nb[axis] += step
                nb = tuple(nb)
                if nb not in visited:
                    neighbors.append(nb)
        if not neighbors:
            return None # Failed to generate
        next_pos = random.choice(neighbors)
        chain.append(next_pos)
        visited.add(next_pos)
        
    return chain

def _find_valid_flips(chain, i):
    """
    Find all valid positions for bead i such that the chain remains a valid SAW.
    The new position must be a neighbor of chain[i-1] and chain[i+1].
    """
    if i <= 0 or i >= len(chain) - 1:
        return []
        
    prev = chain[i-1]
    curr = chain[i]
    next_bead = chain[i+1]
    
    # Find common neighbors of prev and next_bead
    # prev and next_bead are distance 2 apart (via curr) or distance sqrt(2) (if turn)
    # Actually, prev and next_bead are not necessarily adjacent to each other.
    # We need pos p such that |p - prev| = 1 and |p - next_bead| = 1.
    
    candidates = []
    # Generate all neighbors of prev
    for axis in range(len(prev)):
        for step in (-1, 1):
            nb = list(prev)
            nb[axis] += step
            nb = tuple(nb)
            # Check if nb is a neighbor of next_bead
            # Distance squared must be 1
            dist_sq = sum((a-b)**2 for a, b in zip(nb, next_bead))
            if dist_sq == 1:
                # Check if nb is not curr (we want a move, not stay)
                if nb == curr:
                    continue
                # Check if nb is not visited by any other bead
                # We need a fast way to check this. 
                # For simplicity in this snippet, we might pass visited set or check against chain.
                # Since we are in a tight loop, we should maintain a visited set.
                # However, _find_valid_flips doesn't have the visited set.
                # We will modify the caller to pass visited set or we will just check against chain in a helper.
                # For now, return candidates, let caller check visited.
                candidates.append(nb)
                
    return candidates

def _simulated_annealing_local(sequence, dimension, time_limit, visited_set=None):
    """Run a single simulated annealing session using local flips."""
    n = len(sequence)
    if n <= 1:
        return [[0]*dimension for _ in range(n)], 0

    # Initial state
    chain = None
    for _ in range(20):
        chain = _generate_random_saw(n, dimension)
        if chain:
            break
    if not chain:
        chain = [(i, 0, 0) for i in range(n)]
        
    visited_set = set(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    T_start = 2.0
    T_end = 0.001
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Pick a random internal index
        i = random.randint(1, n - 2)
        
        # Find valid flips for i
        prev = chain[i-1]
        next_bead = chain[i+1]
        
        # Find common neighbors of prev and next_bead
        # Optimization: Instead of iterating all 6 neighbors of prev, 
        # we can just check the 6 neighbors of prev and see if they are also neighbors of next_bead.
        
        valid_moves = []
        for axis in range(dimension):
            for step in (-1, 1):
                nb = list(prev)
                nb[axis] += step
                nb = tuple(nb)
                
                # Check if nb is neighbor of next_bead
                dist_sq = sum((a-b)**2 for a, b in zip(nb, next_bead))
                if dist_sq == 1:
                    if nb == chain[i]:
                        continue
                    if nb not in visited_set:
                        valid_moves.append(nb)
                        
        if not valid_moves:
            # No valid local flip. Try a different index or skip.
            # To avoid getting stuck, we could also try an edge pivot here.
            # For now, just skip.
            continue
            
        new_pos = random.choice(valid_moves)
        
        # Perform the move
        old_pos = chain[i]
        chain[i] = new_pos
        visited_set.remove(old_pos)
        visited_set.add(new_pos)
        
        # Calculate energy change
        # Only contacts involving i, i-1, i+1 and their neighbors can change.
        # For simplicity and correctness, recompute full energy? 
        # For n~100, full recompute is O(n^2) or O(n) with hash map. 
        # O(n) is fast enough.
        new_energy = _compute_energy(sequence, chain)
        delta_e = new_energy - current_energy
        
        # Calculate current temperature
        elapsed = time.time() - start_time
        if time_limit > 0:
            cooling_const = math.log(T_start / T_end) / (0.5 * time_limit)
            T = T_start * math.exp(-cooling_const * elapsed)
        else:
            T = T_end
            
        if delta_e < 0 or random.random() < math.exp(-delta_e / T):
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
        else:
            # Reject move
            chain[i] = old_pos
            visited_set.remove(new_pos)
            visited_set.add(old_pos)
            
    return best_chain, best_energy

def fold(sequence, dimension):
    n = len(sequence)
    if n <= 1:
        return [[0]*dimension for _ in range(n)]
        
    # Time budget allocation
    if n < 50:
        time_budget = 4.0
    elif n < 100:
        time_budget = 8.0
    else:
        time_budget = 12.0
        
    start_time = time.time()
    
    best_chain_global = None
    best_energy_global = 0
    
    # Run multiple episodes
    while time.time() - start_time < time_budget * 0.95:
        remaining = time_budget * 0.95 - (time.time() - start_time)
        if remaining <= 0:
            break
            
        episode_time = min(1.0, remaining)
        
        # Run one SA episode with local flips
        chain, energy = _simulated_annealing_local(sequence, dimension, episode_time)
        
        if best_energy_global == 0 or energy < best_energy_global:
            best_energy_global = energy
            best_chain_global = chain
            
    if best_chain_global is None:
        chain = _generate_random_saw(n, dimension)
        if not chain:
            chain = [(i, 0, 0) for i in range(n)]
        best_chain_global = chain
        
    return [list(p) for p in best_chain_global]
