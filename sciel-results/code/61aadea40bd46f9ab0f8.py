"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import time


def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    Uses O(1) energy delta updates for fast simulated annealing.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    if n == 2:
        return [[0] * dimension, [1] + [0] * (dimension - 1)]
    
    is_h = [res == 'H' for res in sequence]
    start_time = time.time()
    time_budget = 11.0  # Leave margin from 12s per sequence
    
    def _grow_initial(rng):
        """Generate an initial self-avoiding walk."""
        chain = [[0, 0]]
        pos_map = {(0, 0): 0}
        occupied = {(0, 0)}
        
        for i in range(1, n):
            last = tuple(chain[-1])
            options = []
            for axis in range(2):
                for step in (-1, 1):
                    q = list(last)
                    q[axis] += step
                    q = tuple(q)
                    if q not in occupied:
                        options.append(q)
            
            if not options:
                return None
            
            site = rng.choice(options)
            chain.append(list(site))
            pos_map[site] = i
            occupied.add(site)
        
        return chain, pos_map
    
    def _compute_energy(chain, pos_map):
        """Compute total energy."""
        contacts = 0
        for i in range(n):
            if not is_h[i]:
                continue
            p = tuple(chain[i])
            for axis in range(2):
                for step in (-1, 1):
                    q = list(p)
                    q[axis] += step
                    q = tuple(q)
                    j = pos_map.get(q)
                    if j is not None and abs(i - j) > 1 and is_h[j]:
                        if i < j:
                            contacts += 1
        return -contacts
    
    def _local_search(chain, pos_map, current_energy, rng, max_iters):
        """Simulated annealing with O(1) energy delta updates."""
        best_chain = [list(p) for p in chain]
        best_energy = current_energy
        
        t_start = 2.0
        t_end = 0.01
        
        for it in range(max_iters):
            alpha = it / max_iters
            temp = t_start * math.exp(-8 * alpha) + t_end
            
            i = rng.randint(1, n - 2)
            
            old_pos = tuple(chain[i])
            prev_pos = tuple(chain[i - 1])
            next_pos = tuple(chain[i + 1])
            
            # Find valid new positions: adjacent to both prev and next, not occupied
            candidates = []
            for axis in range(2):
                for step in (-1, 1):
                    q = list(prev_pos)
                    q[axis] += step
                    q = tuple(q)
                    dist = abs(q[0] - next_pos[0]) + abs(q[1] - next_pos[1])
                    if dist == 1:
                        if q != old_pos and q not in pos_map:
                            candidates.append(q)
            
            if not candidates:
                continue
            
            new_pos = rng.choice(candidates)
            
            # O(1) energy delta computation
            delta = 0
            if is_h[i]:
                # Remove old contacts involving i
                for axis in range(2):
                    for step in (-1, 1):
                        q = list(old_pos)
                        q[axis] += step
                        q = tuple(q)
                        j = pos_map.get(q)
                        if j is not None and abs(i - j) > 1 and is_h[j]:
                            delta += 1  # energy increases (less negative)
                
                # Add new contacts involving i
                for axis in range(2):
                    for step in (-1, 1):
                        q = list(new_pos)
                        q[axis] += step
                        q = tuple(q)
                        j = pos_map.get(q)
                        if j is not None and abs(i - j) > 1 and is_h[j]:
                            delta -= 1  # energy decreases (more negative)
            
            # Accept or reject
            if delta < 0 or rng.random() < math.exp(-delta / temp):
                # Apply move
                del pos_map[old_pos]
                chain[i] = list(new_pos)
                pos_map[new_pos] = i
                current_energy += delta
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = [list(p) for p in chain]
        
        return best_chain, best_energy
    
    # Main: try multiple initial configurations
    best_chain_overall = None
    best_energy_overall = float('inf')
    
    num_trials = 15
    iters_per_trial = 30000
    
    for trial in range(num_trials):
        if time.time() - start_time > time_budget:
            break
        
        rng = random.Random(12345 + len(sequence) + trial * 1000)
        
        result = _grow_initial(rng)
        if result is None:
            continue
        
        chain, pos_map = result
        energy = _compute_energy(chain, pos_map)
        
        improved_chain, improved_energy = _local_search(chain, pos_map, energy, rng, iters_per_trial)
        
        if improved_energy < best_energy_overall:
            best_energy_overall = improved_energy
            best_chain_overall = improved_chain
    
    if best_chain_overall is None:
        # Fallback: straight line
        best_chain_overall = [[i, 0] for i in range(n)]
    
    return best_chain_overall
# EVOLVE-BLOCK-END
