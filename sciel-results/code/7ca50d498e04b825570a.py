"""
HP Lattice Protein Folding on Cubic Lattice
Uses Monte Carlo Chain Growth (MCCG) with backtracking and contact-based energy bias.
"""
import math
import random
import time

# EVOLVE-BLOCK-START

def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D cubic lattice."""
    x, y, z = point
    return [(x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z), (x, y, z + 1), (x, y, z - 1)]

def _calculate_energy_delta(sequence, index, pos, occupied):
    """
    Calculate the change in energy if residue at 'index' is placed at 'pos'.
    Energy = - (number of H-H contacts).
    A contact is formed if:
    1. Current residue is H.
    2. Neighbor residue is H.
    3. Neighbor index is not adjacent (|i - j| > 1).
    
    Returns delta_E (negative for favorable contacts).
    """
    if sequence[index] != 'H':
        return 0
    
    delta = 0
    for n_pos in _neighbours(pos):
        if n_pos in occupied:
            n_idx = occupied[n_pos]
            # Check if not adjacent in sequence
            if abs(index - n_idx) > 1:
                if sequence[n_idx] == 'H':
                    delta -= 1  # Energy decreases (becomes more negative)
    return delta

def _mccg_fold(sequence, dimension, time_limit, seed):
    """
    Monte Carlo Chain Growth folding algorithm with backtracking.
    """
    rng = random.Random(seed)
    n = len(sequence)
    start_time = time.time()
    
    best_chain = None
    best_energy = 0  # 0 is unfolded, we want most negative
    
    # Temperature parameters for Boltzmann selection
    # We will anneal temperature from high to low to balance exploration/exploitation
    T_start = 2.0
    T_end = 0.5
    
    # We will run multiple rounds of MCCG with different temperature schedules
    # Each round uses backtracking to explore the search space more thoroughly
    
    def _mccg_with_backtracking(T):
        """
        Run MCCG with backtracking at temperature T.
        Returns the best chain and energy found.
        """
        start_pos = (0, 0, 0)
        chain = [start_pos]
        occupied = {start_pos: 0}
        energy = 0
        idx = 0
        
        # Stack for backtracking: stores (pos_added, idx_added, energy_before, available_neighbors)
        # available_neighbors is the list of neighbors that were available at that step
        # We need to track which ones we've tried
        stack = []
        
        local_best_chain = None
        local_best_energy = 0
        
        # We'll use a depth-first approach with backtracking
        # When we get stuck or complete, we backtrack to the last branching point
        
        # To avoid infinite loops, we limit the number of backtracks
        max_backtracks = 10000
        
        backtrack_count = 0
        
        while True:
            if time.time() - start_time > time_limit:
                break
            
            if idx == n - 1:
                # Complete chain
                if energy < local_best_energy:
                    local_best_energy = energy
                    local_best_chain = chain[:]
                
                # Backtrack to explore other paths
                if not stack:
                    break  # We've exhausted the search from the root
                
                # Pop the last decision
                pos_added, idx_added, e_before, available = stack.pop()
                chain.pop()
                occupied.pop(pos_added)
                energy = e_before
                idx = idx_added - 1
                
                # We need to try the next available neighbor
                # But we need to know which ones we've already tried
                # For simplicity, we'll just restart from this point with a different choice
                # This is a simplified backtracking that may revisit some states
                # But it's much faster than full state tracking
                backtrack_count += 1
                if backtrack_count > max_backtracks:
                    break
                continue
            
            # Try to grow
            last = chain[-1]
            neighbors = _neighbours(last)
            available = [p for p in neighbors if p not in occupied]
            
            if not available:
                # Stuck, backtrack
                if not stack:
                    break  # No more options from root
                pos_added, idx_added, e_before, _ = stack.pop()
                chain.pop()
                occupied.pop(pos_added)
                energy = e_before
                idx = idx_added - 1
                backtrack_count += 1
                if backtrack_count > max_backtracks:
                    break
                continue
            
            # Calculate scores for each available position
            scores = []
            for p in available:
                dE = _calculate_energy_delta(sequence, idx + 1, p, occupied)
                # Boltzmann weight: favor lower energy (more negative dE)
                # P ~ exp(-dE / T)
                # If dE = -1, T=1, weight = exp(1) ≈ 2.7
                # If dE = 0, weight = 1
                weight = math.exp(-dE / T) if dE < 0 else 1.0
                scores.append((weight, p))
            
            # Normalize and pick
            total_w = sum(w for w, _ in scores)
            r = rng.random() * total_w
            cumulative = 0
            chosen = scores[0][1]
            chosen_idx = 0
            for i, (w, p) in enumerate(scores):
                cumulative += w
                if r <= cumulative:
                    chosen = p
                    chosen_idx = i
                    break
            
            # Save state for backtracking
            # We save the position we're about to add and the available neighbors
            stack.append((chosen, idx + 1, energy, [s[1] for s in scores if s[1] != chosen]))
            
            # Add the chosen position
            chain.append(chosen)
            occupied[chosen] = idx + 1
            energy += _calculate_energy_delta(sequence, idx + 1, chosen, {k: v for k, v in occupied.items() if k != chosen})
            idx += 1
        
        return local_best_chain, local_best_energy
    
    # Run multiple rounds with different temperature schedules
    num_rounds = 5
    for round_num in range(num_rounds):
        if time.time() - start_time > time_limit:
            break
        
        # Anneal temperature from T_start to T_end
        T = T_start + (T_end - T_start) * (round_num / max(1, num_rounds - 1))
        
        chain, energy = _mccg_with_backtracking(T)
        
        if chain is not None and energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    # If no good chain found, do a simple biased growth without backtracking
    if best_chain is None:
        def _grow_biased_simple(T=1.0):
            n = len(sequence)
            start = (0, 0, 0)
            chain = [start]
            occupied = {start: 0}
            
            for i in range(1, n):
                last = chain[-1]
                neighbors = _neighbours(last)
                available = [p for p in neighbors if p not in occupied]
                
                if not available:
                    return None
                
                scores = []
                for p in available:
                    dE = _calculate_energy_delta(sequence, i, p, occupied)
                    weight = math.exp(-dE / T) if dE < 0 else 1.0
                    scores.append((weight, p))
                
                total_w = sum(w for w, _ in scores)
                r = rng.random() * total_w
                cumulative = 0
                chosen = scores[0][1]
                for w, p in scores:
                    cumulative += w
                    if r <= cumulative:
                        chosen = p
                        break
                
                chain.append(chosen)
                occupied[chosen] = i
            
            return chain
        
        for _ in range(100):
            if time.time() - start_time > time_limit:
                break
            chain = _grow_biased_simple(1.0)
            if chain is not None:
                e = _calculate_energy(sequence, chain)
                if e < best_energy:
                    best_energy = e
                    best_chain = chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return best_chain, best_energy

def _calculate_energy(sequence, chain):
    """Calculate total energy of a chain."""
    n = len(chain)
    if n == 0:
        return 0
    occ = {p: i for i, p in enumerate(chain)}
    energy = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for n_pos in _neighbours(p):
                j = occ.get(n_pos)
                if j is not None and abs(i - j) > 1:
                    if sequence[j] == 'H':
                        energy -= 1
    return energy

def fold(sequence, dimension):
    """
    Fold the sequence using MCCG with backtracking and temperature annealing.
    """
    n = len(sequence)
    
    # Dynamic time limit based on sequence length
    # Longer sequences need more time
    time_limit = 2.0 + 0.05 * n
    if time_limit > 10.0:
        time_limit = 10.0
    if time_limit < 1.0:
        time_limit = 1.0
    
    # Use a seed based on hash of sequence for reproducibility
    seed = hash(sequence) % (2**32)
    
    chain, energy = _mccg_fold(sequence, dimension, time_limit, seed)
    
    # Ensure format: list of lists
    return [list(p) for p in chain]

# EVOLVE-BLOCK-END
