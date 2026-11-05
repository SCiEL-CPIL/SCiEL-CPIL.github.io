import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute gcd table for all pairs of moduli in [20, 200]
    moduli = list(range(20, 201))
    n_mods = len(moduli)
    mod_to_idx = {m: i for i, m in enumerate(moduli)}
    
    # Precompute gcd for all pairs
    gcd_table = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(i + 1, n_mods):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def is_compatible_fast(sol_indices, sol_residues, m_idx, r_new):
        """Check if (m_idx, r_new) is compatible with all in solution."""
        for i in range(len(sol_indices)):
            g = gcd_table[m_idx][sol_indices[i]]
            if r_new % g == sol_residues[i] % g:
                return False
        return True
    
    def find_compatible_residues(sol_indices, sol_residues, m_idx):
        """Find all compatible residues for m_idx given current solution."""
        m = moduli[m_idx]
        compatible = []
        for r in range(m):
            if is_compatible_fast(sol_indices, sol_residues, m_idx, r):
                compatible.append(r)
        return compatible
    
    def score_of(sol_indices):
        return sum(1.0 / moduli[i] for i in sol_indices)
    
    # Beam search
    # Each beam entry: (score, sol_indices, sol_residues)
    beam_width = 50
    max_beam = 200
    
    # Initialize with greedy solution
    def greedy_initial():
        sol_indices = []
        sol_residues = []
        used_mods = set()
        # Process moduli in order of decreasing 1/m
        for m_idx in range(n_mods - 1, -1, -1):
            m = moduli[m_idx]
            for r in range(m):
                if is_compatible_fast(sol_indices, sol_residues, m_idx, r):
                    sol_indices.append(m_idx)
                    sol_residues.append(r)
                    used_mods.add(m_idx)
                    break
        return sol_indices, sol_residues
    
    init_indices, init_residues = greedy_initial()
    init_score = score_of(init_indices)
    
    beam = [(init_score, init_indices, init_residues)]
    best_score = init_score
    best_solution = (init_indices, init_residues)
    
    # Beam search with local improvements
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        new_beam = []
        
        for score, sol_indices, sol_residues in beam:
            # Generate neighbors
            neighbors = []
            
            # Move 1: Add a new modulus
            used_mods = set(sol_indices)
            for m_idx in range(n_mods):
                if m_idx in used_mods:
                    continue
                residues = find_compatible_residues(sol_indices, sol_residues, m_idx)
                if residues:
                    # Pick a few random compatible residues
                    for r in residues[:3]:  # Limit to first 3 for speed
                        new_indices = sol_indices + [m_idx]
                        new_residues = sol_residues + [r]
                        new_score = score + 1.0 / moduli[m_idx]
                        neighbors.append((new_score, new_indices, new_residues))
            
            # Move 2: Remove a modulus
            for i in range(len(sol_indices)):
                new_indices = sol_indices[:i] + sol_indices[i+1:]
                new_residues = sol_residues[:i] + sol_residues[i+1:]
                new_score = score - 1.0 / moduli[sol_indices[i]]
                neighbors.append((new_score, new_indices, new_residues))
            
            # Move 3: Change residue of an existing modulus
            for i in range(len(sol_indices)):
                m_idx = sol_indices[i]
                m = moduli[m_idx]
                # Try a few random residues
                for _ in range(5):
                    r_new = random.randint(0, m - 1)
                    if r_new == sol_residues[i]:
                        continue
                    # Check compatibility with others
                    compatible = True
                    for j in range(len(sol_indices)):
                        if j == i:
                            continue
                        g = gcd_table[m_idx][sol_indices[j]]
                        if r_new % g == sol_residues[j] % g:
                            compatible = False
                            break
                    if compatible:
                        new_indices = list(sol_indices)
                        new_residues = list(sol_residues)
                        new_residues[i] = r_new
                        new_score = score  # Same score
                        neighbors.append((new_score, new_indices, new_residues))
            
            # Move 4: Swap a modulus
            if len(sol_indices) > 0:
                for i in range(len(sol_indices)):
                    m_idx_old = sol_indices[i]
                    for m_idx_new in range(n_mods):
                        if m_idx_new == m_idx_old or m_idx_new in used_mods:
                            continue
                        # Check if we can find a compatible residue
                        residues = find_compatible_residues(
                            sol_indices[:i] + sol_indices[i+1:],
                            sol_residues[:i] + sol_residues[i+1:],
                            m_idx_new
                        )
                        if residues:
                            r_new = residues[0]
                            new_indices = list(sol_indices)
                            new_residues = list(sol_residues)
                            new_indices[i] = m_idx_new
                            new_residues[i] = r_new
                            new_score = score - 1.0 / moduli[m_idx_old] + 1.0 / moduli[m_idx_new]
                            neighbors.append((new_score, new_indices, new_residues))
            
            # Sort neighbors by score and keep top ones
            if neighbors:
                neighbors.sort(key=lambda x: -x[0])
                # Keep top 5 from this beam entry
                for nb in neighbors[:5]:
                    new_beam.append(nb)
        
        # Prune new beam to beam_width
        if new_beam:
            new_beam.sort(key=lambda x: -x[0])
            # Keep top beam_width, but also ensure diversity
            beam = new_beam[:beam_width]
            
            # Update best
            if beam[0][0] > best_score:
                best_score = beam[0][0]
                best_solution = (beam[0][1], beam[0][2])
        else:
            break
    
    # Convert best solution to output format
    result = []
    for i in range(len(best_solution[0])):
        m = moduli[best_solution[0][i]]
        r = best_solution[1][i]
        result.append([m, r])
    
    return result
