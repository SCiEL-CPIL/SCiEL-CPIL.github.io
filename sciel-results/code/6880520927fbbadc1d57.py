"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math

def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _compute_energy(sequence, chain):
    """Compute the hydrophobic contact energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _is_valid_chain(chain, sequence):
    """Check if chain is self-avoiding and has correct length."""
    if len(chain) != len(sequence):
        return False
    if len(set(chain)) != len(chain):
        return False
    # Check unit steps
    for i in range(len(chain) - 1):
        x1, y1 = chain[i]
        x2, y2 = chain[i + 1]
        if abs(x1 - x2) + abs(y1 - y2) != 1:
            return False
    return True

def _hill_climb(sequence, chain, max_steps=5000):
    """Perform local search to improve energy via single-residue moves."""
    if chain is None:
        return None
    
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    for step in range(max_steps):
        # Pick a random residue to move (not the first or last)
        i = random.randint(1, len(chain) - 2)
        old_pos = chain[i]
        
        # Try all 4 neighbours of old_pos
        for new_pos in _neighbours(old_pos):
            # Check if new_pos is occupied by a non-adjacent residue
            if new_pos in where:
                j = where[new_pos]
                if abs(i - j) > 1:
                    continue
            
            # Tentatively move residue i to new_pos
            old_pos_copy = old_pos
            chain[i] = new_pos
            where[new_pos] = i
            del where[old_pos]
            
            # Check self-avoidance for the moved residue
            valid = True
            for neighbor in _neighbours(new_pos):
                if neighbor in where:
                    j = where[neighbor]
                    if abs(i - j) == 1:
                        # Adjacent residues can be neighbours
                        pass
                    else:
                        # Non-adjacent residues cannot be neighbours
                        valid = False
                        break
            
            if not valid:
                # Revert
                chain[i] = old_pos_copy
                where[old_pos_copy] = i
                del where[new_pos]
                continue
            
            # Compute new energy
            new_energy = _compute_energy(sequence, chain)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
            else:
                # Revert if not improving (or accept with small probability for simulated annealing effect)
                # For simplicity, revert if not strictly better
                chain[i] = old_pos_copy
                where[old_pos_copy] = i
                del where[new_pos]
    
    return best_chain

def _random_walk(sequence, dimension):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n == 0:
        return []
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        neighbours = list(_neighbours(last))
        random.shuffle(neighbours)
        
        placed = False
        for pos in neighbours:
            if pos not in occupied:
                chain.append(pos)
                occupied[pos] = i
                placed = True
                break
        
        if not placed:
            return None
    
    return chain

def _greedy_grow(sequence, dimension):
    """Grow chain greedily preferring positions with more H-H contacts."""
    n = len(sequence)
    if n == 0:
        return []
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        neighbours = list(_neighbours(last))
        
        best_score = -1
        best_pos = None
        
        for pos in neighbours:
            if pos in occupied:
                continue
            
            # Score: number of H-H contacts this would create
            score = 0
            if sequence[i] == "H":
                for neighbor in _neighbours(pos):
                    if neighbor in occupied:
                        j = occupied[neighbor]
                        if j < i - 1 and sequence[j] == "H":
                            score += 1
            
            # Add small random tiebreaker
            score += random.random() * 0.01
            
            if score > best_score:
                best_score = score
                best_pos = pos
        
        if best_pos is None:
            return None
        
        chain.append(best_pos)
        occupied[best_pos] = i
    
    return chain

def fold(sequence, dimension):
    """Find a low-energy conformation for the given HP sequence."""
    n = len(sequence)
    
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    best_chain = None
    best_energy = float('inf')
    
    # Try multiple strategies
    num_trials = 50
    
    for trial in range(num_trials):
        # Mix of random and greedy walks
        if trial % 2 == 0:
            chain = _greedy_grow(sequence, dimension)
        else:
            chain = _random_walk(sequence, dimension)
        
        if chain is None:
            continue
        
        # Improve with hill climbing
        improved = _hill_climb(sequence, chain, max_steps=2000)
        
        if improved is not None:
            energy = _compute_energy(sequence, improved)
            if energy < best_energy:
                best_energy = energy
                best_chain = improved
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
