"""Rewritten program: HP lattice protein folding via simulated annealing on square lattice."""
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
    """Compute the energy (negative number of H-H contacts) of a conformation."""
    n = len(chain)
    # Map position to index
    pos_to_idx = {chain[i]: i for i in range(n)}
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
            j = pos_to_idx.get((nx, ny))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _valid_move(chain, i, new_pos, occupied_set):
    """Check if moving residue i to new_pos is valid (self-avoiding, unit step to neighbors)."""
    if new_pos in occupied_set:
        return False
    # Check unit step with previous
    if i > 0:
        px, py = chain[i-1]
        if abs(new_pos[0] - px) + abs(new_pos[1] - py) != 1:
            return False
    # Check unit step with next
    if i < len(chain) - 1:
        nx, ny = chain[i+1]
        if abs(new_pos[0] - nx) + abs(new_pos[1] - ny) != 1:
            return False
    return True


def _init_chain(sequence):
    """Initialize a straight-line chain."""
    n = len(sequence)
    return [(i, 0) for i in range(n)]


def _simulate_annealing(sequence, rng, iterations=5000, t_start=1.0, t_end=0.01):
    """Run simulated annealing to minimize energy."""
    n = len(sequence)
    chain = _init_chain(sequence)
    occupied_set = set(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    for it in range(iterations):
        # Linear cooling
        t = t_start * (t_end / t_start) ** (it / max(1, iterations - 1))
        
        # Pick a random residue to move
        i = rng.randint(1, n - 2)  # Don't move endpoints to keep it simple
        
        # Try each of the 4 possible new positions
        x, y = chain[i]
        candidates = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        rng.shuffle(candidates)
        
        moved = False
        for new_pos in candidates:
            if new_pos in occupied_set:
                continue
            
            # Check unit step constraints
            valid = True
            if i > 0:
                px, py = chain[i-1]
                if abs(new_pos[0] - px) + abs(new_pos[1] - py) != 1:
                    valid = False
            if valid and i < n - 1:
                nx, ny = chain[i+1]
                if abs(new_pos[0] - nx) + abs(new_pos[1] - ny) != 1:
                    valid = False
            
            if not valid:
                continue
            
            # Perform the move
            old_pos = chain[i]
            occupied_set.remove(old_pos)
            occupied_set.add(new_pos)
            chain[i] = new_pos
            
            new_energy = _compute_energy(sequence, chain)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / max(t, 1e-10)):
                # Accept the move
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = chain[:]
                moved = True
                break
            else:
                # Reject, revert
                chain[i] = old_pos
                occupied_set.remove(new_pos)
                occupied_set.add(old_pos)
        
        if not moved:
            # No valid move found, try a different residue or skip
            pass
    
    return best_chain, best_energy


def _greedy_init(sequence):
    """Greedy chain growth preferring H-H contacts."""
    n = len(sequence)
    if n == 0:
        return []
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        last_x, last_y = chain[-1]
        candidates = [(last_x+1, last_y), (last_x-1, last_y), (last_x, last_y+1), (last_x, last_y-1)]
        
        best_score = -1
        best_pos = None
        for pos in candidates:
            if pos in occupied:
                continue
            # Score: prefer H-H contacts
            score = 0
            if sequence[i] == 'H':
                px, py = pos
                for nx, ny in [(px+1, py), (px-1, py), (px, py+1), (px, py-1)]:
                    if (nx, ny) in occupied:
                        j = occupied[(nx, ny)]
                        if j < i - 1 and sequence[j] == 'H':
                            score += 10
            # Prefer positions with more free space
            free = 0
            px, py = pos
            for nx, ny in [(px+1, py), (px-1, py), (px, py+1), (px, py-1)]:
                if (nx, ny) not in occupied:
                    free += 1
            score += free * 0.1
            if score > best_score:
                best_score = score
                best_pos = pos
        
        if best_pos is None:
            # Failed, return what we have
            break
        
        chain.append(best_pos)
        occupied[best_pos] = i
    
    return chain


def fold(sequence, dimension):
    """Find a low-energy self-avoiding conformation for the given HP sequence."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    rng = random.Random(42 + len(sequence))
    
    # Start with a greedy initialization
    chain = _greedy_init(sequence)
    if len(chain) < n:
        # Fallback to straight line
        chain = _init_chain(sequence)
    
    best_chain = chain[:]
    best_energy = _compute_energy(sequence, chain)
    
    # Run multiple simulated annealing attempts
    num_runs = 5
    iters_per_run = 8000
    
    for run in range(num_runs):
        # Perturb the initial chain
        if run == 0:
            current_chain = chain[:]
        else:
            # Start from a randomized greedy or perturbed version
            current_chain = _init_chain(sequence)
        
        sa_chain, sa_energy = _simulate_annealing(
            sequence, rng, 
            iterations=iters_per_run,
            t_start=1.0,
            t_end=0.005
        )
        
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_chain = sa_chain[:]
    
    # Final local search on best chain
    occupied_set = set(best_chain)
    for _ in range(2000):
        i = rng.randint(1, n - 2)
        x, y = best_chain[i]
        candidates = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        rng.shuffle(candidates)
        
        improved = False
        for new_pos in candidates:
            if new_pos in occupied_set:
                continue
            valid = True
            if i > 0:
                px, py = best_chain[i-1]
                if abs(new_pos[0] - px) + abs(new_pos[1] - py) != 1:
                    valid = False
            if valid and i < n - 1:
                nx, ny = best_chain[i+1]
                if abs(new_pos[0] - nx) + abs(new_pos[1] - ny) != 1:
                    valid = False
            if not valid:
                continue
            
            old_pos = best_chain[i]
            occupied_set.remove(old_pos)
            occupied_set.add(new_pos)
            best_chain[i] = new_pos
            new_energy = _compute_energy(sequence, best_chain)
            
            if new_energy < best_energy:
                best_energy = new_energy
                improved = True
                break
            else:
                best_chain[i] = old_pos
                occupied_set.remove(new_pos)
                occupied_set.add(old_pos)
        
        if not improved:
            continue
    
    return [list(p) for p in best_chain]
