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
    """Compute the HP contact energy: -1 for each H-H non-bonded contact."""
    n = len(chain)
    if n < 3:
        return 0
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i

    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j_pos = (x + dx, y + dy)
            j = pos_to_idx.get(j_pos)
            if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts // 2  # Each contact counted twice


def _greedy_fold(sequence, dimension, rng, temperature=0.05):
    """
    Simulated annealing style chain growth.
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
            if site not in occupied:
                # Calculate immediate gain (new H-H contacts)
                gain = 0
                if sequence[i] == 'H':
                    for nb in _neighbours(site):
                        idx = occupied.get(nb)
                        if idx is not None and idx < i - 1 and sequence[idx] == 'H':
                            gain += 1
                # Calculate free space (heuristic for avoiding dead ends)
                free = sum(1 for nb in _neighbours(site) if nb not in occupied)
                
                # Score: gain is primary, free space secondary, random noise
                score = gain * 10.0 + free * 0.5 + rng.random() * temperature
                candidates.append((score, site))
        
        if not candidates:
            return None
        
        # Pick best candidate (with some randomness if close scores)
        # Sort by score descending
        candidates.sort(key=lambda x: -x[0])
        
        # With some probability, pick a random top-3 to allow exploration
        if len(candidates) > 1 and rng.random() < 0.2:
            k = min(3, len(candidates))
            choice_idx = rng.randint(0, k - 1)
        else:
            choice_idx = 0
            
        best_site = candidates[choice_idx][1]
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain


def _local_search(sequence, chain, rng, max_iterations=5000, time_budget=5.0):
    """
    Try to improve the chain energy via local moves.
    Uses a simple simulated annealing approach with single-residue moves.
    Optimized with position-to-index mapping for O(1) validity checks.
    """
    if not chain or len(chain) < 3:
        return chain
    
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    start_time = time.time()
    
    # Build position-to-index map for O(1) lookup
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
    
    # Simulated annealing parameters
    t_start = 1.0
    t_end = 0.01
    cooling = (t_start - t_end) / max(max_iterations, 1)
    t = t_start
    
    for it in range(max_iterations):
        t = max(t - cooling, t_end)
        
        if time.time() - start_time > time_budget:
            break
        
        # Try to move a single residue (not endpoints)
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        
        # Try each of the 4 neighbours
        for new_pos in _neighbours(old_pos):
            # Check if new position is valid (not occupied by other residues)
            if new_pos in pos_to_idx:
                continue
            
            # Check if the move maintains connectivity
            # The residue at i must still be adjacent to i-1 and i+1
            prev_pos = chain[i-1]
            next_pos = chain[i+1]
            
            # Check adjacency to previous
            dx1 = abs(new_pos[0] - prev_pos[0])
            dy1 = abs(new_pos[1] - prev_pos[1])
            if dx1 + dy1 != 1:
                continue
                
            # Check adjacency to next
            dx2 = abs(new_pos[0] - next_pos[0])
            dy2 = abs(new_pos[1] - next_pos[1])
            if dx2 + dy2 != 1:
                continue
            
            # Compute new energy incrementally
            # Remove old position, add new position
            new_chain = chain[:]
            new_chain[i] = new_pos
            new_energy = _compute_energy(sequence, new_chain)
            
            delta_e = new_energy - current_energy
            
            # Accept if better, or with probability exp(-delta_e / t) if worse
            if delta_e <= 0 or rng.random() < pow(2.718281828, -delta_e / t):
                # Update pos_to_idx
                del pos_to_idx[old_pos]
                pos_to_idx[new_pos] = i
                chain = new_chain
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_chain = chain[:]
                    best_energy = current_energy
                break
    
    return best_chain


def fold(sequence, dimension):
    """
    Find the best conformation for the given HP sequence.
    Returns a list of [x, y] coordinates.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Fallback: straight line
    straight = [[i, 0] for i in range(n)]
    
    best_chain = None
    best_energy = 0  # Energy is negative, so we want the most negative (lowest)
    
    rng = random.Random(42)
    
    # Number of restarts based on time budget
    # We have 180s total. Let's use a fixed number of restarts that is safe.
    # For n=100, greedy is O(n^2) roughly. 100 restarts should be fast.
    num_restarts = 500
    
    for r in range(num_restarts):
        if time.time() > _DEADLINE - 15:
            break
            
        # Vary temperature slightly for diversity
        temp = 0.01 + rng.random() * 0.1
        chain = _greedy_fold(sequence, dimension, rng, temperature=temp)
        
        if chain is None:
            continue
        
        # Apply local search to improve the chain
        chain = _local_search(sequence, chain, rng, max_iterations=1000, time_budget=2.0)
        
        energy = _compute_energy(sequence, chain)
        
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
            
            # If we hit a very good energy, we might stop early
            # But we don't know the best possible, so keep going.
            
    if best_chain is None:
        return straight
    
    # Final high-precision polish pass: run more iterations with tighter tolerance
    if time.time() < _DEADLINE - 5:
        best_chain = _local_search(sequence, best_chain, rng, max_iterations=2000, time_budget=5.0)
    
    # Convert tuples to lists
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
