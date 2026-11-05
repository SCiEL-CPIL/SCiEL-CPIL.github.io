"""Rewritten program for HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import random
import time

# Global deadline tracking
_START_TIME = time.time()
_TOTAL_BUDGET = 320.0  # seconds, leaving margin before 330s kill
_DEADLINE = _START_TIME + _TOTAL_BUDGET


def _time_remaining():
    return _DEADLINE - time.time()


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _greedy_growth(sequence, dimension, rng, temperature=0.0):
    """Randomized greedy chain growth with contact-based scoring."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Count new H-H contacts this site would make
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            # Count free neighbours (avoidance of trapping)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Score: prioritize contacts, then free space, then randomness
            score = gain * 10.0 + free * 0.5 + rng.random() * temperature
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick the best option (with some randomness if temperature > 0)
        if temperature > 0 and len(options) > 1:
            # Boltzmann-like selection
            max_score = max(o[0] for o in options)
            weights = [exp((o[0] - max_score) / max(temperature, 0.1)) for o in options]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen = options[0][1]
            for w, o in zip(weights, options):
                cum += w
                if r <= cum:
                    chosen = o[1]
                    break
            site = chosen
        else:
            site = max(options)[1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain


def exp(x):
    """Safe exponential for Boltzmann weights."""
    if x < -50:
        return 0.0
    if x > 50:
        return 1e21
    import math
    return math.exp(x)


def _local_search(sequence, chain, dimension, rng, max_moves=5000):
    """Simple local search: try to improve by moving single residues."""
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    for move_count in range(max_moves):
        if _time_remaining() < 5.0:
            break
        
        # Pick a random internal residue to try moving
        i = rng.randint(1, len(chain) - 2)
        old_pos = chain[i]
        
        # Try each neighbour of old_pos (excluding current position)
        for new_pos in _neighbours(old_pos):
            if new_pos == old_pos:
                continue
            if new_pos in where:
                # Check if it's an adjacent residue in the chain
                j = where[new_pos]
                if abs(j - i) <= 1:
                    continue
                continue
            
            # Tentatively move
            del where[old_pos]
            where[new_pos] = i
            chain[i] = new_pos
            
            new_energy = _energy(sequence, chain)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
            else:
                # Revert
                del where[new_pos]
                where[old_pos] = i
                chain[i] = old_pos
    
    return best_chain, best_energy


def _simulated_annealing(sequence, chain, dimension, rng, start_temp=1.0, end_temp=0.01, max_iter=20000):
    """Simulated annealing to improve a chain conformation."""
    where = {p: i for i, p in enumerate(chain)}
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    n = len(sequence)
    for it in range(max_iter):
        if _time_remaining() < 5.0:
            break
        
        temp = start_temp * (end_temp / start_temp) ** (it / max_iter)
        
        # Pick a random internal residue
        i = rng.randint(1, n - 2)
        old_pos = current_chain[i]
        
        # Try a random neighbour
        neighbours = list(_neighbours(old_pos))
        rng.shuffle(neighbours)
        
        moved = False
        for new_pos in neighbours:
            if new_pos == old_pos:
                continue
            if new_pos in where:
                j = where[new_pos]
                if abs(j - i) <= 1:
                    continue
                continue
            
            # Tentatively move
            del where[old_pos]
            where[new_pos] = i
            current_chain[i] = new_pos
            
            new_energy = _energy(sequence, current_chain)
            delta_e = new_energy - current_energy
            
            if delta_e <= 0 or rng.random() < exp(-delta_e / max(temp, 0.01)):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(current_chain)
                moved = True
                break
            else:
                # Revert
                del where[new_pos]
                where[old_pos] = i
                current_chain[i] = old_pos
        
        if not moved:
            continue
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Find the best conformation using multiple strategies with time budgeting."""
    global _DEADLINE
    _DEADLINE = time.time() + _TOTAL_BUDGET
    
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Multiple greedy growths with varying temperatures
    remaining = _time_remaining()
    if remaining > 10:
        temps = [0.0, 0.1, 0.2, 0.5, 1.0]
        per_temp = max(5, int(remaining / 10 / len(temps)))
        
        for temp in temps:
            for _ in range(per_temp):
                if _time_remaining() < 10:
                    break
                chain = _greedy_growth(sequence, dimension, rng, temperature=temp)
                if chain is None:
                    continue
                energy = _energy(sequence, chain)
                if best_chain is None or energy < best_energy:
                    best_chain = chain
                    best_energy = energy
            if _time_remaining() < 10:
                break
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = 0
    
    # Phase 2: Local search on best chain
    remaining = _time_remaining()
    if remaining > 10:
        max_moves = int(remaining / 50)
        best_chain, best_energy = _local_search(sequence, best_chain, dimension, rng, max_moves)
    
    # Phase 3: Simulated annealing if time remains
    remaining = _time_remaining()
    if remaining > 10:
        max_iter = int(remaining / 50)
        best_chain, best_energy = _simulated_annealing(sequence, best_chain, dimension, rng, max_iter=max_iter)
    
    # Final check: if still no valid chain, use straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
