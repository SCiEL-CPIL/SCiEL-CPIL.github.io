"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = 320.0
_start_time = None


def _get_remaining_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _DEADLINE - (time.time() - _start_time)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy: -number of H-H contacts on non-adjacent lattice neighbours."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng, temp=0.0):
    """One chain growth with contact-based scoring."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.5 + rng.random() * (1.0 + temp)
            options.append((score, site))
        
        if not options:
            return None
        
        if temp > 0 and rng.random() < 0.3:
            options.sort(key=lambda x: -x[0])
            k = min(len(options), 1 + int(rng.expovariate(1.0 / max(1, temp))))
            site = options[rng.randint(0, k - 1)][1]
        else:
            site = max(options)[1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain


def _pivot_move(chain, pivot_idx, rng, dimension):
    """
    Perform a pivot move: rotate the tail segment (pivot_idx+1 to end)
    around the pivot point. Returns new chain or None if invalid.
    """
    n = len(chain)
    if pivot_idx < 0 or pivot_idx >= n - 1:
        return None
    
    # The tail is chain[pivot_idx+1:]
    # We rotate it around chain[pivot_idx]
    # Generate all possible rotations of the tail
    
    tail = chain[pivot_idx + 1:]
    pivot_point = chain[pivot_idx]
    
    # Try all 6 possible directions for the first bond of the rotated tail
    # The first bond direction determines the rotation
    best_new_tail = None
    
    for axis in range(dimension):
        for step in (-1, 1):
            # New first position relative to pivot
            new_first = list(pivot_point)
            new_first[axis] += step
            
            # Check if new_first is not occupied by the head part
            head_set = set(chain[:pivot_idx + 1])
            if tuple(new_first) in head_set:
                continue
            
            # Build the rotated tail
            # The tail bonds are: tail[i+1] - tail[i] for i in range(len(tail)-1)
            # We need to rotate these bonds so the first bond goes in the new direction
            # This is complex; instead, let's use a simpler approach:
            # Just try moving the entire tail by a translation that makes
            # the first bond go in the desired direction.
            
            # Actually, for a proper pivot move, we need to rotate the coordinate frame.
            # Let's use a different approach: try all 24 rotations of the tail.
            # For 3D, there are 24 proper rotations of the cube.
            
            # Simpler: try translating the tail so that tail[0] maps to new_first
            # This is a translation, not a rotation, but it's a valid move
            # that preserves the relative geometry of the tail.
            
            dx = new_first[0] - tail[0][0]
            dy = new_first[1] - tail[0][1]
            dz = new_first[2] - tail[0][2]
            
            rotated_tail = []
            valid = True
            for p in tail:
                new_p = (p[0] + dx, p[1] + dy, p[2] + dz)
                rotated_tail.append(new_p)
            
            # Check self-avoidance of the rotated tail
            if len(set(rotated_tail)) != len(rotated_tail):
                continue
            
            # Check no overlap with head
            for p in rotated_tail:
                if p in head_set:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check unit steps within rotated tail
            for i in range(len(rotated_tail) - 1):
                diff = sum(abs(rotated_tail[i][d] - rotated_tail[i+1][d]) for d in range(dimension))
                if diff != 1:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check connection: chain[pivot_idx] to rotated_tail[0]
            diff = sum(abs(pivot_point[d] - rotated_tail[0][d]) for d in range(dimension))
            if diff != 1:
                continue
            
            # Valid move!
            new_chain = chain[:pivot_idx + 1] + rotated_tail
            return new_chain
    
    return None


def _monte_carlo(sequence, chain, dimension, rng, time_budget=10.0):
    """
    Monte Carlo with pivot moves and simulated annealing.
    """
    n = len(chain)
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    start_temp = 2.0
    end_temp = 0.01
    total_steps = 5000
    start_time = time.time()
    
    for step in range(total_steps):
        elapsed = time.time() - start_time
        if elapsed > time_budget or _get_remaining_time() < 2.0:
            break
        
        # Annealing schedule
        progress = step / total_steps
        temp = start_temp * math.exp(-progress * 5.0)
        
        # Choose pivot point (bias towards middle for larger moves)
        pivot_idx = rng.randint(0, n - 2)
        
        new_chain = _pivot_move(current_chain, pivot_idx, rng, dimension)
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Metropolis acceptance
        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-6)):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_chain = list(current_chain)
                best_energy = current_energy
    
    return best_chain, best_energy


def _local_search(sequence, chain, dimension, rng, max_moves=2000):
    """
    Local search with single-residue moves for fine-tuning.
    """
    best_chain = list(chain)
    n = len(best_chain)
    best_energy = _energy(sequence, best_chain)
    
    for _ in range(max_moves):
        if _get_remaining_time() < 1.0:
            break
        
        improved = False
        for i in range(1, n - 1):
            current = best_chain[i]
            prev = best_chain[i-1]
            next_ = best_chain[i+1]
            
            for site in _neighbours(current):
                if site in best_chain:
                    continue
                
                d1 = sum(abs(site[d] - prev[d]) for d in range(dimension))
                d2 = sum(abs(site[d] - next_[d]) for d in range(dimension))
                
                if d1 != 1 or d2 != 1:
                    continue
                
                new_chain = list(best_chain)
                new_chain[i] = site
                new_energy = _energy(sequence, new_chain)
                
                if new_energy < best_energy:
                    best_chain = new_chain
                    best_energy = new_energy
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Best conformation using:
    1. Multiple greedy growths for initial solutions
    2. Monte Carlo with pivot moves and simulated annealing
    3. Local search for fine-tuning
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Greedy growths to get good initial solutions
    if n <= 48:
        n_restarts = 100
    elif n <= 80:
        n_restarts = 60
    else:
        n_restarts = 30
    
    candidates = []
    for restart in range(n_restarts):
        if _get_remaining_time() < 10.0:
            break
        
        temp = 0.0 if restart < n_restarts // 2 else rng.uniform(0.1, 0.5)
        chain = _greedy_grow(sequence, dimension, rng, temp)
        if chain is None:
            continue
        
        value = _energy(sequence, chain)
        if best_chain is None or value < best_energy:
            best_chain = chain
            best_energy = value
        
        candidates.append((value, chain))
    
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = 0
    
    # Phase 2: Monte Carlo with pivot moves on top candidates
    if _get_remaining_time() > 20.0:
        # Sort candidates by energy, take top few
        candidates.sort(key=lambda x: x[0])
        top_k = min(3, len(candidates))
        
        for k in range(top_k):
            if _get_remaining_time() < 5.0:
                break
            
            chain = candidates[k][1]
            time_budget = min(15.0, _get_remaining_time() - 3.0)
            
            improved_chain, improved_energy = _monte_carlo(
                sequence, chain, dimension, rng, time_budget
            )
            
            if improved_energy < best_energy:
                best_chain = improved_chain
                best_energy = improved_energy
    
    # Phase 3: Local search for fine-tuning
    if best_chain is not None and _get_remaining_time() > 5.0:
        best_chain, best_energy = _local_search(
            sequence, best_chain, dimension, rng, max_moves=1000
        )
    
    # Fallback
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
