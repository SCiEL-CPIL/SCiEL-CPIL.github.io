# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE_START = time.time()
_DEADLINE_SECONDS = 300.0  # Safety margin before 330s hard kill

def _remaining_time():
    return _DEADLINE_START + _DEADLINE_SECONDS - time.time()

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _manhattan(p, q):
    return sum(abs(a - b) for a, b in zip(p, q))

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

def _fast_grow(sequence, dimension, rng, aggressive=False):
    """
    Greedy growth with a bias towards H-H contacts.
    If aggressive, it tries to fill space more tightly.
    """
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            # Calculate potential contacts
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            
            # Count free neighbors to avoid dead ends
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # Scoring: prioritize contacts, then free space
            # Add small noise to break ties
            score = gain * 10.0 + free * 0.5 + rng.random() * 0.1
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick best option
        # If aggressive, maybe pick from top 3? No, greedy is faster.
        best_site = max(options, key=lambda x: x[0])[1]
        occupied[best_site] = i
        chain.append(best_site)
    
    return chain

def _simulated_annealing(sequence, chain, dimension, rng, time_limit):
    """
    Improve chain using Simulated Annealing with single-residue moves.
    """
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)

    current_chain = list(chain)
    current_energy = _energy(sequence, chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    # Initial temperature
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.999
    
    start_time = time.time()
    
    while _remaining_time() > 5.0:
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random internal residue
        i = rng.randint(1, n - 2)
        old_pos = current_chain[i]
        
        # Try moving to a random neighbor of old_pos
        # We only consider moves that keep the chain connected
        # Neighbors of old_pos that are valid new positions for i
        # must be distance 1 from i-1 and i+1
        
        prev_pos = current_chain[i-1]
        next_pos = current_chain[i+1]
        
        # Valid new positions are neighbors of prev_pos AND neighbors of next_pos
        # Actually, it's easier to just iterate neighbors of old_pos and check connectivity
        # But that's 6 options.
        # A better neighborhood: Pick a random neighbor of prev_pos that is also 
        # a neighbor of next_pos? No, that's too restrictive.
        # Standard move: Move residue i to a random empty neighbor of its current position?
        # No, that breaks connectivity usually.
        
        # Correct move: Move residue i to a position p such that
        # dist(p, i-1) == 1 and dist(p, i+1) == 1.
        # These positions are the "kinks".
        # The neighbors of i-1 are candidates. Check if they are neighbors of i+1.
        
        candidates = []
        for p in _neighbours(prev_pos):
            if p == old_pos:
                continue
            if p in occupied:
                continue
            if _manhattan(p, next_pos) == 1:
                candidates.append(p)
                
        if not candidates:
            continue
            
        new_pos = rng.choice(candidates)
        
        # Calculate Delta Energy
        delta = 0
        
        # Remove contacts involving i at old_pos
        if sequence[i] == 'H':
            for q in _neighbours(old_pos):
                j = occupied.get(q)
                if j is not None and j != i-1 and j != i+1 and sequence[j] == 'H':
                    delta += 1 # Lose contact -> Energy goes up (less negative)
                    
        # Add contacts involving i at new_pos
        if sequence[i] == 'H':
            for q in _neighbours(new_pos):
                j = occupied.get(q)
                if j is not None and j != i-1 and j != i+1 and sequence[j] == 'H':
                    delta -= 1 # Gain contact -> Energy goes down (more negative)
        
        new_energy = current_energy + delta
        
        # Acceptance Probability
        if new_energy < current_energy:
            accept = True
        else:
            if T > T_min:
                # delta is positive (energy increase)
                # P = exp(-delta / T)
                # If delta is large and T is small, exp(-large) is 0
                try:
                    prob = math.exp(-delta / T)
                except OverflowError:
                    prob = 0.0
                accept = rng.random() < prob
            else:
                accept = False
        
        if accept:
            # Update state
            occupied.pop(old_pos)
            occupied[new_pos] = i
            current_chain[i] = new_pos
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
        # Cool down
        T *= cooling_rate
        
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Best conformation using fast greedy seeds + Simulated Annealing.
    """
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    
    # 1. Generate a few diverse seeds using fast greedy
    # We want seeds that are different, so we use different random seeds for the noise
    seeds = []
    num_seeds = 5
    for k in range(num_seeds):
        seed_rng = random.Random(1000 + k * 137)
        chain = _fast_grow(sequence, dimension, seed_rng)
        if chain:
            seeds.append(chain)
            
    if not seeds:
        # Fallback: straight line
        return [list(p) for p in [tuple([i] + [0] * (dimension - 1)) for i in range(n)]]
    
    # Evaluate seeds
    best_seed = None
    best_seed_energy = 0
    for chain in seeds:
        e = _energy(sequence, chain)
        if e < best_seed_energy:
            best_seed_energy = e
            best_seed = chain
            
    # 2. Allocate time to Simulated Annealing
    # We have ~300s total. 
    # Greedy is very fast. 
    # We will run SA on the best seed for the majority of the time.
    # We might also run SA on the 2nd best seed if time permits.
    
    # Let's just run SA on the best seed for a long time.
    # If we have multiple good seeds, we could run SA on them in parallel? No, single thread.
    # We will run SA on the best seed for 80% of the time, and if there's time, 
    # try to improve the 2nd best seed briefly.
    
    # Actually, a single long SA run is usually better than many short ones for this problem size.
    # Let's give SA 280 seconds.
    
    # Sort seeds by energy
    seeds_with_energy = []
    for chain in seeds:
        e = _energy(sequence, chain)
        seeds_with_energy.append((e, chain))
    seeds_with_energy.sort()
    
    best_chain, best_energy = seeds_with_energy[0][1], seeds_with_energy[0][0]
    
    # Run SA on the best seed
    if _remaining_time() > 10.0:
        best_chain, best_energy = _simulated_annealing(
            sequence, best_chain, dimension, rng, time_limit=250.0
        )
        
    # If we have time left, try SA on the second best seed?
    # It might find a different local minimum.
    if len(seeds_with_energy) > 1 and _remaining_time() > 30.0:
        second_chain, second_energy = seeds_with_energy[1]
        # Only try if it's significantly different or if we have lots of time
        # Just run a shorter SA
        second_chain, second_energy = _simulated_annealing(
            sequence, second_chain, dimension, rng, time_limit=20.0
        )
        if second_energy < best_energy:
            best_chain, best_energy = second_chain, second_energy
            
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
