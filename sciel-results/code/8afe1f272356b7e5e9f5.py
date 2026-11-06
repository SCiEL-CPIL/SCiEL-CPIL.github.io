"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline: 320s safety margin before 330s kill
_DEADLINE = time.time() + 320


def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute HP energy: -number of H-H contacts (non-consecutive neighbours)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow_greedy(sequence, dimension, rng, temperature=0.5):
    """One randomized greedy chain growth: prefer sites with new H-H contacts."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain + 0.1 * free + rng.random() * temperature
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move(chain, dimension, rng):
    """Attempt a pivot move: rotate a subchain around a pivot point."""
    n = len(chain)
    if n < 4:
        return None
    
    # Choose pivot index (not too close to ends)
    pivot = rng.randint(1, n - 3)
    
    # Get the subchain to rotate
    subchain = chain[pivot:]
    
    # Try all 24 rotations in 3D (or just 3 axes * 4 rotations each)
    # For simplicity, try rotating around each axis by 90 degrees
    # and also try reflection
    for axis in range(3):
        for sign in (-1, 1):
            # Rotation by 90 degrees around axis
            new_subchain = []
            for i, p in enumerate(subchain):
                q = list(p)
                # Rotate around pivot
                rel = [p[d] - chain[pivot][d] for d in range(3)]
                if axis == 0:
                    rel_new = [rel[0], -rel[2] * sign, rel[1] * sign]
                elif axis == 1:
                    rel_new = [rel[2] * sign, rel[1], -rel[0] * sign]
                else:
                    rel_new = [-rel[1] * sign, rel[0] * sign, rel[2]]
                q = [chain[pivot][d] + rel_new[d] for d in range(3)]
                new_subchain.append(tuple(q))
            
            # Check self-avoidance
            new_chain = chain[:pivot] + new_subchain
            occupied = set()
            valid = True
            for i, p in enumerate(new_chain):
                if p in occupied:
                    valid = False
                    break
                occupied.add(p)
            
            if valid:
                return new_chain
    
    return None


def _local_move(chain, dimension, rng):
    """Attempt a local move: move a single residue to a nearby free site."""
    n = len(chain)
    if n < 4:
        return None
    
    # Choose a random interior residue
    idx = rng.randint(1, n - 2)
    p = chain[idx]
    
    # Try moving to a neighbouring site
    for site in _neighbours(p):
        if site in chain:
            continue
        
        # Check if the move is valid (no clashes)
        new_chain = chain[:]
        new_chain[idx] = site
        
        # Verify self-avoidance
        occupied = set()
        valid = True
        for i, q in enumerate(new_chain):
            if q in occupied:
                valid = False
                break
            occupied.add(q)
        
        if valid:
            return new_chain
    
    return None


def _mc_fold(sequence, dimension, rng, max_steps=5000):
    """Monte Carlo folding with pivot and local moves."""
    # Start with a random self-avoiding walk
    chain = None
    for _ in range(50):
        chain = _grow_greedy(sequence, dimension, rng, temperature=1.0)
        if chain is not None:
            break
    
    if chain is None:
        # Fallback: straight line
        chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    temperature = 1.0
    cooling = 0.999
    
    for step in range(max_steps):
        if time.time() > _DEADLINE - 10:
            break
        
        # Choose move type
        if rng.random() < 0.7:
            new_chain = _pivot_move(chain, dimension, rng)
        else:
            new_chain = _local_move(chain, dimension, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Metropolis acceptance
        if delta_e < 0 or rng.random() < math.exp(-delta_e / temperature):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_chain = chain[:]
                best_energy = current_energy
        
        temperature *= cooling
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using hybrid greedy + Monte Carlo search."""
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 1
    
    # Phase 1: Multiple greedy restarts with different temperatures
    for temp in [0.1, 0.3, 0.5, 0.8, 1.0]:
        if time.time() > _DEADLINE - 60:
            break
        for _ in range(30):
            if time.time() > _DEADLINE - 60:
                break
            chain = _grow_greedy(sequence, dimension, rng, temperature=temp)
            if chain is None:
                continue
            value = _energy(sequence, chain)
            if value < best_energy:
                best_chain, best_energy = chain, value
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Monte Carlo refinement from best greedy solution
    if time.time() < _DEADLINE - 20:
        mc_chain, mc_energy = _mc_fold(sequence, dimension, rng, max_steps=20000)
        if mc_energy < best_energy:
            best_chain, best_energy = mc_chain, mc_energy
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
