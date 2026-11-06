"""Improved HP lattice folding: Greedy growth + Simulated Annealing with Reversals."""
# EVOLVE-BLOCK-START
import random
import time
import math

_T0 = time.time()
# 120s CPU budget, 180s wall clock limit. Use 170s as safe deadline.
_DEADLINE = 170.0

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _neighbours(point):
    """Yield the 4 neighbors of a point on the square lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _compute_energy(sequence, chain):
    """Compute the energy of a conformation.
    Energy = - (number of H-H contacts).
    """
    # Map position to index for O(1) lookup
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _delta_energy_reversal(sequence, chain, i, j):
    """Calculate the change in energy if we reverse the segment from i to j.
    Returns None if the move is invalid (collision or non-unit step).
    Returns delta_E if valid.
    """
    # Check connectivity at the boundaries
    # The segment is chain[i...j].
    # After reversal, chain[i] becomes old chain[j], and chain[j] becomes old chain[i].
    # The rest of the chain is unchanged.
    
    # 1. Check unit step between i-1 and new i (which is old j)
    if i > 0:
        p_prev = chain[i-1]
        p_new_i = chain[j]
        dist = sum(abs(a-b) for a, b in zip(p_prev, p_new_i))
        if dist != 1:
            return None
            
    # 2. Check unit step between new j (which is old i) and j+1
    if j < len(chain) - 1:
        p_next = chain[j+1]
        p_new_j = chain[i]
        dist = sum(abs(a-b) for a, b in zip(p_next, p_new_j))
        if dist != 1:
            return None

    # 3. Check for collisions with the rest of the chain
    # The positions occupied by the segment i...j are the same set, just reversed.
    # So we only need to check if any position in the segment collides with
    # positions in the rest of the chain (indices < i or > j).
    
    # Create a set of positions in the rest of the chain for fast lookup
    # However, doing this from scratch every time is O(N).
    # For N=100, it's fine.
    
    rest_positions = set()
    for k in range(len(chain)):
        if k < i or k > j:
            rest_positions.add(chain[k])
    
    # Check if any position in the segment is in rest_positions
    # Note: The segment positions are chain[i], ..., chain[j].
    # Since the chain was self-avoiding, no position in the segment collides with 
    # itself. But we must ensure no position in the segment is in rest_positions.
    # Wait, the set of positions is identical. So if the original chain was self-avoiding,
    # the positions in the segment are disjoint from the positions in the rest.
    # So collision check is trivially satisfied?
    # NO. The reversal changes WHICH residue is at WHICH position?
    # No, the residues move to new positions?
    # Let's clarify:
    # Original: ... A - B - C - D ...
    # Segment i..j corresponds to positions P_i, P_{i+1}, ..., P_j.
    # Residues R_i, R_{i+1}, ..., R_j are at these positions.
    # Reversal:
    # The chain becomes ... A - R_j - R_{j-1} - ... - R_i - D ...
    # The positions occupied by the segment are still {P_i, ..., P_j}.
    # The residues are permuted.
    # Since the set of positions is unchanged, and the original chain was self-avoiding,
    # there are NO collisions with the rest of the chain.
    # There are NO internal collisions within the segment (set of points is same).
    # So the ONLY validity checks are the unit step constraints at the boundaries!
    
    # Let's double check this logic.
    # Self-avoiding walk: all positions distinct.
    # Reversal of a sub-walk: The sub-walk visits the same set of vertices.
    # The rest of the walk visits a disjoint set of vertices.
    # Therefore, no collisions are introduced.
    # The only thing that can break the walk is the connectivity at the ends.
    
    # So, if the unit steps at the boundaries are valid, the move is valid.
    
    # Now calculate energy change.
    # Energy depends on H-H contacts.
    # Contacts are pairs of H residues at neighboring lattice sites.
    # Reversing the segment changes the adjacency of residues within the segment?
    # No, the lattice adjacencies (geometric neighbors) remain the same set of points.
    # However, the RESIDUES at those points change.
    # Specifically:
    # Old: R_k at P_k.
    # New: R_{i+j-k} at P_k (for k in i..j).
    
    # We need to count H-H contacts in the new configuration and subtract old.
    # This is expensive if done from scratch O(N^2) or O(N*deg).
    # Let's just compute the full energy for simplicity if N is small, 
    # or optimize delta.
    
    # Given N <= 100, computing full energy is O(N * 4) = O(400) ops.
    # This is very fast.
    # Let's construct the new chain and compute energy.
    
    new_chain = chain[:i] + chain[i:j+1][::-1] + chain[j+1:]
    
    # Quick self-avoidance check just in case (should be redundant but safe)
    if len(set(new_chain)) != len(new_chain):
        return None
        
    e_old = _compute_energy(sequence, chain)
    e_new = _compute_energy(sequence, new_chain)
    
    return e_new - e_old

def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            # Heuristic: prefer sites with H neighbors
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _local_search(sequence, chain, rng, time_budget):
    """Simulated Annealing with Reversal moves."""
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    n = len(chain)
    if n < 4:
        return best_chain, best_energy
    
    # Initial temperature
    T = 2.0
    # Cooling rate
    alpha = 0.995
    
    start_time = time.time()
    
    while _remaining() > 5.0: # Keep 5s margin
        # Randomly pick i, j for reversal
        # i < j
        # To make moves more effective, maybe pick shorter segments first?
        # Or random lengths?
        # Let's pick random i and j with i < j.
        # To ensure i < j, pick i, then pick j > i.
        i = rng.randint(0, n - 2)
        j = rng.randint(i + 1, n - 1)
        
        # Calculate delta E
        delta_e = _delta_energy_reversal(sequence, chain, i, j)
        
        if delta_e is None:
            continue
            
        # Acceptance criterion
        if delta_e < 0:
            accept = True
        else:
            if T > 1e-6:
                prob = math.exp(-delta_e / T)
                accept = rng.random() < prob
            else:
                accept = False
                
        if accept:
            # Perform reversal
            chain[i:j+1] = reversed(chain[i:j+1])
            current_energy += delta_e
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        # Cool down
        T *= alpha
        
        # Periodically check time
        if time.time() - start_time > 0.1:
             start_time = time.time()
             
    return best_chain, best_energy

def fold(sequence, dimension):
    """Best conformation using multiple restarts and local search."""
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0 # Energy is negative, so 0 is worse than any valid folded chain
    
    n = len(sequence)
    
    # Determine number of restarts based on time
    # Each restart + SA takes some time.
    # Let's do a few quick restarts first, then longer SA on the best.
    
    # Phase 1: Multiple short restarts to find a good initial state
    # Phase 2: Long SA on the best state found
    
    # Estimate time per restart.
    # Let's just run as many as we can within a budget, then refine.
    
    restarts_done = 0
    max_restarts = 20
    
    # Run initial greedy growths
    while _remaining() > 30.0 and restarts_done < max_restarts:
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        restarts_done += 1
        
        # Quick local search? Or just evaluate?
        # Let's do a very short SA or just accept if it's good.
        # Actually, let's just store the best greedy result so far.
        e = _compute_energy(sequence, chain)
        if e < best_energy:
            best_energy = e
            best_chain = chain
            
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
        return [list(p) for p in best_chain]
        
    # Phase 2: Refine the best chain with SA
    # Allocate remaining time to SA
    best_chain, best_energy = _local_search(sequence, best_chain, rng, _remaining())
    
    # If we have time left, maybe try another restart + SA?
    # For now, one good SA run on the best greedy start is a solid improvement.
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
