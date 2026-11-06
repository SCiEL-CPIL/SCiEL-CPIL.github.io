# EVOLVE-BLOCK-START
import random
import time
import sys
import os
import numpy as np

# Global state for time management across multiple calls to fold()
# The contract states 240s CPU budget for all sequences together.
# We assume sequential execution in a single process or subprocess.
_global_start_time = None
_global_deadline_margin = 10.0  # Safety margin in seconds

def _get_deadline():
    """Returns the global deadline for the entire evaluation run."""
    global _global_start_time
    if _global_start_time is None:
        _global_start_time = time.time()
    # The total budget is 240s. We use a conservative limit to avoid timeout.
    # If the evaluator kills at 330s, we want to finish well before that.
    # Assuming 26 calls, and some overhead, we allocate a dynamic deadline.
    # However, since we don't know the exact call count or order, we use a global clock.
    # Let's set the deadline to 230s from the first call.
    return _global_start_time + 230.0

def _energy(sequence, chain):
    """
    Compute the energy of a folded chain.
    Energy = - (number of H-H contacts on non-adjacent lattice sites)
    """
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            # Check 6 neighbors
            for axis in range(len(p)):
                for step in (-1, 1):
                    q = list(p)
                    q[axis] += step
                    q = tuple(q)
                    j = where.get(q)
                    if j is not None and j > i + 1 and sequence[j] == 'H':
                        contacts += 1
    return -contacts

def _grow(sequence, dimension, rng):
    """
    Randomized greedy growth with bias towards H-H contacts.
    Returns a list of tuples representing the chain.
    """
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        for axis in range(dimension):
            for step in (-1, 1):
                site = list(last)
                site[axis] += step
                site = tuple(site)
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == 'H':
                    # Check if this site forms a contact with any previous H
                    for axis2 in range(dimension):
                        for step2 in (-1, 1):
                            q = list(site)
                            q[axis2] += step2
                            q = tuple(q)
                            j = occupied.get(q)
                            if j is not None and j < i - 1 and sequence[j] == 'H':
                                gain += 1
                free = 0
                for axis2 in range(dimension):
                    for step2 in (-1, 1):
                        q = list(site)
                        q[axis2] += step2
                        q = tuple(q)
                        if q not in occupied:
                            free += 1
                
                # Score: strongly favor contacts, weakly favor free space, add noise
                score = gain * 10.0 + free * 0.1 + rng.random()
                options.append((score, site))
        
        if not options:
            return None
        
        # Pick the best option
        best_site = max(options, key=lambda x: x[0])[1]
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain

def _pivot_optimize(sequence, chain, dimension, rng, deadline):
    """
    Optimize the chain using pivot moves (rotating the tail around a pivot bond).
    Uses Simulated Annealing-like acceptance to escape local minima.
    """
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Initial temperature
    T = 5.0
    # Cooling factor
    alpha = 0.999
    
    while time.time() < deadline:
        # Check deadline frequently
        if time.time() > deadline - 0.1:
            break
            
        # Choose a random pivot index (1 to n-2)
        pivot_idx = rng.randint(1, n - 2)
        
        # Determine the bond axis for the pivot
        p_prev = current_chain[pivot_idx - 1]
        p_pivot = current_chain[pivot_idx]
        
        v = [p_pivot[i] - p_prev[i] for i in range(dimension)]
        bond_axis = -1
        for i in range(dimension):
            if v[i] != 0:
                bond_axis = i
                break
        if bond_axis == -1:
            continue
            
        # Choose a rotation axis (one of the other 2 axes)
        rot_axes = [i for i in range(dimension) if i != bond_axis]
        if not rot_axes:
            continue
        r_axis = rng.choice(rot_axes)
        angle = rng.choice([1, -1])
        
        # Rotate the tail [pivot_idx+1 .. n-1]
        # The pivot point p_pivot stays fixed.
        # The head [0 .. pivot_idx-1] stays fixed.
        
        head_set = set(current_chain[:pivot_idx])
        # Note: p_pivot is at index pivot_idx, it is part of the "head" for collision purposes
        # but it is the center of rotation. The tail starts at pivot_idx+1.
        
        new_tail = []
        valid = True
        occupied_tail = set()
        
        for i in range(pivot_idx + 1, n):
            orig = current_chain[i]
            # Translate to origin relative to pivot
            dx = orig[0] - p_pivot[0]
            dy = orig[1] - p_pivot[1]
            dz = orig[2] - p_pivot[2]
            
            if r_axis == 0: # Rotate around X
                if angle == 1:
                    nx, ny, nz = dx, -dz, dy
                else:
                    nx, ny, nz = dx, dz, -dy
            elif r_axis == 1: # Rotate around Y
                if angle == 1:
                    nx, ny, nz = dz, dy, -dx
                else:
                    nx, ny, nz = -dz, dy, dx
            else: # Rotate around Z
                if angle == 1:
                    nx, ny, nz = -dy, dx, dz
                else:
                    nx, ny, nz = dy, -dx, dz
            
            new_point = (p_pivot[0] + nx, p_pivot[1] + ny, p_pivot[2] + nz)
            
            if new_point in head_set:
                valid = False
                break
            if new_point in occupied_tail:
                valid = False
                break
                
            # Check unit step from previous new point
            if new_tail:
                prev = new_tail[-1]
                if abs(new_point[0]-prev[0]) + abs(new_point[1]-prev[1]) + abs(new_point[2]-prev[2]) != 1:
                    valid = False
                    break
            
            occupied_tail.add(new_point)
            new_tail.append(new_point)
            
        if not valid:
            T *= alpha
            continue
            
        candidate_chain = current_chain[:pivot_idx + 1] + new_tail
        new_energy = _energy(sequence, candidate_chain)
        
        # Acceptance criterion
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            current_chain = candidate_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        T *= alpha
        if T < 0.1:
            T = 0.1
            
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold the sequence using randomized growth and pivot optimization.
    """
    # Set up global deadline
    global _global_start_time
    if _global_start_time is None:
        _global_start_time = time.time()
    
    deadline = _get_deadline()
    
    # Check if we are out of time
    if time.time() > deadline:
        # Return a trivial valid chain (straight line) to avoid error
        # This is a fallback, should not happen if time management is correct
        return [[i, 0, 0] for i in range(len(sequence))]
    
    # Allocate time for this specific fold call
    # We don't know how many calls are left, so we use a heuristic:
    # If we have used > 50% of time, reduce budget for this call.
    elapsed = time.time() - _global_start_time
    total_budget = 230.0
    remaining_budget = total_budget - elapsed
    
    # Assume ~26 calls. If we are early, give more time. If late, give less.
    # A simple strategy: give each call 1/10th of remaining time, capped.
    # Or just use a fixed chunk if time allows.
    # Let's use a dynamic budget: min(10s, remaining_budget * 0.5)
    # But we need to ensure we finish.
    # Let's just use the global deadline and let the loop inside _pivot_optimize handle it.
    # We will do a few restarts.
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = float('inf')
    
    # Number of restarts depends on time available
    # Let's do at least 1 restart, and more if time permits.
    # We will try to fit as many restarts as possible within the deadline.
    
    # Estimate time per restart.
    # Growth is fast. Optimization is the bottleneck.
    # Let's do 5 restarts, each with a sub-deadline.
    
    n_restarts = 5
    
    # If time is tight, reduce restarts
    if remaining_budget < 20:
        n_restarts = 1
    elif remaining_budget < 50:
        n_restarts = 2
    elif remaining_budget < 100:
        n_restarts = 3
    
    for i in range(n_restarts):
        # Check time
        if time.time() > deadline - 2:
            break
            
        # Generate initial chain
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            # Try a simple random walk if growth fails
            chain = [(0,0,0)]
            occupied = {chain[0]: 0}
            for j in range(1, len(sequence)):
                while True:
                    axis = rng.randint(0, dimension-1)
                    step = rng.choice([-1, 1])
                    new_p = list(chain[-1])
                    new_p[axis] += step
                    new_p = tuple(new_p)
                    if new_p not in occupied:
                        chain.append(new_p)
                        occupied[new_p] = j
                        break
            
        # Optimize
        # Allocate remaining time / remaining restarts
        time_per_restart = (deadline - time.time() - 1.0) / (n_restarts - i)
        if time_per_restart < 0.1:
            time_per_restart = 0.1
            
        opt_chain, opt_energy = _pivot_optimize(sequence, chain, dimension, rng, time.time() + time_per_restart)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_chain = opt_chain
            
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(len(sequence))]
        
    # Convert to list of lists
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
