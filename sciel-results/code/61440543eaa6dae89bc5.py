"""HP lattice protein folding: Monte Carlo Move Simulated Annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = time.time()
_DEADLINE = 205.0  # Safety margin before 210s kill

def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute the energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _generate_initial_chain(sequence, dimension, rng):
    """
    Generate a random self-avoiding walk.
    Uses a biased random walk to try to keep the chain compact,
    but falls back to simple random walk if stuck.
    """
    n = len(sequence)
    if n <= 2:
        return [(0, 0)] + [(i, 0) for i in range(1, n)]
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    # Try to grow a valid chain
    for i in range(1, n):
        candidates = []
        last = chain[-1]
        for pos in _neighbours(last):
            if pos not in occupied:
                candidates.append(pos)
        
        if not candidates:
            # Stuck. This shouldn't happen often for small n, 
            # but if it does, we restart with a different seed/logic.
            # For robustness, just return a straight line if we fail multiple times.
            # In a real SA, we'd restart the whole process.
            # Here, we just try a simple straight line as a fallback to ensure validity.
            return [(i, 0) for i in range(n)]
        
        # Bias towards moving "inwards" or towards center to encourage compaction?
        # Simple random choice is sufficient for initialization.
        pos = rng.choice(candidates)
        chain.append(pos)
        occupied[pos] = i
        
    return chain

def _try_mc_move(chain, rng):
    """
    Attempt a Monte Carlo move: pick a pivot point and rotate the tail.
    Returns the new chain if valid, otherwise None.
    """
    n = len(chain)
    if n < 4:
        return None
    
    # Pick a pivot index. 
    # Uniform distribution over [1, n-3] allows rotating the tail.
    # We exclude the very ends to ensure the move is a valid rotation of a sub-segment
    # that maintains connectivity with the head and the rest of the tail.
    pivot = rng.randint(1, n - 3)
    
    # The tail starts at pivot + 1
    tail_start = pivot + 1
    tail = chain[tail_start:]
    
    # Direction from pivot to tail[0]
    dx = tail[0][0] - chain[pivot][0]
    dy = tail[0][1] - chain[pivot][1]
    
    # Possible rotations: 90, 180, 270 degrees
    # (dx, dy) -> (-dy, dx) [90 deg CCW]
    # (dx, dy) -> (-dx, -dy) [180 deg]
    # (dx, dy) -> (dy, -dx) [90 deg CW]
    
    rotations = [
        (-dy, dx),
        (-dx, -dy),
        (dy, -dx)
    ]
    
    # Shuffle to pick a random rotation
    rng.shuffle(rotations)
    
    for rd, ry in rotations:
        # Check if the new first step is valid (unit step, not colliding with head)
        new_first = (chain[pivot][0] + rd, chain[pivot][1] + ry)
        
        # The head is chain[0:pivot+1]
        head_set = set(chain[:pivot+1])
        
        if new_first in head_set:
            continue
            
        # Construct the new tail
        new_tail = [None] * len(tail)
        new_tail[0] = new_first
        
        valid = True
        for k in range(1, len(tail)):
            # The direction of the subsequent steps in the tail remains the same
            # relative to the previous step in the tail, because we are rotating
            # the entire rigid body of the tail.
            odx = tail[k][0] - tail[k-1][0]
            ody = tail[k][1] - tail[k-1][1]
            
            new_tail[k] = (new_tail[k-1][0] + odx, new_tail[k-1][1] + ody)
            
            # Check collisions with head
            if new_tail[k] in head_set:
                valid = False
                break
            # Check collisions within the new tail itself
            # Since the original tail was self-avoiding and we applied a rigid rotation,
            # internal collisions in the new tail are impossible unless the rotation
            # causes it to fold onto itself in a way that wasn't present before?
            # No, rigid rotation preserves distances. If tail was self-avoiding, 
            # rotated tail is self-avoiding.
            # So we only need to check against the head.
            # Wait, what if new_tail[k] collides with new_tail[j] for j < k?
            # No, rigid body rotation preserves the relative positions.
            # So internal self-avoidance is guaranteed if the original was self-avoiding.
            
        if valid:
            return chain[:pivot+1] + new_tail
            
    return None

def _anneal(sequence, chain, max_steps, rng):
    """
    Perform Simulated Annealing using MC moves.
    """
    n = len(chain)
    if n < 4:
        return chain
    
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    # Temperature schedule
    # Start high enough to accept bad moves, end low for refinement
    T_start = 3.0
    T_end = 0.1
    if max_steps <= 0:
        return best_chain
    
    cooling = (T_end / T_start) ** (1.0 / max_steps)
    
    for step in range(max_steps):
        if _remaining_time() < 0.5:
            break
        
        T = T_start * (cooling ** step)
        
        new_chain = _try_mc_move(chain, rng)
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
    
    return best_chain

def fold(sequence, dimension):
    """
    Fold the sequence using Simulated Annealing with Monte Carlo Moves.
    """
    n = len(sequence)
    if n <= 2:
        return [list(p) for p in [(i, 0) for i in range(n)]]
    
    # Allocate time budget
    # We will run multiple independent SA runs, restarting from random initial chains
    # or from the best found so far, to explore different basins.
    
    best_chain = None
    best_energy = 0 # 0 is unfolded, we want negative
    
    # Determine time per run
    # Let's aim for ~5 seconds per run, allowing ~40 runs in 200s.
    # However, for long sequences, MC moves are slower.
    # Let's just loop until time runs out, restarting periodically.
    
    run_count = 0
    rng = random.Random(42)
    
    while _remaining_time() > 5.0:
        run_count += 1
        
        # Generate initial chain
        # For the first run, use a specific seed. For others, use run_count.
        init_rng = random.Random(12345 + run_count)
        
        # We can try to start from the best found so far in subsequent runs
        # to refine it, but also sometimes start fresh to explore new areas.
        if best_chain is None or run_count % 3 == 0:
            # Fresh start
            initial_chain = _generate_initial_chain(sequence, dimension, init_rng)
        else:
            # Perturb best
            initial_chain = best_chain[:]
            # Apply a few random MC moves to kick it out of local minimum
            pert_rng = random.Random(run_count * 100)
            for _ in range(10):
                moved = _try_mc_move(initial_chain, pert_rng)
                if moved:
                    initial_chain = moved
        
        # Calculate steps for this run
        # We want to spend about 4 seconds per run.
        # Estimate cost of one step: O(n) for collision check in _try_mc_move?
        # Actually _try_mc_move is O(n) in the worst case due to head_set construction 
        # and the loop. _energy is O(n).
        # So one step is ~ O(n).
        # If n=100, 100 ops per step. Python does ~10^6-10^7 simple ops/sec.
        # So ~10^4-10^5 steps/sec.
        # Let's be conservative: 1000 steps/sec for n=100.
        # For n=20, maybe 10000 steps/sec.
        # Let's just run until time slice is up.
        
        start_run_time = time.time()
        time_slice = 4.0
        # Estimate steps
        # Let's use a fixed number of steps scaled by 1/n to keep total time similar
        # But dynamic time check is better.
        
        # Run SA
        # We need a max_steps for the cooling schedule.
        # Let's estimate: 2000 steps for short, 500 for long.
        est_steps = int(2000 * (20.0 / n))
        est_steps = max(100, est_steps)
        
        improved_chain = _anneal(sequence, initial_chain, est_steps, rng)
        
        e = _energy(sequence, improved_chain)
        if e < best_energy:
            best_energy = e
            best_chain = improved_chain
        
        # Check time
        elapsed = time.time() - start_run_time
        if elapsed > time_slice:
            # We might have overshot slightly due to slow steps, but _anneal checks time.
            pass

    if best_chain is None:
        # Fallback
        return [list(p) for p in [(i, 0) for i in range(n)]]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
