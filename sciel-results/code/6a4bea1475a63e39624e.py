"""HP lattice protein folding using Monte Carlo local search with Simulated Annealing."""
# EVOLVE-BLOCK-START
import random
import time

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Calculate the energy of the chain. Energy = - (number of H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                # Contact must be between non-adjacent residues (j > i + 1)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _initial_chain(sequence, dimension, rng):
    """
    Generate an initial self-avoiding walk.
    Uses a simple greedy approach to get a valid starting point quickly.
    """
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Prefer sites that are not too close to the start to avoid tight coils initially?
            # Actually, for SA, we just need a valid SAS.
            # Greedy: pick randomly among valid neighbors, maybe bias towards free space
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((free + rng.random(), site))
        
        if not options:
            # If stuck, we might need to backtrack or just fail this init
            # For robustness, if we get stuck, return a straight line (always valid)
            # But straight line has high energy. 
            # Let's try a simple "snake" pattern if greedy fails.
            pass
            
        if options:
            # Pick the one with most free space (to avoid trapping)
            site = max(options)[1]
            occupied[site] = i
            chain.append(site)
        else:
            # Fallback: this shouldn't happen often with random bias, but if it does
            # we return a valid straight line
            return [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
            
    return chain

def _sa_search(sequence, dimension, rng, deadline):
    """
    Perform Simulated Annealing to minimize energy.
    """
    n = len(sequence)
    if n <= 2:
        return [tuple([0] + [0] * (dimension - 1))] * n

    # 1. Initialize
    chain = _initial_chain(sequence, dimension, rng)
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # 2. Annealing Schedule
    # We have a deadline. We estimate steps per second based on n.
    # A kink move is O(1) to check validity, O(n) to update energy if we recalculate from scratch.
    # To be fast, we should do delta energy calculation.
    
    # For simplicity and correctness first, let's implement delta energy.
    # Moving residue i to new_site:
    # - Remove i from old_site
    # - Add i to new_site
    # - Update contacts involving i and its neighbors in the chain (i-1, i+1) and spatial neighbors.
    
    # Actually, recalculating full energy is O(n). With n=100, O(n) is 100 ops.
    # 100k steps * 100 ops = 10M ops. Python can do ~10M-50M simple ops/sec.
    # So 100k steps might take ~1-5 seconds.
    # We have 120s budget. We can afford ~2-5 million steps.
    
    start_time = time.time()
    max_time = deadline - start_time - 5.0 # 5s safety margin
    
    # Initial temperature
    T = 1.0
    # Cooling rate: we want T to drop to near 0 by end of time.
    # Let's estimate max steps based on time.
    # Assume ~10000 steps/sec for Python loop with dict lookups.
    est_steps = int(max_time * 10000)
    
    if est_steps <= 0:
        return best_chain
        
    # Geometric cooling: T(t) = T0 * (T_end/T0)^(t/max_steps)
    # T_end = 0.01
    # alpha = (T_end / T0) ** (1.0 / est_steps)
    # To avoid underflow issues, we can just use a simple linear or geometric decay.
    # Let's use a standard SA loop.
    
    # Optimize: Precompute H positions? No, sequence is fixed.
    
    # We will run until time runs out.
    # To speed up energy calculation, we can maintain a count of contacts.
    # But delta calculation is tricky with the "j > i+1" constraint.
    # Let's stick to full recalculation first to ensure correctness, 
    # but optimize the loop structure.
    
    # Actually, full recalculation of _energy is O(N^2) in worst case? No, _energy is O(N * 4) = O(N).
    # _energy iterates over N residues, checks 4 neighbors. Dict lookup is O(1).
    # So _energy is O(N).
    # Total cost per step: O(N) for energy + O(1) for move check.
    # With N=100, 100 ops per step.
    # 100k steps -> 10M ops. ~1 sec.
    # We have 115s. So we can do ~10-15 million steps.
    
    # Let's do a simple loop checking time every 1000 steps.
    
    T = 1.0
    # We don't know exact steps, so we adjust T based on time?
    # Or just run as many steps as possible with a fixed cooling schedule based on estimated steps.
    # If we finish early, we can restart with a lower T? 
    # Better: Use a time-based cooling.
    # T(t) = T0 * exp(-beta * t)
    # Let's pick beta such that T drops to 0.01 at max_time.
    # 0.01 = 1.0 * exp(-beta * max_time)
    # ln(0.01) = -beta * max_time
    # beta = -ln(0.01) / max_time = 4.6 / max_time
    
    beta = 4.6 / max_time if max_time > 0 else 1.0
    
    # To avoid exp() call every step (expensive), we can update T multiplicatively.
    # T_new = T * exp(-beta * dt)
    # If we do 1 step every ~10us, dt is small.
    # Alternatively, just use a fixed number of steps and geometric cooling.
    # Let's estimate steps more accurately.
    # Let's just run a fixed number of steps per second?
    
    # Simple approach:
    # Run in blocks.
    steps_per_block = 1000
    T = 1.0
    # We will cool T by a factor of 0.999 every 100 steps? 
    # Let's try: T = 1.0 initially. 
    # If we do 1M steps, and want T to go to 0.01.
    # 0.01 = 1.0 * r^1000000
    # r = 0.01^(1/1000000) = exp(ln(0.01)/1000000) = exp(-4.6/1000000) ~ 0.9999954
    
    # We don't know total steps. 
    # Let's use a dynamic T based on elapsed time.
    # T = max(0.01, 1.0 * exp(-beta * elapsed))
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_time:
            break
            
        T = max(0.01, 1.0 * (0.01) ** (elapsed / max_time))
        
        for _ in range(steps_per_block):
            # Pick a random residue to move (kink move)
            # Exclude ends (0 and n-1) to keep it simple? 
            # Moving ends is also valid but less effective for packing.
            # Let's allow moving any residue 1..n-2
            if n < 4:
                break
            i = rng.randint(1, n - 2)
            
            old_site = chain[i]
            
            # Try to move to a random neighbor
            # Pick 1 random neighbor to try (or all 4? 1 is faster, 4 is better mixing)
            # Let's pick 1 random neighbor to keep step cost low.
            neighbors = list(_neighbours(old_site))
            rng.shuffle(neighbors)
            
            moved = False
            for new_site in neighbors:
                if new_site in occupied:
                    continue
                # Check if new_site is adjacent to i-1 or i+1 in chain?
                # If new_site is adjacent to chain[i-1] or chain[i+1], it's a valid move 
                # (it just changes the path locally). 
                # Wait, if new_site is occupied by i-1 or i+1, it's not in occupied set 
                # because we are moving i? 
                # No, occupied contains ALL current positions.
                # If new_site is occupied by j, and j != i, it's invalid.
                # Since we check `if new_site in occupied`, and i is currently at old_site,
                # new_site cannot be old_site. 
                # If new_site is occupied by i-1 or i+1, that's a collision.
                # So `if new_site in occupied` is sufficient to ensure self-avoidance 
                # for the new position, provided we remove i from old_site first?
                # No, we check before removing.
                # If new_site is occupied by i-1, `new_site in occupied` is True. So we skip.
                # This is correct.
                
                # Perform move
                del occupied[old_site]
                occupied[new_site] = i
                chain[i] = new_site
                
                # Calculate energy
                new_energy = _energy(sequence, chain)
                delta = new_energy - current_energy
                
                if delta < 0 or rng.random() < (2.718281828 ** (-delta / T)):
                    # Accept
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                    moved = True
                    break # Move successful, break neighbor loop
                else:
                    # Reject: undo move
                    del occupied[new_site]
                    occupied[old_site] = i
                    chain[i] = old_site
                    # Don't break, try next neighbor? 
                    # Usually we try one move per step. 
                    # If rejected, we just stay put.
                    # Breaking here means we tried 1 move.
                    # If we don't break, we might try another neighbor in the same "step".
                    # Let's break to keep step definition consistent (1 attempt).
                    break
            
            # If no move was made (all neighbors occupied or rejected), we just continue
            # The loop continues to next iteration.
        
        # Check time at end of block
        if time.time() - start_time > max_time:
            break

    return best_chain

def fold(sequence, dimension):
    """Best conformation found via Simulated Annealing."""
    # Deadline guard: 120s budget, leave 10s for overhead/return
    deadline = time.time() + 110.0
    
    # If sequence is very short, just return a valid path
    if len(sequence) <= 2:
        return [[0, 0] for _ in range(len(sequence))]
    
    rng = random.Random(12345 + len(sequence))
    
    # Run SA
    best_chain = _sa_search(sequence, dimension, rng, deadline)
    
    # Ensure we return a valid self-avoiding walk
    # _sa_search should return a valid one, but just in case:
    if best_chain is None or len(best_chain) != len(sequence):
        # Fallback to straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        
    # Convert tuples to lists as required
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
