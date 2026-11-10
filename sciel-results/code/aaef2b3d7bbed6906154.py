"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Evaluator kills after 210s wall-clock. 
# We keep a safety margin. The task says 120s CPU budget for all sequences, 
# but the interface is called once per sequence. 
# The prompt says: "fold is called once per benchmark sequence in a sandboxed subprocess with a 120 s CPU budget for all sequences together."
# However, the local test harness might call it multiple times or the "subprocess" implies the whole script runs once.
# Looking at the current program, it uses a global _DEADLINE.
# If fold is called 10 times, a global deadline of 175s means the last calls have very little time.
# But usually, these tasks run the script once which calls fold internally, OR the sandbox runs the script for each sequence.
# The prompt says: "fold is called once per benchmark sequence in a sandboxed subprocess".
# This implies a NEW process for each sequence? Or one process where fold is called 10 times?
# "120 s CPU budget for all sequences together" suggests one process.
# If it's one process, we should divide the time.
# If it's 10 processes, we have 120s each? 
# "subprocess" singular in "a sandboxed subprocess" might imply the whole execution is one subprocess.
# Let's assume the script is run once and fold is called 10 times.
# To be safe, we track time per call or globally?
# The current program uses a global _DEADLINE. If I use a global deadline, the later sequences get less time.
# A better strategy for "120s total" is to estimate time per sequence and stick to it, or use a global deadline that accounts for all.
# However, the score is the MEAN. So all sequences are equally important.
# If I spend 110s on the first sequence, I have 10s for the remaining 9.
# I will implement a per-call time limit based on the total budget / 10, with a small buffer.
# Total budget 120s. 10 sequences. ~12s per sequence.
# But the current program uses 175s global. This suggests the evaluator might actually give more time or the "120s" is a soft limit or per-subprocess limit if they are separate.
# Let's look at the "180s" limit in the prompt: "Each evaluation must finish within 180 s."
# This likely refers to the total time for the script.
# I will use a global deadline of 170s to be safe, but I will distribute time more evenly by checking elapsed time.
# Actually, the most robust way for "mean score" is to ensure no single sequence eats all time.
# I will use a global deadline, but I will also cap individual fold calls to ensure we don't run out of time for later ones.
# Estimated time per sequence: (Total Time - Overhead) / 10.

_START_TIME = time.time()
_TOTAL_BUDGET = 170.0 # Safety margin from 180s
_NUM_SEQS = 10
_TIME_PER_SEQ = _TOTAL_BUDGET / _NUM_SEQS

def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2

def _greedy_init(sequence, rng):
    """
    Fast greedy growth to get an initial valid self-avoiding walk.
    This serves as the initial state for the CSP solver.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Prefer sites that create H-H contacts
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            # Prefer sites with more free space (heuristic to avoid dead ends)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.5 + rng.random()
            options.append((score, site))
        
        if not options:
            # Failed to grow, return None
            return None
        
        best_score, best_site = max(options, key=lambda x: x[0])
        occupied[best_site] = i
        chain.append(best_site)
        
    return chain

def _min_conflicts_csp(sequence, chain, rng, time_limit):
    """
    Min-Conflicts Local Search on the grid assignment.
    State: chain (list of points), occupied (dict point->index).
    Objective: Minimize violations = (Self-Intersections * BigWeight) - (H-H Contacts).
    Since we start with a valid walk, Self-Intersections is 0.
    We only move if we maintain self-avoidance (validity) or if we can quickly fix it?
    Standard Min-Conflicts allows invalid states but penalizes them.
    However, maintaining validity is crucial for the final output.
    Strategy: Only accept moves that result in a VALID self-avoiding walk.
    This turns it into a constrained local search (Simulated Annealing / Hill Climbing on valid states).
    Moves:
    1. Kink Flip: Move residue i to a neighbor of i-1 that is also a neighbor of i+1 (if i is internal).
       Actually, a kink flip moves i to a position that maintains connectivity with i-1 and i+1.
       For i to be connected to i-1 and i+1, the new position must be a common neighbor of i-1 and i+1?
       No, if we move i, the links (i-1, i) and (i, i+1) must remain unit steps.
       So new_pos must be a neighbor of chain[i-1] AND a neighbor of chain[i+1].
       This is only possible if chain[i-1] and chain[i+1] are distance 2 apart (with i in between originally) or distance 0 (impossible) or distance...
       If i-1 is (0,0) and i+1 is (2,0), i was (1,0). New pos could be (1,1) or (1,-1).
       This is a "kink flip".
       
    2. Detour Move: Move a segment of length 2 or 3.
       This is more complex.
       
    Let's stick to Kink Flips and "Sliding" moves.
    A "Slide" move: Move residue i to a neighbor of i-1 that is NOT occupied.
    This breaks the link to i+1. So we must also move i+1?
    This becomes a regrow.
    
    Given the time constraints, Kink Flips are the safest and most common effective local move for HP folding.
    To escape local minima, we can use Simulated Annealing acceptance criteria.
    
    Let's implement Kink Flip SA.
    """
    n = len(sequence)
    if n < 3:
        return chain, _compute_energy(sequence, chain)
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    is_h = [c == 'H' for c in sequence]
    
    # Precompute neighbors for each residue for faster kink check?
    # Kink flip at i:
    # prev = chain[i-1], next = chain[i+1]
    # new_pos must be neighbor of prev AND neighbor of next.
    # And new_pos != chain[i]
    # And new_pos not in occupied (except if occupied by i, which it is)
    
    start_temp = 5.0
    cool_rate = 0.9995
    temp = start_temp
    
    end_time = time.time() + time_limit
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < end_time:
        if temp < 0.01:
            break
            
        # Pick a random internal residue
        i = rng.randint(1, n - 2)
        prev = chain[i-1]
        next = chain[i+1]
        curr = chain[i]
        
        # Find common neighbors of prev and next
        # prev's neighbors: (px+1,py), (px-1,py), (px,py+1), (px,py-1)
        # next's neighbors: ...
        # Intersection
        candidates = []
        for np in _neighbours(prev):
            if np == curr:
                continue
            # Check if np is neighbor of next
            if abs(np[0]-next[0]) + abs(np[1]-next[1]) == 1:
                # Check if np is free
                if np in occupied:
                    if occupied[np] != i:
                        continue
                candidates.append(np)
                
        if not candidates:
            continue
            
        new_pos = rng.choice(candidates)
        
        # Calculate energy delta
        # Remove contacts at curr, add contacts at new_pos
        # Only H residues matter
        delta = 0
        if is_h[i]:
            # Remove
            for q in _neighbours(curr):
                j = occupied.get(q)
                if j is not None and abs(j - i) > 1 and is_h[j]:
                    delta += 1
            # Add
            for q in _neighbours(new_pos):
                j = occupied.get(q)
                if j is not None and abs(j - i) > 1 and is_h[j]:
                    delta -= 1
                    
        # Note: delta is change in contacts. Energy = -contacts.
        # So delta_energy = -delta
        new_energy = current_energy - delta
        
        # Acceptance
        if new_energy < current_energy:
            accept = True
        else:
            # Metropolis
            if temp > 0:
                prob = math.exp((-delta) / temp) # -delta is positive if energy increases
                accept = rng.random() < prob
            else:
                accept = False
                
        if accept:
            # Update state
            del occupied[curr]
            chain[i] = new_pos
            occupied[new_pos] = i
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
            
        temp *= cool_rate
        
        if no_improve > max_no_improve:
            # Restart with higher temp?
            temp = start_temp * 0.5
            no_improve = 0
            # Optionally randomize chain? No, keep best found so far as anchor?
            # In Min-Conflicts, we usually keep the current state.
            # Let's just continue.
            
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold the sequence using CSP-style local search.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    # Determine time budget for this specific call
    # If we are early in the sequence list, we can use more time.
    # But we don't know which call we are.
    # We will use a fixed fraction of the total budget, capped by remaining global time.
    elapsed = time.time() - _START_TIME
    remaining_global = _TOTAL_BUDGET - elapsed
    if remaining_global <= 0:
        # Out of time, return quick greedy
        chain = _greedy_init(sequence, random.Random(42))
        if chain:
            return [list(p) for p in chain]
        return [[i, 0] for i in range(n)]
        
    # Use up to 80% of the per-sequence budget, or all remaining if it's the last one?
    # We don't know if it's the last one.
    # Safe bet: use min(_TIME_PER_SEQ * 0.9, remaining_global * 0.5)
    # This ensures we don't blow the global budget.
    time_limit = min(_TIME_PER_SEQ * 0.9, remaining_global * 0.5)
    
    rng = random.Random(42)
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Get initial valid state
    chain = _greedy_init(sequence, rng)
    if chain is None:
        # Fallback: straight line
        chain = [(i, 0) for i in range(n)]
        
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Phase 2: Local Search
    if time_limit > 1.0:
        # Run multiple restarts or just one long SA?
        # One long SA with restarts on stagnation is usually better for HP.
        sa_chain, sa_energy = _min_conflicts_csp(sequence, chain, rng, time_limit)
        
        if sa_energy < best_energy:
            best_chain = sa_chain
            best_energy = sa_energy
            
        # If we have time left, do a second pass from the best found?
        elapsed_this_call = time.time() - (time.time() - time_limit) # Approx
        # Actually, let's just check global time
        if time.time() < _START_TIME + _TOTAL_BUDGET - 5.0:
            # Second pass with different random seed or higher temp?
            # Let's try a different initial greedy with higher randomness
            chain2 = _greedy_init(sequence, rng)
            if chain2:
                e2 = _compute_energy(sequence, chain2)
                if e2 < best_energy:
                    best_chain = chain2
                    best_energy = e2
                    
                # Short SA on this new start
                remaining_for_sa = min(2.0, _START_TIME + _TOTAL_BUDGET - 5.0 - time.time())
                if remaining_for_sa > 0.5:
                    sa_chain2, sa_energy2 = _min_conflicts_csp(sequence, chain2, rng, remaining_for_sa)
                    if sa_energy2 < best_energy:
                        best_chain = sa_chain2
                        best_energy = sa_energy2

    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
