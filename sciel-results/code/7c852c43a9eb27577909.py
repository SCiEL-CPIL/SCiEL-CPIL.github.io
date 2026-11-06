"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

def construct():
    start_time = time.time()
    TIME_LIMIT = 85.0  # Safety margin before 90s kill
    
    n = 256
    max_sum = 2 * n
    
    # 1. Precompute squares
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
        
    # 2. Identify valid nodes (diagonal safe: 2*a not in squares)
    all_candidates = [a for a in range(1, n + 1) if 2 * a not in squares]
    M = len(all_candidates)
    
    # 3. Build Conflict Graph
    # conflicts[v] = set of nodes u such that v+u is a square
    conflicts = {a: set() for a in all_candidates}
    # To speed up, we can precompute for each a, which b in candidates conflict
    # But building the full graph is O(M^2). M ~ 245. 245^2 ~ 60k ops. Very fast.
    for idx, a in enumerate(all_candidates):
        for b in all_candidates[idx+1:]:
            if (a + b) in squares:
                conflicts[a].add(b)
                conflicts[b].add(a)

    # Helper: Check if node v can be added to set s
    def try_add_fast(s, v):
        # If v conflicts with any node in s, it cannot be added
        # s is a set of nodes
        # conflicts[v] is a set of nodes
        return len(conflicts[v] & s) == 0

    def greedy_min_degree(candidates):
        remaining = set(candidates)
        independent_set = set()
        while remaining:
            min_deg = float('inf')
            chosen = None
            for v in remaining:
                deg = len(conflicts[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    chosen = v
            independent_set.add(chosen)
            remaining.remove(chosen)
            # Remove neighbors of chosen from remaining
            for neighbor in conflicts[chosen]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
        return independent_set

    def local_search_improve(s):
        s = set(s)
        improved = True
        # Limit iterations to prevent infinite loops in case of oscillation
        max_iters = 500
        iters = 0
        all_candidates_set = set(all_candidates)
        
        while improved and iters < max_iters:
            iters += 1
            improved = False
            
            # 1. Try to add any single node
            for v in all_candidates:
                if v not in s and try_add_fast(s, v):
                    s.add(v)
                    improved = True
            
            # 2. Try 1-swap: Remove u, add v.
            best_swap = None
            best_score = -1
            
            # Optimization: Precompute complement of s for blocking calculation
            # Note: s changes slightly in the loop, so we recompute "outside" set once per u
            for u in list(s):
                s.remove(u)
                outside = all_candidates_set - s
                
                # Find all v in outside that can be added
                addable = []
                for v in outside:
                    if try_add_fast(s, v):
                        addable.append(v)
                        
                if addable:
                    for v in addable:
                        # v is compatible with s
                        # Score: Minimize blocked nodes
                        blocked = len(conflicts[v] & outside)
                        score = -blocked
                        if score > best_score:
                            best_score = score
                            best_swap = (u, v)
                            
                s.add(u) # Restore u for next iteration of u loop
                
            if best_swap:
                u, v = best_swap
                s.remove(u)
                s.add(v)
                improved = True
                
        return s

    # Main Search Loop
    best_set = greedy_min_degree(all_candidates)
    best_size = len(best_set)
    
    random.seed(42)
    
    # Try multiple random restarts
    # We can also try different greedy strategies
    strategies = [
        lambda: greedy_min_degree(all_candidates),
        lambda: set(random.sample(all_candidates, min(50, len(all_candidates)))), # Random small set
    ]
    
    # Add a "weighted" greedy: pick node with min degree, but break ties randomly
    def greedy_min_degree_random():
        remaining = set(all_candidates)
        independent_set = set()
        while remaining:
            min_deg = float('inf')
            candidates_for_choice = []
            for v in remaining:
                deg = len(conflicts[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    candidates_for_choice = [v]
                elif deg == min_deg:
                    candidates_for_choice.append(v)
            chosen = random.choice(candidates_for_choice)
            independent_set.add(chosen)
            remaining.remove(chosen)
            for neighbor in conflicts[chosen]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
        return independent_set

    strategies.append(greedy_min_degree_random)
    
    for trial in range(500): # Max 500 trials
        if time.time() - start_time > TIME_LIMIT:
            break
            
        # Choose a strategy
        strat = random.choice(strategies)
        current_set = strat()
        
        # Refine
        current_set = local_search_improve(current_set)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
            # If we hit a high score, we can tighten the time limit for subsequent checks?
            # Or just continue.
            
    result = sorted(best_set)
    
    # Final Verification
    for i, a in enumerate(result):
        for b in result[i:]:
            s = a + b
            root = math.isqrt(s)
            if root * root == s:
                # This should never happen if our logic is correct
                # But if it does, we have a bug.
                # For safety, we could remove b, but that breaks the set.
                # Raise error to signal invalidity.
                raise ValueError(f"Conflict: {a} + {b} = {s}")
                
    return result
# EVOLVE-BLOCK-END
