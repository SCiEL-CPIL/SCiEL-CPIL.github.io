import math
import time
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # Initial candidates: Small Golomb rulers / Sidon-like sets
    initial_sets = [
        [0],
        [0, 1],
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 18, 23],
        [0, 1, 4, 10, 18, 23, 31],
        [0, 1, 4, 10, 18, 23, 31, 39],
        [0, 1, 4, 10, 18, 23, 31, 39, 49],
        [0, 1, 4, 10, 18, 23, 31, 39, 49, 59],
        [0, 2, 3, 5, 8, 13, 21, 34, 55, 89], # Fibonacci-like
        [0, 1, 3, 7, 12, 20, 31, 45, 62, 82]  # Quadratic-like
    ]
    
    best_set = initial_sets[0]
    best_score = -1.0
    
    def get_score(u):
        if not u:
            return 1.0
        u = np.array(u, dtype=np.int64)
        if len(u) == 1:
            return 1.0
        # Ensure unique and sorted
        u = np.unique(u)
        if u[0] != 0:
             # The problem requires 0 in U. If not, we should probably penalize or fix.
             # The local search will maintain 0.
             pass
        
        diffs = np.unique(np.subtract.outer(u, u))
        sums = np.unique(np.add.outer(u, u))
        
        max_u = u.max()
        m = 2 * max_u + 1
        
        if m <= 1:
            return 1.0
            
        r = len(diffs) / len(sums)
        # Avoid log(0) or log(negative)
        if r <= 0:
            return 0.0
        score = 1 + math.log(r) / math.log(m)
        return score

    # Evaluate initial sets
    for u in initial_sets:
        s = get_score(u)
        if s > best_score:
            best_score = s
            best_set = u[:]

    # Local Search
    # We will try to improve the best set found so far.
    # We might also restart from other initial sets if time permits.
    
    # To be efficient, we'll focus on improving the current best.
    # We'll also keep a history of top sets to potentially restart from.
    
    top_sets = [(best_score, best_set)]
    
    # Define a function to try a perturbation
    def try_perturb(u, max_val):
        # u is a list
        n = len(u)
        u_set = set(u)
        
        # Strategy 1: Add a new element
        # Try adding a random element in [1, max_val]
        # Or try adding elements that complete Golomb/Sidon properties?
        # Random is simpler.
        
        # Strategy 2: Replace an element
        # Strategy 3: Remove an element
        
        # We'll try a few random moves
        moves = []
        
        # Add
        for _ in range(10):
            new_val = np.random.randint(1, max_val + 1)
            if new_val not in u_set:
                moves.append((u + [int(new_val)], "add"))
        
        # Replace
        for _ in range(10):
            idx = np.random.randint(0, n)
            new_val = np.random.randint(0, max_val + 1)
            if new_val not in u_set or new_val == u[idx]:
                new_u = u[:]
                new_u[idx] = int(new_val)
                # Ensure 0 is still in the set
                if 0 not in new_u:
                    # If we replaced 0, we must put it back or swap
                    if u[idx] == 0:
                        # Cannot replace 0 with non-0 if we must keep 0?
                        # The task says "containing 0".
                        # So if we remove 0, the set is invalid.
                        # So we only replace non-zero elements, or ensure 0 is present.
                        if 0 in u:
                             # If 0 was in u and we replaced a non-zero, 0 is still there.
                             # If we replaced 0, we are invalid.
                             pass
                if 0 in new_u:
                    moves.append((sorted(new_u), "replace"))
        
        # Remove
        if n > 1:
            for _ in range(5):
                idx = np.random.randint(0, n)
                if u[idx] != 0: # Don't remove 0
                    new_u = u[:]
                    new_u.pop(idx)
                    moves.append((sorted(new_u), "remove"))
        
        return moves

    # Search loop
    current_set = best_set[:]
    current_score = best_score
    no_improvement_count = 0
    max_no_improvement = 1000
    
    while time.time() - start_time < time_limit:
        max_val = max(current_set) * 2 + 100
        # Cap max_val to avoid huge search spaces if max grows too fast
        if max_val > 10000:
            max_val = 10000
            
        moves = try_perturb(current_set, max_val)
        
        found_improvement = False
        for move, _ in moves:
            s = get_score(move)
            if s > current_score:
                current_score = s
                current_set = move
                found_improvement = True
                no_improvement_count = 0
                
                # Update global best
                if s > best_score:
                    best_score = s
                    best_set = current_set[:]
                    
                    # Add to top sets
                    top_sets.append((s, current_set[:]))
                    # Keep only top 5
                    top_sets.sort(key=lambda x: x[0], reverse=True)
                    top_sets = top_sets[:5]
                
                break # Restart perturbation with new set
        
        if not found_improvement:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Try restarting from another top set
                if len(top_sets) > 1:
                    # Pick a random top set that is not the current one
                    candidates = [ts for ts in top_sets if ts[1] != current_set]
                    if candidates:
                        next_set = np.random.choice(len(candidates))
                        current_set = candidates[next_set][1][:]
                        current_score = candidates[next_set][0]
                        no_improvement_count = 0
                else:
                    # Stuck, maybe try a larger jump or break
                    # For now, just keep trying with larger max_val
                    max_val *= 2
                    if max_val > 100000:
                         break

    return best_set

