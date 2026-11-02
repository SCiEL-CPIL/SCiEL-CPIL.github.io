import time
import math
import random
import numpy as np

def _score_set_bitset(U):
    """
    Compute the score for a finite set U containing 0 using bitsets for efficiency.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)
    """
    U = sorted(set(U))
    if not U:
        return 0.0
    
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Bitset representation
    # Bit i is set if i is in the set.
    # We need bits up to 2*max_u for sums, and -max_u to max_u for diffs.
    # To handle negative differences, we shift by max_u.
    # Diff range: -max_u to max_u. Size: 2*max_u + 1.
    # Sum range: 0 to 2*max_u. Size: 2*max_u + 1.
    
    # Create bitsets
    sum_mask = 0
    diff_mask = 0
    
    # Optimization: We can build the bitsets by OR-ing shifted versions of the U bitset.
    # But a simpler way for small U is to just iterate pairs.
    # Given U is small (size < 20 usually), iterating pairs is fast enough if we use bit ops.
    
    # Let's represent U as a bitset too for potential optimization, 
    # but since n is small, direct pair iteration is O(n^2) which is fine.
    
    # However, the bottleneck in the original was np.unique. 
    # Bitsets avoid sorting/unique.
    
    # We'll construct the masks directly.
    # sum_mask will have bits set for all a+b
    # diff_mask will have bits set for all a-b + max_u (shifted)
    
    for a in U:
        for b in U:
            sum_mask |= (1 << (a + b))
            diff_mask |= (1 << (a - b + max_u))
    
    sumset_size = sum_mask.bit_count()
    diffset_size = diff_mask.bit_count()
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    if ratio == 1.0:
        return 1.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_denom

def _local_search(U, deadline):
    """
    Perform local search starting from set U.
    Moves: 
    1. Change value of an existing element.
    2. Add a new element.
    3. Remove an element (if size > 1).
    """
    U = sorted(set(U))
    best_score = _score_set_bitset(U)
    best_U = U[:]
    
    # Determine search bounds for new values
    # We want to explore values that are not too large, but can grow if beneficial.
    # Start with a bound based on current max.
    max_u = max(best_U) if best_U else 1
    search_max = max(50, 2 * max_u + 10)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_move = None
        best_new_score = best_score
        
        n = len(best_U)
        
        # 1. Try changing existing elements
        for i in range(n):
            old_val = best_U[i]
            # Try a range of new values around old_val and globally
            # To keep it fast, we don't try EVERY value up to search_max.
            # We try a sampled set or a local window.
            # Let's try a local window and some random global values.
            candidates = set()
            
            # Local window
            low = max(1, old_val - 5)
            high = min(search_max, old_val + 5)
            candidates.update(range(low, high + 1))
            
            # Random global samples
            for _ in range(10):
                candidates.add(random.randint(1, search_max))
            
            # Always try 1 and small primes?
            candidates.update([1, 2, 3, 4, 5, 10, 20, 50, 100])
            
            for new_val in candidates:
                if new_val == old_val:
                    continue
                if new_val == 0:
                    continue # 0 is already in U, and we assume 0 is fixed at index 0?
                             # Actually, the problem says "containing 0". 
                             # If we move the element that is 0, we must ensure 0 is still present.
                             # Let's enforce that 0 is always in the set.
                             # So we only move elements that are NOT 0.
                             # If U[0] is 0, we don't move it.
                             # If we remove an element, we don't remove 0.
                             
                # Check for duplicates
                if new_val in best_U:
                    # If it's a duplicate of another element, the set size decreases or stays same.
                    # If we are changing U[i] to a value already in U, it's effectively a removal of U[i].
                    # We handle removals separately, so let's skip if new_val is in U (and new_val != old_val).
                    continue
                
                new_U = best_U[:]
                new_U[i] = new_val
                new_U.sort()
                
                # Ensure 0 is in the set
                if 0 not in new_U:
                    continue
                    
                s = _score_set_bitset(new_U)
                if s > best_new_score + 1e-9:
                    best_new_score = s
                    best_move = ('change', i, new_val)
        
        # 2. Try adding a new element
        if len(best_U) < 15: # Limit size to keep search space manageable
            candidates = set()
            # Random samples
            for _ in range(20):
                candidates.add(random.randint(1, search_max))
            # Some structured candidates
            candidates.update([1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500])
            # Gaps in current set?
            current_vals = set(best_U)
            # Try values just outside current range
            if best_U:
                candidates.add(max(best_U) + 1)
                candidates.add(max(best_U) + 2)
                candidates.add(max(best_U) + 5)
            
            for new_val in candidates:
                if new_val in best_U:
                    continue
                if new_val == 0:
                    continue
                new_U = best_U + [new_val]
                new_U.sort()
                s = _score_set_bitset(new_U)
                if s > best_new_score + 1e-9:
                    best_new_score = s
                    best_move = ('add', new_val)
        
        # 3. Try removing an element
        if len(best_U) > 2: # Keep at least {0, x, y}
            for i in range(len(best_U)):
                if best_U[i] == 0:
                    continue
                new_U = best_U[:i] + best_U[i+1:]
                s = _score_set_bitset(new_U)
                if s > best_new_score + 1e-9:
                    best_new_score = s
                    best_move = ('remove', i)
        
        if best_move:
            improved = True
            if best_move[0] == 'change':
                _, i, new_val = best_move
                best_U[i] = new_val
                best_U.sort()
            elif best_move[0] == 'add':
                _, new_val = best_move
                best_U.append(new_val)
                best_U.sort()
            elif best_move[0] == 'remove':
                _, i = best_move
                best_U.pop(i)
            
            best_score = best_new_score
            # Update search bounds
            max_u = max(best_U) if best_U else 1
            search_max = max(search_max, 2 * max_u + 10)
            # Cap search_max to avoid infinite growth
            search_max = min(search_max, 2000)
        else:
            # No improvement found, try a perturbation to escape local optimum
            # Randomly modify the set
            if random.random() < 0.5:
                # Add a random element
                if len(best_U) < 15:
                    new_val = random.randint(1, search_max)
                    if new_val not in best_U and new_val != 0:
                        best_U.append(new_val)
                        best_U.sort()
                        best_score = _score_set_bitset(best_U)
                else:
                    # Change a random element
                    i = random.randint(1, len(best_U)-1) # Don't change 0
                    new_val = random.randint(1, search_max)
                    if new_val not in best_U:
                        best_U[i] = new_val
                        best_U.sort()
                        best_score = _score_set_bitset(best_U)
            else:
                # Remove a random element
                if len(best_U) > 2:
                    i = random.randint(1, len(best_U)-1)
                    best_U.pop(i)
                    best_score = _score_set_bitset(best_U)
                
            # If perturbation didn't help or made it worse, we might get stuck.
            # For simplicity, we just break and rely on restarts.
            break
            
    return best_U, best_score

def _generate_heuristic_seeds():
    """Generate a variety of initial sets for local search."""
    seeds = []
    
    # Small sets
    seeds.append([0, 1, 2])
    seeds.append([0, 1, 3])
    seeds.append([0, 1, 4])
    seeds.append([0, 2, 3])
    seeds.append([0, 1, 2, 4])
    seeds.append([0, 1, 2, 3, 4])
    
    # Geometric progressions
    for base in [2, 3, 4]:
        U = [0]
        val = 1
        while val < 100:
            U.append(val)
            val *= base
        seeds.append(U)
        
    # Random sets
    for _ in range(20):
        size = random.randint(3, 10)
        max_val = random.randint(20, 200)
        U = [0] + random.sample(range(1, max_val + 1), size - 1)
        U.sort()
        seeds.append(U)
        
    # Sets with gaps
    for _ in range(10):
        U = [0]
        val = 1
        for _ in range(random.randint(2, 5)):
            val += random.randint(1, 10)
            U.append(val)
        U.sort()
        seeds.append(U)
        
    return seeds

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    best_U = [0, 1, 4]
    best_score = _score_set_bitset(best_U)
    
    seeds = _generate_heuristic_seeds()
    
    # Randomly shuffle seeds to ensure diversity in case time runs out
    random.shuffle(seeds)
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        # Run local search from this seed
        U_sol, s_sol = _local_search(seed, deadline)
        
        if s_sol > best_score:
            best_score = s_sol
            best_U = U_sol[:]
            
        # If we have a very good score, we might stop early?
        # Theoretical max is unbounded? No, log ratio is bounded.
        # Ratio |U-U|/|U+U| can be large.
        # For arithmetic progression, |U-U| ~ n, |U+U| ~ n, ratio ~ 1.
        # For geometric progression, |U-U| ~ n^2, |U+U| ~ n^2, ratio ~ 1.
        # To maximize ratio, we want |U-U| large and |U+U| small.
        # Small sumset implies structure (e.g. AP). Large diffset implies spread.
        # This is a trade-off.
        
    # Final polish: Run a few more local searches from the best found so far
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        for _ in range(5):
            if time.time() > deadline:
                break
            # Perturb best_U
            U_perturbed = best_U[:]
            # Randomly add/remove/change
            op = random.choice(['add', 'remove', 'change'])
            if op == 'add' and len(U_perturbed) < 15:
                U_perturbed.append(random.randint(1, 2 * max(U_perturbed) + 10))
            elif op == 'remove' and len(U_perturbed) > 2:
                idx = random.randint(1, len(U_perturbed)-1)
                U_perturbed.pop(idx)
            elif op == 'change' and len(U_perturbed) > 1:
                idx = random.randint(1, len(U_perturbed)-1)
                U_perturbed[idx] = random.randint(1, 2 * max(U_perturbed) + 10)
            
            U_perturbed.sort()
            U_sol, s_sol = _local_search(U_perturbed, deadline)
            if s_sol > best_score:
                best_score = s_sol
                best_U = U_sol[:]
                
    return sorted(set(best_U))

# Test
if __name__ == "__main__":
    start = time.time()
    U = construct()
    end = time.time()
    score = _score_set_bitset(U)
    print(f"Time: {end-start:.2f}s")
    print(f"Set: {U}")
    print(f"Score: {score}")
