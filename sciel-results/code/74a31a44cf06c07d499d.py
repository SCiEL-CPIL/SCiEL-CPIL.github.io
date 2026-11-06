import math
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    N = 512
    MAX_A = 511
    deadline = start_time + 110.0
    
    # Precompute prime factors for each number 1..511
    factors = [set() for _ in range(MAX_A + 1)]
    factors[1] = set()
    for i in range(2, MAX_A + 1):
        if not factors[i]:
            for j in range(i, MAX_A + 1, i):
                factors[j].add(i)
    
    # Weights: w[a] = 1/(512-a)
    # Use Fraction for exactness
    weights = [Fraction(0)] * (MAX_A + 1)
    for a in range(1, MAX_A + 1):
        weights[a] = Fraction(1, N - a)
        
    # Build conflict: conflict[a] is set of b such that gcd(a,b) > 1
    conflict = [set() for _ in range(MAX_A + 1)]
    for a in range(1, MAX_A + 1):
        for b in range(a + 1, MAX_A + 1):
            if factors[a] & factors[b]:
                conflict[a].add(b)
                conflict[b].add(a)
                
    # Sort vertices by weight descending
    # weights are 1/(512-a), so larger a -> larger weight
    vertices = list(range(MAX_A, 0, -1))
    
    # Greedy Initial Solution
    chosen = set()
    for a in vertices:
        if all(a not in conflict[b] for b in chosen):
            chosen.add(a)
            
    best_score = sum(weights[a] for a in chosen)
    best_set = set(chosen)
    
    # Helper: Check if a can be added to a set
    def can_add(a, s):
        for b in s:
            if a in conflict[b]:
                return False
        return True

    # Local Search
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        current_set = set(best_set)
        current_score = best_score
        
        # 1. Try adding any valid node
        # Find all nodes not in current_set that are coprime to all in current_set
        valid_adds = []
        for a in range(1, MAX_A + 1):
            if a not in current_set and can_add(a, current_set):
                valid_adds.append(a)
                
        if valid_adds:
            # Add the one with highest weight (which is the largest a)
            # Since weights are increasing with a, just pick max
            a_add = max(valid_adds)
            current_set.add(a_add)
            current_score += weights[a_add]
            improved = True
            continue # Restart loop as set changed
            
        # 2. Try 1-opt swaps (remove u, add v)
        # For each u in current_set, find best v not in current_set
        # such that v is coprime to current_set \ {u}
        # and weight(v) > weight(u)
        swap_done = False
        for u in list(current_set):
            best_v = None
            best_gain = Fraction(0)
            # Precompute conflict with current_set \ {u}
            # We can just check can_add(v, current_set - {u})
            rest_set = current_set - {u}
            for v in range(MAX_A, 0, -1): # Iterate from highest weight
                if v in current_set:
                    continue
                if can_add(v, rest_set):
                    gain = weights[v] - weights[u]
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                        break # Found best v for this u (highest weight)
                        
            if best_v is not None and best_gain > 0:
                current_set.remove(u)
                current_set.add(best_v)
                current_score += best_gain
                improved = True
                swap_done = True
                break # Restart loop
                
        if swap_done:
            continue
            
        # 3. Try 2-opt swaps (remove u, v; add x, y)
        # Iterate over all pairs u, v in current_set
        # For each pair, find best pair x, y not in current_set
        # such that x, y coprime to current_set \ {u, v} and to each other
        # and weight(x) + weight(y) > weight(u) + weight(v)
        
        # Optimization: Precompute for each pair u,v the best x,y?
        # Given small size, we can just do it.
        two_opt_done = False
        for u, v in combinations(current_set, 2):
            rest_set = current_set - {u, v}
            # Find all candidates z not in current_set, coprime to rest_set
            candidates = []
            for z in range(MAX_A, 0, -1):
                if z in current_set:
                    continue
                if can_add(z, rest_set):
                    candidates.append(z)
                    
            if len(candidates) < 2:
                continue
                
            # Find best pair x, y from candidates such that gcd(x,y)=1
            # and weight(x)+weight(y) > weight(u)+weight(v)
            # Sort candidates by weight desc (already sorted by index desc)
            # We want max sum.
            # Check pairs
            target_score = weights[u] + weights[v]
            best_pair_score = Fraction(0)
            best_x, best_y = None, None
            
            # Efficient search for best pair in candidates
            # Since candidates are sorted by weight desc, we can try to find 
            # if candidates[0] + candidates[i] > target
            # Actually, we just need ANY pair that improves.
            # To find the BEST pair to ensure convergence, we might need more.
            # But any improvement is fine for hill climbing.
            
            # Let's just find the best pair.
            # Candidates size <= 512. Pairs <= 512*511/2 ~ 130k.
            # This is too slow if done for every u,v pair (80*79/2 ~ 3000 pairs).
            # 3000 * 130k = 400M ops. Too slow.
            
            # Optimization:
            # For a fixed rest_set, the set of valid candidates is fixed.
            # We can precompute the best pair of coprime numbers from the candidate set?
            # No, candidate set depends on rest_set.
            
            # Alternative:
            # Iterate x in candidates (high weight first).
            # For each x, find best y in candidates (x+1..end) such that gcd(x,y)=1.
            # Since candidates are sorted by weight desc, the best y for a given x 
            # is the first one in candidates that is coprime to x.
            
            found_improvement = False
            for i in range(len(candidates)):
                x = candidates[i]
                # Find best y > x in candidates (index > i) coprime to x
                # We want max weight(y), so scan from j=i+1 upwards
                for j in range(i + 1, len(candidates)):
                    y = candidates[j]
                    if x in conflict[y]:
                        continue
                    # If we find a coprime y, it's the best y for this x 
                    # because candidates are sorted by weight desc.
                    score = weights[x] + weights[y]
                    if score > target_score:
                        best_x, best_y = x, y
                        best_pair_score = score
                        found_improvement = True
                        break
                    # If score <= target, we can break? 
                    # No, a smaller y might allow a larger x? 
                    # No, x is fixed. y is decreasing. So score decreases.
                    # So if best y for x doesn't beat target, no y for x will.
                    break 
                    
                if found_improvement:
                    break
            # Wait, the logic above assumes that for a fixed x, the best y is the first coprime one.
            # And if that pair doesn't beat target, no other y for this x will.
            # This is correct.
            # So we just need to check if the best pair for the best x works?
            # No, maybe x[0] doesn't work with any y, but x[1] does.
            # The loop above checks x[0], finds best y. If fails, breaks inner loop.
            # Then moves to x[1].
            # This is O(C^2) worst case, but breaks early.
            
            if found_improvement:
                current_set.remove(u)
                current_set.remove(v)
                current_set.add(best_x)
                current_set.add(best_y)
                current_score = current_score - target_score + best_pair_score
                improved = True
                two_opt_done = True
                break
                
        if two_opt_done:
            continue
            
        # 4. Try 3-opt swaps (remove u, v, w; add x, y, z)
        # Similar to 2-opt, but with 3 elements.
        # This is much more expensive.
        # Given time constraints, 2-opt might be sufficient.
        # If we have time left, we can add 3-opt.
        
        # Update best if improved
        if current_score > best_score:
            best_score = current_score
            best_set = current_set
        else:
            # If no improvement found in this iteration, we are at a local optimum
            # for these move types.
            break

    return list(best_set)
