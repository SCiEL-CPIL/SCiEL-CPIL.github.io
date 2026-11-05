import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    # The condition is: 1/a = 1/b + 1/c for distinct a, b, c in S.
    # This is equivalent to a(b+c) = bc, or (a-b)(a-c) = a^2.
    # Note: If a < b and a < c, then a is the smallest element in the triple.
    # Let's precompute all potential triples (a, b, c) with a < b < c such that 
    # 1/a = 1/b + 1/c.
    
    N = 180
    all_numbers = set(range(1, N + 1))
    
    # Precompute conflict triples
    # For each a, we need b, c such that b > a, c > b, and a(b+c) = bc.
    # => ab + ac = bc => ab = c(b-a) => c = ab / (b-a).
    # For c to be an integer, (b-a) must divide ab.
    # Let d = b-a. Then b = a+d. c = a(a+d)/d = a^2/d + a.
    # So d must be a divisor of a^2.
    # Also we need c <= N and c > b.
    # c > b => a^2/d + a > a+d => a^2/d > d => a^2 > d^2 => d < a.
    
    triples = []
    # Map each element to the list of triples it participates in
    # This helps in greedy removal
    
    # We will store triples as tuples (a, b, c) with a < b < c
    # And also a reverse index: element -> set of triple_indices
    
    # To be efficient, we only generate triples once.
    for a in range(1, N + 1):
        a_sq = a * a
        # d must be a divisor of a_sq and d < a
        # Also c = a + a_sq/d must be <= N
        # a + a_sq/d <= N => a_sq/d <= N - a => d >= a_sq / (N - a)
        
        min_d = max(1, math.ceil(a_sq / (N - a))) if N > a else a_sq + 1
        max_d = a - 1
        
        if min_d > max_d:
            continue
            
        # Find divisors of a_sq in [min_d, max_d]
        # Iterate divisors of a_sq
        # To avoid checking all numbers, we can iterate up to sqrt(a_sq)
        limit = int(math.isqrt(a_sq))
        divisors = []
        for d in range(1, limit + 1):
            if a_sq % d == 0:
                divisors.append(d)
                other = a_sq // d
                if other != d:
                    divisors.append(other)
                    
        for d in divisors:
            if min_d <= d <= max_d:
                b = a + d
                c = a + a_sq // d
                if b <= N and c <= N and b < c:
                    # Check distinctness: a < b < c is guaranteed by d>0 and d<a
                    triples.append((a, b, c))
                    
    # If no triples, return all
    if not triples:
        return list(range(1, N + 1))
        
    # Greedy Removal Strategy
    # Start with all elements
    S = set(range(1, N + 1))
    
    # Build conflict graph dynamically or use static?
    # Static triples are easier. 
    # We want to remove elements to break all triples.
    # This is a hitting set problem.
    # Heuristic: Remove the element in the "most active" triple.
    # "Active" means both other elements are still in S.
    
    # To speed up, maintain for each element the count of active triples it's in.
    # And for each triple, whether it is active.
    
    # Since N=180 is small, we can afford a simpler approach if optimized.
    # Let's try a local search / greedy removal.
    
    # Initialize state
    in_set = [True] * (N + 1)
    # active_triple_count[i] = number of triples containing i that are currently active (all 3 elements in S)
    active_triple_count = [0] * (N + 1)
    
    # Map element to list of triple indices
    elem_to_triples = {i: [] for i in range(1, N + 1)}
    
    for idx, t in enumerate(triples):
        a, b, c = t
        elem_to_triples[a].append(idx)
        elem_to_triples[b].append(idx)
        elem_to_triples[c].append(idx)
        # Initially all are in S, so all triples are active
        active_triple_count[a] += 1
        active_triple_count[b] += 1
        active_triple_count[c] += 1
        
    # If there are no active triples, we are done.
    # But initially there are.
    
    # We will remove elements one by one.
    # Which one? The one whose removal breaks the most active triples.
    # This is a greedy maximal independent set / vertex cover heuristic.
    
    # To avoid O(N^3) per step, we update counts.
    
    # However, simple greedy might get stuck in local minima.
    # Let's try multiple restarts with randomization or different tie-breaking.
    
    best_size = 0
    best_set = None
    
    num_restarts = 20 # Adjust based on time
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Reset state
        in_set = [True] * (N + 1)
        active_triple_count = [0] * (N + 1)
        
        for idx, t in enumerate(triples):
            a, b, c = t
            active_triple_count[a] += 1
            active_triple_count[b] += 1
            active_triple_count[c] += 1
            
        # Greedy removal loop
        # We maintain a list of active elements
        # At each step, pick element x with max active_triple_count[x].
        # If max count is 0, break.
        # Remove x.
        # Update counts for all triples involving x.
        
        # To make it faster, we can just recalculate counts for neighbors?
        # Or simply:
        # While there is an active triple:
        #   Find triple t that is active.
        #   Pick element in t with highest degree (in active subgraph).
        #   Remove it.
        
        # Let's use a priority queue or just scan. N=180 is small.
        # Scanning 180 elements is fast.
        
        active_elements = set(range(1, N + 1))
        
        # We need to know which triples are active.
        # A triple is active if all its elements are in active_elements.
        
        # Optimization: Instead of checking all triples, we can just look at the elements.
        # But we need to know if removing x breaks any active triple.
        
        # Let's stick to the count-based greedy.
        # Re-initialize counts for each restart is O(E). E is small (approx 180*180/2 worst case? No, much less).
        
        # Let's run the greedy
        while True:
            if time.time() > deadline:
                break
                
            # Find element with max active_triple_count
            # If all are 0, no active triples
            max_count = 0
            candidate = -1
            
            # Randomize tie-breaking for diversity
            candidates = []
            for i in active_elements:
                if active_triple_count[i] > max_count:
                    max_count = active_triple_count[i]
                    candidates = [i]
                elif active_triple_count[i] == max_count and max_count > 0:
                    candidates.append(i)
                    
            if not candidates:
                break
                
            x = random.choice(candidates)
            
            # Remove x
            active_elements.remove(x)
            in_set[x] = False
            
            # Update counts for triples involving x
            for idx in elem_to_triples[x]:
                a, b, c = triples[idx]
                # Check if this triple was active before removal
                # It is active if all three were in active_elements (before removing x)
                # Since x was in active_elements, we check the other two.
                # Note: The other two might have been removed earlier.
                # So we need to check if they are in active_elements NOW (after x removed, but they weren't x).
                # Actually, if the triple was active, it means a,b,c were all in S.
                # We are removing x.
                # The triple becomes inactive.
                # We must decrement the count for the *other* two elements IF the triple was active.
                
                # How to know if it was active?
                # It was active if the other two elements are still in active_elements.
                # (x is about to be removed, so we check the others).
                
                others = [y for y in (a, b, c) if y != x]
                if others[0] in active_elements and others[1] in active_elements:
                    # Triple was active
                    # Decrement count for others
                    active_triple_count[others[0]] -= 1
                    active_triple_count[others[1]] -= 1
                    # x's count doesn't matter anymore as it's removed
            
        current_size = len(active_elements)
        if current_size > best_size:
            best_size = current_size
            best_set = list(active_elements)
            
        # Optional: Local Search / Repair
        # Try to add back removed elements if possible.
        # This is complex to implement correctly with counts.
        # Given time constraints, the greedy hitting set is often good for small N.
        
        # Let's try a simple "add back" heuristic on the best set found so far
        # Actually, let's just do more restarts.
        
    if best_set is None:
        # Fallback: The current program's best known is 90 (91-180).
        # Let's ensure we return at least that.
        # The interval [91, 180] is known to be valid.
        return list(range(91, 181))
        
    return best_set

# Verification helper (not used in construct but good for debugging)
def _is_reciprocal_triple(a, b, c):
    return a * (b + c) == b * c

def _validate_full(subset):
    s = set(subset)
    n = len(s)
    # Check all triples? O(N^3) is too slow for verification if done naively, 
    # but for N=180, 180^3 is ~5M, which is fine in Python if optimized or just for small checks.
    # But we trust the construction.
    # Just a quick check for the final output.
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a, b, c = subset[i], subset[j], subset[k]
                # Assume a < b < c
                if _is_reciprocal_triple(a, b, c):
                    return False
    return True

# The construct function above is the main logic.
# One potential issue: The greedy removal might be too aggressive.
# Is there a better initial set?
# The set [91, 180] has size 90.
# The greedy starts with 180 and removes.
# If it removes 90, it's tied. If it removes fewer, it wins.
# The conflict graph for [1, 180] is sparse enough that greedy should do well.

# Let's refine the "Add Back" phase to improve quality.
# After greedy removal, we have a valid set S.
# We can try to add back elements one by one.
# For each removed element x, check if adding x to S creates any new triple.
# A new triple would be (x, y, z) where y, z in S and 1/x = 1/y + 1/z.
# Or (y, x, z) where y, z in S... wait, x must be the smallest in the triple 1/a=1/b+1/c.
# So if we add x, we only need to check triples where x is the smallest (a=x).
# i.e., 1/x = 1/b + 1/c with b, c in S.
# This is much faster to check.

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 180
    all_numbers = set(range(1, N + 1))
    
    # Precompute triples (a, b, c) with a < b < c
    triples = []
    elem_to_triples = {i: [] for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        a_sq = a * a
        if N <= a: continue
        min_d = max(1, math.ceil(a_sq / (N - a)))
        max_d = a - 1
        if min_d > max_d: continue
        
        limit = int(math.isqrt(a_sq))
        divisors = []
        for d in range(1, limit + 1):
            if a_sq % d == 0:
                divisors.append(d)
                other = a_sq // d
                if other != d:
                    divisors.append(other)
                    
        for d in divisors:
            if min_d <= d <= max_d:
                b = a + d
                c = a + a_sq // d
                if b <= N and c <= N and b < c:
                    triples.append((a, b, c))
                    idx = len(triples) - 1
                    elem_to_triples[a].append(idx)
                    elem_to_triples[b].append(idx)
                    elem_to_triples[c].append(idx)
                    
    if not triples:
        return list(range(1, N + 1))
        
    best_size = 0
    best_set = None
    
    num_restarts = 50 # Increase restarts, each is fast
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Greedy Removal
        in_set = [True] * (N + 1)
        active_triple_count = [0] * (N + 1)
        
        for idx, t in enumerate(triples):
            a, b, c = t
            active_triple_count[a] += 1
            active_triple_count[b] += 1
            active_triple_count[c] += 1
            
        active_elements = set(range(1, N + 1))
        
        while True:
            if time.time() > deadline:
                break
                
            max_count = 0
            candidates = []
            for i in active_elements:
                if active_triple_count[i] > max_count:
                    max_count = active_triple_count[i]
                    candidates = [i]
                elif active_triple_count[i] == max_count and max_count > 0:
                    candidates.append(i)
                    
            if not candidates:
                break
                
            x = random.choice(candidates)
            active_elements.remove(x)
            in_set[x] = False
            
            for idx in elem_to_triples[x]:
                a, b, c = triples[idx]
                others = [y for y in (a, b, c) if y != x]
                if others[0] in active_elements and others[1] in active_elements:
                    active_triple_count[others[0]] -= 1
                    active_triple_count[others[1]] -= 1
                    
        # Current valid set
        current_set = list(active_elements)
        
        # Local Repair: Try to add back removed elements
        # We iterate through removed elements in random order
        removed = [i for i in range(1, N + 1) if i not in active_elements]
        random.shuffle(removed)
        
        for x in removed:
            if time.time() > deadline:
                break
                
            # Check if adding x creates a triple where x is the smallest (a=x)
            # We need to check if there exist b, c in current_set such that 1/x = 1/b + 1/c
            # This is equivalent to checking triples (x, b, c) in our precomputed list.
            
            can_add = True
            for idx in elem_to_triples[x]:
                a, b, c = triples[idx]
                # Only care if x is the smallest 'a'
                if a != x:
                    continue
                # Check if b and c are in current_set
                if b in active_elements and c in active_elements:
                    can_add = False
                    break
                    
            if can_add:
                active_elements.add(x)
                in_set[x] = True
                # Note: We don't need to update active_triple_count for future additions 
                # because we are just doing a one-pass repair. 
                # However, adding x might enable adding other elements? 
                # No, adding an element only creates new conflicts, it doesn't resolve them. 
                # So we don't need to re-evaluate other removed elements based on this addition 
                # unless we are doing iterative repair. 
                # Since we are just adding back if safe *at this moment*, and adding x 
                # makes the set "more constrained" for other additions, the order matters.
                # Random shuffle handles this.
                
        current_size = len(active_elements)
        if current_size > best_size:
            best_size = current_size
            best_set = list(active_elements)
            
    if best_set is None:
        return list(range(91, 181))
        
    return best_set

