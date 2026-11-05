import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 50.0 # Leave 10s margin
    
    # 1. Generate all forbidden triples (a, b, c) with a < b < c (or any order)
    # Condition: 1/a = 1/b + 1/c => a = bc / (b+c)
    # Parameterization:
    # b = k * x * (x + y)
    # c = k * y * (x + y)
    # a = k * x * y * (x + y)
    # with gcd(x, y) = 1, x < y, k >= 1
    
    triples = set()
    
    # Max value is 180
    for x in range(2, 181):
        # If x=1, a=c, not distinct. So x>=2.
        for y in range(x + 1, 181):
            if gcd(x, y) != 1:
                continue
            
            base_b = x * (x + y)
            base_c = y * (x + y)
            base_a = x * y * (x + y)
            
            # b, c, a must be <= 180
            # k * base_b <= 180 => k <= 180 // base_b
            # k * base_c <= 180 => k <= 180 // base_c
            # k * base_a <= 180 => k <= 180 // base_a
            
            max_k = min(180 // base_b, 180 // base_c, 180 // base_a)
            
            if max_k < 1:
                continue
                
            for k in range(1, max_k + 1):
                a = k * base_a
                b = k * base_b
                c = k * base_c
                
                # Check distinctness
                if a == b or b == c or a == c:
                    continue
                
                # Store as sorted tuple to handle permutations
                t = tuple(sorted((a, b, c)))
                triples.add(t)
                
            # Optimization: if base_b > 180, then for larger y, base_b will only increase.
            # base_b = x(x+y) = x^2 + xy. Increases with y.
            if base_b > 180:
                break
                
    # Convert to list for easier access
    triples_list = list(triples)
    
    # 2. Greedy Removal to find a valid set
    # We want to keep as many numbers as possible.
    # This is equivalent to finding a minimum vertex cover in the 3-uniform hypergraph.
    # Greedy: Repeatedly pick the vertex (number) that covers the most uncovered edges (triples).
    
    # Map number -> set of triples it belongs to
    node_to_triples = {i: [] for i in range(1, 181)}
    for t in triples_list:
        for node in t:
            node_to_triples[node].append(t)
    
    # Active triples: those not yet "hit" (broken) by a removed node
    # A triple is "active" if none of its nodes have been removed.
    # Initially all are active.
    active_triples = set(triples_list)
    
    removed = set()
    
    # To speed up, we can pre-compute degrees or just iterate.
    # Since N=180 and E is small, this is fast.
    
    while active_triples:
        # Check time
        if time.time() > deadline:
            break
            
        # Find the node that appears in the most active triples
        # We can iterate through all nodes 1..180 and count how many of their triples are in active_triples
        best_node = -1
        max_count = -1
        
        # Optimization: Only consider nodes that are part of at least one active triple
        # We can maintain a set of candidate nodes.
        # For simplicity and given small N, just loop 1..180
        for node in range(1, 181):
            if node in removed:
                continue
                
            count = 0
            for t in node_to_triples[node]:
                if t in active_triples:
                    count += 1
                    
            if count > max_count:
                max_count = count
                best_node = node
                
        if best_node == -1:
            # No node covers any active triple? This shouldn't happen if active_triples is not empty
            break
            
        # Remove the best node
        removed.add(best_node)
        
        # Update active triples: remove all triples containing best_node
        # This is O(E) per removal. Total O(V * E).
        # E is likely small (a few hundred). V is 180. 180 * 500 = 90,000 ops. Very fast.
        new_active = set()
        for t in active_triples:
            if best_node not in t:
                new_active.add(t)
        active_triples = new_active
        
    # 3. Construct result
    # The remaining set is [1..180] \ removed
    result = [i for i in range(1, 181) if i not in removed]
    
    # Final verification (safety net)
    # If the greedy loop timed out, we might have active triples remaining.
    # In that case, we must break them arbitrarily to ensure validity.
    # Let's do a simple pass to ensure no triples remain.
    # If active_triples is not empty, we just pick any node from each remaining triple and remove it.
    if active_triples:
        for t in active_triples:
            # Pick the first node in the triple
            n = t[0]
            if n not in removed:
                removed.add(n)
                # We don't need to update active_triples further if we just want validity,
                # but let's be safe.
    
    # Re-verify
    final_set = set(result)
    for t in triples_list:
        if all(x in final_set for x in t):
            # Still a valid triple? Remove one more.
            # This should not happen if the loop above finished or the safety net worked.
            # But if the safety net added nodes, `result` was calculated before that.
            pass
            
    # Recalculate result after safety net
    result = [i for i in range(1, 181) if i not in removed]
    
    # One last check to be absolutely sure
    # If any triple remains, remove one element from it.
    valid = True
    for t in triples_list:
        if all(x in result for x in t):
            valid = False
            # Remove the smallest element to maximize chance of keeping larger ones?
            # Or just the first one.
            rem_val = t[0]
            result = [x for x in result if x != rem_val]
            
    if not result:
        # Fallback: return a known valid set (e.g., just {180} or {1, 180})
        # {180} is always valid (no triples possible with 1 element)
        return [180]
        
    return result
