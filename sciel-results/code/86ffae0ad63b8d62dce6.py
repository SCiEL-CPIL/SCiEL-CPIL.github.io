import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill
    
    limit = 180
    
    # Precompute all forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    triples = []
    triple_set = set()
    for a in range(1, limit + 1):
        for b in range(a + 1, limit + 1):
            num = a * b
            den = b - a
            if num % den == 0:
                c = num // den
                if c > b and c <= limit:
                    t = (a, b, c)
                    if t not in triple_set:
                        triple_set.add(t)
                        triples.append(t)
    
    # For each vertex, precompute which triples it participates in
    vertex_triples = {v: [] for v in range(1, limit + 1)}
    for i, (a, b, c) in enumerate(triples):
        vertex_triples[a].append(i)
        vertex_triples[b].append(i)
        vertex_triples[c].append(i)
    
    # Precompute conflicts for fast lookup
    conflicts = {}
    for a in range(1, limit + 1):
        conflicts[a] = {}
        for b in range(a + 1, limit + 1):
            num = a * b
            den = b - a
            if num % den == 0:
                c = num // den
                if c > a and c != b and c <= limit:
                    conflicts[a][b] = c
    
    # Greedy initialization: add elements from largest to smallest
    S = set()
    for a in range(limit, 0, -1):
        can_add = True
        for b in S:
            if b > a:
                if b in conflicts[a] and conflicts[a][b] in S:
                    can_add = False
                    break
            elif b < a:
                if a in conflicts[b] and conflicts[b][a] in S:
                    can_add = False
                    break
        
        if can_add:
            S.add(a)
        
        if time.time() > deadline:
            break
    
    # Local search: minimize violated constraints (violations) while maximizing |S|
    # Objective: maximize |S| - lambda * violations, or equivalently minimize violations with penalty for small |S|
    
    def get_violations(S_set):
        """Return list of indices of violated triples"""
        vlist = []
        for i, (a, b, c) in enumerate(triples):
            if a in S_set and b in S_set and c in S_set:
                vlist.append(i)
        return vlist
    
    # Start with a valid solution from greedy
    # Now do local search: try moves that reduce violations or increase |S|
    
    best_S = set(S)
    best_size = len(S)
    
    improved = True
    stall_count = 0
    while improved and time.time() < deadline - 5:
        improved = False
        stall_count += 1
        
        current_violations = get_violations(S)
        
        # Try 1: Add an element that doesn't create violations
        for a in range(1, limit + 1):
            if a in S:
                continue
            can_add = True
            for b in S:
                if b > a:
                    if b in conflicts[a] and conflicts[a][b] in S:
                        can_add = False
                        break
                elif b < a:
                    if a in conflicts[b] and conflicts[b][a] in S:
                        can_add = False
                        break
            
            if can_add:
                S.add(a)
                improved = True
                break
        
        # Try 2: Remove an element that is in many violations, then try to add others
        if not improved and time.time() < deadline - 15:
            # Count how many violations each element in S participates in
            viol_count = {v: 0 for v in S}
            for vi in current_violations:
                a, b, c = triples[vi]
                viol_count[a] = viol_count.get(a, 0) + 1
                viol_count[b] = viol_count.get(b, 0) + 1
                viol_count[c] = viol_count.get(c, 0) + 1
            
            # Try removing the element with highest violation count
            if current_violations:
                # Sort by violation count descending
                candidates = sorted(S, key=lambda x: viol_count.get(x, 0), reverse=True)
                
                for remove_val in candidates[:20]:  # Try top 20 candidates
                    if time.time() > deadline - 18:
                        break
                    S.remove(remove_val)
                    
                    # Try adding any element not in S
                    found_add = False
                    for a in range(1, limit + 1):
                        if a in S:
                            continue
                        can_add = True
                        for b in S:
                            if b > a:
                                if b in conflicts[a] and conflicts[a][b] in S:
                                    can_add = False
                                    break
                            elif b < a:
                                if a in conflicts[b] and conflicts[b][a] in S:
                                    can_add = False
                                    break
                        
                        if can_add:
                            S.add(a)
                            found_add = True
                            improved = True
                            break
                    
                    if not found_add:
                        S.add(remove_val)
                    
                    if improved:
                        break
        
        # Try 3: 2-swap - remove two elements, add one or two
        if not improved and time.time() < deadline - 25:
            current_violations = get_violations(S)
            if current_violations:
                # Find pairs of elements that appear together in violations
                pair_count = {}
                for vi in current_violations:
                    a, b, c = triples[vi]
                    for x, y in [(a,b), (a,c), (b,c)]:
                        key = (min(x,y), max(x,y))
                        pair_count[key] = pair_count.get(key, 0) + 1
                
                # Try removing pairs with high co-occurrence
                top_pairs = sorted(pair_count.items(), key=lambda x: x[1], reverse=True)[:10]
                
                for (r1, r2), _ in top_pairs:
                    if time.time() > deadline - 28:
                        break
                    if r1 not in S or r2 not in S:
                        continue
                    
                    S.remove(r1)
                    S.remove(r2)
                    
                    # Try adding elements
                    added_any = False
                    for a in range(1, limit + 1):
                        if a in S:
                            continue
                        can_add = True
                        for b in S:
                            if b > a:
                                if b in conflicts[a] and conflicts[a][b] in S:
                                    can_add = False
                                    break
                            elif b < a:
                                if a in conflicts[b] and conflicts[b][a] in S:
                                    can_add = False
                                    break
                        
                        if can_add:
                            S.add(a)
                            added_any = True
                            improved = True
                            break
                    
                    if not added_any:
                        S.add(r1)
                        S.add(r2)
                    
                    if improved:
                        break
        
        # Track best solution
        if len(S) > best_size:
            best_S = set(S)
            best_size = len(S)
        
        if time.time() > deadline:
            break
    
    # Use best solution found
    if len(best_S) > len(S):
        S = best_S
    
    # Final polish: try to add any remaining elements that don't create violations
    for a in range(1, limit + 1):
        if a in S:
            continue
        can_add = True
        for b in S:
            if b > a:
                if b in conflicts[a] and conflicts[a][b] in S:
                    can_add = False
                    break
            elif b < a:
                if a in conflicts[b] and conflicts[b][a] in S:
                    can_add = False
                    break
        
        if can_add:
            S.add(a)
    
    # Verify no violations
    for (a, b, c) in triples:
        if a in S and b in S and c in S:
            S.remove(c)
    
    return list(S)
