import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 85  # Safety margin
    
    N = 180
    
    # 1. Generate all forbidden triples (a, b, c) distinct in [1, N] with 1/a = 1/b + 1/c
    # Using a = bc/(b+c) => (b-a)(c-a) = a^2
    triples = []
    seen = set()
    
    for a in range(1, N + 1):
        a_sq = a * a
        # Find factors k, m of a_sq such that k*m = a_sq
        # b = a + k, c = a + m
        # We need b, c <= N and b != c
        for k in range(1, int(a_sq**0.5) + 1):
            if a_sq % k == 0:
                m = a_sq // k
                b1 = a + k
                c1 = a + m
                if b1 <= N and c1 <= N and b1 != c1 and b1 != a and c1 != a:
                    t = tuple(sorted([a, b1, c1]))
                    if t not in seen:
                        seen.add(t)
                        triples.append(t)
                
                # Check the other orientation if k != m (already covered by sorted tuple, 
                # but we must ensure we don't miss cases where b,c are swapped? 
                # No, the triple {a,b,c} is the same. 
                # The equation is symmetric in b and c.
                # So generating (k,m) and (m,k) yields the same set {a, a+k, a+m}.
                # Since we sort the triple, duplicates are handled by 'seen'.
    
    # 2. Precompute conflict pairs for each element
    # conflict_pairs[x] = list of (y, z) such that {x, y, z} is a forbidden triple
    conflict_pairs = defaultdict(list)
    for t in triples:
        a, b, c = t
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))
    
    def count_violations(S_set):
        count = 0
        for t in triples:
            if all(x in S_set for x in t):
                count += 1
        return count
    
    def can_add(x, S_set):
        """Check if x can be added to S_set without creating a violation"""
        for (y, z) in conflict_pairs[x]:
            if y in S_set and z in S_set:
                return False
        return True

    # 3. Initial Greedy Construction
    # Strategy: Start with all elements, remove those in most violated triples
    S = set(range(1, N + 1))
    
    # Calculate initial violations
    violated_triples = []
    for t in triples:
        if all(x in S for x in t):
            violated_triples.append(t)
            
    # Greedy removal
    while violated_triples and time.time() < deadline - 10:
        elem_count = defaultdict(int)
        for t in violated_triples:
            for x in t:
                if x in S:
                    elem_count[x] += 1
        
        if not elem_count:
            break
            
        # Pick element with max count
        best_elem = max(elem_count, key=elem_count.get)
        S.discard(best_elem)
        
        # Update violated triples
        new_violated = []
        for t in triples:
            if all(x in S for x in t):
                new_violated.append(t)
        violated_triples = new_violated
        
    best_set = set(S)
    best_size = len(S)
    
    # 4. Greedy Addition
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for x in range(1, N + 1):
            if x not in best_set and can_add(x, best_set):
                best_set.add(x)
                improved = True
                best_size = len(best_set)
                
    # 5. Simulated Annealing Local Search
    random.seed(42)
    current_set = set(best_set)
    current_violations = count_violations(current_set)
    
    temp = 5.0
    cooling = 0.995
    min_temp = 0.1
    
    # Precompute list of elements for random selection
    all_elements = list(range(1, N + 1))
    
    for sa_iter in range(5000):
        if time.time() > deadline - 10:
            break
            
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add a random element
            not_in_set = [x for x in all_elements if x not in current_set]
            if not not_in_set:
                continue
            x = random.choice(not_in_set)
            
            # Calculate change in violations
            new_violations = 0
            for (y, z) in conflict_pairs[x]:
                if y in current_set and z in current_set:
                    new_violations += 1
            
            if new_violations == 0:
                current_set.add(x)
                current_violations = 0 # Actually, if it was 0 before, it stays 0. 
                # Wait, if current_violations > 0, adding x might not change it if x wasn't in the violating triples.
                # But can_add checks if x creates NEW violations.
                # If current_set already has violations, adding x might add more.
                # My can_add logic: returns False if x creates a violation.
                # It does NOT check if x resolves a violation (impossible for addition).
                # So if can_add is True, new_violations added is 0.
                # So current_violations remains the same.
                if len(current_set) > best_size and current_violations == 0:
                    best_set = set(current_set)
                    best_size = len(current_set)
            else:
                # Accept with probability
                delta_viol = new_violations
                if random.random() < (1.0 / (1.0 + delta_viol)) * (temp / 5.0):
                    current_set.add(x)
                    current_violations += delta_viol
                    
        elif move_type < 0.8:
            # Try to remove a random element
            if not current_set:
                continue
            x = random.choice(list(current_set))
            
            # Calculate change in violations
            # Removing x might resolve violations involving x
            resolved = 0
            for (y, z) in conflict_pairs[x]:
                if y in current_set and z in current_set:
                    resolved += 1
            
            current_set.discard(x)
            current_violations -= resolved
            
            # After removal, try to greedily add back as many as possible
            # This is a "shake" move
            added_back = 0
            for y in all_elements:
                if y not in current_set and can_add(y, current_set):
                    current_set.add(y)
                    added_back += 1
            
            if len(current_set) > best_size and current_violations == 0:
                best_set = set(current_set)
                best_size = len(current_set)
                
        else:
            # Try a swap: remove x, add y
            if len(current_set) < 1:
                continue
            x = random.choice(list(current_set))
            not_in_set = [z for z in all_elements if z not in current_set]
            if not not_in_set:
                continue
            y = random.choice(not_in_set)
            
            # Tentative swap
            old_set = set(current_set)
            current_set.discard(x)
            current_set.add(y)
            
            new_violations = count_violations(current_set)
            
            if new_violations < current_violations or (new_violations == current_violations and len(current_set) > best_size and new_violations == 0):
                current_violations = new_violations
                if len(current_set) > best_size and current_violations == 0:
                    best_set = set(current_set)
                    best_size = len(current_set)
            else:
                # Revert
                current_set = old_set
        
        temp *= cooling
        
    # 6. Final Polish
    # Ensure best_set is valid and maximal
    # Remove any violations from best_set
    violated = [t for t in triples if all(x in best_set for x in t)]
    while violated:
        elem_count = defaultdict(int)
        for t in violated:
            for x in t:
                elem_count[x] += 1
        if not elem_count:
            break
        best_elem = max(elem_count, key=elem_count.get)
        best_set.discard(best_elem)
        violated = [t for t in triples if all(x in best_set for x in t)]
        
    # Add back any possible elements
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        for x in range(1, N + 1):
            if x not in best_set and can_add(x, best_set):
                best_set.add(x)
                changed = True
                
    return list(best_set)
