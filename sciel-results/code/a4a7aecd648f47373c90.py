import time
import random
import itertools

# Task: Find a nonempty subset of [1, 180] with no THREE DISTINCT elements a,b,c
# satisfying 1/a = 1/b + 1/c. Maximize cardinality.
# 1/a = 1/b + 1/c  <=>  bc = a(b+c)

def construct():
    deadline = time.time() + 55  # Safety margin for 60s limit
    
    N = 180
    universe = list(range(1, N + 1))
    
    # Precompute for each element x, the set of pairs (u,v) that would form a conflict
    # conflict_pairs[x] = set of frozenset({u,v}) such that {x,u,v} satisfies bc=a(b+c)
    # with x,u,v distinct
    conflict_pairs = {x: set() for x in range(1, N + 1)}
    
    # For each pair (u,v) with u<v, at most 2 x values conflict:
    # Role 1: x = uv/(u+v) if integer
    # Role 2: x = uv/(v-u) if integer
    
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            # Role 1: x = uv/(u+v)
            num1 = u * v
            den1 = u + v
            if num1 % den1 == 0:
                x1 = num1 // den1
                if 1 <= x1 <= N and x1 != u and x1 != v:
                    conflict_pairs[x1].add(frozenset((u, v)))
            
            # Role 2: x = uv/(v-u)
            num2 = u * v
            den2 = v - u
            if den2 > 0 and num2 % den2 == 0:
                x2 = num2 // den2
                if 1 <= x2 <= N and x2 != u and x2 != v:
                    conflict_pairs[x2].add(frozenset((u, v)))
    
    def is_conflict_with_addition(S, x):
        """Check if adding x to S creates a conflict."""
        for pair in conflict_pairs[x]:
            u, v = tuple(pair)
            if u in S and v in S:
                return True
        return False
    
    # Count conflicts for each element (for ordering heuristic)
    conflict_count = {x: len(conflict_pairs[x]) for x in range(1, N + 1)}
    
    best_set = set()
    best_size = 0
    
    def greedy_construct(order):
        S = set()
        for x in order:
            if not is_conflict_with_addition(S, x):
                S.add(x)
        return S
    
    def local_search(S):
        """Try to improve S by adding, swapping, or 2-for-1 replacements."""
        S = set(S)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Phase 1: Try adding each element not in S
            for x in range(1, N + 1):
                if x in S:
                    continue
                if not is_conflict_with_addition(S, x):
                    S.add(x)
                    improved = True
                    break  # Restart to try more additions
            
            if improved:
                continue
            
            # Phase 2: Try 1-for-1 swaps
            for x in range(1, N + 1):
                if x in S:
                    continue
                if not is_conflict_with_addition(S, x):
                    continue
                for y in list(S):
                    S.remove(y)
                    if not is_conflict_with_addition(S, x):
                        S.add(x)
                        improved = True
                        break
                    S.add(y)
                if improved:
                    break
            
            if improved:
                continue
            
            # Phase 3: Try 2-for-1 swaps (remove two, add one) - only if it increases size
            # This won't increase size, but might enable future additions
            # Actually, let's try: remove y1,y2, add x1,x2 where net gain >= 0
            # For now, just try removing pairs that block additions
            for x in range(1, N + 1):
                if x in S:
                    continue
                if not is_conflict_with_addition(S, x):
                    continue
                # Find pairs (y1,y2) whose removal allows adding x
                blocking_pairs = []
                for pair in conflict_pairs[x]:
                    u, v = tuple(pair)
                    if u in S and v in S:
                        blocking_pairs.append((u, v))
                
                if len(blocking_pairs) <= 2:
                    # Try removing all blockers
                    to_remove = set()
                    for u, v in blocking_pairs:
                        to_remove.add(u)
                        to_remove.add(v)
                    
                    if len(to_remove) <= 2:
                        for y in to_remove:
                            S.remove(y)
                        if not is_conflict_with_addition(S, x):
                            S.add(x)
                            improved = True
                            break
                        else:
                            # Restore
                            for y in to_remove:
                                S.add(y)
            
            if improved:
                continue
            
            # Phase 4: Try 1-for-2 swaps (remove one, add two)
            for x1 in range(1, N + 1):
                if x1 in S:
                    continue
                for x2 in range(x1 + 1, N + 1):
                    if x2 in S:
                        continue
                    # Check if we can add both x1 and x2 after removing one y
                    for y in list(S):
                        S.remove(y)
                        can_add_x1 = not is_conflict_with_addition(S, x1)
                        if can_add_x1:
                            S.add(x1)
                        can_add_x2 = not is_conflict_with_addition(S, x1 if can_add_x1 else S, x2)
                        # Simpler: check both independently after removal
                        S2 = set(S)
                        ok1 = not is_conflict_with_addition(S2, x1)
                        ok2 = not is_conflict_with_addition(S2, x2)
                        if ok1 and ok2:
                            # Also check x1 and x2 don't conflict with each other
                            # They don't directly conflict (need a third element)
                            S.add(x1)
                            S.add(x2)
                            improved = True
                            break
                        S.add(y)  # Restore
                    if improved:
                        break
                if improved:
                    break
        
        return S
    
    # Initial orderings
    orderings = []
    # Descending (large numbers first)
    orderings.append(sorted(universe, reverse=True))
    # Ascending
    orderings.append(sorted(universe))
    # By conflict count (fewest conflicts first)
    orderings.append(sorted(universe, key=lambda x: conflict_count[x]))
    # By conflict count (most conflicts first)
    orderings.append(sorted(universe, key=lambda x: -conflict_count[x]))
    # Random
    for _ in range(5):
        orderings.append(random.sample(universe, N))
    
    for order in orderings:
        S = greedy_construct(order)
        S = local_search(S)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
        
        if time.time() > deadline - 10:
            break
    
    # Randomized restarts with local search
    while time.time() < deadline - 5:
        order = random.sample(universe, N)
        S = greedy_construct(order)
        S = local_search(S)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # Final polish: try to add any remaining elements
    for x in range(1, N + 1):
        if x not in best_set and not is_conflict_with_addition(best_set, x):
            best_set.add(x)
    
    return list(best_set)
