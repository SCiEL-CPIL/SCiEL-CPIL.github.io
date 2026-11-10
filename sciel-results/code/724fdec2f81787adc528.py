"""Rewritten program: ILS approach for large sets avoiding reciprocal Schur triples."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    N = 180
    
    # Precompute forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    # a = bc / (b+c)
    # Condition: (b+c) divides (b*c)
    
    a_to_bc = {i: [] for i in range(N + 1)}
    b_to_ac = {i: [] for i in range(N + 1)}
    c_to_ab = {i: [] for i in range(N + 1)}
    
    triples = []
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            s = b + c
            prod = b * c
            if prod % s == 0:
                a = prod // s
                if 1 <= a < b:
                    triples.append((a, b, c))
                    a_to_bc[a].append((b, c))
                    b_to_ac[b].append((a, c))
                    c_to_ab[c].append((a, b))
    
    # Use sets for fast membership testing
    # S is a set of integers
    
    def can_add(x, S):
        """Check if x can be added to set S without creating a forbidden triple."""
        # x as 'a'
        for (b, c) in a_to_bc[x]:
            if b in S and c in S:
                return False
        # x as 'b'
        for (a, c) in b_to_ac[x]:
            if a in S and c in S:
                return False
        # x as 'c'
        for (a, b) in c_to_ab[x]:
            if a in S and b in S:
                return False
        return True

    def is_valid(S):
        """Verify S has no forbidden triple."""
        for a in S:
            for (b, c) in a_to_bc[a]:
                if b in S and c in S:
                    return False
        return True

    def count_violations(S):
        """Count number of forbidden triples fully contained in S (for violation-based local search)."""
        cnt = 0
        for a in S:
            for (b, c) in a_to_bc[a]:
                if b in S and c in S:
                    cnt += 1
        return cnt

    def maximize(S):
        """Greedy addition until no more elements can be added."""
        improved = True
        while improved:
            if time.time() > deadline:
                return S
            improved = False
            # Try adding each element not in S
            for x in range(1, N + 1):
                if x not in S:
                    if can_add(x, S):
                        S.add(x)
                        improved = True
        return S

    def local_search_violation(S, max_iter=200):
        """Violation-minimizing local search: minimize number of violated triples,
        then maximize |S| among solutions with minimal violations.
        
        Objective: (violations, -|S|) lexicographic.
        Moves: add, remove, or swap elements to reduce violations first, then grow size.
        """
        if time.time() > deadline:
            return S
        S = set(S)
        cur_viol = count_violations(S)
        cur_size = len(S)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            
            # Try all single-element moves: add, remove, or swap
            # First pass: try to reduce violations (add or remove)
            for x in range(1, N + 1):
                if x not in S:
                    # Try adding x
                    S.add(x)
                    new_viol = count_violations(S)
                    if new_viol < cur_viol:
                        cur_viol = new_viol
                        cur_size = len(S)
                        improved = True
                    elif new_viol == cur_viol and len(S) > cur_size:
                        cur_size = len(S)
                        improved = True
                    else:
                        S.remove(x)
                else:
                    # Try removing x
                    S.remove(x)
                    new_viol = count_violations(S)
                    if new_viol < cur_viol:
                        cur_viol = new_viol
                        cur_size = len(S)
                        improved = True
                    else:
                        S.add(x)
            
            # Second pass: try swaps (remove one, add another) if still stuck
            if not improved:
                for x in list(S):
                    if time.time() > deadline:
                        break
                    S.remove(x)
                    base_viol = count_violations(S)
                    for y in range(1, N + 1):
                        if y not in S:
                            S.add(y)
                            new_viol = count_violations(S)
                            new_size = len(S)
                            # Accept if fewer violations, or same violations but we'll keep size same (swap)
                            if new_viol < base_viol:
                                # Good swap: reduces violations
                                cur_viol = new_viol
                                cur_size = new_size
                                improved = True
                                break
                            else:
                                S.remove(y)
                    else:
                        # No good swap found, put x back
                        S.add(x)
                        continue
                    break  # Found a good swap, x is already removed and y added
            
            if not improved:
                break
        
        return S

    def local_search_1swap(S, max_iter=100):
        """1-swap local search: try swapping one element for another to increase size while maintaining validity."""
        if time.time() > deadline:
            return S
        S = set(S)
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            # Try removing each element in S and adding a different one
            for x in list(S):
                if time.time() > deadline:
                    return S
                S.remove(x)
                # Try adding each element not in S
                best_add = None
                for y in range(1, N + 1):
                    if y not in S:
                        if can_add(y, S):
                            best_add = y
                            break
                if best_add is not None:
                    S.add(best_add)
                    improved = True
                else:
                    # Put x back if we couldn't improve
                    S.add(x)
            if not improved:
                break
        return S

    best_set = set()
    best_size = 0
    rng = random.Random(42)
    
    # Initial Greedy
    S = set()
    maximize(S)
    if len(S) > best_size:
        best_set = set(S)
        best_size = len(S)
    
    # Iterated Local Search
    consecutive_no_improve = 0
    while time.time() < deadline:
        # Perturb: Remove k random elements from current S (or best if S is empty)
        if not S or len(S) == 0:
            S = set(best_set)
        
        # Adaptive kick size: increase when stuck
        kick_size = max(1, min(15, consecutive_no_improve // 3 + 1))
        
        # Remove kick_size elements
        if len(S) > kick_size:
            elements_to_remove = rng.sample(list(S), kick_size)
            for e in elements_to_remove:
                S.remove(e)
        elif len(S) > 0:
            if len(S) <= kick_size:
                 S.clear()
        
        # Maximize again
        maximize(S)
        
        # Violation-based local search (primary)
        S = local_search_violation(S, max_iter=100)
        
        # Then 1-swap polish
        S = local_search_1swap(S, max_iter=50)
        
        # Update best
        if len(S) > best_size:
            best_set = set(S)
            best_size = len(S)
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            
        # If we have used all elements, stop
        if best_size == N:
            break

    # Final verification
    if not is_valid(best_set):
        # Fallback: use a known valid construction
        # All numbers > 90: if a,b,c > 90, then 1/a + 1/b < 2/90 = 1/45 < 1/90, 
        # so 1/a = 1/b + 1/c is impossible for a,b,c > 90
        best_set = set(range(91, N + 1))
        if not is_valid(best_set):
            best_set = {180}
            
    # One final polish: try to add any remaining elements
    if time.time() < deadline - 5:
        for x in range(1, N + 1):
            if x not in best_set:
                if can_add(x, best_set):
                    best_set.add(x)
        # Verify again
        if not is_valid(best_set):
            # Roll back to last valid state
            best_set = set(range(91, N + 1))
            
    return list(best_set)

# EVOLVE-BLOCK-END
