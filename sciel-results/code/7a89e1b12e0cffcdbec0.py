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

    def local_search(S, max_iter=100):
        """2-opt style local search: try swapping one element for another."""
        if time.time() > deadline:
            return S
        S = set(S)
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            # Try removing each element in S
            for x in list(S):
                if time.time() > deadline:
                    return S
                S.remove(x)
                # Try adding each element not in S
                best_gain = -1
                best_add = None
                for y in range(1, N + 1):
                    if y not in S:
                        if can_add(y, S):
                            gain = 1  # We added one element
                            if gain > best_gain:
                                best_gain = gain
                                best_add = y
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
        if not S:
            S = set(best_set)
        
        # Adaptive kick size: increase when stuck
        kick_size = max(1, min(10, consecutive_no_improve // 5 + 1))
        
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
        
        # Local search to escape shallow local optima
        S = local_search(S, max_iter=50)
        
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
