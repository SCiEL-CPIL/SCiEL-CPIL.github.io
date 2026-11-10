import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 60s limit
    
    N = 180
    full_set = set(range(1, N + 1))
    
    # Precompute all forbidden triples (a, b, c) with a < b < c, 1/a = 1/b + 1/c
    # 1/a = 1/b + 1/c => c = a*b / (b-a)
    triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            den = b - a
            num = a * b
            if num % den == 0:
                c = num // den
                if c > b and c <= N:
                    triples.append((a, b, c))
    
    # Precompute conflict data: for each x, the set of pairs that would form a triple with x
    # If x is in set S, then for each pair (u,v) in conflict_pairs[x], we must NOT have both u and v in S
    conflict_pairs = {x: [] for x in range(1, N + 1)}
    conflict_count = [0] * (N + 1)
    
    for a, b, c in triples:
        # a is blocked by pair (b, c)
        conflict_pairs[a].append((b, c))
        conflict_count[a] += 1
        # b is blocked by pair (a, c)
        conflict_pairs[b].append((a, c))
        conflict_count[b] += 1
        # c is blocked by pair (a, b)
        conflict_pairs[c].append((a, b))
        conflict_count[c] += 1
    
    # Also build: for each x, which triples contain x (for counting active conflicts)
    triples_with_x = {x: [] for x in range(1, N + 1)}
    for a, b, c in triples:
        triples_with_x[a].append((a, b, c))
        triples_with_x[b].append((a, b, c))
        triples_with_x[c].append((a, b, c))
    
    def is_valid_addition(x, S):
        """Check if x can be added to S without creating a forbidden triple."""
        for u, v in conflict_pairs[x]:
            if u in S and v in S:
                return False
        return True
    
    def count_active_conflicts(x, S):
        """Count how many triples containing x are fully active in S."""
        count = 0
        for a, b, c in triples_with_x[x]:
            others = [y for y in (a, b, c) if y != x]
            if others[0] in S and others[1] in S:
                count += 1
        return count
    
    def greedy_removal(tie_breaker):
        """Start from full set, repeatedly remove element in most active conflicts."""
        current = set(range(1, N + 1))
        
        while True:
            if time.time() > deadline - 5:
                break
            
            # Find all active triples
            active_triples = [t for t in triples if all(x in current for x in t)]
            if not active_triples:
                break
            
            # Count how many active triples each element is in
            elem_count = Counter()
            for t in active_triples:
                for x in t:
                    elem_count[x] += 1
            
            # Remove element with highest count (tie-break by tie_breaker)
            if elem_count:
                max_count = max(elem_count.values())
                candidates = [x for x, c in elem_count.items() if c == max_count]
                if tie_breaker == 'min':
                    worst_elem = min(candidates)
                elif tie_breaker == 'max':
                    worst_elem = max(candidates)
                elif tie_breaker == 'rand':
                    worst_elem = random.choice(candidates)
                else:
                    worst_elem = max(candidates, key=lambda x: (elem_count[x], -x))
                current.remove(worst_elem)
        
        return current
    
    def local_search_additions(S, time_limit):
        """Try to add back elements that no longer conflict."""
        S = set(S)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for y in range(1, N + 1):
                if y not in S and is_valid_addition(y, S):
                    S.add(y)
                    improved = True
        return S
    
    def local_search_swaps(S, time_limit):
        """Try swaps: remove one element from S, add one from outside."""
        S = set(S)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try to add any missing element
            added_any = False
            for y in range(1, N + 1):
                if y not in S and is_valid_addition(y, S):
                    S.add(y)
                    improved = True
                    added_any = True
                    break
            if added_any:
                continue
            
            # Try swaps: for each y not in S, find blockers and try removing one
            candidates_y = [y for y in range(1, N + 1) if y not in S]
            # Sort by conflict count (prefer adding low-conflict elements)
            candidates_y.sort(key=lambda y: conflict_count[y])
            
            for y in candidates_y:
                if time.time() > time_limit:
                    break
                
                # Find blockers: elements in S that form a triple with y and another in S
                blockers = set()
                for u, v in conflict_pairs[y]:
                    if u in S and v in S:
                        blockers.add(u)
                        blockers.add(v)
                
                if not blockers:
                    continue
                
                # Try removing each blocker
                for x in blockers:
                    if time.time() > time_limit:
                        break
                    S.remove(x)
                    if is_valid_addition(y, S):
                        S.add(y)
                        improved = True
                        break
                    S.add(x)  # restore
                if improved:
                    break
        
        return S
    
    best_set = set()
    best_size = 0
    
    random.seed(42)
    
    # Try multiple removal strategies
    strategies = ['min', 'max', 'rand', 'default']
    
    for strat in strategies:
        if time.time() > deadline - 10:
            break
        
        S = greedy_removal(strat)
        
        # Local search: additions then swaps
        S = local_search_additions(S, deadline - 5)
        S = local_search_swaps(S, deadline - 5)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Final additions pass
    best_set = local_search_additions(best_set, deadline - 2)
    
    return sorted(best_set)
