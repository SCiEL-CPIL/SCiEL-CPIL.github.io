import time
import math
import random
import sys

# EVOLVE-BLOCK-START

def _precompute_triples(N=180):
    """
    Precompute all (a, b, c) with distinct a,b,c in [1,N] such that 1/a = 1/b + 1/c.
    Equation: 1/a = 1/b + 1/c => c = ab / (b-a).
    We require b > a and c to be an integer in [1, N] distinct from a and b.
    Note: Since 1/a = 1/b + 1/c, a must be smaller than both b and c.
    """
    triples = []
    seen = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    # Normalize triple to sorted tuple for deduplication if needed,
                    # but (a,b,c) structure is fixed by a < b.
                    # However, 1/a = 1/b + 1/c is not symmetric in b,c.
                    # But the set {a,b,c} is what matters for the constraint.
                    # The equation defines a specific relationship.
                    # Let's just store the set of elements.
                    key = frozenset((a, b, c))
                    if key not in seen:
                        seen.add(key)
                        triples.append((a, b, c))
    return triples

import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 88.0
    N = 180
    
    triples = _precompute_triples(N)
    T = len(triples)
    
    # Build index: element -> list of triple indices
    elem_triples = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(triples):
        elem_triples[a].append(idx)
        elem_triples[b].append(idx)
        elem_triples[c].append(idx)
    
    # Precompute degree of each element
    elem_degree = {i: len(elem_triples[i]) for i in range(1, N + 1)}
    
    best_set = set()
    best_size = 0
    
    def greedy_removal(random_seed, strategy):
        """
        Strategy 0: Remove element with max active triple count (ties broken randomly).
        Strategy 1: Remove element with max active triple count, prefer smaller numbers.
        Strategy 2: Remove element with max (active_count + degree) as tiebreaker.
        """
        random.seed(random_seed)
        S = set(range(1, N + 1))
        counts = [3] * T  # Initially all in S
        
        active_count = [0] * (N + 1)
        for t_idx in range(T):
            a, b, c = triples[t_idx]
            active_count[a] += 1
            active_count[b] += 1
            active_count[c] += 1
            
        # Use a heap for efficient max extraction: (-active_count, element)
        # We need to handle lazy deletion since active_count changes
        heap = [(-active_count[x], x) for x in range(1, N + 1)]
        heapq.heapify(heap)
        
        while True:
            if time.time() > deadline - 5.0:
                return S
                
            # Find element with max active_count among those in S
            x_to_remove = None
            max_ac = 0
            
            while heap:
                neg_ac, x = heapq.heappop(heap)
                ac = -neg_ac
                if x in S and ac == active_count[x]:
                    if ac > 0:
                        x_to_remove = x
                        max_ac = ac
                        break
                    else:
                        # No more active triples
                        return S
                # Else: stale entry, skip
            
            if x_to_remove is None:
                # Rebuild heap if we ran out
                heap = [(-active_count[x], x) for x in range(1, N + 1) if x in S]
                heapq.heapify(heap)
                if not heap:
                    return S
                neg_ac, x_to_remove = heapq.heappop(heap)
                if -neg_ac <= 0:
                    return S
                
            S.remove(x_to_remove)
            
            # Update counts and active_count for triples containing x_to_remove
            for t_idx in elem_triples[x_to_remove]:
                if counts[t_idx] == 3:
                    a, b, c = triples[t_idx]
                    active_count[a] -= 1
                    active_count[b] -= 1
                    active_count[c] -= 1
                    counts[t_idx] = 2
                    # Push updated values to heap
                    for y in (a, b, c):
                        if y in S:
                            heapq.heappush(heap, (-active_count[y], y))
                elif counts[t_idx] == 2:
                    counts[t_idx] = 1
                
        return S

    def local_search(S, random_seed):
        """
        Try to add back removed elements.
        """
        random.seed(random_seed)
        S = set(S)
        
        counts = [0] * T
        for x in S:
            for t_idx in elem_triples[x]:
                counts[t_idx] += 1
        
        removed = [x for x in range(1, N + 1) if x not in S]
        random.shuffle(removed)
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 5.0:
                break
            for x in removed:
                if x in S:
                    continue
                can_add = True
                for t_idx in elem_triples[x]:
                    if counts[t_idx] >= 2:
                        can_add = False
                        break
                if can_add:
                    S.add(x)
                    improved = True
                    for t_idx in elem_triples[x]:
                        counts[t_idx] += 1
        
        return S

    def swap_local_search(S, random_seed):
        """
        Try 2-opt moves: remove one element x from S, then add as many removed elements as possible.
        If we can add more than 1, accept the move (net gain > 0).
        """
        random.seed(random_seed)
        S = set(S)
        
        counts = [0] * T
        for x in S:
            for t_idx in elem_triples[x]:
                counts[t_idx] += 1
        
        removed = [x for x in range(1, N + 1) if x not in S]
        
        # Try each x in S as a candidate to remove
        # For efficiency, only try x that are "blocking" some removed elements
        # An element x blocks y if there exists a triple t containing both x and y,
        # and counts[t] == 2 (so adding y would make it 3).
        
        # Build a map: for each removed y, which elements in S block it?
        # blocker[x] = set of removed elements that x blocks
        # Actually, we want: for each x in S, which removed elements become addable if x is removed?
        
        # Let's just try a few random x's from S
        S_list = list(S)
        random.shuffle(S_list)
        
        # Limit number of swap attempts based on time
        max_swaps = 50
        for x in S_list[:max_swaps]:
            if time.time() > deadline - 5.0:
                break
                
            # Temporarily remove x
            S.remove(x)
            for t_idx in elem_triples[x]:
                counts[t_idx] -= 1
            
            # Try to add as many removed elements as possible
            added = []
            for y in removed:
                if y in S:
                    continue
                can_add = True
                for t_idx in elem_triples[y]:
                    if counts[t_idx] >= 2:
                        can_add = False
                        break
                if can_add:
                    S.add(y)
                    added.append(y)
                    for t_idx in elem_triples[y]:
                        counts[t_idx] += 1
            
            # If we added more than 1, accept (net gain)
            if len(added) > 1:
                # Keep the improvement
                pass
            else:
                # Revert: remove added elements, put x back
                for y in added:
                    S.remove(y)
                    for t_idx in elem_triples[y]:
                        counts[t_idx] -= 1
                S.add(x)
                for t_idx in elem_triples[x]:
                    counts[t_idx] += 1
        
        # Final pass: try to add any remaining removable elements
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 5.0:
                break
            for y in removed:
                if y in S:
                    continue
                can_add = True
                for t_idx in elem_triples[y]:
                    if counts[t_idx] >= 2:
                        can_add = False
                        break
                if can_add:
                    S.add(y)
                    improved = True
                    for t_idx in elem_triples[y]:
                        counts[t_idx] += 1
        
        return S

    # Run multiple restarts
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
            
        strategy = i % 3
        S_greedy = greedy_removal(i, strategy)
        S_local = local_search(S_greedy, i + 1000)
        S_final = swap_local_search(S_local, i + 2000)
        
        if len(S_final) > best_size:
            best_size = len(S_final)
            best_set = set(S_final)
            
    # Final validity check
    if not best_set:
        best_set = set(range(90, 181))
        
    valid = True
    for (a, b, c) in triples:
        if a in best_set and b in best_set and c in best_set:
            valid = False
            break
            
    if not valid:
        for (a, b, c) in triples:
            if a in best_set and b in best_set and c in best_set:
                best_set.remove(c)

    return list(best_set)

# EVOLVE-BLOCK-END
