import time
import math
import random
import sys
import heapq

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
                    key = frozenset((a, b, c))
                    if key not in seen:
                        seen.add(key)
                        triples.append((a, b, c))
    return triples

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 180
    
    triples = _precompute_triples(N)
    T = len(triples)
    
    # Build index: element -> list of triple indices
    elem_triples = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(triples):
        elem_triples[a].append(idx)
        elem_triples[b].append(idx)
        elem_triples[c].append(idx)
    
    best_set = set()
    best_size = 0
    
    def greedy_removal(random_seed, strategy):
        """
        Greedy removal using priority queue for efficiency.
        """
        random.seed(random_seed)
        S = set(range(1, N + 1))
        counts = [3] * T  # All triples start with count 3
        
        active_count = [0] * (N + 1)
        for t_idx in range(T):
            a, b, c = triples[t_idx]
            active_count[a] += 1
            active_count[b] += 1
            active_count[c] += 1
            
        # Use a max-heap (negate for min-heap behavior)
        # Heap entries: (-active_count, x)
        # We'll use lazy deletion: push new entries, skip stale ones on pop
        heap = []
        for x in range(1, N + 1):
            if active_count[x] > 0:
                heapq.heappush(heap, (-active_count[x], x))
                
        removed_from_S = set()
        
        while heap:
            if time.time() > deadline - 5.0:
                return S
                
            neg_ac, x = heapq.heappop(heap)
            ac = -neg_ac
            
            # Lazy deletion check
            if x not in S or active_count[x] != ac:
                continue
                
            if ac == 0:
                break  # No more active triples involving remaining elements
                
            # Choose removal
            if strategy == 0:
                x_to_remove = x
            elif strategy == 1:
                # Collect all with same max ac for random tie-breaking
                candidates = [x]
                # Scan for others with same ac (only among those in S with active_count == ac)
                for y in range(1, N + 1):
                    if y != x and y in S and active_count[y] == ac:
                        candidates.append(y)
                x_to_remove = random.choice(candidates)
            else:
                x_to_remove = x
                
            S.discard(x_to_remove)
            
            # Update counts and active_count
            for t_idx in elem_triples[x_to_remove]:
                if counts[t_idx] == 3:
                    a, b, c = triples[t_idx]
                    active_count[a] -= 1
                    active_count[b] -= 1
                    active_count[c] -= 1
                    counts[t_idx] = 2
                    # Push updated entries for a, b, c if they're still in S
                    for y in (a, b, c):
                        if y in S and active_count[y] > 0:
                            heapq.heappush(heap, (-active_count[y], y))
                elif counts[t_idx] == 2:
                    counts[t_idx] = 1
                    
        return S

    def local_search_add(S, random_seed):
        """
        Try to add back removed elements greedily.
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

    def two_opt_swap(S, random_seed):
        """
        Try 2-opt moves: remove one element from S, then add as many removed elements as possible.
        Accept if net gain > 0.
        """
        random.seed(random_seed)
        S = set(S)
        
        counts = [0] * T
        for x in S:
            for t_idx in elem_triples[x]:
                counts[t_idx] += 1
                
        removed = [x for x in range(1, N + 1) if x not in S]
        if not removed or not S:
            return S
            
        # Try a limited number of 2-opt attempts
        max_attempts = 30
        attempts = 0
        best_S = set(S)
        best_size = len(S)
        
        while attempts < max_attempts:
            if time.time() > deadline - 5.0:
                break
            attempts += 1
            
            # Pick a random element to remove from S
            x_remove = random.choice(list(S))
            S.discard(x_remove)
            
            # Update counts
            for t_idx in elem_triples[x_remove]:
                counts[t_idx] -= 1
                
            # Try to add as many removed elements as possible
            added = 0
            random.shuffle(removed)
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
                    added += 1
                    for t_idx in elem_triples[y]:
                        counts[t_idx] += 1
                    
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
            else:
                # Revert
                S = set(best_S)
                # Rebuild counts from best_S
                counts = [0] * T
                for x in best_S:
                    for t_idx in elem_triples[x]:
                        counts[t_idx] += 1
                # Rebuild removed
                removed = [x for x in range(1, N + 1) if x not in best_S]
                
        return best_S

    # Run multiple restarts
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > deadline - 3.0:
            break
            
        strategy = i % 3
        S_greedy = greedy_removal(i, strategy)
        S_added = local_search_add(S_greedy, i + 1000)
        S_final = two_opt_swap(S_added, i + 2000)
        
        if len(S_final) > best_size:
            best_size = len(S_final)
            best_set = set(S_final)
            
    # Fallback
    if not best_set:
        best_set = set(range(90, 181))
        
    # Verify and fix if needed
    for (a, b, c) in triples:
        if a in best_set and b in best_set and c in best_set:
            best_set.discard(c)
    
    # Final verification pass
    for (a, b, c) in triples:
        if a in best_set and b in best_set and c in best_set:
            best_set.discard(a)
            best_set.discard(b)
            best_set.discard(c)
    
    return list(best_set)

# EVOLVE-BLOCK-END
