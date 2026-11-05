import time
import random
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    n = 128
    numbers = list(range(1, n + 1))

    # 1. Identify all bad triples
    bad_triples = []
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm(a, b)
            for c in range(b + 1, n + 1):
                lac = lcm(a, c)
                if lac != lab:
                    continue
                lbc = lcm(b, c)
                if lbc == lab:
                    bad_triples.append((a, b, c))
                    
    if time.time() > deadline:
        return numbers

    # 2. Precompute conflict structures
    conflicts = {x: set() for x in numbers}
    triple_membership = {x: set() for x in numbers}  # x -> set of triple indices containing x
    
    for i, (a, b, c) in enumerate(bad_triples):
        conflicts[a].add((b, c))
        conflicts[b].add((a, c))
        conflicts[c].add((a, b))
        triple_membership[a].add(i)
        triple_membership[b].add(i)
        triple_membership[c].add(i)

    def can_add(x, current_set):
        for u, v in conflicts[x]:
            if u in current_set and v in current_set:
                return False
        return True

    best_set = set()
    best_size = -1

    def greedy_remove(S, active_triples, counts, strategy='max'):
        """Greedy removal of elements from S to eliminate all bad triples."""
        S = set(S)
        active_triples = [t for t in active_triples]
        counts = dict(counts)
        
        while True:
            if time.time() > deadline - 5.0:
                break
                
            # Find elements with highest count
            if not any(counts[x] > 0 for x in S):
                break
                
            if strategy == 'max':
                max_count = max(counts[x] for x in S)
                candidates = [x for x in S if counts[x] == max_count]
            elif strategy == 'random':
                candidates = [x for x in S if counts[x] > 0]
            elif strategy == 'weighted':
                # Weight by count / (1 + len(conflicts[x])) to prefer elements with fewer conflicts
                weighted = [(counts[x] / (1 + len(conflicts[x])), x) for x in S if counts[x] > 0]
                weighted.sort(reverse=True)
                candidates = [x for _, x in weighted[:max(1, len(S)//10)]]
            else:
                candidates = [x for x in S if counts[x] > 0]
                
            if not candidates:
                break
                
            target = random.choice(candidates)
            S.remove(target)
            
            # Update counts for remaining elements
            for i in triple_membership[target]:
                a, b, c = bad_triples[i]
                for x in (a, b, c):
                    if x != target and x in S:
                        counts[x] -= 1
                        if counts[x] < 0:
                            counts[x] = 0
        
        return S

    def local_search(S, max_iters=30):
        """Local search: try removing one element and adding many back."""
        S = set(S)
        
        for _ in range(max_iters):
            if time.time() > deadline - 3.0:
                break
                
            improved = False
            s_list = list(S)
            random.shuffle(s_list)
            
            for y in s_list:
                if time.time() > deadline - 3.0:
                    break
                    
                S_tentative = S - {y}
                
                # Try to add elements not in S, in random order
                candidates_to_add = [x for x in numbers if x not in S]
                random.shuffle(candidates_to_add)
                
                added = 0
                for z in candidates_to_add:
                    if can_add(z, S_tentative):
                        S_tentative.add(z)
                        added += 1
                        
                if len(S_tentative) > len(S):
                    S = S_tentative
                    improved = True
                    break
                    
            if not improved:
                break
                
        return S

    # Multiple strategies with different random seeds
    strategies = ['max', 'random', 'weighted']
    num_restarts_per_strategy = 40
    
    for strategy in strategies:
        for r in range(num_restarts_per_strategy):
            if time.time() > deadline - 10.0:
                break
                
            # Initial greedy removal
            S = set(numbers)
            counts = {x: len(triple_membership[x]) for x in numbers}
            active_triples = list(range(len(bad_triples)))
            
            S = greedy_remove(S, active_triples, counts, strategy)
            
            # Local search
            S = local_search(S, max_iters=20)
            
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
                
            # Early exit if we found a very large set
            if best_size > 115:
                break
                
        if best_size > 115:
            break

    if best_size == -1:
        return [1]
        
    return list(best_set)
