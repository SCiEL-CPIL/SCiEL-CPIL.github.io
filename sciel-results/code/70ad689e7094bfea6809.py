import time
import random
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin for 60s limit
    
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
        
    # Group bad triples by element for fast lookup
    triples_by_elem = {x: [] for x in numbers}
    for t in bad_triples:
        a, b, c = t
        triples_by_elem[a].append((b, c))
        triples_by_elem[b].append((a, c))
        triples_by_elem[c].append((a, b))
        
    # Helper: check if x can be added to current_set
    def can_add(x, current_set):
        for u, v in triples_by_elem[x]:
            if u in current_set and v in current_set:
                return False
        return True
    
    # Helper: greedy add all possible elements from candidates
    def greedy_add(current_set, candidates):
        for x in candidates:
            if can_add(x, current_set):
                current_set.add(x)
    
    # 2. Heuristic Search
    best_size = -1
    best_set = set()
    
    num_restarts = 2000 # More restarts with improved local search
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Randomize order with bias towards low-degree elements
        order = numbers[:]
        random.shuffle(order)
        
        current_set = set()
        
        # Greedy Add
        for x in order:
            if can_add(x, current_set):
                current_set.add(x)
                
        # Local Search with proper copy/revert
        max_kicks = 30
        for _ in range(max_kicks):
            if time.time() > deadline - 2.0:
                break
            if not current_set:
                break
            
            size_before = len(current_set)
            saved_set = set(current_set)
            
            # Try kicking out a random element
            to_kick = random.choice(list(current_set))
            current_set.remove(to_kick)
            
            # Re-add all possible elements
            candidates = [x for x in numbers if x not in current_set]
            random.shuffle(candidates)
            greedy_add(current_set, candidates)
            
            if len(current_set) > size_before:
                # Accept improvement, continue
                continue
            else:
                # Revert to saved state
                current_set = saved_set
                # If no improvement after trying a kick, try another kick
                # But to avoid infinite loops, break if we've tried several
                # Actually, let's just continue to next kick attempt
                pass
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
        if best_size == n:
            break
            
    # 3. Final polish pass on best_set
    # Try systematic improvements: for each element not in best_set,
    # try swapping it with each element in best_set
    if best_size > 0 and time.time() < deadline - 10.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for x in numbers:
                if x in best_set:
                    continue
                # Try adding x by removing some y from best_set
                for y in list(best_set):
                    if time.time() > deadline - 5.0:
                        break
                    test_set = set(best_set)
                    test_set.remove(y)
                    test_set.add(x)
                    # Check validity
                    valid = True
                    for u, v in triples_by_elem[x]:
                        if u in test_set and v in test_set:
                            valid = False
                            break
                    if valid:
                        # Also need to check that removing y doesn't break anything
                        # (it can't, since we're removing an element)
                        # And adding x is valid as checked above
                        best_set = test_set
                        improved = True
                        break
                if improved:
                    break
            
    if best_size == -1:
        return [1]
        
    return list(best_set)
