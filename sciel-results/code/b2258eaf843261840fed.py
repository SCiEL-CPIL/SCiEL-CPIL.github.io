import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 256
    
    # Build conflict graph
    conflict = [set() for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict[a].add(b)
                conflict[b].add(a)
    
    # Initial set: All odd numbers (no two odds conflict since a+b is even, a*b is odd)
    current_set = set(range(1, N + 1, 2))
    current_size = len(current_set)
    best_set = set(current_set)
    best_size = current_size
    
    def can_add(x, s):
        return not (conflict[x] & s)
    
    # Phase 1: Greedy add - repeatedly add any node with no conflicts
    improved = True
    while improved:
        improved = False
        for x in range(1, N + 1):
            if x not in current_set and can_add(x, current_set):
                current_set.add(x)
                current_size += 1
                improved = True
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(current_set)
    
    # Phase 2: Local search with 1-swap and random perturbation
    random.seed(42)
    
    for iteration in range(500):
        if time.time() > deadline - 5:
            break
        
        made_progress = False
        
        # 1-swap: remove x, add y (y was blocked by x)
        for x in list(current_set):
            for y in conflict[x]:
                if y in current_set:
                    continue
                if can_add(y, current_set - {x}):
                    old_size = current_size
                    current_set.remove(x)
                    current_set.add(y)
                    
                    # Try to add more after swap
                    for z in range(1, N + 1):
                        if z not in current_set and can_add(z, current_set):
                            current_set.add(z)
                            current_size += 1
                            if current_size > best_size:
                                best_size = current_size
                                best_set = set(current_set)
                    
                    if current_size > old_size:
                        made_progress = True
                        break
                    else:
                        # Revert
                        current_set.remove(y)
                        current_set.add(x)
                        current_size = old_size
            
            if made_progress:
                break
        
        if not made_progress:
            # Random perturbation: remove 1-2 random elements, re-greedy
            if len(current_set) > 5:
                n_remove = 1 if random.random() < 0.5 else 2
                to_remove = random.sample(list(current_set), min(n_remove, len(current_set)))
                old_set = set(current_set)
                old_size = current_size
                
                for x in to_remove:
                    current_set.discard(x)
                    current_size -= 1
                
                # Re-add greedily
                for z in range(1, N + 1):
                    if z not in current_set and can_add(z, current_set):
                        current_set.add(z)
                        current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_set = set(current_set)
                
                if current_size < old_size:
                    current_set = old_set
                    current_size = old_size
    
    # Final polish
    for x in range(1, N + 1):
        if x not in current_set and can_add(x, current_set):
            current_set.add(x)
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
    
    if current_size > best_size:
        best_set = current_set
    
    return list(best_set)
