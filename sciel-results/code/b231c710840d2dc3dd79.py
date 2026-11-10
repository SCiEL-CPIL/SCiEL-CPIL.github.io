import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    deadline = time.time() + 580  # safety margin before 600s limit
    
    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        r = (x * x) % m
        if r != 0:
            forbidden.add(r)
    
    # Precompute adjacency
    adj = [set() for _ in range(m)]
    for i in range(m):
        for d in forbidden:
            j = (i + d) % m
            adj[i].add(j)
    
    def conflicts_with_set(c, s_set):
        """Return True if c conflicts with any element in s_set."""
        for s in s_set:
            if c in adj[s]:
                return True
        return False
    
    # Greedy init
    def greedy_init(order):
        selected = []
        sel_set = set()
        for c in order:
            if not conflicts_with_set(c, sel_set):
                selected.append(c)
                sel_set.add(c)
        return selected
    
    # Generate best greedy init
    best_init = []
    for trial in range(50):
        order = list(range(m))
        random.shuffle(order)
        sol = greedy_init(order)
        if len(sol) > len(best_init):
            best_init = sol[:]
    
    # Simulated annealing local search
    current = best_init[:]
    current_set = set(current)
    best_sol = current[:]
    best_size = len(current)
    
    T = 5.0
    cooling = 0.9995
    min_T = 0.05
    
    while time.time() < deadline:
        if T < min_T:
            T = 5.0
            current = best_sol[:]
            current_set = set(current)
        
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add an element
            candidates = [c for c in range(m) if c not in current_set]
            if not candidates:
                continue
            c = random.choice(candidates)
            if not conflicts_with_set(c, current_set):
                # Valid add - always accept
                current.append(c)
                current_set.add(c)
                if len(current) > best_size:
                    best_size = len(current)
                    best_sol = current[:]
            else:
                # Count conflicts for acceptance
                n_conf = sum(1 for s in current_set if c in adj[s])
                if random.random() < np.exp(-n_conf / T):
                    # Tentatively add (will be invalid, but SA allows)
                    # Actually, we should only maintain valid sets
                    # So skip this - only accept valid moves
                    pass
        elif move_type < 0.6:
            # Remove a random element
            if current:
                idx = random.randrange(len(current))
                removed = current.pop(idx)
                current_set.discard(removed)
                # Check if this allows an add
                candidates = [c for c in range(m) if c not in current_set]
                if candidates:
                    random.shuffle(candidates)
                    for c in candidates[:30]:
                        if not conflicts_with_set(c, current_set):
                            current.append(c)
                            current_set.add(c)
                            break
                if len(current) > best_size:
                    best_size = len(current)
                    best_sol = current[:]
        else:
            # Swap: remove one, add one
            if current:
                idx = random.randrange(len(current))
                removed = current[idx]
                temp_set = set(current_set)
                temp_set.discard(removed)
                candidates = [c for c in range(m) if c not in temp_set]
                if candidates:
                    random.shuffle(candidates)
                    for c in candidates[:30]:
                        if not conflicts_with_set(c, temp_set):
                            current[idx] = c
                            current_set.discard(removed)
                            current_set.add(c)
                            break
                if len(current) > best_size:
                    best_size = len(current)
                    best_sol = current[:]
        
        T *= cooling
    
    if not best_sol:
        return np.array([0], dtype=np.int64)
    
    return np.array(best_sol, dtype=np.int64)
