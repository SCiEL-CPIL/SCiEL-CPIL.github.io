import random
import numpy as np

def construct():
    n = 128
    target_min, target_max = 128, 256
    
    random.seed(42)
    
    # Start with a dense valid set
    A = set(range(64, 129))
    
    # Incremental count array for sums 0..256
    counts = [0] * 257
    
    def rebuild_counts(A_set):
        nonlocal counts
        counts = [0] * 257
        A_list = sorted(A_set)
        for i in range(len(A_list)):
            a = A_list[i]
            for j in range(i, len(A_list)):
                b = A_list[j]
                s = a + b
                if s <= 256:
                    counts[s] += 1

    def is_valid_counts():
        for t in range(target_min, target_max + 1):
            if counts[t] == 0:
                return False
        return True

    def get_max_rep():
        m = 0
        for t in range(target_min, target_max + 1):
            if counts[t] > m:
                m = counts[t]
        return m

    # Initialize counts
    rebuild_counts(A)
    
    # Phase 1: Greedy Pruning
    removed_any = True
    while removed_any:
        removed_any = False
        A_list = list(A)
        random.shuffle(A_list)
        for a in A_list:
            if a not in A:
                continue
            # Try removing a
            A.remove(a)
            counts[2*a] -= 1
            for b in A:
                counts[a+b] -= 1
            
            if is_valid_counts():
                removed_any = True
            else:
                # Revert
                A.add(a)
                counts[2*a] += 1
                for b in A:
                    if b != a:
                        counts[a+b] += 1
        
    # Update best after pruning
    best_A = set(A)
    best_max = get_max_rep()
    
    # Phase 2: Local Search
    # We will perform swaps and single removals/adds
    
    for iteration in range(15000):
        # Choose a move type
        move_type = random.random()
        
        if move_type < 0.7:
            # Swap: remove a, add b
            if len(A) < 2: continue
            a_remove = random.choice(list(A))
            candidates = [x for x in range(0, 129) if x not in A]
            if not candidates: continue
            a_add = random.choice(candidates)
            
            # Apply swap
            A.remove(a_remove)
            counts[2*a_remove] -= 1
            for b in A:
                counts[a_remove + b] -= 1
            
            A.add(a_add)
            counts[2*a_add] += 1
            for b in A:
                if b != a_add:
                    counts[a_add + b] += 1
            
            if is_valid_counts():
                new_max = get_max_rep()
                if new_max < best_max:
                    best_max = new_max
                    best_A = set(A)
                elif new_max == best_max and random.random() < 0.1:
                    pass # Accept to explore
                else:
                    # Revert
                    A.remove(a_add)
                    counts[2*a_add] -= 1
                    for b in A:
                        if b != a_add:
                            counts[a_add + b] -= 1
                    A.add(a_remove)
                    counts[2*a_remove] += 1
                    for b in A:
                        if b != a_remove:
                            counts[a_remove + b] += 1
                            
        elif move_type < 0.85:
            # Remove
            if not A: continue
            a_remove = random.choice(list(A))
            A.remove(a_remove)
            counts[2*a_remove] -= 1
            for b in A:
                counts[a_remove + b] -= 1
                
            if is_valid_counts():
                new_max = get_max_rep()
                if new_max <= best_max:
                    if new_max < best_max:
                        best_max = new_max
                        best_A = set(A)
                else:
                    # Revert
                    A.add(a_remove)
                    counts[2*a_remove] += 1
                    for b in A:
                        if b != a_remove:
                            counts[a_remove + b] += 1
            else:
                # Revert
                A.add(a_remove)
                counts[2*a_remove] += 1
                for b in A:
                    if b != a_remove:
                        counts[a_remove + b] += 1
                            
        else:
            # Add (rarely, to allow restructuring)
            candidates = [x for x in range(0, 129) if x not in A]
            if not candidates: continue
            a_add = random.choice(candidates)
            
            A.add(a_add)
            counts[2*a_add] += 1
            for b in A:
                if b != a_add:
                    counts[a_add + b] += 1
            
            # Adding increases counts, so max rep likely increases.
            # Only accept if it allows future improvements? 
            # For now, reject unless it doesn't increase max rep (unlikely) 
            # or we use it as a step in a 2-move sequence.
            # Since we don't do 2-move sequences explicitly, we reject if max rep increases.
            new_max = get_max_rep()
            if new_max > best_max:
                # Revert
                A.remove(a_add)
                counts[2*a_add] -= 1
                for b in A:
                    if b != a_add:
                        counts[a_add + b] -= 1
            else:
                # If max rep didn't increase, maybe keep it?
                # But it makes set larger. 
                # Let's just revert to keep set small.
                A.remove(a_add)
                counts[2*a_add] -= 1
                for b in A:
                    if b != a_add:
                        counts[a_add + b] -= 1

    return sorted(best_A)
