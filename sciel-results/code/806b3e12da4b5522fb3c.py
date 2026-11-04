import math
import random

def calc_score(U):
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num = len(diffs) / len(sums)
    den = math.log(2 * max_u + 1)
    if den == 0:
        return 0
    return 1 + math.log(num) / den

def construct():
    best = [0, 1, 3]
    best_score = calc_score(best)
    
    # Strategy: Local search on subsets of the base-B hypercube
    # U is a subset of V = { sum b_j B^j | b_j in {0,1} }
    
    random.seed(42)
    
    # Try a range of bases and depths
    # Larger n allows for more complex structures
    for B in [3, 4, 5, 7, 10]:
        for n in [6, 8, 10]:
            # Generate all possible values in the hypercube
            V = []
            for mask in range(1 << n):
                val = 0
                for j in range(n):
                    if mask & (1 << j):
                        val += B ** j
                V.append(val)
            
            V_set = set(V)
            
            # Initialize with a random subset containing 0
            # Size of U should be moderate to keep calc_score fast
            # Let's try sizes between 5 and 30
            target_size = random.randint(5, 25)
            
            # Ensure 0 is in U (mask 0 is val 0)
            # Indices in V: 0 corresponds to val 0
            current_indices = {0}
            for _ in range(target_size - 1):
                idx = random.randint(1, len(V) - 1)
                current_indices.add(idx)
            
            current_U = [V[i] for i in current_indices]
            current_score = calc_score(current_U)
            
            # Local search
            # We will try to swap elements or add/remove
            # To save time, we limit iterations
            for _ in range(2000):
                # Randomly pick a move
                move_type = random.choice(['add', 'remove', 'swap'])
                
                new_indices = set(current_indices)
                
                if move_type == 'add':
                    if len(new_indices) < target_size + 5: # Allow growing
                        candidate = random.randint(0, len(V) - 1)
                        if candidate not in new_indices:
                            new_indices.add(candidate)
                        else:
                            continue
                elif move_type == 'remove':
                    if len(new_indices) > 2:
                        to_remove = random.choice(list(new_indices))
                        if to_remove == 0: # Keep 0
                            continue
                        new_indices.remove(to_remove)
                    else:
                        continue
                else: # swap
                    if len(new_indices) == 0:
                        continue
                    to_remove = random.choice(list(new_indices))
                    if to_remove == 0:
                        # Swap 0 with something? No, 0 must stay.
                        # So we swap a non-zero element with a new element
                        non_zero_indices = [i for i in new_indices if i != 0]
                        if not non_zero_indices:
                            continue
                        to_remove = random.choice(non_zero_indices)
                    
                    candidate = random.randint(0, len(V) - 1)
                    if candidate in new_indices:
                        continue
                    
                    new_indices.remove(to_remove)
                    new_indices.add(candidate)
                
                new_U = [V[i] for i in new_indices]
                new_score = calc_score(new_U)
                
                if new_score > current_score:
                    current_score = new_score
                    current_indices = new_indices
                    current_U = new_U
                    
                    if current_score > best_score:
                        best_score = current_score
                        best = current_U
                        # Optional: break early if we hit a very high score?
                        # But let's keep searching a bit more.
                
                # Random restarts? Maybe if stuck.
                # For now, simple hill climbing.
                
            # Try a few more random restarts for this B, n
            for restart in range(3):
                current_indices = {0}
                for _ in range(target_size - 1):
                    idx = random.randint(1, len(V) - 1)
                    current_indices.add(idx)
                
                current_U = [V[i] for i in current_indices]
                current_score = calc_score(current_U)
                
                for _ in range(1000):
                    move_type = random.choice(['add', 'remove', 'swap'])
                    new_indices = set(current_indices)
                    
                    if move_type == 'add':
                        if len(new_indices) < target_size + 5:
                            candidate = random.randint(0, len(V) - 1)
                            if candidate not in new_indices:
                                new_indices.add(candidate)
                            else:
                                continue
                    elif move_type == 'remove':
                        if len(new_indices) > 2:
                            to_remove = random.choice(list(new_indices))
                            if to_remove == 0:
                                continue
                            new_indices.remove(to_remove)
                        else:
                            continue
                    else:
                        if len(new_indices) == 0:
                            continue
                        to_remove = random.choice(list(new_indices))
                        if to_remove == 0:
                            non_zero_indices = [i for i in new_indices if i != 0]
                            if not non_zero_indices:
                                continue
                            to_remove = random.choice(non_zero_indices)
                        
                        candidate = random.randint(0, len(V) - 1)
                        if candidate in new_indices:
                            continue
                        
                        new_indices.remove(to_remove)
                        new_indices.add(candidate)
                    
                    new_U = [V[i] for i in new_indices]
                    new_score = calc_score(new_U)
                    
                    if new_score > current_score:
                        current_score = new_score
                        current_indices = new_indices
                        current_U = new_U
                        
                        if current_score > best_score:
                            best_score = current_score
                            best = current_U

    return best
