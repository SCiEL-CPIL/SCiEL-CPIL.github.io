import random
import numpy as np

N = 256

def _is_sidon(s):
    """Check if set s is a Sidon set (all pairwise sums a+b with a<=b are distinct)."""
    sums = set()
    for i in range(len(s)):
        for j in range(i, len(s)):
            sm = s[i] + s[j]
            if sm in sums:
                return False
            sums.add(sm)
    return True

def _is_maximal(s, n):
    """Check if Sidon set s is inclusion-maximal in [1, n]."""
    s_set = set(s)
    sums = set()
    for i in range(len(s)):
        for j in range(i, len(s)):
            sums.add(s[i] + s[j])
    for x in range(1, n + 1):
        if x in s_set:
            continue
        # Check if adding x would break Sidon property
        new_sums = set()
        for a in s:
            new_sums.add(x + a)
        new_sums.add(2 * x)
        if sums.intersection(new_sums) or len(new_sums) != len(new_sums):
            # x cannot be added, so maximality holds for this x
            continue
        else:
            # x CAN be added, so s is not maximal
            return False
    return True

def _greedy_maximal(order):
    """Build a maximal Sidon set by processing elements in given order."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    # Now make it maximal: try adding remaining elements
    s_set = set(selected)
    for x in order:
        if x in s_set:
            continue
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
            s_set.add(x)
    return selected

def construct():
    # Use simulated annealing to find a small maximal Sidon set
    best_set = None
    best_size = N + 1
    
    # Initialize with greedy solution
    greedy = _greedy_maximal(list(range(1, N + 1)))
    if _is_sidon(greedy) and _is_maximal(greedy, N):
        best_set = greedy
        best_size = len(greedy)
    
    # Simulated annealing
    current = list(greedy) if greedy else []
    current_size = len(current)
    
    for restart in range(50):
        if restart == 0:
            current = list(greedy) if greedy else []
        else:
            # Random restart: build a random maximal Sidon set
            order = list(range(1, N + 1))
            random.shuffle(order)
            current = _greedy_maximal(order)
        
        current_size = len(current)
        if current_size < best_size:
            best_size = current_size
            best_set = list(current)
        
        # Simulated annealing iterations
        temp = 10.0
        cooling = 0.999
        for it in range(2000):
            # Random move
            move = random.choice(['add', 'remove', 'swap'])
            new_current = list(current)
            
            if move == 'add':
                candidates = [x for x in range(1, N + 1) if x not in new_current]
                if not candidates:
                    continue
                x = random.choice(candidates)
                # Check if adding x keeps it Sidon
                added = [x + a for a in new_current] + [2 * x]
                if len(set(added)) != len(added):
                    continue
                # Check against existing sums
                sums = set()
                for i in range(len(new_current)):
                    for j in range(i, len(new_current)):
                        sums.add(new_current[i] + new_current[j])
                if sums.intersection(added):
                    continue
                new_current.append(x)
                
            elif move == 'remove':
                if not new_current:
                    continue
                idx = random.randint(0, len(new_current) - 1)
                new_current.pop(idx)
                
            elif move == 'swap':
                if len(new_current) < 2:
                    continue
                idx1 = random.randint(0, len(new_current) - 1)
                candidates = [x for x in range(1, N + 1) if x not in new_current]
                if not candidates:
                    continue
                x = random.choice(candidates)
                # Check if swap keeps it Sidon
                new_current[idx1] = x
                # Verify Sidon
                valid = True
                sums = set()
                for i in range(len(new_current)):
                    for j in range(i, len(new_current)):
                        sm = new_current[i] + new_current[j]
                        if sm in sums:
                            valid = False
                            break
                        sums.add(sm)
                    if not valid:
                        break
                if not valid:
                    new_current = list(current)
                    continue
            
            # Evaluate
            new_size = len(new_current)
            # Check maximality
            is_max = _is_maximal(new_current, N) if new_current else False
            # Cost: size + penalty for non-maximality
            cost = new_size + (100 if not is_max else 0)
            curr_cost = current_size + (100 if not _is_maximal(current, N) else 0)
            
            if cost <= curr_cost or random.random() < np.exp(-(cost - curr_cost) / temp):
                current = new_current
                current_size = new_size
                if current_size < best_size and is_max:
                    best_size = current_size
                    best_set = list(current)
            
            temp *= cooling
    
    if best_set is None or not _is_sidon(best_set) or not _is_maximal(best_set, N):
        # Fallback to greedy
        return greedy
    
    return best_set
