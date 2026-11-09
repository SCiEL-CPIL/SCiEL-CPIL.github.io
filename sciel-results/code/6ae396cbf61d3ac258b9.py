import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 55.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    fracs = [Fraction(1, d) for d in denominators]
    
    # Greedy initialization
    selected = []
    total = Fraction(0, 1)
    for i, d in enumerate(denominators):
        if total + fracs[i] < 1:
            selected.append(i)
            total += fracs[i]
    
    best_indices = selected[:]
    best_sum = total
    
    current_indices = selected[:]
    current_sum = total
    current_set = set(current_indices)
    
    # Local Search
    while time.time() < deadline:
        improved = False
        
        # 1. Try 1-Swaps (Remove one, Add one)
        for i_idx in range(len(current_indices)):
            i = current_indices[i_idx]
            
            for j in range(n):
                if j in current_set:
                    continue
                
                new_sum = current_sum - fracs[i] + fracs[j]
                if new_sum < 1 and new_sum > current_sum:
                    # Apply swap
                    current_indices[i_idx] = j
                    current_set.remove(i)
                    current_set.add(j)
                    current_sum = new_sum
                    improved = True
                    break
            if improved:
                break
        
        # 2. If no 1-swap, try 2-Swaps (Remove two, Add two)
        if not improved:
            # Generate removal pairs
            rem_pairs = []
            for a_idx in range(len(current_indices)):
                for b_idx in range(a_idx + 1, len(current_indices)):
                    rem_pairs.append((a_idx, b_idx))
                    
            # Generate addition pairs (not in current_set)
            add_pairs = []
            for a in range(n):
                if a in current_set:
                    continue
                for b in range(a + 1, n):
                    if b in current_set:
                        continue
                    add_pairs.append((a, b))
                    
            for r_idx1, r_idx2 in rem_pairs:
                i1 = current_indices[r_idx1]
                i2 = current_indices[r_idx2]
                rem_sum = fracs[i1] + fracs[i2]
                
                for a, b in add_pairs:
                    add_sum = fracs[a] + fracs[b]
                    
                    if add_sum <= rem_sum:
                        continue
                    
                    new_sum = current_sum - rem_sum + add_sum
                    if new_sum < 1 and new_sum > current_sum:
                        # Apply 2-swap
                        new_indices = [x for idx, x in enumerate(current_indices) if idx != r_idx1 and idx != r_idx2]
                        new_indices.append(a)
                        new_indices.append(b)
                        
                        current_indices = new_indices
                        current_set = set(new_indices)
                        current_sum = new_sum
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
        
        # Update best
        if current_sum > best_sum:
            best_sum = current_sum
            best_indices = current_indices[:]
            
        # Time check
        if time.time() > deadline:
            break
            
    return best_indices
