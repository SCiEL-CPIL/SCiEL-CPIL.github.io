import time

def construct():
    start_time = time.time()
    DEADLINE = 85  # 90s limit, leave 5s margin
    
    # 1. Generate all 2,3,5-smooth numbers up to 128
    smooth = set()
    for a in range(0, 8):
        val_a = 2 ** a
        for b in range(0, 5):
            val_ab = val_a * (3 ** b)
            if val_ab > 128:
                break
            for c in range(0, 4):
                val_abc = val_ab * (5 ** c)
                if val_abc > 128:
                    break
                smooth.add(val_abc)
    
    smooth_list = sorted(smooth)
    n_smooth = len(smooth_list)
    
    # Precompute incompatibility mask for each smooth number
    # incompat[i] is a bitmask of indices j where smooth_list[i] and smooth_list[j] are incompatible (one divides the other)
    incompat = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            a, b = smooth_list[i], smooth_list[j]
            if a % b == 0 or b % a == 0:
                incompat[i] |= (1 << j)
                incompat[j] |= (1 << i)
    
    # 2. Generate all antichains and find min size for each sum
    # best_rep[sum] = (size, tuple_of_indices)
    best_rep = {}
    
    def gen_antichains(start_idx, current_mask, current_sum, current_size, current_indices):
        if time.time() - start_time > DEADLINE:
            return
            
        if current_sum > 128:
            return
            
        # Update best for this sum
        if current_sum in best_rep:
            if current_size < best_rep[current_sum][0]:
                best_rep[current_sum] = (current_size, tuple(current_indices))
        else:
            best_rep[current_sum] = (current_size, tuple(current_indices))
            
        # Try adding new elements
        for i in range(start_idx, n_smooth):
            # Check if i is compatible with all in current_mask
            if incompat[i] & current_mask:
                continue
            
            new_mask = current_mask | (1 << i)
            new_sum = current_sum + smooth_list[i]
            new_size = current_size + 1
            new_indices = current_indices + [i]
            
            gen_antichains(i + 1, new_mask, new_sum, new_size, new_indices)
            
    gen_antichains(0, 0, 0, 0, [])
    
    # 3. Construct answers
    answers = []
    for n in range(1, 129):
        if n in best_rep:
            size, indices = best_rep[n]
            ans = [smooth_list[i] for i in indices]
            answers.append(ans)
        else:
            # Fallback: should not happen
            if n in smooth_list:
                answers.append([n])
            else:
                answers.append([n])
                
    return answers
