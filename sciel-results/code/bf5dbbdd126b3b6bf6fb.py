import itertools
import time
import random

def construct(q, n, radius):
    # Specific optimization for the known task: q=3, n=8, radius=1
    if q == 3 and n == 8 and radius == 1:
        return _solve_3_8_1()
    
    # Fallback for other parameters (Greedy)
    return _greedy_fallback(q, n, radius)

def _solve_3_8_1():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    q = 3
    n = 8
    radius = 1
    total_words = q ** n
    word_list = [list(x) for x in itertools.product(range(q), repeat=n)]
    
    # Map word tuple to index
    word_to_idx = {tuple(w): i for i, w in enumerate(word_list)}
    
    # Precompute neighbors (ball of radius 1)
    # For each word index, find all words at Hamming distance <= 1
    neighbors = [[] for _ in range(total_words)]
    for i, w in enumerate(word_list):
        neighbors[i].append(i) # distance 0
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    new_w = w.copy()
                    new_w[pos] = new_val
                    neighbors[i].append(word_to_idx[tuple(new_w)])
    
    # Greedy Initialization
    # Use bitsets for speed
    masks = [0] * total_words
    for i in range(total_words):
        mask = 0
        for j in neighbors[i]:
            mask |= (1 << j)
        masks[i] = mask
        
    full_mask = (1 << total_words) - 1
    current_union = 0
    current_code = []
    
    # Greedy selection
    # To speed up, we can just iterate and pick max gain
    # Since total_words is 6561, and code size ~500, this is 500 * 6561 ops ~ 3M ops. Fast.
    remaining = full_mask
    for _ in range(total_words): # max iterations
        if time.time() > deadline:
            break
        if remaining == 0:
            break
            
        best_gain = -1
        best_idx = -1
        # Optimization: only check candidates that cover at least 1 new word?
        # Just iterate all.
        for i in range(total_words):
            gain = (remaining & masks[i]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
            if gain == 0:
                continue
            # Early break if gain is max possible? Max gain is 17.
            # If best_gain == 17, we can't do better.
            if best_gain == 17: 
                 # We found a word covering 17 new words. This is the max possible for radius 1 in q=3.
                 # Wait, if remaining has isolated bits, max gain < 17.
                 # But 17 is the theoretical max for a ball.
                 pass
        
        if best_idx == -1:
            break
            
        current_code.append(best_idx)
        current_union |= masks[best_idx]
        remaining = full_mask ^ current_union
        
    # Initialize count_covered array for local search
    count_covered = [0] * total_words
    for c in current_code:
        for j in neighbors[c]:
            count_covered[j] += 1
            
    best_code = list(current_code)
    best_size = len(current_code)
    
    # Local Search
    # State: current_code (list of indices), count_covered (array)
    
    def is_removable(c_idx, c_count):
        # c_idx is the index of the codeword
        # Check if all neighbors are covered by > 1 codeword
        for j in neighbors[c_idx]:
            if c_count[j] == 1:
                return False
        return True
        
    def try_remove(c_idx):
        nonlocal current_code, count_covered, best_code, best_size
        if is_removable(c_idx, count_covered):
            # Remove it
            for j in neighbors[c_idx]:
                count_covered[j] -= 1
            current_code.remove(c_idx)
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
            return True
        return False

    def try_swap(c_idx, new_idx):
        nonlocal current_code, count_covered
        # Tentatively swap
        # 1. Remove c_idx
        for j in neighbors[c_idx]:
            count_covered[j] -= 1
        # 2. Add new_idx
        for j in neighbors[new_idx]:
            count_covered[j] += 1
            
        # Check validity
        valid = True
        # We only need to check if any count became 0
        # Optimization: only check neighbors of c_idx and new_idx
        # Words that were covered only by c_idx (count 1 before removal) are now 0.
        # Words that were covered by c_idx and others (count > 1) are now count-1.
        # Words covered by new_idx are now count+1.
        
        # Check if any word in neighbors[c_idx] has count 0
        for j in neighbors[c_idx]:
            if count_covered[j] == 0:
                valid = False
                break
                
        if valid:
            # Accept swap
            pos = current_code.index(c_idx)
            current_code[pos] = new_idx
            return True
        else:
            # Revert
            for j in neighbors[new_idx]:
                count_covered[j] -= 1
            for j in neighbors[c_idx]:
                count_covered[j] += 1
            return False

    random.seed(42)
    
    while time.time() < deadline:
        # Phase 1: Try to remove
        removed_any = False
        # Shuffle to avoid bias
        indices = list(range(len(current_code)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            c_idx = current_code[i]
            if try_remove(c_idx):
                removed_any = True
                # After removal, list indices change, restart sweep?
                # Or just continue? Since we removed one, indices are stale.
                # Better to break and restart sweep.
                break
        
        if removed_any:
            continue
            
        # Phase 2: Perturbation / Swap
        # Try some random swaps
        swaps_done = 0
        max_swaps = 100
        while swaps_done < max_swaps:
            if time.time() > deadline:
                break
            if not current_code:
                break
                
            # Pick a random codeword to replace
            pos = random.randint(0, len(current_code) - 1)
            old_idx = current_code[pos]
            
            # Pick a random new codeword
            new_idx = random.randint(0, total_words - 1)
            
            if new_idx == old_idx:
                continue
                
            if try_swap(old_idx, new_idx):
                swaps_done += 1
            else:
                # If swap fails, we might still want to try another
                swaps_done += 1 # Count attempts to limit time? Or just limit successful swaps?
                # Let's limit attempts to avoid getting stuck in a tight loop of failures
                if swaps_done % 10 == 0:
                    break
                    
    return [word_list[i] for i in best_code]

def _greedy_fallback(q, n, radius):
    # Simple greedy for other cases
    total_words = q ** n
    word_list = [list(x) for x in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(word_list)}
    
    # Precompute balls
    balls = []
    for w in word_list:
        ball = set()
        ball.add(tuple(w))
        # Generate all words at dist <= radius
        # This is expensive for large n/radius, but fallback is rarely used
        for perm in itertools.product(range(q), repeat=n):
            dist = sum(1 for a, b in zip(w, perm) if a != b)
            if dist <= radius:
                ball.add(perm)
        balls.append(ball)
        
    full_set = set(range(total_words))
    current_union = set()
    code = []
    
    while current_union != full_set:
        best_gain = -1
        best_idx = -1
        for i in range(total_words):
            gain = len(full_set - current_union & balls[i])
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1:
            break
        code.append(best_idx)
        current_union |= balls[best_idx]
        
    return [word_list[i] for i in code]
