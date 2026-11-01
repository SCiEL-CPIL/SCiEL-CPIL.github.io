import time
import random
import math

def construct(n):
    if n < 7:
        # Use standard construction for n < 7 as it is optimal
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                # Find max overlap
                max_ov = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0:
                        max_ov = 0
                        break
                    if new[len(new) - o:] == block[:o]:
                        max_ov = o
                        break
                new += block[max_ov:]
            word = new
        return "".join(map(str, word))
    
    # For n=7, use Hill Climbing on the permutation order
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    # 1. Generate standard S6 and the list of permutations P
    word6 = [1]
    for m in range(2, 7):
        perms, seen = [], set()
        for i in range(len(word6) - m + 2):
            window = tuple(word6[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_ov = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if o == 0:
                    max_ov = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    max_ov = o
                    break
            new += block[max_ov:]
        word6 = new
        
    # Extract permutations P from word6
    # The permutations are the windows of length 6 in word6 that are unique and valid
    # Actually, the standard construction ensures that the permutations appear in a specific order.
    # We can extract them by scanning word6.
    P_list = []
    seen = set()
    for i in range(len(word6) - 6 + 1):
        window = tuple(word6[i:i + 6])
        if window not in seen:
            seen.add(window)
            P_list.append(window)
            
    # P_list contains the 720 permutations of 1..6 in the order they appear in S6.
    # We need to insert 7 into these to form S7.
    # The standard method uses this order.
    
    # Prepare blocks for n=7
    # Block for permutation p is p + [7] + p
    blocks = []
    for p in P_list:
        block = list(p) + [7] + list(p)
        blocks.append(block)
        
    # Precompute overlaps? 
    # Overlap between block i and block j
    # We want to maximize total overlap.
    # Length = sum(len(block)) - sum(overlap(i, i+1))
    # len(block) is always 13.
    # So we want to minimize sum(13 - overlap(i, i+1))
    # Which is equivalent to maximizing sum(overlap(i, i+1))
    
    def get_overlap(b1, b2):
        # Max overlap of suffix of b1 and prefix of b2
        max_ov = 0
        # Max possible overlap is 12
        for o in range(min(len(b1), len(b2)), -1, -1):
            if o == 0:
                return 0
            if b1[-o:] == b2[:o]:
                return o
        return 0

    # Initial order is 0..719
    order = list(range(len(P_list)))
    
    # Calculate initial cost (total overlap)
    total_overlap = 0
    for i in range(len(order) - 1):
        total_overlap += get_overlap(blocks[order[i]], blocks[order[i+1]])
        
    best_overlap = total_overlap
    best_order = order[:]
    
    # Hill Climbing / Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Pick a random move
        # 1. Swap two adjacent
        # 2. Swap two random
        # 3. Reverse a segment
        
        move_type = random.randint(0, 2)
        if move_type == 0:
            i = random.randint(0, len(order) - 2)
            j = i + 1
        elif move_type == 1:
            i = random.randint(0, len(order) - 1)
            j = random.randint(0, len(order) - 1)
            if i == j: continue
        else:
            i = random.randint(0, len(order) - 2)
            j = random.randint(i + 1, len(order) - 1)
            
        # Apply move to a copy
        new_order = order[:]
        if move_type == 0:
            new_order[i], new_order[j] = new_order[j], new_order[i]
        elif move_type == 1:
            new_order[i], new_order[j] = new_order[j], new_order[i]
        else:
            new_order[i:j+1] = reversed(new_order[i:j+1])
            
        # Calculate delta overlap
        # Only edges affected are those involving i and j (and neighbors for reverse)
        # For simplicity, recalculate total overlap if N is small? 
        # N=720. Recalc is 719 overlaps. 
        # 719 * 12 comparisons = ~8000 ops. 
        # 10000 iterations * 8000 = 80M ops. Might be too slow in Python.
        # We need to calculate delta efficiently.
        
        # Let's try to calculate delta for swap i, j
        # Edges removed: (i-1, i), (i, i+1), (j-1, j), (j, j+1)
        # Edges added: (i-1, j), (j, i+1), (i-1? no, i is now at j's pos)
        
        # This is tricky with indices. 
        # Given the time, let's just recalculate if we can afford it.
        # 720 overlaps * 12 ops = 8640 ops per eval.
        # 10k evals = 86M ops. Python might take 10-20s. 
        # So we can do ~10k-50k iterations.
        
        new_overlap = 0
        for k in range(len(new_order) - 1):
            new_overlap += get_overlap(blocks[new_order[k]], blocks[new_order[k+1]])
            
        delta = new_overlap - total_overlap
        
        if delta > 0 or random.random() < math.exp(delta / T):
            order = new_order
            total_overlap = new_overlap
            if total_overlap > best_overlap:
                best_overlap = total_overlap
                best_order = order[:]
                
        T *= cooling
        
    # Reconstruct word from best_order
    final_word = []
    for idx in best_order:
        block = blocks[idx]
        if not final_word:
            final_word.extend(block)
        else:
            ov = get_overlap(final_word, block)
            final_word.extend(block[ov:])
            
    return "".join(map(str, final_word))
