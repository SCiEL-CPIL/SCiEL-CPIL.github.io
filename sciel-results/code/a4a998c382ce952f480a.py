import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    if n <= 6:
        # Standard construction is optimal for n <= 5, and very close for n=6
        # We can just use the standard recursive construction
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
                # Calculate overlap
                overlap = 0
                max_overlap = min(len(new), len(block))
                for o in range(max_overlap, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    # For n=7, we need to optimize
    # Step 1: Get standard superpermutation for n=6
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
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word6 = new
        
    # Step 2: Extract permutations of 1..6 from word6
    perms6, seen = [], set()
    for i in range(len(word6) - 6 + 1):
        window = tuple(word6[i:i + 6])
        if len(set(window)) == 6 and window not in seen:
            seen.add(window)
            perms6.append(window)
            
    # Step 3: Heuristic to reorder perms6 to minimize length when inserting 7
    # We want to order perms6 such that the overlap between 
    # (p_i, 7, p_i) and (p_{i+1}, 7, p_{i+1}) is maximized.
    # The overlap is determined by the suffix of p_i and prefix of p_{i+1}.
    # Specifically, we append 7 to p_i, then we want to start p_{i+1} as soon as possible.
    # The block is p_i + [7] + p_i.
    # The next block is p_{i+1} + [7] + p_{i+1}.
    # We check overlap between the end of (p_i + [7] + p_i) and start of (p_{i+1} + [7] + p_{i+1}).
    
    # A simpler metric: distance between p_i and p_{i+1}.
    # If p_i and p_{i+1} are "close" in some sense, the overlap might be better.
    # Let's use a greedy nearest neighbor approach based on a cost function.
    # Cost(i, j) = number of matching suffix of p_i and prefix of p_j.
    # We want to maximize this.
    
    def overlap_score(p1, p2):
        # p1 and p2 are tuples of length 6
        # We are appending 7 to p1, so the end of the current segment is ... p1
        # We are starting with p2.
        # So we check if suffix of p1 matches prefix of p2.
        max_ov = 0
        for k in range(6, -1, -1):
            if p1[-k:] == p2[:k]:
                max_ov = k
                break
        return max_ov

    # Greedy construction of the order
    remaining = set(perms6)
    order = [remaining.pop()]
    while remaining:
        last = order[-1]
        # Find best next
        best_next = None
        best_score = -1
        # To speed up, we might just pick the first one, but for n=7, 720 items is small
        # We can do a full scan
        for p in remaining:
            score = overlap_score(last, p)
            if score > best_score:
                best_score = score
                best_next = p
        order.append(best_next)
        remaining.remove(best_next)
        
    # Construct word7
    word7 = []
    for p in order:
        block = list(p) + [7] + list(p)
        overlap = 0
        max_overlap = min(len(word7), len(block))
        for o in range(max_overlap, -1, -1):
            if word7[len(word7) - o:] == block[:o]:
                overlap = o
                break
        word7 += block[overlap:]
        
    # Check time
    if time.time() > deadline:
         # Return standard construction if we ran out of time, though unlikely
         return "".join(map(str, word7))
         
    # Optional: Local search on the order to improve further
    # Swap adjacent elements in 'order' and see if length decreases
    # This might be too slow if not careful, but 720 elements is small.
    # Let's try a few iterations.
    
    def calc_length(order):
        w = []
        for p in order:
            block = list(p) + [7] + list(p)
            overlap = 0
            max_overlap = min(len(w), len(block))
            for o in range(max_overlap, -1, -1):
                if w[len(w) - o:] == block[:o]:
                    overlap = o
                    break
            w += block[overlap:]
        return len(w)
        
    current_len = calc_length(order)
    best_len = current_len
    best_order = order[:]
    
    # Simple local search: try swapping i, i+1
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(len(best_order) - 1):
            new_order = best_order[:]
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            new_len = calc_length(new_order)
            if new_len < best_len:
                best_len = new_len
                best_order = new_order
                improved = True
                
    # Reconstruct with best order
    word7 = []
    for p in best_order:
        block = list(p) + [7] + list(p)
        overlap = 0
        max_overlap = min(len(word7), len(block))
        for o in range(max_overlap, -1, -1):
            if word7[len(word7) - o:] == block[:o]:
                overlap = o
                break
        word7 += block[overlap:]
        
    return "".join(map(str, word7))
