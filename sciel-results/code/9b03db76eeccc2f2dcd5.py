# EVOLVE-BLOCK-START
import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    if n <= 6:
        # Standard construction is optimal for n <= 6
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
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    if n == 7:
        # Generate S_6 standard
        word = [1]
        for m in range(2, 7):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        
        # Extract perms of 1..6
        perms, seen = [], set()
        for i in range(len(word) - 5):
            window = tuple(word[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # TSP heuristic to maximize overlaps
        # Overlap(p, q) = longest suffix of p that is prefix of q
        # Precompute overlap matrix? 720x720 = 518400 entries.
        # Each calc is O(6). Total ~3M ops. Fast.
        
        N = len(perms)
        # dist[i][j] = -overlap(perms[i], perms[j]) to be minimized
        # Or just store overlap.
        
        # Heuristic: Nearest Neighbor
        # Start with a random perm
        unvisited = list(range(N))
        random.shuffle(unvisited)
        path = [unvisited.pop()]
        
        while unvisited:
            last = path[-1]
            # Find neighbor with max overlap
            best_idx = -1
            best_overlap = -1
            for idx in unvisited:
                ov = 0
                p = perms[last]
                q = perms[idx]
                # Check suffix of p vs prefix of q
                # Max possible is 5 (since they are distinct perms, 6 is impossible)
                for k in range(5, -1, -1):
                    if p[-k:] == q[:k]:
                        ov = k
                        break
                if ov > best_overlap:
                    best_overlap = ov
                    best_idx = idx
            path.append(best_idx)
            unvisited.remove(best_idx)
            
        # 2-opt improvement
        # We want to maximize sum of overlaps.
        # Calculate current total overlap
        def calc_total_overlap(path):
            total = 0
            for i in range(len(path) - 1):
                p = perms[path[i]]
                q = perms[path[i+1]]
                ov = 0
                for k in range(5, -1, -1):
                    if p[-k:] == q[:k]:
                        ov = k
                        break
                total += ov
            return total
        
        current_overlap = calc_total_overlap(path)
        
        improved = True
        while improved and time.time() < deadline - 20: # Leave time for construction
            improved = False
            for i in range(len(path) - 2):
                if time.time() > deadline - 20: break
                for j in range(i + 1, len(path) - 1):
                    # Reverse path[i+1..j]
                    # Check change in overlap
                    # Old: (path[i], path[i+1]) and (path[j], path[j+1])
                    # New: (path[i], path[j]) and (path[i+1], path[j+1])
                    # Note: path[j+1] exists if j < len(path)-1
                    
                    if j == len(path) - 1:
                        # Only (path[i], path[i+1]) changes to (path[i], path[j])
                        # No (path[j], path[j+1])
                        old_ov = 0
                        p1 = perms[path[i]]
                        q1 = perms[path[i+1]]
                        for k in range(5, -1, -1):
                            if p1[-k:] == q1[:k]: old_ov = k; break
                        
                        new_ov = 0
                        p2 = perms[path[i]]
                        q2 = perms[path[j]]
                        for k in range(5, -1, -1):
                            if p2[-k:] == q2[:k]: new_ov = k; break
                        
                        if new_ov > old_ov:
                            path[i+1:j+1] = reversed(path[i+1:j+1])
                            current_overlap += new_ov - old_ov
                            improved = True
                            break # Restart loop? Or continue?
                    else:
                        # Both edges change
                        old_ov1 = 0
                        p1 = perms[path[i]]
                        q1 = perms[path[i+1]]
                        for k in range(5, -1, -1):
                            if p1[-k:] == q1[:k]: old_ov1 = k; break
                        
                        old_ov2 = 0
                        p2 = perms[path[j]]
                        q2 = perms[path[j+1]]
                        for k in range(5, -1, -1):
                            if p2[-k:] == q2[:k]: old_ov2 = k; break
                        
                        new_ov1 = 0
                        p3 = perms[path[i]]
                        q3 = perms[path[j]]
                        for k in range(5, -1, -1):
                            if p3[-k:] == q3[:k]: new_ov1 = k; break
                        
                        new_ov2 = 0
                        p4 = perms[path[i+1]]
                        q4 = perms[path[j+1]]
                        for k in range(5, -1, -1):
                            if p4[-k:] == q4[:k]: new_ov2 = k; break
                        
                        if new_ov1 + new_ov2 > old_ov1 + old_ov2:
                            path[i+1:j+1] = reversed(path[i+1:j+1])
                            current_overlap += (new_ov1 + new_ov2) - (old_ov1 + old_ov2)
                            improved = True
                            break # Restart i loop?
                if improved:
                    break # Restart i loop
                
        # Construct S_7 using the optimized path
        new_word = []
        for idx in path:
            p = perms[idx]
            block = list(p) + [7] + list(p)
            overlap = 0
            # Overlap with current new_word
            # Suffix of new_word vs prefix of block
            # new_word ends with previous block's tail
            # We can just do the generic overlap check
            min_len = min(len(new_word), len(block))
            for o in range(min_len, -1, -1):
                if new_word[len(new_word)-o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
            
        return "".join(map(str, new_word))
    
    # Fallback for n > 7 (should not happen per task, but good practice)
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
