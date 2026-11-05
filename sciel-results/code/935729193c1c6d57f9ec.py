# EVOLVE-BLOCK-START
import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    if n <= 6:
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
        # Build S_6 standard construction
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
        
        # Extract all 720 permutations of 1..6 in order of appearance
        perms, seen = [], set()
        for i in range(len(word) - 5):
            window = tuple(word[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        N = len(perms)
        
        # Precompute overlap matrix
        ov = [[0]*N for _ in range(N)]
        for i in range(N):
            p = perms[i]
            for j in range(N):
                if i == j:
                    ov[i][j] = 6
                    continue
                q = perms[j]
                best = 0
                for k in range(5, 0, -1):
                    if p[-k:] == q[:k]:
                        best = k
                        break
                ov[i][j] = best
        
        def calc_total_overlap(path):
            total = 0
            for i in range(len(path) - 1):
                total += ov[path[i]][path[i+1]]
            return total
        
        def build_word_from_path(path):
            new_word = []
            for idx in path:
                p = perms[idx]
                block = list(p) + [7] + list(p)
                overlap = 0
                min_len = min(len(new_word), len(block))
                for o in range(min_len, -1, -1):
                    if new_word[len(new_word)-o:] == block[:o]:
                        overlap = o
                        break
                new_word += block[overlap:]
            return new_word
        
        # Initialize with standard order
        best_path = list(range(N))
        best_overlap = calc_total_overlap(best_path)
        
        # Local search: 2-opt and Or-Opt
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current_path = best_path[:]
            current_overlap = best_overlap
            
            # 2-opt
            for i in range(N - 1):
                if time.time() > deadline - 5: break
                for j in range(i + 1, N):
                    if j == N - 1:
                        old_ov = ov[current_path[i]][current_path[i+1]]
                        new_ov = ov[current_path[i]][current_path[j]]
                        if new_ov > old_ov:
                            current_path[i+1:j+1] = reversed(current_path[i+1:j+1])
                            current_overlap += new_ov - old_ov
                            improved = True
                            break
                    else:
                        old_ov = ov[current_path[i]][current_path[i+1]] + ov[current_path[j]][current_path[j+1]]
                        new_ov = ov[current_path[i]][current_path[j]] + ov[current_path[i+1]][current_path[j+1]]
                        if new_ov > old_ov:
                            current_path[i+1:j+1] = reversed(current_path[i+1:j+1])
                            current_overlap += new_ov - old_ov
                            improved = True
                            break
                if improved:
                    break
            
            # Or-Opt (move 1 or 2 elements)
            if not improved and time.time() < deadline - 5:
                for length in [1, 2]:
                    for i in range(N - length + 1):
                        if time.time() > deadline - 5: break
                        block = current_path[i:i+length]
                        # Remove block
                        if i > 0:
                            left = current_path[i-1]
                        else:
                            left = -1
                        if i + length < N:
                            right = current_path[i+length]
                        else:
                            right = -1
                        
                        # Calculate cost of removing block
                        remove_cost = 0
                        if left != -1 and right != -1:
                            remove_cost = ov[left][right]
                        elif left != -1:
                            remove_cost = 0
                        elif right != -1:
                            remove_cost = 0
                        else:
                            remove_cost = 0
                        
                        # Original cost involving block
                        orig_cost = 0
                        if i > 0:
                            orig_cost += ov[current_path[i-1]][current_path[i]]
                        if i + length < N:
                            orig_cost += ov[current_path[i+length-1]][current_path[i+length]]
                        for k in range(length-1):
                            orig_cost += ov[current_path[i+k]][current_path[i+k+1]]
                        
                        # Try inserting at all positions
                        for k in range(N - length + 1):
                            if k == i: continue
                            # Calculate cost of inserting at k
                            insert_cost = 0
                            if k > 0:
                                insert_cost += ov[current_path[k-1]][block[0]]
                            if k < N - length + 1:
                                # Careful with indices after removal
                                if k <= i:
                                    # Insert before original position
                                    if k + length < N - length + 1:
                                        # This logic is getting complex with indices shifting
                                        pass
                            # Simplified: just try and check if it improves
                            # To keep it simple and fast, we'll do a direct check
                            new_path = current_path[:]
                            del new_path[i:i+length]
                            # Insert at k in new_path
                            new_path[k:k] = block
                            new_overlap = calc_total_overlap(new_path)
                            if new_overlap > current_overlap:
                                current_path = new_path
                                current_overlap = new_overlap
                                improved = True
                                break
                    if improved:
                        break
            
            if improved:
                best_path = current_path
                best_overlap = current_overlap
        
        # Build final word
        final_word = build_word_from_path(best_path)
        return "".join(map(str, final_word))
    
    # Fallback for n > 7
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
