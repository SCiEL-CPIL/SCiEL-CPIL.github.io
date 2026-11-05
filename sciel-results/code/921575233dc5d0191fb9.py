import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Standard construction for n <= 6
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
                # Find max overlap
                overlap = 0
                min_len = min(len(new), len(block))
                for o in range(min_len, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    if n == 7:
        # 1. Build S_6 standard construction
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
                overlap = 0
                min_len = min(len(new), len(block))
                for o in range(min_len, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        
        # 2. Extract all 720 permutations of 1..6
        perms, seen = [], set()
        for i in range(len(word) - 5):
            window = tuple(word[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        N = len(perms)
        
        # 3. Precompute overlap matrix
        # ov[i][j] is overlap length between perms[i] and perms[j]
        ov = [[0]*N for _ in range(N)]
        for i in range(N):
            p = perms[i]
            for j in range(N):
                if i == j:
                    ov[i][j] = 6
                    continue
                q = perms[j]
                best = 0
                # Max overlap is 5 (suffix of p matches prefix of q)
                # Actually, can it be more?
                # Block ends with p. Next block starts with q.
                # Overlap is suffix of p matching prefix of q.
                # Max length 5.
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
                    if new_word[len(new_word) - o:] == block[:o]:
                        overlap = o
                        break
                new_word += block[overlap:]
            return new_word
        
        # 4. Local Search
        # Initialize with standard order
        best_path = list(range(N))
        best_overlap = calc_total_overlap(best_path)
        
        # Or-Opt local search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current_path = best_path[:]
            current_overlap = best_overlap
            
            # Or-Opt (move 1, 2, or 3 elements)
            for length in [1, 2, 3]:
                if time.time() > deadline - 5:
                    break
                for i in range(N - length + 1):
                    if time.time() > deadline - 5:
                        break
                    block = current_path[i:i+length]
                    
                    # Cost of removing block
                    orig_cost = 0
                    if i > 0:
                        orig_cost += ov[current_path[i-1]][current_path[i]]
                    if i + length < N:
                        orig_cost += ov[current_path[i+length-1]][current_path[i+length]]
                    for k in range(length-1):
                        orig_cost += ov[block[k]][block[k+1]]
                    
                    # Cost of connecting gap
                    rem_cost = 0
                    if i > 0 and i + length < N:
                        rem_cost = ov[current_path[i-1]][current_path[i+length]]
                    
                    delta_rem = rem_cost - orig_cost
                    
                    # Try inserting at k
                    found_improvement = False
                    for k in range(N - length + 1):
                        if k == i or k == i + 1:
                            continue
                        
                        # Determine neighbors in the "gap" path
                        # If k < i:
                        #   Left: current_path[k-1] (if k>0)
                        #   Right: current_path[k]
                        # If k > i + length:
                        #   Left: current_path[k-1]
                        #   Right: current_path[k]
                        
                        # Case 1: k < i
                        if k < i:
                            ins_cost = 0
                            if k > 0:
                                ins_cost += ov[current_path[k-1]][block[0]]
                            ins_cost += ov[block[-1]][current_path[k]]
                            
                            # Broken edge at k
                            broken_cost = 0
                            if k > 0:
                                broken_cost = ov[current_path[k-1]][current_path[k]]
                            
                            delta_ins = ins_cost - broken_cost
                            total_delta = delta_rem + delta_ins
                        else:
                            # k > i + length
                            # After removal, the insertion position shifts
                            # The element that was at k is now at k - length
                            # Left neighbor is current_path[k-1] (which is now at k-1 in the modified path)
                            # Right neighbor is current_path[k]
                            ins_cost = 0
                            if k - 1 >= i + length:
                                # Left neighbor is current_path[k-1]
                                ins_cost += ov[current_path[k-1]][block[0]]
                            else:
                                # Left neighbor is current_path[i-1] if i > 0
                                if i > 0:
                                    ins_cost += ov[current_path[i-1]][block[0]]
                            
                            ins_cost += ov[block[-1]][current_path[k]]
                            
                            # Broken edge at k
                            broken_cost = 0
                            if k - 1 >= i + length:
                                broken_cost = ov[current_path[k-1]][current_path[k]]
                            else:
                                # The edge that was broken is between i-1 and i+length
                                if i > 0 and i + length < N:
                                    broken_cost = ov[current_path[i-1]][current_path[i+length]]
                                elif i > 0:
                                    broken_cost = 0
                                elif i + length < N:
                                    broken_cost = 0
                            
                            delta_ins = ins_cost - broken_cost
                            total_delta = delta_rem + delta_ins
                        
                        if total_delta > 0:
                            # Perform move
                            # Remove block
                            del current_path[i:i+length]
                            # Insert at k (adjust k if k > i)
                            if k > i:
                                k -= length
                            current_path[k:k] = block
                            current_overlap += total_delta
                            best_overlap = current_overlap
                            best_path = current_path[:]
                            improved = True
                            found_improvement = True
                            break
                    
                    if found_improvement:
                        break
                
                if improved:
                    break
        
        # 2-Opt local search (reverse segments)
        if time.time() < deadline - 5:
            current_path = best_path[:]
            current_overlap = best_overlap
            improved_2opt = True
            while improved_2opt and time.time() < deadline - 5:
                improved_2opt = False
                for i in range(N - 1):
                    if time.time() > deadline - 5:
                        break
                    for j in range(i + 1, N):
                        if time.time() > deadline - 5:
                            break
                        # Calculate delta for reversing [i, j]
                        old_ov = 0
                        new_ov = 0
                        
                        if i > 0:
                            old_ov += ov[current_path[i-1]][current_path[i]]
                            new_ov += ov[current_path[i-1]][current_path[j]]
                        if j < N - 1:
                            old_ov += ov[current_path[j]][current_path[j+1]]
                            new_ov += ov[current_path[i]][current_path[j+1]]
                        
                        # Internal edges change direction
                        internal_old = 0
                        internal_new = 0
                        for k in range(i, j):
                            internal_old += ov[current_path[k]][current_path[k+1]]
                            internal_new += ov[current_path[k+1]][current_path[k]]
                        
                        delta = (new_ov + internal_new) - (old_ov + internal_old)
                        
                        if delta > 0:
                            # Perform reversal
                            current_path[i:j+1] = reversed(current_path[i:j+1])
                            current_overlap += delta
                            best_overlap = current_overlap
                            best_path = current_path[:]
                            improved_2opt = True
                            break
                    if improved_2opt:
                        break
        
        # 5. Build final word
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
