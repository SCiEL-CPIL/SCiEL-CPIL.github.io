import time
import random
import math

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # 1. Generate standard recursive superpermutation
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
        
    S = list(word)
    best_S = S[:]
    best_len = len(S)
    
    def build_counts(S, n):
        counts = {}
        for i in range(len(S) - n + 1):
            p = tuple(S[i:i+n])
            counts[p] = counts.get(p, 0) + 1
        return counts

    def is_valid(S, n):
        seen = set()
        for i in range(len(S) - n + 1):
            seen.add(tuple(S[i:i+n]))
        total = 1
        for k in range(2, n+1):
            total *= k
        return len(seen) == total

    def local_search(S, n, deadline):
        """
        Perform local search on S using deletion passes.
        Returns the optimized S.
        """
        counts = build_counts(S, n)
        
        def try_remove_pass(S, counts, n, deadline, forward=False):
            removed = False
            if forward:
                i = 0
                while i < len(S):
                    if time.time() > deadline:
                        break
                    # ... same logic as seed ...
                    i += 1 # or not, depending on removal
            else:
                i = len(S) - 1
                while i >= 0:
                    if time.time() > deadline:
                        break
                    # ... same logic as seed ...
                    i -= 1 # or not
            return S, counts, removed

        # Re-implementing the efficient removal pass from seed, but cleaner
        def remove_pass(S, counts, n, deadline, forward=False):
            removed_any = False
            if forward:
                i = 0
                while i < len(S):
                    if time.time() > deadline:
                        break
                    start_old = max(0, i - n + 1)
                    end_old = min(i, len(S) - n)
                    
                    valid = True
                    old_perms = []
                    for j in range(start_old, end_old + 1):
                        p = tuple(S[j:j+n])
                        if counts.get(p, 0) == 1:
                            valid = False
                            break
                        old_perms.append(p)
                    
                    if not valid:
                        i += 1
                        continue
                    
                    # Remove
                    for p in old_perms:
                        counts[p] -= 1
                        if counts[p] == 0:
                            del counts[p]
                    
                    new_len = len(S) - 1
                    start_new = max(0, i - n + 1)
                    end_new = min(i - 1, new_len - n)
                    
                    for k in range(start_new, end_new + 1):
                        window_chars = []
                        for idx in range(k, k+n):
                            s_idx = idx if idx < i else idx + 1
                            window_chars.append(S[s_idx])
                        p_new = tuple(window_chars)
                        counts[p_new] = counts.get(p_new, 0) + 1
                    
                    S.pop(i)
                    removed_any = True
                    # Don't increment i
            else:
                i = len(S) - 1
                while i >= 0:
                    if time.time() > deadline:
                        break
                    start_old = max(0, i - n + 1)
                    end_old = min(i, len(S) - n)
                    
                    valid = True
                    old_perms = []
                    for j in range(start_old, end_old + 1):
                        p = tuple(S[j:j+n])
                        if counts.get(p, 0) == 1:
                            valid = False
                            break
                        old_perms.append(p)
                    
                    if not valid:
                        i -= 1
                        continue
                    
                    for p in old_perms:
                        counts[p] -= 1
                        if counts[p] == 0:
                            del counts[p]
                    
                    new_len = len(S) - 1
                    start_new = max(0, i - n + 1)
                    end_new = min(i - 1, new_len - n)
                    
                    for k in range(start_new, end_new + 1):
                        window_chars = []
                        for idx in range(k, k+n):
                            s_idx = idx if idx < i else idx + 1
                            window_chars.append(S[s_idx])
                        p_new = tuple(window_chars)
                        counts[p_new] = counts.get(p_new, 0) + 1
                    
                    S.pop(i)
                    removed_any = True
                    # Don't decrement i
            
            return S, counts, removed_any

        # Run alternating passes until no more removals or time out
        while time.time() < deadline:
            S, counts, r1 = remove_pass(S, counts, n, deadline, forward=False)
            if not r1:
                break
            S, counts, r2 = remove_pass(S, counts, n, deadline, forward=True)
            if not r2:
                # Check if another backward pass finds anything
                S, counts, r3 = remove_pass(S, counts, n, deadline, forward=False)
                if not r3:
                    break
        
        return S

    # ILS Loop
    current_S = best_S[:]
    
    while time.time() < deadline:
        # Perturb current_S
        S_perturbed = current_S[:]
        
        # Strategy: Insert a few random characters
        num_inserts = random.randint(1, 5)
        for _ in range(num_inserts):
            if len(S_perturbed) == 0:
                S_perturbed.append(random.randint(1, n))
            else:
                pos = random.randint(0, len(S_perturbed))
                S_perturbed.insert(pos, random.randint(1, n))
        
        # Run Local Search
        S_ls = local_search(S_perturbed, n, deadline)
        
        # Update Best
        if len(S_ls) < best_len:
            best_S = S_ls
            best_len = len(S_ls)
            current_S = S_ls[:]
        else:
            # Accept with some probability to explore
            if random.random() < 0.3:
                current_S = S_ls[:]
        
        # Safety: if we are taking too long on one LS, break
        if time.time() > deadline - 5:
            break

    # Final Polish: Run local search on best_S one last time
    if time.time() < deadline - 2:
        best_S = local_search(best_S, n, deadline)
        best_len = len(best_S)
        
    # Final Validity Check
    if not is_valid(best_S, n):
        return "".join(map(str, word))
    
    return "".join(map(str, best_S))
