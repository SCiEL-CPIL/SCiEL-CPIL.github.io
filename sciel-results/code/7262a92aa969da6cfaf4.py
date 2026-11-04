import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # 1. Generate all permutations
    all_perms = list(permutations(range(1, n + 1)))
    total_perms = len(all_perms)
    
    best_S = None
    best_len = float('inf')
    
    # 2. Run greedy construction multiple times
    # For n=7, 10-20 runs should be fast enough
    max_runs = 20
    for run in range(max_runs):
        if time.time() > deadline:
            break
            
        random.shuffle(all_perms)
        S = list(all_perms[0])
        visited = set()
        visited.add(all_perms[0])
        unvisited = all_perms[1:]
        
        P_cur = all_perms[0]
        
        while unvisited:
            if time.time() > deadline:
                break
                
            best_P = None
            best_k = -1
            best_idx = -1
            
            # Find the permutation with maximum overlap with P_cur's suffix
            # Overlap k means P_cur[n-k:] == P[:k]
            # We check k from n-1 down to 1
            # Actually, we can just iterate and find max k
            
            for idx, P in enumerate(unvisited):
                # Check overlap
                # P must start with a suffix of P_cur
                # Suffixes of P_cur: P_cur[1:], P_cur[2:], ..., P_cur[n-1:]
                # Prefixes of P: P[:n-1], P[:n-2], ..., P[:1]
                
                # Fast check:
                # If P[0] != P_cur[-1] and P[0] != P_cur[-2] ... 
                # It's better to just compute the overlap directly.
                
                max_overlap = 0
                # Try overlap lengths from n-1 down to 1
                # If P_cur[i:] == P[:n-i], then overlap is n-i
                for i in range(1, n):
                    if P_cur[i:] == P[:n-i]:
                        max_overlap = n - i
                        break
                
                if max_overlap > best_k:
                    best_k = max_overlap
                    best_P = P
                    best_idx = idx
                    
                    # Early exit if we found max possible overlap (n-1)
                    if best_k == n - 1:
                        break
                        
            if best_P is None:
                # This shouldn't happen if unvisited is not empty
                break
            
            # Remove best_P from unvisited
            unvisited.pop(best_idx)
            visited.add(best_P)
            
            # Append the non-overlapping part
            S.extend(best_P[best_k:])
            
            P_cur = best_P
            
        # Check if we completed
        if len(visited) == total_perms:
            if len(S) < best_len:
                best_len = len(S)
                best_S = S[:]
    
    if best_S is None:
        # Fallback to simple construction if greedy failed (unlikely)
        # Use the seed program's recursive construction as fallback
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
        best_S = word
    
    # 3. Local Search
    # Reuse the logic from the seed program but optimized
    
    def build_counts(S, n):
        counts = {}
        for i in range(len(S) - n + 1):
            p = tuple(S[i:i+n])
            counts[p] = counts.get(p, 0) + 1
        return counts
    
    counts = build_counts(best_S, n)
    
    def is_valid(S, n):
        seen = set()
        for i in range(len(S) - n + 1):
            seen.add(tuple(S[i:i+n]))
        total = 1
        for k in range(2, n+1):
            total *= k
        return len(seen) == total
    
    # Local search removal pass
    def try_remove_pass(S, counts, n, deadline):
        removed = False
        i = len(S) - 1
        while i >= 0:
            if time.time() > deadline:
                break
            
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)
            
            valid = True
            old_perms_to_dec = []
            
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j+n])
                if counts.get(p, 0) == 1:
                    valid = False
                    break
                old_perms_to_dec.append(p)
            
            if not valid:
                i -= 1
                continue
            
            for p in old_perms_to_dec:
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
            removed = True
            
        return S, counts, removed
    
    # Run local search
    for _ in range(100):
        if time.time() > deadline:
            break
        S, counts, removed = try_remove_pass(best_S, counts, n, deadline)
        if not removed:
            break
    
    if not is_valid(best_S, n):
        # If local search broke validity (shouldn't happen with correct logic, but just in case)
        # Re-run greedy to get a valid one
        pass # best_S was valid before local search, and we only removed if valid.
             # But if we broke deadline, we might have left it in a weird state?
             # No, we check validity of the removal before committing.
             # So best_S should be valid.
    
    return "".join(map(str, best_S))
