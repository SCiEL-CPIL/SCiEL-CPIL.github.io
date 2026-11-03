import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = 115.0
    
    if n < 2:
        return "1"
    
    def generate_standard(n):
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
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word

    def get_counts(word, n, all_perms):
        counts = {p: 0 for p in all_perms}
        for i in range(len(word) - n + 1):
            p = tuple(word[i:i+n])
            if p in counts:
                counts[p] += 1
        return counts

    def try_delete(word, counts, i, n, all_perms):
        # Returns (success, new_word, new_counts)
        affected = []
        start_min = max(0, i - n + 1)
        start_max = min(i, len(word) - n)
        
        for j in range(start_min, start_max + 1):
            p = tuple(word[j:j+n])
            if p in counts:
                affected.append(p)
        
        # Decrement
        for p in affected:
            counts[p] -= 1
            if counts[p] < 0:
                # Restore and fail
                for q in affected:
                    counts[q] += 1
                return False, word, counts
        
        new_word = word[:i] + word[i+1:]
        return True, new_word, counts

    def greedy_delete(word, counts, n, all_perms, reverse=False):
        w = word
        c = counts.copy()
        i = 0
        step = 1 if not reverse else -1
        limit = len(w) if not reverse else -1
        # Simpler: just use a list of indices
        indices = list(range(len(w)))
        if reverse:
            indices.reverse()
            
        # We need to handle index shifting.
        # Easiest: Re-scan from start every time? No, too slow.
        # Let's stick to the sequential scan with index adjustment.
        
        # Actually, the sequential scan `i` logic is:
        # If delete, don't increment i.
        # If not, increment i.
        
        if reverse:
             # Reverse scan is tricky with the "don't increment" logic.
             # Let's just do LTR and RTL separately.
             pass
             
        # LTR
        i = 0
        while i < len(w):
            success, new_w, new_c = try_delete(w, c, i, n, all_perms)
            if success:
                w = new_w
                c = new_c
            else:
                i += 1
        return w, c

    def optimize(word, n, all_perms, time_limit):
        best_w = word
        best_c = get_counts(best_w, n, all_perms)
        best_len = len(best_w)
        
        # LTR
        w, c = greedy_delete(best_w, best_c, n, all_perms)
        if len(w) < best_len:
            best_w, best_c, best_len = w, c, len(w)
            
        # RTL (Implement by reversing, doing LTR, then reversing back? 
        # No, deletion in reverse is different.
        # Let's implement a reverse greedy delete.
        def greedy_delete_reverse(word, counts, n, all_perms):
            w = word
            c = counts.copy()
            i = len(w) - 1
            while i >= 0:
                success, new_w, new_c = try_delete(w, c, i, n, all_perms)
                if success:
                    w = new_w
                    c = new_c
                else:
                    i -= 1
            return w, c
            
        w, c = greedy_delete_reverse(best_w, best_c, n, all_perms)
        if len(w) < best_len:
            best_w, best_c, best_len = w, c, len(w)
            
        # Random Swaps + Deletion
        while time.time() < time_limit:
            candidate_w = best_w
            candidate_c = best_c.copy()
            
            # Try a few random swaps
            for _ in range(10):
                if len(candidate_w) < 2: break
                i = random.randint(0, len(candidate_w) - 2)
                # Swap i and i+1
                old_i, old_ip1 = candidate_w[i], candidate_w[i+1]
                candidate_w[i], candidate_w[i+1] = old_ip1, old_i
                
                # Check validity of swap
                # Only windows containing i or i+1 are affected
                # Start j in [i-n+1, i+1]
                affected = []
                start_min = max(0, i - n + 1)
                start_max = min(i + 1, len(candidate_w) - n)
                
                valid_swap = True
                for j in range(start_min, start_max + 1):
                    p = tuple(candidate_w[j:j+n])
                    if p in candidate_c:
                         affected.append((j, p))
                         candidate_c[p] -= 1
                         if candidate_c[p] < 0:
                             valid_swap = False
                             break
                
                if not valid_swap:
                    # Restore
                    for j, p in affected:
                        candidate_c[p] += 1
                    # Undo swap
                    candidate_w[i], candidate_w[i+1] = old_i, old_ip1
                else:
                    # Accept swap
                    pass
            
            # Now delete
            w, c = greedy_delete(candidate_w, candidate_c, n, all_perms)
            if len(w) < best_len:
                best_w, best_c, best_len = w, c, len(w)
                
        return best_w

    # Stage 1: Optimize n=6
    if n == 7:
        all_perms_6 = set(permutations(range(1, 7)))
        std_6 = generate_standard(6)
        opt_6 = optimize(std_6, 6, all_perms_6, 20.0)
        
        # Generate L(7) from opt_6
        # Extract perms from opt_6
        perms, seen = [], set()
        for i in range(len(opt_6) - 6 + 1):
            window = tuple(opt_6[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Build L(7)
        word = []
        for p in perms:
            block = list(p) + [7] + list(p)
            max_overlap = min(len(word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or word[len(word) - o:] == block[:o]:
                    overlap = o
                    break
            word += block[overlap:]
        
        # Stage 2: Optimize L(7)
        all_perms_7 = set(permutations(range(1, 8)))
        best_7 = optimize(word, 7, all_perms_7, deadline - time.time())
        return "".join(map(str, best_7))
        
    else:
        all_perms = set(permutations(range(1, n + 1)))
        std = generate_standard(n)
        best = optimize(std, n, all_perms, deadline - time.time())
        return "".join(map(str, best))

