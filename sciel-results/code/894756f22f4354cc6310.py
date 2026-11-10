import itertools
import time

def construct(n):
    if n <= 6:
        # Use the classic recursive construction for n <= 6 as it is optimal
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
                max_ov = min(len(new), len(block))
                for o in range(max_ov, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    if n == 7:
        # Generate S6
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
                max_ov = min(len(new), len(block))
                for o in range(max_ov, -1, -1, ):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word6 = new
        
        # Get the sequence of permutations in S6
        # The permutations are the length-6 windows
        perms_in_s6 = []
        seen = set()
        for i in range(len(word6) - 6 + 1):
            window = tuple(word6[i:i + 6])
            if window not in seen:
                seen.add(window)
                perms_in_s6.append(window)
        
        # The last permutation in S6 is the one that ends the string
        # Actually, the last window is the last 6 elements
        p_last = tuple(word6[-6:])
        
        # We need to order all 720 permutations of 1..6
        # Let's generate all permutations
        all_perms = list(itertools.permutations(range(1, 7)))
        
        # Precompute overlap matrix
        # Overlap between p_i and p_j is the longest suffix of p_i that is a prefix of p_j
        # We want to find a path P_1...P_720 that maximizes sum of overlaps
        # Start overlap is with p_last
        
        # Greedy approach:
        # Start with the permutation that has the best overlap with p_last?
        # Or just start with any and pick next best.
        
        # Let's use a simple greedy: 
        # 1. Find P_1 that maximizes overlap(p_last, P_1)
        # 2. For i from 1 to 719, find P_{i+1} (not used) that maximizes overlap(P_i, P_{i+1})
        
        # To speed up, we can precompute overlaps.
        # But 720*720 is small.
        
        # Let's try to improve with local search (2-opt)
        
        # Initialize path
        used = [False] * 720
        path = [0] * 720
        
        # Find best start
        best_start = 0
        best_ov = -1
        for i, p in enumerate(all_perms):
            ov = 0
            # overlap between p_last and p
            # p_last is tuple, p is tuple
            max_k = 6
            for k in range(max_k, -1, -1):
                if p_last[6-k:] == p[:k]:
                    ov = k
                    break
            if ov > best_ov:
                best_ov = ov
                best_start = i
        
        path[0] = best_start
        used[best_start] = True
        
        # Greedy fill
        for i in range(1, 720):
            p_prev = all_perms[path[i-1]]
            best_next = -1
            best_ov = -1
            for j in range(720):
                if not used[j]:
                    p_curr = all_perms[j]
                    ov = 0
                    for k in range(6, -1, -1):
                        if p_prev[6-k:] == p_curr[:k]:
                            ov = k
                            break
                    if ov > best_ov:
                        best_ov = ov
                        best_next = j
            path[i] = best_next
            used[best_next] = True
        
        # Calculate total length
        # Length = len(word6) + sum(12 - overlap)
        # Overlaps:
        # 1. p_last to path[0]
        # 2. path[i] to path[i+1]
        
        total_len = len(word6)
        p_prev = p_last
        for i in range(720):
            p_curr = all_perms[path[i]]
            ov = 0
            for k in range(6, -1, -1):
                if p_prev[6-k:] == p_curr[:k]:
                    ov = k
                    break
            total_len += 12 - ov
            p_prev = p_curr
        
        # Construct the string
        # Start with word6
        result = word6[:]
        p_prev = p_last
        for i in range(720):
            p_curr = all_perms[path[i]]
            block = list(p_curr) + [7] + list(p_curr)
            # Find overlap between current result suffix and block prefix
            # Current result ends with p_prev
            # Block starts with p_curr
            ov = 0
            for k in range(6, -1, -1):
                if p_prev[6-k:] == p_curr[:k]:
                    ov = k
                    break
            result += block[ov:]
            p_prev = p_curr
            
        return "".join(map(str, result))
    
    # Fallback for n > 7 (not expected)
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
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
