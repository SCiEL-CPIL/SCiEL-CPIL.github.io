import time
import itertools

def construct(n):
    if n == 1:
        return "1"
    
    # Helper to get standard GL word and perms
    def get_gl(m):
        if m == 1:
            return [1], [(1,)]
        prev_word, prev_perms = get_gl(m-1)
        new_word = []
        for p in prev_perms:
            block = list(p) + [m] + list(p)
            # overlap
            max_ov = min(len(new_word), len(block))
            ov = 0
            for o in range(max_ov, -1, -1):
                if new_word[len(new_word)-o:] == block[:o]:
                    ov = o
                    break
            new_word += block[ov:]
        
        # extract perms of length m
        perms = []
        seen = set()
        for i in range(len(new_word) - m + 1):
            w = tuple(new_word[i:i+m])
            if len(set(w)) == m and w not in seen:
                seen.add(w)
                perms.append(w)
        return new_word, perms

    if n <= 6:
        word, _ = get_gl(n)
        return "".join(map(str, word))
    
    if n == 7:
        # Get perms of 1..6 in standard order
        _, perms = get_gl(6)
        num_perms = len(perms)
        
        # Precompute overlap matrix
        # L[i][j] = max k such that perms[i][-k:] == perms[j][:k]
        # k in 0..5
        L = [[0]*num_perms for _ in range(num_perms)]
        for i in range(num_perms):
            pi = perms[i]
            for j in range(num_perms):
                pj = perms[j]
                max_k = 5 # max possible overlap for distinct perms of len 6
                # actually can be up to 5
                for k in range(max_k, -1, -1):
                    if pi[-k:] == pj[:k]:
                        L[i][j] = k
                        break
        
        # Initial order: 0..num_perms-1
        order = list(range(num_perms))
        
        def calc_overlap(order):
            total = 0
            for k in range(len(order)-1):
                total += L[order[k]][order[k+1]]
            return total
        
        best_overlap = calc_overlap(order)
        best_order = order[:]
        
        # Local Search: 2-opt
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 100.0 # 100s for search, 20s buffer
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try all 2-opt moves
            for i in range(num_perms - 1):
                if time.time() > deadline: break
                for j in range(i + 1, num_perms):
                    # Reverse order[i:j+1]
                    # Calculate change in overlap
                    # Old links: (i-1, i), (j, j+1)
                    # New links: (i-1, j), (i, j+1)
                    # Note: if i==0, no (i-1, i). if j==n-1, no (j, j+1).
                    
                    old_score = 0
                    new_score = 0
                    
                    # Link before i
                    if i > 0:
                        old_score += L[order[i-1]][order[i]]
                        new_score += L[order[i-1]][order[j]]
                    # Link after j
                    if j < num_perms - 1:
                        old_score += L[order[j]][order[j+1]]
                        new_score += L[order[i]][order[j+1]]
                    
                    # Internal links in reversed segment
                    # Old: L[order[k]][order[k+1]] for k in i..j-1
                    # New: L[order[k]][order[k-1]] for k in i..j-1 (which is L[order[k-1]][order[k]] reversed? No)
                    # The segment is order[i], order[i+1], ..., order[j].
                    # Reversed: order[j], order[j-1], ..., order[i].
                    # Old internal sum: sum_{k=i}^{j-1} L[order[k]][order[k+1]]
                    # New internal sum: sum_{k=i}^{j-1} L[order[j-(k-i)]][order[j-(k-i)-1]]
                    # Let m = j-i.
                    # New internal sum: sum_{t=0}^{m-1} L[order[j-t]][order[j-t-1]]
                    
                    # This calculation is O(j-i).
                    # Total 2-opt is O(N^3) per pass. N=720. 720^3 ~ 370M. Too slow in Python.
                    
                    # Optimization: Precompute prefix sums of overlaps?
                    # Or just do simple swaps (1-opt) first, which is O(N^2) per pass.
                    # Or use a faster heuristic.
                    
                    pass # skip full 2-opt for now in thought process, implement 1-opt (adjacent swap) which is O(N^2) per pass.
            
            # Let's stick to adjacent swaps (1-opt) for simplicity and speed in Python.
            # It might not find the global optimum but should improve significantly.
            pass
            
        # Actually, let's implement 2-opt with a time check inside the inner loop, but break early if too slow.
        # Or just do random restarts of 2-opt?
        # Given 100s, we can do several passes of 2-opt.
        # But 720^3 is heavy.
        # Let's use "Or-opt" or just 2-opt with early termination.
        
        # Alternative: Simulated Annealing?
        # Let's try 2-opt but optimized.
        
        # We'll implement 2-opt properly.
        
        def do_2opt(order, deadline):
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(num_perms - 1):
                    if time.time() > deadline:
                        return order
                    for j in range(i + 1, num_perms):
                        # Calculate delta
                        # Old: (i-1, i), (j, j+1) + internal
                        # New: (i-1, j), (i, j+1) + internal_rev
                        
                        delta = 0
                        
                        # Boundary 1
                        if i > 0:
                            delta += L[order[i-1]][order[j]] - L[order[i-1]][order[i]]
                        # Boundary 2
                        if j < num_perms - 1:
                            delta += L[order[i]][order[j+1]] - L[order[j]][order[j+1]]
                        
                        # Internal
                        # Sum of L[a][b] - L[b][a] for the segment
                        # We need sum_{k=i}^{j-1} ( L[order[k]][order[k+1]] - L[order[k+1]][order[k]] )? No.
                        # Old internal: S_old = sum_{k=i}^{j-1} L[order[k]][order[k+1]]
                        # New internal: S_new = sum_{k=i}^{j-1} L[order[j-(k-i)]][order[j-(k-i)-1]]
                        # Let's iterate.
                        # If j-i is large, this is slow.
                        
                        # Optimization: Precompute prefix sums of L?
                        # L is not symmetric.
                        
                        # For now, if j-i > 50, skip? Or just compute.
                        # 720*720/2 ~ 250k pairs.
                        # Average length 360.
                        # 250k * 360 ~ 90M operations.
                        # In Python, 90M ops might take ~10-20s.
                        # So one pass of 2-opt is feasible.
                        
                        s_old = 0
                        s_new = 0
                        # Unroll for speed?
                        for k in range(i, j):
                            s_old += L[order[k]][order[k+1]]
                            s_new += L[order[j-(k-i)]][order[j-(k-i)-1]]
                        
                        delta += s_new - s_old
                        
                        if delta > 0:
                            # Apply reversal
                            order[i:j+1] = reversed(order[i:j+1])
                            improved = True
            return order

        order = do_2opt(order, deadline)
        
        # Construct final word
        final_word = []
        for idx in order:
            p = perms[idx]
            block = list(p) + [n] + list(p)
            max_overlap = min(len(final_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if final_word[len(final_word) - o:] == block[:o]:
                    overlap = o
                    break
            final_word += block[overlap:]
            
        return "".join(map(str, final_word))
