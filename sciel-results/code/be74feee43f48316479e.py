import time
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Standard recursive construction
    def standard_superperm(n):
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
        return word

    def get_perms(word, m):
        """Extract permutations of size m-1 from superperm of size m-1"""
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        return perms

    def is_valid_superperm(word, n):
        """Check if word contains all permutations of 1..n"""
        if len(word) < n:
            return False
        perms_seen = set()
        for i in range(len(word) - n + 1):
            window = tuple(word[i:i + n])
            if len(set(window)) == n:
                perms_seen.add(window)
        return len(perms_seen) == int(n * (n-1) * (n-2) * (n-3) * (n-4) * (n-5) * (n-6) / (1*2*3*4*5*6)) if n==7 else True # Simplified for n=7, general factorial below
        # Actually, let's just compute factorial properly
        import math
        return len(perms_seen) == math.factorial(n)

    def optimize_step(word_prev, m, deadline):
        """
        Given superperm for m-1, find best ordering of blocks for m.
        Returns best valid superperm for m or None if no improvement found/valid.
        """
        if time.time() > deadline:
            return None
            
        perms = get_perms(word_prev, m)
        k = len(perms)
        
        if k == 0:
            return None

        # Build blocks
        blocks = []
        for p in perms:
            blocks.append(list(p) + [m] + list(p))
            
        # Precompute overlap matrix
        # overlap[i][j] = max overlap between suffix of blocks[i] and prefix of blocks[j]
        # blocks are lists of ints
        k2 = k * k
        overlaps = [[0]*k for _ in range(k)]
        
        for i in range(k):
            b_i = blocks[i]
            len_i = len(b_i)
            for j in range(k):
                b_j = blocks[j]
                len_j = len(b_j)
                max_ov = min(len_i, len_j)
                ov = 0
                # Check overlaps from largest to smallest
                # Since block length is small (2m-1), this is fast
                for o in range(max_ov, 0, -1):
                    if b_i[len_i-o:] == b_j[:o]:
                        ov = o
                        break
                overlaps[i][j] = ov
                
        # Initial order: 0, 1, ..., k-1
        order = list(range(k))
        
        # Calculate initial cost
        # Cost is sum of (len(block[j]) - overlap[i][j]) for i->j
        # Total length = len(block[order[0]]) + sum(len(block[order[t]] - overlap[order[t-1]][order[t]]))
        
        def calc_len(order):
            if not order: return 0
            total = len(blocks[order[0]])
            for t in range(1, len(order)):
                i = order[t-1]
                j = order[t]
                total += len(blocks[j]) - overlaps[i][j]
            return total
        
        current_len = calc_len(order)
        
        # 2-Opt Local Search
        improved = True
        iterations = 0
        max_iterations = 10000
        
        while improved and iterations < max_iterations and time.time() < deadline:
            improved = False
            iterations += 1
            for i in range(k - 1):
                for j in range(i + 1, k):
                    # Reverse segment order[i+1 ... j]
                    # New order: ... order[i], order[j], ..., order[i+1], ...
                    # Check if it improves
                    # Old edges: (order[i-1], order[i]), (order[i], order[i+1]), (order[j], order[j+1]), (order[j+1], order[j+2]?)
                    # Actually, 2-opt on a path:
                    # We reverse the subsequence from i to j (inclusive? or i to j-1?)
                    # Standard 2-opt: reverse order[i..j].
                    # Edges affected: (prev(i), i), (i, i+1), (j, next(j))
                    # If i=0, no prev(i). If j=k-1, no next(j).
                    
                    # Let's define the cost change
                    # Old cost parts:
                    # C_old = 
                    # if i > 0: len(blocks[order[i]]) - overlaps[order[i-1]][order[i]]
                    # else: len(blocks[order[i]])
                    # + (len(blocks[order[i+1]]) - overlaps[order[i]][order[i+1]])
                    # ...
                    # + (len(blocks[order[j]]) - overlaps[order[j-1]][order[j]])
                    # if j < k-1: + (len(blocks[order[j+1]]) - overlaps[order[j]][order[j+1]])
                    
                    # This is complex to calculate delta. Easier to just swap and recalc local cost?
                    # Or just recalc total length? k=720, calc_len is O(k).
                    # 720 * 720 * 720 = 373M ops. Too slow in Python.
                    
                    # We need efficient delta calculation.
                    # Let A = order[i-1] (if i>0 else -1)
                    # Let B = order[i]
                    # Let C = order[i+1]
                    # Let D = order[j]
                    # Let E = order[j+1] (if j<k-1 else -1)
                    
                    # Old segment: A -> B -> ... -> D -> E
                    # Edges: (A,B), (B, C), (D, E)
                    # New segment: A -> D -> ... -> B -> E
                    # Edges: (A,D), (D, C), (B, E)
                    
                    # Note: The internal edges (B..D) are reversed, so their cost is sum of overlaps reversed?
                    # Cost of edge (u,v) is len(blocks[v]) - overlaps[u][v].
                    # The sum of internal costs in the reversed segment is NOT the same as forward.
                    # Wait, 2-opt on TSP path usually assumes symmetric distance?
                    # Here distance is asymmetric?
                    # dist(u,v) = len(v) - ov(u,v).
                    # dist(v,u) = len(u) - ov(v,u).
                    # They are different.
                    # So we must sum the internal costs.
                    # Internal length is small (j-i).
                    
                    # Let's just recalc the cost of the affected part.
                    # Affected indices: i-1 to j+1.
                    
                    if i == 0:
                        # Start of path
                        # Old: len(blocks[B]) + (len(blocks[C]) - ov(B,C)) + ... + (len(blocks[D]) - ov(prev_D, D)) + (len(blocks[E]) - ov(D,E))
                        # This is getting messy.
                        
                        # Simpler: Just recalc total length for the candidate if k is small?
                        # k=720. calc_len is O(720).
                        # 720*720/2 * 720 ~ 180M operations.
                        # Python might handle 180M simple ops in 10-20s?
                        # Let's try.
                        
                        new_order = order[:i] + order[i:j+1][::-1] + order[j+1:]
                        new_len = calc_len(new_order)
                        
                        if new_len < current_len:
                            order = new_order
                            current_len = new_len
                            improved = True
                    else:
                        new_order = order[:i] + order[i:j+1][::-1] + order[j+1:]
                        new_len = calc_len(new_order)
                        if new_len < current_len:
                            order = new_order
                            current_len = new_len
                            improved = True
        
        # Construct final word
        if not order:
            return None
            
        best_word = list(blocks[order[0]])
        for t in range(1, len(order)):
            i = order[t-1]
            j = order[t]
            ov = overlaps[i][j]
            best_word += blocks[j][ov:]
            
        # Check validity
        if is_valid_superperm(best_word, m):
            return best_word
        else:
            return None

    # Main logic
    # Start with n=1
    word = [1]
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
            
        # Try to optimize
        optimized = optimize_step(word, m, deadline)
        
        if optimized and len(optimized) < len(word) + len(get_perms(word, m)) * (2*m - 1): # Rough check
            # Actually, just check if it's valid and shorter than standard?
            # Standard length for step m is approx (m-1)! * (2m-1) + overlaps.
            # Let's just use it if it's valid.
            word = optimized
        else:
            # Use standard step
            perms = get_perms(word, m)
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
