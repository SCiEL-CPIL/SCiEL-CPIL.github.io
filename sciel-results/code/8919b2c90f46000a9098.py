import itertools
import random

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    if n > 7:
        # Fallback to standard construction for larger n
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

    # Heuristic for n=7 (and potentially 4,5,6)
    # Generate all permutations of 1..n-1
    prev_perms = list(itertools.permutations(range(1, n)))
    
    # Create blocks: p + [n] + p
    # Block length is 2*(n-1) + 1
    blocks = []
    for p in prev_perms:
        b = list(p) + [n] + list(p)
        blocks.append(b)
        
    num_blocks = len(blocks)
    
    # Precompute overlaps
    # M[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    # Max possible overlap is len(block) - 1? 
    # Actually, blocks are distinct permutations of 1..n-1 with n in center.
    # Overlap can be up to len(block)-1? 
    # If block i ends with ... X and block j starts with X ...
    # Since blocks contain unique numbers 1..n-1 and n, the overlap is constrained.
    
    overlaps = [[0]*num_blocks for _ in range(num_blocks)]
    
    # Optimization: Use tuples for faster comparison?
    # blocks are lists. Convert to tuples for hashing/comparison?
    # Let's stick to lists but optimize the loop.
    
    for i in range(num_blocks):
        bi = blocks[i]
        for j in range(num_blocks):
            if i == j: continue
            bj = blocks[j]
            # Max overlap is limited by the structure.
            # bi ends with p_i, bj starts with p_j.
            # p_i is a perm of 1..n-1. p_j is a perm of 1..n-1.
            # The overlap k means bi[-k:] == bj[:k].
            # Since bi contains n at index n-1, and bj contains n at index n-1.
            # If k > n-1, the n in bi would have to match a number in bj's prefix.
            # bj's prefix of length n-1 is p_j (no n).
            # bi's suffix of length n-1 is p_i (no n).
            # So overlap can be at most n-1? 
            # Wait, bi = p_i + [n] + p_i.
            # Suffix of length n-1 is the last n-1 elements of p_i.
            # Prefix of length n-1 is the first n-1 elements of p_j.
            # If these match, overlap is n-1.
            # Can overlap be larger?
            # If k = n, bi[-n:] includes the 'n' from bi. bj[:n] is first n elements of bj.
            # bj starts with p_j (length n-1), then n.
            # So bj[:n] is p_j + [n].
            # bi[-n:] is last n-1 of p_i + [n].
            # So we need last n-1 of p_i == p_j AND n == n.
            # So max overlap is n.
            # Can it be n+1?
            # bi[-(n+1):] includes part of p_i, n, and start of p_i? No, bi is p_i, n, p_i.
            # Length 2n-1.
            # If k = n+1, bi[-(n+1):] is (last n-2 of p_i) + [n] + (first 1 of p_i).
            # bj[:n+1] is p_j + [n] + (first 1 of p_j).
            # This requires complex matching.
            # Generally, overlap is small.
            
            max_k = min(len(bi), len(bj))
            found = 0
            # We can limit max_k to n+1 or so? 
            # Let's just check up to n+2 to be safe, or full length if n is small.
            # For n=7, block len 13.
            for k in range(min(max_k, n + 2), 0, -1):
                if bi[-k:] == bj[:k]:
                    found = k
                    break
            overlaps[i][j] = found

    # Greedy TSP Heuristic
    # Start with block 0
    current = 0
    unvisited = set(range(1, num_blocks))
    path = [0]
    
    while unvisited:
        best_next = -1
        best_overlap = -1
        for j in unvisited:
            if overlaps[current][j] > best_overlap:
                best_overlap = overlaps[current][j]
                best_next = j
        path.append(best_next)
        unvisited.remove(best_next)
        current = best_next
        
    # Calculate total length
    def calc_length(path):
        total = len(blocks[path[0]])
        for k in range(len(path)-1):
            total += len(blocks[path[k+1]]) - overlaps[path[k]][path[k+1]]
        return total

    # 2-Opt Local Search
    # We want to maximize total overlap (minimize length).
    # Total Length = Sum(len(block)) - Sum(overlaps).
    # Sum(len(block)) is constant.
    # So we maximize Sum(overlaps[path[k], path[k+1]]).
    
    improved = True
    iterations = 0
    while improved and iterations < 10: # Limit iterations for time
        improved = False
        iterations += 1
        for i in range(len(path)-1):
            for j in range(i+2, len(path)):
                # Reverse path[i+1:j+1]
                # Old segment: ... A, B, ..., C, D ...
                # A = path[i], B = path[i+1], C = path[j], D = path[j+1]
                # Note: If i==0, no left neighbor. If j==len(path)-1, no right neighbor.
                
                # Old Overlaps involved:
                # ov(A, B)
                # ov(C, D)
                # (Internal overlaps B...C are reversed, so sum remains same? 
                #  Sum(ov(B, next) ... ov(prev, C))
                #  New Sum(ov(C, prev) ... ov(next, B))
                #  Since ov(X, Y) != ov(Y, X) generally, the internal sum changes.
                #  Wait, in TSP, cost is symmetric? No, overlap is directed.
                #  ov(X, Y) is suffix of X == prefix of Y.
                #  ov(Y, X) is suffix of Y == prefix of X.
                #  They are different.
                #  So reversing a segment changes all internal overlaps.
                #  This makes delta calculation expensive.
                #  For 720 nodes, O(N^3) is too slow if we recalc everything.
                #  However, N=720 is small enough that we can just recalc the total length
                #  if we do few iterations?
                #  Recalc length is O(N).
                #  2-opt checks O(N^2) pairs.
                #  Total O(N^3) = 720^3 ~ 370 million. Too slow in Python.
                
                # Alternative: 1-Swap (Or-opt) or just accept greedy?
                # Greedy is often decent.
                # Let's try a simple Hill Climbing: Try swapping adjacent pairs?
                # Or try moving a single block (Or-opt 1).
                
                pass

    # Let's implement a faster local search: Or-opt 1 (move 1 block)
    # Try to move path[k] to position j.
    # This is O(N^2) moves, each O(1) delta if we precompute?
    # Delta for moving block B from pos i to pos j:
    # Remove B:
    #   Lose ov(prev, B) and ov(B, next)
    #   Gain ov(prev, next)
    # Insert B at j:
    #   Lose ov(prev_j, next_j)
    #   Gain ov(prev_j, B) and ov(B, next_j)
    # Total Delta = Gain - Loss.
    
    # Let's implement Or-opt 1 and 2.
    
    def get_overlap(i, j):
        return overlaps[i][j]
        
    improved = True
    while improved:
        improved = False
        for i in range(len(path)):
            for j in range(len(path)):
                if i == j or (i > 0 and j == i-1) or (i < len(path)-1 and j == i+1):
                    continue
                # Move path[i] to position j
                # Be careful with indices when removing/inserting
                
                block_to_move = path[i]
                
                # Determine neighbors in old position
                prev_old = path[i-1] if i > 0 else None
                next_old = path[i+1] if i < len(path)-1 else None
                
                # Determine neighbors in new position
                # If we remove i first, indices shift.
                # Let's simulate on a copy or calculate logically.
                
                # Old cost contributions:
                old_gain = 0
                if prev_old is not None:
                    old_gain += get_overlap(prev_old, block_to_move)
                if next_old is not None:
                    old_gain += get_overlap(block_to_move, next_old)
                if prev_old is not None and next_old is not None:
                    old_gain += get_overlap(prev_old, next_old) # This is the "skip" overlap we gain by removing? No.
                    # Wait.
                    # Current sum includes: ov(prev, B) + ov(B, next).
                    # After removal, sum includes: ov(prev, next).
                    # So we lose ov(prev, B) + ov(B, next) and gain ov(prev, next).
                    # Delta_remove = ov(prev, next) - ov(prev, B) - ov(B, next).
                    
                new_gain = 0
                # Inserting at j.
                # If j < i, the index of the insertion point in the *original* array is j.
                # If j > i, the index is j-1 (because i was removed).
                # Let's define insertion index in the *reduced* array (size N-1).
                # Let reduced = path without i.
                # We want to insert B at position j_reduced.
                # If j == i, skip.
                # If j < i, j_reduced = j.
                # If j > i, j_reduced = j - 1.
                
                # Neighbors in reduced array at j_reduced:
                # prev_new = reduced[j_reduced - 1] if j_reduced > 0 else None
                # next_new = reduced[j_reduced] if j_reduced < len(reduced) else None
                
                # This logic is getting complex.
                # Given time constraints, maybe just use the Greedy result?
                # The Greedy result for 720 nodes is likely close to optimal for this metric.
                # The seed program uses Greedy.
                # Can we do better?
                # Random restarts?
                
                pass

    # Let's just use the greedy path for now, as implementing efficient 2-opt in Python for 720 nodes within 120s is risky.
    # However, the current code *has* the 2-opt loop but it's empty (`pass`).
    # I will implement a simple 2-opt that only checks a subset or uses a faster delta calculation if possible.
    # Actually, for N=720, a pure Python 2-opt is too slow.
    # I will stick to the Greedy approach but maybe add a "nearest neighbor" refinement or just accept it.
    
    # Wait, the current code returns the greedy path.
    # I can improve the *initial* greedy by using a better start or multiple starts.
    # Or use "First Fit Decreasing" style? No, it's TSP.
    
    # Let's try to optimize the overlap calculation speed.
    # And maybe implement a very limited 2-opt (e.g., only check j = i+1, i+2, etc. - local moves).
    
    # Let's implement Or-opt 1 (move 1 block) with efficient delta.
    
    def try_move(i, j):
        # i: index to move, j: target index in original path
        # Return delta in overlap sum (positive is better)
        if i == j: return 0
        if abs(i - j) == 1: return 0 # No change effectively or handled by 2-opt swap
        
        B = path[i]
        
        # Old neighbors
        p_old = path[i-1] if i > 0 else -1
        n_old = path[i+1] if i < len(path)-1 else -1
        
        # New neighbors in the path *after* removing i
        # If j < i:
        #   Insert at j.
        #   Prev in new pos: path[j-1] (if j>0)
        #   Next in new pos: path[j] (if j < len-1)
        # If j > i:
        #   Insert at j-1 in reduced array? 
        #   Actually, if we remove i, the element at j becomes j-1.
        #   So we insert *before* the element that was at j (if j>i).
        #   So Prev: path[j-1] (if j-1 > i-1, i.e., j-1 != i-1)
        #   This is tricky.
        
        # Let's just recalculate the total length for the modified path?
        # O(N) per move.
        # O(N^2) moves.
        # O(N^3) total.
        # 720^3 is too slow.
        
        # I will skip local search and rely on the Greedy TSP which is already implemented.
        # The main improvement I can make is ensuring the Greedy TSP is correct and fast.
        
        return 0

    # Build the string
    final_word = list(blocks[path[0]])
    for k in range(1, len(path)):
        prev_block = blocks[path[k-1]]
        curr_block = blocks[path[k]]
        ov = overlaps[path[k-1]][path[k]]
        final_word += curr_block[ov:]
        
    return "".join(map(str, final_word))
