import itertools
import random

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"

    # For n > 3, use TSP heuristic on blocks of size n-1
    if n > 7:
        # Fallback to standard construction for larger n if needed, though task is n=7
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

    # Heuristic for n=7 (and potentially 4,5,6)
    # Generate all permutations of 1..n-1
    prev_perms = list(itertools.permutations(range(1, n)))
    
    # Create blocks: p + [n] + p
    blocks = []
    for p in prev_perms:
        b = list(p) + [n] + list(p)
        blocks.append(b)
        
    # Precompute overlaps
    # M[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    num_blocks = len(blocks)
    overlaps = [[0]*num_blocks for _ in range(num_blocks)]
    
    # Optimization: Convert blocks to strings for faster comparison? 
    # Or just use list slicing which is implemented in C.
    for i in range(num_blocks):
        bi = blocks[i]
        for j in range(num_blocks):
            if i == j: continue
            bj = blocks[j]
            # Max possible overlap is min(len(bi), len(bj))
            # But actually, since blocks are permutations + n, 
            # the max overlap is bounded by 2*(n-1).
            # We check from max down to 1.
            max_k = min(len(bi), len(bj))
            found = 0
            # We can limit search: if bi[-1] != bj[0], no overlap of length >=1 is possible?
            # No, overlap of length 1 means bi[-1] == bj[0].
            # Overlap of length k means bi[-k:] == bj[:k].
            # So if bi[-1] != bj[0], found is 0.
            if bi[-1] != bj[0]:
                overlaps[i][j] = 0
            else:
                for k in range(max_k, 0, -1):
                    if bi[-k:] == bj[:k]:
                        found = k
                        break
                overlaps[i][j] = found

    # Initial Path: Greedy
    current = 0
    unvisited = set(range(1, num_blocks))
    path = [0]
    
    while unvisited:
        best_next = -1
        best_overlap = -1
        # Optimization: Pre-sort candidates? No, just iterate.
        for j in unvisited:
            if overlaps[current][j] > best_overlap:
                best_overlap = overlaps[current][j]
                best_next = j
        path.append(best_next)
        unvisited.remove(best_next)
        current = best_next
        
    def calc_score(path):
        # Maximize sum of overlaps
        score = 0
        for k in range(len(path)-1):
            score += overlaps[path[k]][path[k+1]]
        return score

    current_score = calc_score(path)
    
    # Or-Opt Local Search
    # Try moving segments of length 1, 2, 3 to new positions
    improved = True
    iterations = 0
    while improved and iterations < 10000: # Limit iterations to prevent infinite loops if stuck
        improved = False
        iterations += 1
        for seg_len in [1, 2, 3]:
            for i in range(len(path) - seg_len):
                # Segment to move: path[i : i+seg_len]
                # Remove it
                seg = path[i : i+seg_len]
                # Indices in the remaining path
                # Before: path[:i] + path[i+seg_len:]
                # We try to insert seg at every position in the remaining path
                
                # Calculate old score contribution
                # This is complex to calculate delta efficiently without rebuilding.
                # Given N=720, rebuilding score O(N) for each move O(N^2 * 3) = 720^2 * 3 ~ 1.5M ops per pass.
                # 10000 passes * 1.5M = 15B ops. Too slow in Python.
                
                # We need delta calculation.
                # Let A = path[i-1], B = path[i], ..., C = path[i+seg_len-1], D = path[i+seg_len]
                # Old links: (A,B), (B, next_B), ..., (C, D)
                # New links: (A, D), (D, ...), ..., (prev_insert, seg[0]), (seg[-1], next_insert)
                
                # This is getting complicated. Let's use a simpler 2-Opt on the reverse direction?
                # Or just a simple "best neighbor" hill climb on single swaps?
                
                # Actually, for N=720, a simple greedy + a few rounds of "random restart + greedy" might be better and faster to implement correctly.
                # Or, since the graph is dense, we can use a fast C-extension? No, standard library only.
                
                # Let's try a simpler Local Search: 2-Opt.
                # Reverse a segment path[i:j].
                # Old links: (path[i-1], path[i]), (path[j], path[j+1])
                # New links: (path[i-1], path[j]), (path[i], path[j+1])
                # Internal links are reversed: (path[k], path[k+1]) becomes (path[k+1], path[k]).
                # Sum of overlaps changes by:
                # Delta = ov(path[i-1], path[j]) + ov(path[i], path[j+1]) - ov(path[i-1], path[i]) - ov(path[j], path[j+1])
                #       + Sum(ov(path[k+1], path[k]) - ov(path[k], path[k+1]) for k in i..j-1)
                
                # This is O(N) per 2-opt move. O(N^2) per pass. 720^2 = 518k ops per pass.
                # 100 passes = 50M ops. Feasible in Python if optimized.
                
                pass # Placeholder for actual 2-opt implementation below

    # Implementing 2-Opt properly
    def two_opt(path):
        n_path = len(path)
        improved = True
        while improved:
            improved = False
            for i in range(n_path - 1):
                for j in range(i + 1, n_path):
                    # Reverse path[i:j+1]
                    # Check boundaries
                    # i=0 is start, j=n_path-1 is end
                    
                    # Old overlaps involved in change:
                    # 1. Connection before i: path[i-1] -> path[i] (if i > 0)
                    # 2. Connection after j: path[j] -> path[j+1] (if j < n_path-1)
                    # 3. Internal connections: path[k] -> path[k+1] for k in [i, j-1]
                    
                    # New overlaps:
                    # 1. Connection before i: path[i-1] -> path[j] (if i > 0)
                    # 2. Connection after j: path[i] -> path[j+1] (if j < n_path-1)
                    # 3. Internal connections reversed: path[k+1] -> path[k] for k in [i, j-1]
                    
                    delta = 0
                    
                    # Internal sum
                    # Sum of ov(path[k+1], path[k]) - ov(path[k], path[k+1])
                    # This part is independent of i,j boundaries, depends only on the segment.
                    # We can precompute this? No, it changes as we reverse.
                    # But we can compute it on the fly.
                    
                    # However, calculating internal sum for every i,j is O(N) per pair, making total O(N^3). 720^3 is too slow.
                    
                    # Alternative: Just do random restarts with Greedy.
                    # Greedy is O(N^2). 
                    # Let's do 100 random restarts.
                    pass
                    
    # Given the complexity of implementing fast 2-opt in Python within the edit limit, 
    # and the risk of timeout, I will stick to a robust Greedy with multiple random restarts.
    # Greedy is O(N^2). 720^2 = 518k. 100 restarts = 50M. Might be tight but possible.
    # Actually, the initial greedy is deterministic. 
    # Let's try a few random initializations.
    
    best_path = path
    best_score = current_score
    
    # Try a few random restarts
    for restart in range(20):
        if restart == 0:
            p = path[:]
        else:
            p = list(range(num_blocks))
            random.shuffle(p)
            
        # Greedy from random start? No, Greedy picks best next.
        # Let's just use the deterministic greedy, but maybe start from different nodes?
        # Or use "Nearest Neighbor" with random tie-breaking.
        
        # Let's just evaluate the current best and return. 
        # The seed program's score is 1.0. 
        # A good TSP heuristic should significantly reduce length.
        # Length of word = Sum(len(block)) - Sum(overlaps).
        # Sum(len(block)) = 720 * 13 = 9360.
        # If we get average overlap of 6, length = 9360 - 719*6 = 9360 - 4314 = 5046.
        # Lower bound is 5884? Wait.
        # Task says: "The shortest known word has 5905 symbols... best published lower bound is 5884."
        # My calculation: 720 blocks of length 13. 
        # If overlap is 1, length = 9360 - 719 = 8641.
        # If overlap is 6, length = 5046.
        # Wait, the "block" construction I am using is: p + [n] + p.
        # This ensures every permutation of 1..n is contained?
        # No. The standard construction for superpermutations is different.
        # The "block" method I'm using (p + n + p) is for *de Bruijn-like* sequences or specific structures?
        
        # Let's re-read the task.
        # "Shortest superpermutation on seven symbols"
        # "contains every one of the 5040 permutations of 1..7 as seven consecutive symbols"
        
        # The current code's logic:
        # Blocks are p + [n] + p.
        # Does this word contain all permutations of 1..n?
        # Let's check n=3.
        # Perms of 1,2: (1,2), (2,1).
        # Blocks: 1,2,3,1,2 and 2,1,3,2,1.
        # Word: 123121321.
        # Perms of 1,2,3 in 123121321:
        # 123 (yes)
        # 231 (yes)
        # 312 (yes)
        # 121 (no)
        # 213 (yes)
        # 132 (yes)
        # 321 (yes)
        # 213... wait.
        # Windows of 123121321:
        # 123
        # 231
        # 312
        # 121
        # 213
        # 132
        # 321
        # Missing? 
        # Perms: 123, 132, 213, 231, 312, 321.
        # All 6 are present.
        
        # So the block construction IS valid for n=3.
        # Is it valid for n=7?
        # The word constructed by merging blocks p+n+p contains every permutation of 1..n?
        # Yes, this is a known construction for superpermutations (though not always shortest).
        # The length is roughly n! * (n-1) + n.
        # For n=7, 5040 * 6 + 7 = 30247? No.
        # Number of blocks is (n-1)! = 720.
        # Length of each block is 2(n-1)+1 = 13.
        # Total length if no overlap: 720 * 13 = 9360.
        # With overlaps, it's shorter.
        # The known shortest is 5905.
        # 9360 - 5905 = 3455 total overlap.
        # Average overlap per link: 3455 / 719 = 4.8.
        # So we need average overlap ~5.
        
        # The greedy TSP on overlaps should find a path with high average overlap.
        
        # Let's ensure the code returns the best path found.
        
        # I will implement a simple "Best Insertion" or just stick to the greedy.
        # Given the time, I will just clean up the code and ensure it runs.
        # The "pass" in the local search means it currently just returns the greedy result.
        # I will add a simple 2-opt that is O(N^2) per pass but limited to a few passes, 
        # or just rely on the greedy which is O(N^2).
        
        # Let's try to implement a fast 2-opt that only checks a subset of moves?
        # Or just return the greedy result.
        
        # The greedy result is likely better than the seed (which is also greedy but maybe with a fixed order?).
        # The seed code in the prompt IS the greedy TSP.
        # So score 1.0 is the greedy TSP.
        # To improve, we MUST do local search.
        
        # I will implement a randomized 2-opt with a time limit.
        
    # Time-limited Local Search
    import time
    start_time = time.time()
    timeout = 100 # 100 seconds
    
    # Simple 2-Opt
    # To make it fast, we only consider moves that might improve.
    # We can't easily precompute deltas for all i,j in O(1).
    # But we can do "First Improvement" 2-opt.
    
    # Actually, for N=720, even O(N^2) per pass is ~500k ops.
    # Python does ~10M simple ops/sec. So 500k is 0.05s.
    # We can do 1000 passes in 50s.
    
    improved = True
    while improved and time.time() - start_time < timeout:
        improved = False
        for i in range(num_blocks - 1):
            for j in range(i + 1, num_blocks):
                # Calculate delta for reversing path[i:j+1]
                # Old: ov(i-1, i) + ov(j, j+1) + Sum_internal
                # New: ov(i-1, j) + ov(i, j+1) + Sum_internal_reversed
                
                # We need Sum_internal and Sum_internal_reversed.
                # This is O(N) to compute.
                # Total O(N^3). 720^3 = 373M. Too slow.
                
                # Optimization: 
                # We can precompute prefix sums of overlaps?
                # No, because the internal links change direction.
                
                # Alternative: Only do 2-opt on small segments?
                # Or use a different heuristic.
                
                # Let's try "Or-Opt" with segment length 1.
                # Move path[i] to position j.
                # This is O(N) to evaluate if we maintain prefix sums?
                
                # Given the difficulty, I will use a "Random Restart Greedy" approach.
                # It is O(N^2) per restart.
                # 720^2 = 518k.
                # 100 restarts = 51M ops. ~5 seconds.
                # This is safe.
                
                pass # Break out to use Random Restart Greedy

    # Re-implementing with Random Restart Greedy
    best_path = path
    best_score = current_score
    
    for restart in range(50):
        if time.time() - start_time > timeout:
            break
            
        if restart == 0:
            p = path[:]
        else:
            p = list(range(num_blocks))
            random.shuffle(p)
            
        # Greedy from this start? 
        # No, standard Greedy starts at 0 and picks best next.
        # If we shuffle, we can't just pick best next from 0.
        # We can do "Greedy Construction" starting from a random node.
        
        current = p[0]
        unvisited = set(p[1:])
        gp = [current]
        
        while unvisited:
            best_next = -1
            best_ov = -1
            for j in unvisited:
                ov = overlaps[current][j]
                if ov > best_ov:
                    best_ov = ov
                    best_next = j
            gp.append(best_next)
            unvisited.remove(best_next)
            current = best_next
            
        score = calc_score(gp)
        if score > best_score:
            best_score = score
            best_path = gp
            
    # Build final word from best_path
    final_word = list(blocks[best_path[0]])
    for k in range(1, len(best_path)):
        prev_block = blocks[best_path[k-1]]
        curr_block = blocks[best_path[k]]
        ov = overlaps[best_path[k-1]][best_path[k]]
        final_word += curr_block[ov:]
        
    return "".join(map(str, final_word))
