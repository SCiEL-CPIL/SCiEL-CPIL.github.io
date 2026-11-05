import itertools
import time
import math
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110
    
    if n == 1:
        return "1"
    
    # Step 1: Construct the standard superpermutation
    word = [1]
    
    # We will store the order of permutations at each level
    # perms_order[m] will be the list of tuples (perms of 1..m) used to build S(m+1)
    # Actually, we just need the current list of perms to generate the next.
    
    current_perms = [(1,)] # Perms of 1
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
            
        # Extract current perms from word? 
        # No, we track them. 
        # In the standard construction, the perms are derived from the windows of the previous word.
        # But if we reorder them, the word changes, so the windows change?
        # Wait. The set of permutations is always all permutations of 1..m-1.
        # The *order* is what we optimize.
        # The "blocks" are determined by the permutation $\pi$.
        # So we just need the set of all permutations of 1..m-1.
        
        # Generate all permutations of 1..m-1
        if m == 2:
            base_perms = [(1,)]
        else:
            # We can generate from itertools, or derive from previous?
            # It's safer to just generate all permutations of 1..m-1
            # For m=7, 6! = 720.
            base_perms = list(itertools.permutations(range(1, m)))
            
        # We need to solve TSP on these base_perms
        # Distance d(i, j) = cost of appending block_j after block_i
        # block_p = list(p) + [m] + list(p)
        
        N = len(base_perms)
        
        # Precompute blocks
        blocks = [list(p) + [m] + list(p) for p in base_perms]
        
        # Precompute distances
        # dist[i][j] = len(blocks[j]) - overlap(blocks[i], blocks[j])
        # Note: overlap is suffix of blocks[i] matching prefix of blocks[j]
        
        # To save memory and time, we can compute on the fly or use a matrix.
        # N=720, matrix is 720x720 = 518k floats. Fine.
        
        dist_matrix = [[0]*N for _ in range(N)]
        
        for i in range(N):
            bi = blocks[i]
            for j in range(N):
                bj = blocks[j]
                max_ov = min(len(bi), len(bj))
                ov = 0
                # Check overlap
                # bi suffix vs bj prefix
                for k in range(max_ov, -1, -1):
                    if k == 0:
                        ov = 0
                        break
                    if bi[-k:] == bj[:k]:
                        ov = k
                        break
                dist_matrix[i][j] = len(bj) - ov
                
        # Solve TSP
        # Use Nearest Neighbor + 2-Opt
        
        def solve_tsp_heuristic():
            best_path = None
            best_cost = float('inf')
            
            # Try a few random starting nodes
            num_starts = 10
            if N > 100:
                num_starts = 5 # Fewer starts for larger N to save time
            
            for start in range(num_starts):
                if time.time() > deadline:
                    break
                start_node = random.randint(0, N-1)
                # Nearest Neighbor
                path = [start_node]
                unvisited = set(range(N))
                unvisited.remove(start_node)
                
                current = start_node
                cost = len(blocks[start_node]) # First block cost
                
                while unvisited:
                    next_node = -1
                    min_d = float('inf')
                    for node in unvisited:
                        d = dist_matrix[current][node]
                        if d < min_d:
                            min_d = d
                            next_node = node
                    path.append(next_node)
                    unvisited.remove(next_node)
                    current = next_node
                    cost += min_d
                    
                # 2-Opt Improvement
                # We only run 2-opt if N is small enough or we have time
                if N <= 200:
                    improved = True
                    passes = 0
                    while improved and passes < 10 and time.time() < deadline:
                        improved = False
                        passes += 1
                        for i in range(N-1):
                            if time.time() > deadline:
                                break
                            for j in range(i+1, N):
                                # Check if reversing path[i..j] improves cost
                                # Cost change: 
                                # Old: d(path[i-1], path[i]) + d(path[j], path[j+1])
                                # New: d(path[i-1], path[j]) + d(path[i], path[j+1])
                                # Handle boundaries
                                prev = path[i-1] if i > 0 else None
                                next = path[j+1] if j < N-1 else None
                                
                                old_cost = 0
                                new_cost = 0
                                
                                if prev is not None:
                                    old_cost += dist_matrix[prev][path[i]]
                                else:
                                    # If i==0, the start node changes? 
                                    # In TSP cycle, yes. But here it's a path (open tour).
                                    # The cost of the first node is fixed?
                                    # No, the cost of the first node is len(block).
                                    # The transition cost is d(prev, curr).
                                    # If i==0, there is no prev.
                                    # The "cost" associated with the start node is just its length.
                                    # Reversing the path changes which node is first.
                                    # So we must re-evaluate the start cost.
                                    # This makes 2-opt on open tours slightly more complex.
                                    # Let's just calculate total cost for the segment.
                                    pass
                                    
                                # Simplified 2-opt for open tour:
                                # Calculate cost of path segment i..j
                                # This is getting complicated. 
                                # Given time constraints, maybe just Nearest Neighbor is enough?
                                # Or a simpler local search: swap adjacent?
                                
                                # Let's skip complex 2-opt and just use NN.
                                # NN is fast.
                                break
                            break
                            
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
                    
            return best_path

        path = solve_tsp_heuristic()
        
        if path is None:
            # Fallback to standard order
            path = list(range(N))
            
        # Construct new word
        new_word = []
        for idx in path:
            block = blocks[idx]
            if not new_word:
                new_word.extend(block)
            else:
                max_overlap = min(len(new_word), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if o == 0:
                        overlap = 0
                        break
                    if new_word[-o:] == block[:o]:
                        overlap = o
                        break
                new_word.extend(block[overlap:])
                
        word = new_word
        
        # Update current_perms for next iteration?
        # The "perms" for the next step are the windows of size m in the new word.
        # We don't strictly need to extract them if we just generate all perms of 1..m.
        # The TSP approach uses all perms of 1..m.
        # So we don't need `current_perms` from the word. We just generate `base_perms` from scratch.
        
    # Step 2: Local Search (Deletion)
    # Same as seed program
    
    def is_valid_fast(w):
        if len(w) < n:
            return False
        seen = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
                if len(seen) == int(math.factorial(n)):
                    return True
        return False

    target_perms = int(math.factorial(n))
    
    changed = True
    while changed:
        changed = False
        L = len(word)
        for i in range(L):
            if time.time() > deadline:
                return "".join(map(str, word))
            
            test_word = word[:i] + word[i+1:]
            if is_valid_fast(test_word):
                word = test_word
                changed = True
                break
        
    return "".join(map(str, word))
