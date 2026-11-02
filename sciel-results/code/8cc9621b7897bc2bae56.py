import itertools

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # Standard construction fallback
    def standard_construct(n):
        if n == 1:
            return "1"
        prev = standard_construct(n-1)
        perms_in_order = []
        seen = set()
        # Extract permutations of 1..n-1 from prev
        # They are substrings of length n-1
        for i in range(len(prev) - (n - 2)):
            w = prev[i:i+n-1]
            if len(set(w)) == n-1 and w not in seen:
                seen.add(w)
                perms_in_order.append(w)
        
        new_word = prev
        for p in perms_in_order:
            new_word += str(n) + p
        return new_word

    # For n=7, use the TSP heuristic on the blocks of size n-1.
    # This assumes n is small enough for the heuristic to run fast.
    # For n=7, n-1=6. 720 blocks.
    
    if n > 7:
        return standard_construct(n)

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
    
    for i in range(num_blocks):
        bi = blocks[i]
        for j in range(num_blocks):
            bj = blocks[j]
            max_k = min(len(bi), len(bj))
            found = 0
            # Optimization: Only check k where bi[-k] == bj[0] etc?
            # Just loop k
            for k in range(max_k, -1, -1):
                if k == 0:
                    found = 0
                    break
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
        
    # Calculate total overlap
    def calc_total_overlap(path):
        total = 0
        for k in range(len(path)-1):
            total += overlaps[path[k]][path[k+1]]
        return total

    current_overlap = calc_total_overlap(path)
    
    # 2-Opt Local Search
    # We want to maximize total overlap.
    # Delta = NewOverlap - OldOverlap
    # If Delta > 0, accept.
    
    improved = True
    max_iterations = 5 # Limit iterations to avoid timeout
    iter_count = 0
    
    while improved and iter_count < max_iterations:
        improved = False
        iter_count += 1
        N = len(path)
        for i in range(N-1):
            for j in range(i+1, N):
                # Reverse path[i+1 : j+1]
                # A = path[i], B = path[i+1], C = path[j], D = path[j+1] (if exists)
                
                # Old boundaries:
                # (A, B) and (C, D)
                old_boundary = overlaps[path[i]][path[i+1]]
                if j + 1 < N:
                    old_boundary += overlaps[path[j]][path[j+1]]
                else:
                    # If j is the last element, there is no D.
                    # But path is a cycle? No, it's a path.
                    # The problem is a path, not a cycle.
                    # So the end of the path doesn't have an outgoing edge cost in the sum?
                    # Wait, the length is sum of blocks - sum of overlaps.
                    # Overlaps are between consecutive blocks.
                    # So we sum overlaps[path[k]][path[k+1]] for k=0 to N-2.
                    # If we reverse a segment at the end, the last overlap changes?
                    # If j == N-1, the segment is path[i+1..N-1].
                    # The overlap at i is (path[i], path[i+1]).
                    # The overlap at i+1..N-2 are internal.
                    # There is no overlap after N-1.
                    # So old_boundary is just overlaps[path[i]][path[i+1]].
                    # New_boundary is overlaps[path[i]][path[j]] (which is path[i], path[N-1]).
                    # And internal overlaps.
                    pass # Handled below

                new_boundary = overlaps[path[i]][path[j]]
                if j + 1 < N:
                    new_boundary += overlaps[path[i+1]][path[j+1]]
                else:
                    # No new boundary at the end
                    pass

                # Internal overlaps
                # Old: sum(overlaps[path[k]][path[k+1]] for k in i+1 to j-1)
                # New: sum(overlaps[path[k+1]][path[k]] for k in i+1 to j-1)
                
                old_internal = 0
                new_internal = 0
                for k in range(i+1, j):
                    old_internal += overlaps[path[k]][path[k+1]]
                    new_internal += overlaps[path[k+1]][path[k]]
                
                delta = (new_boundary + new_internal) - (old_boundary + old_internal)
                
                if delta > 0:
                    # Perform reverse
                    path[i+1:j+1] = reversed(path[i+1:j+1])
                    current_overlap += delta
                    improved = True
                    # Continue searching from current i or restart?
                    # Usually restart or continue. Let's continue to find more improvements in this pass.
                    
    # Build the string
    final_word = list(blocks[path[0]])
    for k in range(1, len(path)):
        prev_block = blocks[path[k-1]]
        curr_block = blocks[path[k]]
        ov = overlaps[path[k-1]][path[k]]
        final_word += curr_block[ov:]
        
    result_str = "".join(map(str, final_word))
    
    # Compare with standard construction
    standard_str = standard_construct(n)
    
    if len(result_str) < len(standard_str):
        return result_str
    else:
        return standard_str

