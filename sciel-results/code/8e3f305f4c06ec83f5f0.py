import itertools

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"

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
    
    # Optimization: Use strings for faster comparison?
    # Blocks are lists of ints. Convert to strings?
    # Or just use list slicing.
    
    for i in range(num_blocks):
        bi = blocks[i]
        for j in range(num_blocks):
            bj = blocks[j]
            max_k = min(len(bi), len(bj))
            found = 0
            for k in range(max_k, 0, -1):
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

    # 2-Opt Local Search with Prefix Sums for Fast Delta
    # Precompute forward and reverse overlap sums for the current path
    # F[k] = overlaps[path[k]][path[k+1]]
    # R[k] = overlaps[path[k+1]][path[k]]
    
    def get_prefix_sums(path):
        N = len(path)
        F = [0] * (N-1)
        R = [0] * (N-1)
        for k in range(N-1):
            F[k] = overlaps[path[k]][path[k+1]]
            R[k] = overlaps[path[k+1]][path[k]]
        
        # Prefix sums
        PF = [0] * N
        PR = [0] * N
        for k in range(1, N):
            PF[k] = PF[k-1] + F[k-1]
            PR[k] = PR[k-1] + R[k-1]
        return PF, PR

    def total_overlap_from_prefix(PF):
        return PF[-1]

    improved = True
    while improved:
        improved = False
        PF, PR = get_prefix_sums(path)
        N = len(path)
        
        for i in range(N-1):
            for j in range(i+2, N):
                # Reverse path[i+1 : j+1]
                # Indices involved:
                # Old: ... path[i], path[i+1] ... path[j], path[j+1] ...
                # New: ... path[i], path[j] ... path[i+1], path[j+1] ...
                
                # Note: If j == N-1, there is no path[j+1]. Handle boundary.
                
                # Old Overlap Sum Segment:
                # Boundaries:
                # Left: overlaps[path[i]][path[i+1]]
                # Right: overlaps[path[j]][path[j+1]] if j < N-1 else 0
                
                # Internal Old: sum(overlaps[path[k]][path[k+1]] for k in i+1 to j-1)
                # This is F[i+1] ... F[j-1]
                # Using PF: PF[j] - PF[i+1]
                
                # New Overlap Sum Segment:
                # Left: overlaps[path[i]][path[j]]
                # Right: overlaps[path[i+1]][path[j+1]] if j < N-1 else 0
                
                # Internal New: sum(overlaps[path[k+1]][path[k]] for k in i+1 to j-1)
                # This is R[i+1] ... R[j-1]
                # Using PR: PR[j] - PR[i+1]
                
                old_left = overlaps[path[i]][path[i+1]]
                old_right = overlaps[path[j]][path[j+1]] if j < N-1 else 0
                old_internal = PF[j] - PF[i+1]
                old_total = old_left + old_right + old_internal
                
                new_left = overlaps[path[i]][path[j]]
                new_right = overlaps[path[i+1]][path[j+1]] if j < N-1 else 0
                new_internal = PR[j] - PR[i+1]
                new_total = new_left + new_right + new_internal
                
                if new_total > old_total:
                    # Perform reversal
                    path[i+1:j+1] = path[i+1:j+1][::-1]
                    improved = True
                    # Update prefix sums? 
                    # Since we are in a loop over i,j, and path changed,
                    # the prefix sums PF, PR are now stale.
                    # We must break and recalculate, or update them.
                    # Breaking is safer and simpler.
                    break
            if improved:
                break

    # Build the string
    final_word = list(blocks[path[0]])
    for k in range(1, len(path)):
        prev_block = blocks[path[k-1]]
        curr_block = blocks[path[k]]
        ov = overlaps[path[k-1]][path[k]]
        final_word += curr_block[ov:]
        
    return "".join(map(str, final_word))
