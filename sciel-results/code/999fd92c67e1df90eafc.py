import itertools
import time

def construct(n):
    """
    Constructs the shortest known superpermutation for n symbols.
    For n <= 4, returns the known optimal length.
    For n = 5, 6, 7, uses a greedy TSP heuristic on the blocks of size n-1
    with 2-opt local search to minimize length.
    
    The approach:
    1. Generate all permutations of {1, ..., n-1}.
    2. For each permutation p, create a block B_p = p + [n] + p.
    3. Find an ordering of these blocks that maximizes the total overlap
       between consecutive blocks.
    4. Concatenate the blocks using the maximum overlaps to form the final word.
    """
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "12341214312431423413241342143214"

    # For n >= 5, use the heuristic
    # Generate all permutations of 1..n-1
    prev_perms = list(itertools.permutations(range(1, n)))
    num_blocks = len(prev_perms)
    
    # Create blocks: p + [n] + p
    blocks = []
    for p in prev_perms:
        b = list(p) + [n] + list(p)
        blocks.append(b)
        
    # Precompute overlaps
    # M[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    # Note: k can be 0 to min(len(bi), len(bj))
    # Since blocks have length 2*(n-1)+1 = 2n-1
    # The max possible overlap is limited by the structure.
    # Actually, for p + [n] + p, the suffix is p and prefix is p.
    # So if p_i == p_j, we can overlap the entire p?
    # Let's just compute it directly.
    
    overlaps = [[0]*num_blocks for _ in range(num_blocks)]
    
    # Optimization: Use string comparison for speed
    block_strs = ["".join(map(str, b)) for b in blocks]
    
    for i in range(num_blocks):
        bi = block_strs[i]
        for j in range(num_blocks):
            bj = block_strs[j]
            # Max overlap length is min(len(bi), len(bj))
            # But we can prune: the overlap must be a suffix of bi and prefix of bj.
            # Also, since bi ends with prev_perms[i] and bj starts with prev_perms[j],
            # if prev_perms[i] != prev_perms[j], the overlap cannot be n-1 or more?
            # Actually, the block is p + [n] + p.
            # Suffix of length n-1 is p.
            # Prefix of length n-1 is p.
            # If p_i != p_j, overlap < n-1.
            # Let's just check k from min_len down to 0.
            
            max_k = min(len(bi), len(bj))
            # We can limit max_k to 2*n-2 effectively, but min is fine.
            # To speed up, we can check only k where bi[-k] == bj[0] etc?
            # Simple loop is okay for 720 blocks (518k pairs).
            found = 0
            for k in range(max_k, -1, -1):
                if k == 0:
                    found = 0
                    break
                if bi[-k:] == bj[:k]:
                    found = k
                    break
            overlaps[i][j] = found

    # Greedy TSP Heuristic
    # Start with a random block or block 0
    # Let's try a few random starts and keep the best
    best_path = None
    best_len = float('inf')
    
    start_time = time.time()
    time_limit = 110  # Leave some time for building the string
    
    num_restarts = 0
    while time.time() - start_time < time_limit:
        num_restarts += 1
        current = num_restarts % num_blocks
        unvisited = set(range(num_blocks))
        unvisited.remove(current)
        path = [current]
        
        while unvisited:
            best_next = -1
            best_overlap = -1
            # Heuristic: choose next block with max overlap
            # To break ties, we can use a secondary metric or random
            for j in unvisited:
                ov = overlaps[current][j]
                if ov > best_overlap:
                    best_overlap = ov
                    best_next = j
            path.append(best_next)
            unvisited.remove(best_next)
            current = best_next
            
        # Calculate total length
        def calc_length(p):
            total = len(blocks[p[0]])
            for k in range(len(p)-1):
                total += len(blocks[p[k+1]]) - overlaps[p[k]][p[k+1]]
            return total

        current_len = calc_length(path)
        if current_len < best_len:
            best_len = current_len
            best_path = path[:]
            
        # 2-Opt Local Search
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Iterate through pairs (i, j) with i < j-1
            # Reverse path[i+1 : j+1]
            # Check if length decreases
            for i in range(len(path)-1):
                if time.time() - start_time > time_limit:
                    break
                for j in range(i+2, len(path)):
                    # Old segment: ... A=path[i], B=path[i+1], ..., C=path[j], D=path[j+1] (if exists)
                    # New segment: ... A, C, ..., B, D ...
                    
                    # Costs involved in the sum of (len - overlap):
                    # We want to maximize sum of overlaps.
                    # Old overlaps to consider:
                    # ov(path[i-1], path[i]) -> unchanged (unless i=0)
                    # ov(path[i], path[i+1]) -> removed
                    # ov(path[j], path[j+1]) -> removed (unless j is last)
                    
                    # Internal overlaps:
                    # ov(path[i+1], path[i+2]), ..., ov(path[j-1], path[j])
                    # These are reversed. ov(X, Y) vs ov(Y, X).
                    # So we need to check if the sum of overlaps increases.
                    
                    # Let's compute the delta in total length.
                    # Delta = (New Len) - (Old Len)
                    # New Len = Old Len - (Old Overlaps Sum) + (New Overlaps Sum)
                    # So we want New Overlaps Sum > Old Overlaps Sum.
                    
                    # Old Overlaps in the segment [i, j+1):
                    # 1. ov(path[i], path[i+1])
                    # 2. ov(path[i+1], path[i+2]) ... ov(path[j-1], path[j])
                    # 3. ov(path[j], path[j+1]) if j+1 < len(path)
                    
                    # New Overlaps:
                    # 1. ov(path[i], path[j])  (A to C)
                    # 2. ov(path[j], path[j-1]) ... ov(path[i+2], path[i+1]) (Reversed internal)
                    # 3. ov(path[i+1], path[j+1]) (B to D) if j+1 < len(path)
                    
                    # Note: The connections to path[i-1] and path[j+1] (if they exist)
                    # path[i-1] connects to path[i] (A). Unchanged.
                    # path[j+1] connects to path[i+1] (B) in new, path[j] (C) in old?
                    # Wait.
                    # Old: ... path[i-1], path[i], path[i+1], ..., path[j], path[j+1] ...
                    # New: ... path[i-1], path[i], path[j], ..., path[i+1], path[j+1] ...
                    
                    # Connections:
                    # (path[i-1], path[i]): Unchanged.
                    # (path[i], path[i+1]) -> Removed.
                    # (path[j], path[j+1]) -> Removed.
                    
                    # (path[i], path[j]) -> Added.
                    # (path[i+1], path[j+1]) -> Added.
                    
                    # Internal:
                    # Old: ov(path[i+1], path[i+2]), ..., ov(path[j-1], path[j])
                    # New: ov(path[j], path[j-1]), ..., ov(path[i+2], path[i+1])
                    
                    # So Delta Overlap = 
                    #   ov(path[i], path[j]) + ov(path[i+1], path[j+1]) 
                    #   + sum_{k=i+1}^{j-1} ov(path[k+1], path[k])
                    #   - ov(path[i], path[i+1]) - ov(path[j], path[j+1])
                    #   - sum_{k=i+1}^{j-1} ov(path[k], path[k+1])
                    
                    # We need to be careful with boundaries.
                    # If i=0, there is no path[i-1].
                    # If j+1 == len(path), there is no path[j+1].
                    
                    # Let's compute the change in total overlap.
                    old_ov = 0
                    new_ov = 0
                    
                    # Boundary 1: i -> i+1
                    old_ov += overlaps[path[i]][path[i+1]]
                    
                    # Boundary 2: j -> j+1
                    if j + 1 < len(path):
                        old_ov += overlaps[path[j]][path[j+1]]
                        new_ov += overlaps[path[i+1]][path[j+1]]
                    else:
                        # No path[j+1], so no overlap to remove/add there?
                        # The last block is path[j].
                        # In new, last block is path[i+1].
                        # So we replace ov(path[j], None) with ov(path[i+1], None)?
                        # No, the length is sum of block lengths - sum of overlaps between consecutive blocks.
                        # If j is the last index, there is no overlap after path[j].
                        # So we don't add/remove that term.
                        pass
                    
                    # New Boundary 1: i -> j
                    new_ov += overlaps[path[i]][path[j]]
                    
                    # Internal
                    for k in range(i+1, j):
                        old_ov += overlaps[path[k]][path[k+1]]
                        new_ov += overlaps[path[k+1]][path[k]]
                    
                    if new_ov > old_ov:
                        # Perform reversal
                        path[i+1:j+1] = reversed(path[i+1:j+1])
                        improved = True
                        # Update best_path if this local path is better?
                        # The local search modifies 'path' in place.
                        # We should update best_path at the end of local search or if we break.
                        # Actually, we just keep searching on the modified path.
                        # We should update best_len/best_path if the current path length is better.
                        # But calc_length is O(N). We can track length incrementally?
                        # For now, let's just update best_path if we finish local search.
                        # Or we can check length occasionally.
                        # Given N=720, O(N) is fast.
                        pass
        
        # After local search, check if this path is the best so far
        final_len = calc_length(path)
        if final_len < best_len:
            best_len = final_len
            best_path = path[:]

    # Build the string from best_path
    if best_path is None:
        # Fallback to simple greedy without local search if something went wrong
        # But best_path should be set.
        best_path = list(range(num_blocks))
        
    final_word = list(blocks[best_path[0]])
    for k in range(1, len(best_path)):
        prev_block = blocks[best_path[k-1]]
        curr_block = blocks[best_path[k]]
        ov = overlaps[best_path[k-1]][best_path[k]]
        final_word += curr_block[ov:]
        
    return "".join(map(str, final_word))
