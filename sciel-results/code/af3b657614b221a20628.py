import itertools
import random

def construct_backtrack(n):
    if n == 1:
        return "1"
    perms = list(itertools.permutations(range(1, n+1)))
    
    def dfs(current, covered):
        if len(covered) == len(perms):
            return current
        for sym in range(1, n+1):
            new_current = current + [sym]
            if len(new_current) >= n:
                window = tuple(new_current[-n:])
                if window in perms and window not in covered:
                    new_covered = covered | {window}
                    res = dfs(new_current, new_covered)
                    if res:
                        return res
            else:
                res = dfs(new_current, covered)
                if res:
                    return res
        return None

    result = dfs([], set())
    if result:
        return "".join(map(str, result))
    return ""

def construct(n):
    if n <= 3:
        return construct_backtrack(n)

    # For n=7, use the TSP heuristic on the blocks of size n-1.
    # This assumes n is small enough for the heuristic to run fast.
    # For n=7, n-1=6. 720 blocks.
    
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
    
    for i in range(num_blocks):
        bi = blocks[i]
        for j in range(num_blocks):
            bj = blocks[j]
            max_k = min(len(bi), len(bj))
            found = 0
            # Check from longest to shortest
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

    # Multi-Start Greedy TSP Heuristic
    best_path = None
    best_len = float('inf')
    
    # Calculate total length helper
    def calc_length(path):
        total = len(blocks[path[0]])
        for k in range(len(path)-1):
            total += len(blocks[path[k+1]]) - overlaps[path[k]][path[k+1]]
        return total

    # Try multiple starting points
    starts = list(range(num_blocks))
    random.shuffle(starts)
    # Limit starts to avoid timeout for n=7
    num_starts = min(50, num_blocks)
    
    for start in starts[:num_starts]:
        current = start
        unvisited = set(range(num_blocks))
        unvisited.remove(start)
        path = [start]
        
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
            
        l = calc_length(path)
        if l < best_len:
            best_len = l
            best_path = path

    path = best_path

    # Build the string
    final_word = list(blocks[path[0]])
    for k in range(1, len(path)):
        prev_block = blocks[path[k-1]]
        curr_block = blocks[path[k]]
        ov = overlaps[path[k-1]][path[k]]
        final_word += curr_block[ov:]
        
    return "".join(map(str, final_word))
