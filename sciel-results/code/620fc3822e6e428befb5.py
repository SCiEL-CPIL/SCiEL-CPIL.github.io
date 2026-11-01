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
    num_blocks = len(blocks)
    overlaps = [[0]*num_blocks for _ in range(num_blocks)]
    
    for i in range(num_blocks):
        bi = blocks[i]
        for j in range(num_blocks):
            bj = blocks[j]
            max_k = min(len(bi), len(bj))
            found = 0
            for k in range(max_k, -1, -1):
                if k == 0:
                    found = 0
                    break
                if bi[-k:] == bj[:k]:
                    found = k
                    break
            overlaps[i][j] = found

    def calc_total_overlap(path):
        total = 0
        for k in range(len(path) - 1):
            total += overlaps[path[k]][path[k+1]]
        return total

    def two_opt(path, max_passes=5):
        n_path = len(path)
        for _ in range(max_passes):
            improved = False
            for i in range(n_path - 1):
                for j in range(i + 1, n_path):
                    # Calculate delta
                    delta = 0
                    if i > 0:
                        delta += overlaps[path[i-1]][path[j]] - overlaps[path[i-1]][path[i]]
                    if j < n_path - 1:
                        delta += overlaps[path[i+1]][path[j+1]] - overlaps[path[j]][path[j+1]]
                    
                    old_int = 0
                    for k in range(i, j):
                        old_int += overlaps[path[k]][path[k+1]]
                        
                    new_int = overlaps[path[i]][path[j]]
                    for k in range(j, i, -1):
                         new_int += overlaps[path[k]][path[k-1]]
                         
                    delta += new_int - old_int
                    
                    if delta > 0:
                        path[i+1 : j+1] = path[i+1 : j+1][::-1]
                        improved = True
            if not improved:
                break
        return path

    best_path = None
    best_score = -1
    
    # Random Restarts
    num_restarts = 30
    for r in range(num_restarts):
        if r == 0:
            # Greedy Start
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
        else:
            path = list(range(num_blocks))
            random.shuffle(path)
        
        path = two_opt(path)
        score = calc_total_overlap(path)
        if score > best_score:
            best_score = score
            best_path = path[:]
            
    # Build the string
    final_word = list(blocks[best_path[0]])
    for k in range(1, len(best_path)):
        prev_block = blocks[best_path[k-1]]
        curr_block = blocks[best_path[k]]
        ov = overlaps[best_path[k-1]][best_path[k]]
        final_word += curr_block[ov:]
        
    return "".join(map(str, final_word))
