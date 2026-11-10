"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time

# EVOLVE-BLOCK-START
def construct(n):
    if n <= 1:
        return "1"
    if n <= 6:
        # Classic construction is optimal for n <= 6
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
                # Find overlap
                overlap = 0
                max_ov = min(len(new), len(block))
                for o in range(max_ov, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    # For n >= 7, use TSP-based ordering of permutations
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Generate permutations of 1..n-1
    symbols = list(range(1, n))
    perms = list(itertools.permutations(symbols))
    m = len(perms)
    
    # Precompute cost matrix (overlap)
    # Overlap[i][j] = max k such that perms[i][-k:] == (perms[i] + (n,) + perms[j])[:k]
    # Note: perms[i] is a prefix of the block, so overlap is at least 0?
    # Actually, we match suffix of perms[i] with prefix of Block.
    # Block = perms[i] + (n,) + perms[j]
    # Since perms[i] is a permutation, it has no non-trivial borders.
    # But we are matching suffix of perms[i] with prefix of (perms[i] + ...).
    
    overlaps = [[0]*m for _ in range(m)]
    for i in range(m):
        pi = perms[i]
        for j in range(m):
            pj = perms[j]
            block_prefix = pi + (n,) + pj
            # Find max k
            max_k = len(pi) # n-1
            # Check k from max_k down to 0
            # pi[-k:] == block_prefix[:k]
            for k in range(max_k, -1, -1):
                if k == 0:
                    overlaps[i][j] = 0
                    break
                if pi[-k:] == block_prefix[:k]:
                    overlaps[i][j] = k
                    break
    
    # Solve TSP: Maximize sum of overlaps
    # Start with a random or greedy path
    # Greedy: Start at 0, go to neighbor with max overlap
    path = [0]
    unvisited = set(range(1, m))
    curr = 0
    while unvisited:
        best_next = -1
        best_ov = -1
        for j in unvisited:
            if overlaps[curr][j] > best_ov:
                best_ov = overlaps[curr][j]
                best_next = j
        path.append(best_next)
        unvisited.remove(best_next)
        curr = best_next
        
    # 2-Opt improvement
    def path_cost(p):
        total = 0
        for k in range(len(p)-1):
            total += overlaps[p[k]][p[k+1]]
        return total
    
    improved = True
    iterations = 0
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(path)-2):
            if time.time() > deadline:
                break
            a = path[i]
            b = path[i+1]
            c = path[i+2]
            # Check if swapping b and c improves
            # Current: a->b->c
            # New: a->c->b
            curr_cost = overlaps[a][b] + overlaps[b][c]
            new_cost = overlaps[a][c] + overlaps[c][b]
            if new_cost > curr_cost:
                path[i+1], path[i+2] = path[i+2], path[i+1]
                improved = True
                iterations += 1
        # Optionally add 3-opt or restarts if time permits
        if not improved:
            break
            
    # Construct word
    word = []
    first_p = perms[path[0]]
    word.extend(list(first_p) + [n] + list(first_p))
    
    for idx in range(1, len(path)):
        p = perms[path[idx]]
        block = list(p) + [n] + list(p)
        # Find overlap with current word end
        # We precomputed overlaps[path[idx-1]][path[idx]]
        ov = overlaps[path[idx-1]][path[idx]]
        word.extend(block[ov:])
        
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
