import numpy as np

def construct(n):
    if n == 1:
        return "1"
    
    def standard_superperm(k):
        if k == 1:
            return [1]
        prev = standard_superperm(k - 1)
        perms = []
        seen = set()
        for i in range(len(prev) - (k - 2)):
            window = tuple(prev[i:i + k - 1])
            if len(set(window)) == k - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [k] + list(p)
            max_overlap = min(len(new_word), len(block))
            best_overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new_word[len(new_word) - o:] == block[:o]:
                    best_overlap = o
                    break
            new_word.extend(block[best_overlap:])
        return new_word

    if n <= 5:
        word = standard_superperm(n)
        return "".join(map(str, word))
    
    # For n >= 6, use optimized construction
    base = standard_superperm(n - 1)
    base_str = "".join(map(str, base))
    
    perms = []
    seen = set()
    for i in range(len(base) - (n - 2)):
        window = tuple(base[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    
    m = len(perms)
    
    # Precompute blocks
    blocks = []
    for p in perms:
        blocks.append(list(p) + [n] + list(p))
    
    # Precompute overlap matrix
    # overlap[i][j] = max o such that blocks[i][-o:] == blocks[j][:o]
    overlap = np.zeros((m, m), dtype=int)
    for i in range(m):
        bi = blocks[i]
        for j in range(m):
            bj = blocks[j]
            max_o = min(len(bi), len(bj))
            best_o = 0
            # Since n is small, simple loop is fine
            for o in range(max_o, -1, -1):
                if o == 0 or bi[len(bi)-o:] == bj[:o]:
                    best_o = o
                    break
            overlap[i][j] = best_o
            
    # Precompute start overlaps
    # start_overlap[i] = overlap between base_str and blocks[i]
    start_overlap = np.zeros(m, dtype=int)
    for i in range(m):
        bi = blocks[i]
        max_o = min(len(base), len(bi))
        best_o = 0
        for o in range(max_o, -1, -1):
            if o == 0 or base[len(base)-o:] == bi[:o]:
                best_o = o
                break
        start_overlap[i] = best_o
        
    # Greedy initialization
    path = []
    unvisited = set(range(m))
    
    # Find start
    start_idx = np.argmax(start_overlap)
    path.append(start_idx)
    unvisited.remove(start_idx)
    current = start_idx
    
    while unvisited:
        # Find best next
        best_next = -1
        best_ov = -1
        for j in unvisited:
            if overlap[current][j] > best_ov:
                best_ov = overlap[current][j]
                best_next = j
        path.append(best_next)
        unvisited.remove(best_next)
        current = best_next
        
    # Local Search
    # Maximize sum of overlaps
    # Total Overlap = start_overlap[path[0]] + sum(overlap[path[i-1]][path[i]])
    
    def get_total_overlap(p):
        s = start_overlap[p[0]]
        for i in range(1, len(p)):
            s += overlap[p[i-1]][p[i]]
        return s
        
    current_overlap = get_total_overlap(path)
    
    improved = True
    while improved:
        improved = False
        for k in range(len(path) - 1):
            u = path[k]
            v = path[k+1]
            
            # Calculate delta for swapping u and v
            # Old terms:
            old_cost = 0
            if k > 0:
                old_cost += overlap[path[k-1]][u]
            else:
                old_cost += start_overlap[u]
                
            old_cost += overlap[u][v]
            
            if k + 1 < len(path) - 1:
                old_cost += overlap[v][path[k+2]]
                
            # New terms:
            new_cost = 0
            if k > 0:
                new_cost += overlap[path[k-1]][v]
            else:
                new_cost += start_overlap[v]
                
            new_cost += overlap[v][u]
            
            if k + 1 < len(path) - 1:
                new_cost += overlap[u][path[k+2]]
                
            delta = new_cost - old_cost
            if delta > 0:
                path[k], path[k+1] = v, u
                current_overlap += delta
                improved = True
                # Restart or continue? Restarting is safer for finding local opt
                # But continuing might be faster. Let's break and restart loop
                break
                
    # Construct final word
    # Start with base
    final_word = list(base)
    
    # Append blocks in path order
    for idx in path:
        block = blocks[idx]
        max_o = min(len(final_word), len(block))
        best_o = 0
        for o in range(max_o, -1, -1):
            if o == 0 or final_word[len(final_word)-o:] == block[:o]:
                best_o = o
                break
        final_word.extend(block[best_o:])
        
    return "".join(map(str, final_word))
