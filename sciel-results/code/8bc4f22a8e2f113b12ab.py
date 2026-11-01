"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _build_superperm_recursive(n):
    """Classic recursive superpermutation construction."""
    if n == 1:
        return [1]
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def construct(n):
    if n == 1:
        return "1"
    
    start_time = time.time()
    deadline = start_time + 140
    
    if n <= 6:
        word = _build_superperm_recursive(n)
        return "".join(map(str, word))
    
    # For n=7: use greedy nearest-neighbor to find better ordering
    base6 = _build_superperm_recursive(6)
    perms6, seen = [], set()
    for i in range(len(base6) - 5):
        window = tuple(base6[i:i + 6])
        if len(set(window)) == 6 and window not in seen:
            seen.add(window)
            perms6.append(window)
    
    m = len(perms6)
    blocks = [list(p) + [7] + list(p) for p in perms6]
    block_len = 13
    
    # Precompute overlap matrix
    overlap_mat = [[0] * m for _ in range(m)]
    for i in range(m):
        bi = blocks[i]
        for j in range(m):
            bj = blocks[j]
            max_ov = min(len(bi), len(bj))
            ov = 0
            for o in range(max_ov, -1, -1):
                if bi[len(bi)-o:] == bj[:o]:
                    ov = o
                    break
            overlap_mat[i][j] = ov
    
    # First overlap with base6
    first_ov = [0] * m
    for i in range(m):
        bi = blocks[i]
        max_ov = min(len(base6), len(bi))
        for o in range(max_ov, -1, -1):
            if base6[len(base6)-o:] == bi[:o]:
                first_ov[i] = o
                break
    
    # Greedy: start with the perm that has max first overlap, then pick next with max overlap
    remaining = set(range(m))
    order = []
    
    # Start with best first overlap
    best_start = max(remaining, key=lambda x: first_ov[x])
    order.append(best_start)
    remaining.remove(best_start)
    
    while remaining and (time.time() - start_time) < deadline - 10:
        last = order[-1]
        # Pick next with max overlap
        best_next = max(remaining, key=lambda x: overlap_mat[last][x])
        order.append(best_next)
        remaining.remove(best_next)
    
    # Compute length
    def compute_length(order):
        total = len(base6) + block_len - first_ov[order[0]]
        for k in range(1, len(order)):
            total += block_len - overlap_mat[order[k-1]][order[k]]
        return total
    
    greedy_len = compute_length(order)
    
    # Build the word
    def build_word(order):
        current = list(base6)
        for idx in order:
            block = blocks[idx]
            max_ov = min(len(current), len(block))
            ov = 0
            for o in range(max_ov, -1, -1):
                if current[len(current)-o:] == block[:o]:
                    ov = o
                    break
            current += block[ov:]
        return current
    
    best_word = build_word(order)
    best_len = greedy_len
    
    # Also try standard order
    standard_order = list(range(m))
    std_len = compute_length(standard_order)
    if std_len < best_len:
        best_len = std_len
        best_word = build_word(standard_order)
    
    # Local search on greedy order: try swapping adjacent elements
    improved = True
    while improved and (time.time() - start_time) < deadline - 5:
        improved = False
        for i in range(len(order) - 1):
            if (time.time() - start_time) > deadline - 5:
                break
            # Try swapping i and i+1
            order[i], order[i+1] = order[i+1], order[i]
            new_len = compute_length(order)
            if new_len < best_len:
                best_len = new_len
                best_word = build_word(order)
            else:
                # Swap back
                order[i], order[i+1] = order[i+1], order[i]
            if new_len < best_len:
                improved = True
    
    return "".join(map(str, best_word))
# EVOLVE-BLOCK-END
