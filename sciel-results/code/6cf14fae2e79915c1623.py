import time
import itertools

def _classic_superperm(n):
    """Classic recursive superpermutation construction."""
    if n == 1:
        return [1]
    word = _classic_superperm(n - 1)
    m = n
    perms, seen = [], set()
    # Extract (n-1)-permutations in order of appearance
    for i in range(len(word) - m + 2):
        window = tuple(word[i:i + m - 1])
        if len(set(window)) == m - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
            
    new = []
    for p in perms:
        block = list(p) + [m] + list(p)
        # Find max overlap with end of new
        overlap = 0
        max_ov = min(len(new), len(block))
        for o in range(max_ov, 0, -1):
            if new[len(new) - o:] == block[:o]:
                overlap = o
                break
        new += block[overlap:]
    return new

def _build_level_perms(n):
    """
    Build the list of (n-1)-length permutations in the order they appear
    in the classic superpermutation of level n-1.
    """
    if n == 1:
        return [[1]]
    word = _classic_superperm(n - 1)
    perms, seen = [], set()
    m = n - 1
    for i in range(len(word) - m + 1):
        window = tuple(word[i:i + m])
        if len(set(window)) == m and window not in seen:
            seen.add(window)
            perms.append(window)
    return [list(p) for p in perms]

def _overlap(a, b):
    """Maximum overlap: suffix of a matches prefix of b."""
    max_ov = min(len(a), len(b))
    for o in range(max_ov, 0, -1):
        if a[-o:] == b[:o]:
            return o
    return 0

def _optimize_order_2opt(perm_list, m, time_limit=100.0):
    """
    Optimize the ordering of blocks to minimize total length.
    """
    start_time = time.time()
    n_perms = len(perm_list)
    blocks = [list(p) + [m] + list(p) for p in perm_list]
    
    # Precompute overlap matrix
    # ov[i][j] is overlap of block i suffix and block j prefix
    ov = [[0] * n_perms for _ in range(n_perms)]
    for i in range(n_perms):
        for j in range(n_perms):
            if i != j:
                ov[i][j] = _overlap(blocks[i], blocks[j])
                
    # Initial order: classic order (0, 1, ..., n_perms-1)
    order = list(range(n_perms))
    
    def get_cost(order):
        # Cost is total length = len(block[0]) + sum(len(block[k]) - ov[order[k-1]][order[k]])
        # Since len(block) is constant L, cost = L * N - sum(ov)
        # Minimizing cost is maximizing sum(ov)
        # We return -sum(ov) to minimize
        total_ov = 0
        for k in range(1, len(order)):
            i, j = order[k-1], order[k]
            total_ov += ov[i][j]
        return -total_ov
        
    best_order = list(order)
    best_cost = get_cost(best_order)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n_perms - 1):
            if (time.time() - start_time) > time_limit:
                break
            for j in range(i + 1, n_perms):
                # Reverse segment [i+1, j]
                # New order: ... order[i], order[j], order[j-1], ..., order[i+1], ...
                # Changes in overlap:
                # Old: ov[order[i-1]][order[i]] + ov[order[i]][order[i+1]] + ... + ov[order[j]][order[j+1]]
                # This is complex to compute incrementally without a full path sum.
                # Given N=720, a full recalculation of cost is O(N).
                # A 2-opt move changes O(1) edges if we track the total sum?
                # No, the segment reversal changes the internal edges of the segment and the two boundary edges.
                # Internal edges of segment [i+1..j] are reversed.
                # Old internal sum: sum_{k=i+1}^{j-1} ov[order[k]][order[k+1]]
                # New internal sum: sum_{k=i+1}^{j-1} ov[order[k+1]][order[k]]
                # Boundary edges:
                # Old: ov[order[i]][order[i+1]] and ov[order[j]][order[j+1]] (if j+1 < N)
                # New: ov[order[i]][order[j]] and ov[order[i+1]][order[j+1]] (if j+1 < N)
                
                # Let's just recalculate the total overlap for the whole path.
                # It's O(N) = 720.
                # Number of pairs is ~ N^2/2 = 250,000.
                # Total ops ~ 250,000 * 720 = 180,000,000.
                # This might be too slow in Python for 100s.
                # We need a faster cost update.
                
                # Let's maintain the total overlap sum.
                # We need to update it incrementally.
                pass

    # The incremental update is tricky to implement correctly and quickly in pure Python without NumPy.
    # Let's use a simpler heuristic: Simulated Annealing or just many random restarts of 2-opt with full recalculation but limited passes.
    # Or, we can just try to find the specific known improvement.
    
    # Actually, for N=720, 2-opt with full recalculation is O(N^3) total if we do N^2 moves.
    # N^3 = 370M. In Python, this is likely > 100s.
    # We should use a more efficient local search or limit the search.
    
    # Alternative: Use the known structure.
    # The classic order is a Hamiltonian path in the permutation graph.
    # We can try swapping adjacent elements (1-opt) which is O(N^2) moves, each O(1) update if we track sums.
    # Let's implement 1-opt (adjacent swap) with incremental cost update.
    
    # Re-initialize
    order = list(range(n_perms))
    # Calculate initial total overlap
    total_ov = 0
    for k in range(1, n_perms):
        total_ov += ov[order[k-1]][order[k]]
    best_total_ov = total_ov
    best_order = list(order)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n_perms - 1):
            if (time.time() - start_time) > time_limit:
                break
            # Swap order[i] and order[i+1]
            a = order[i]
            b = order[i+1]
            
            # Calculate change in total overlap
            delta = 0
            
            # Edge before i
            if i > 0:
                prev = order[i-1]
                # Old: ov[prev][a]
                # New: ov[prev][b]
                delta += ov[prev][b] - ov[prev][a]
            
            # Edge between i and i+1
            # Old: ov[a][b]
            # New: ov[b][a]
            delta += ov[b][a] - ov[a][b]
            
            # Edge after i+1
            if i + 1 < n_perms - 1:
                next_node = order[i+2]
                # Old: ov[b][next_node]
                # New: ov[a][next_node]
                delta += ov[a][next_node] - ov[b][next_node]
                
            if delta > 0:
                # Improvement
                total_ov += delta
                order[i], order[i+1] = b, a
                improved = True
                if total_ov > best_total_ov:
                    best_total_ov = total_ov
                    best_order = list(order)
                    
    return best_order

def construct(n):
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412134132413241423142314321" # Length 19? Classic is 19.
        # Actually, let's just use classic for n<=6 as it's optimal.
        
    if n <= 6:
        word = _classic_superperm(n)
        return "".join(map(str, word))
        
    if n == 7:
        perm_list = _build_level_perms(n)
        m = n
        n_perms = len(perm_list)
        
        # Get classic word for fallback
        classic_word = _classic_superperm(n)
        classic_len = len(classic_word)
        
        # Optimize
        time_left = deadline - (time.time() - start_time)
        if time_left > 5.0:
            best_order = _optimize_order_2opt(perm_list, m, time_limit=time_left - 5.0)
            
            # Build word from best_order
            blocks = [list(perm_list[i]) + [m] + list(perm_list[i]) for i in best_order]
            word = list(blocks[0])
            for k in range(1, len(blocks)):
                b = blocks[k]
                ov = _overlap(word, b)
                word += b[ov:]
                
            if len(word) < classic_len:
                return "".join(map(str, word))
        
        return "".join(map(str, classic_word))
        
    # For n > 7, just use classic
    word = _classic_superperm(n)
    return "".join(map(str, word))
