"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from functools import lru_cache

# EVOLVE-BLOCK-START

def _classic_superperm(n):
    """Classic recursive superpermutation construction."""
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
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if o > 0 and new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
                elif o == 0:
                    overlap = 0
                    break
            new += block[overlap:]
        word = new
    return word

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

def _overlap_fast(a_tuple, b_tuple):
    """Fast overlap using tuple comparison (caches better)."""
    max_ov = min(len(a_tuple), len(b_tuple))
    for o in range(max_ov, 0, -1):
        if a_tuple[-o:] == b_tuple[:o]:
            return o
    return 0

def _build_overlap_matrix(blocks):
    """Build full overlap matrix efficiently using tuples for faster comparison."""
    n = len(blocks)
    block_tuples = [tuple(b) for b in blocks]
    ov_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                ov_matrix[i][j] = _overlap_fast(block_tuples[i], block_tuples[j])
    return ov_matrix

def _block_for_perm(p, m):
    """The block for permutation p with new symbol m: p + [m] + p."""
    return list(p) + [m] + list(p)

def _total_length_given_order(order, perm_list, m):
    """
    Given an ordering of permutations, compute the total length of the
    resulting superpermutation segment (starting from empty, but we
    account for the first block fully).
    """
    if not order:
        return 0
    blocks = [_block_for_perm(perm_list[i], m) for i in order]
    total = len(blocks[0])
    for k in range(1, len(blocks)):
        ov = _overlap(blocks[k-1], blocks[k])
        total += len(blocks[k]) - ov
    return total

def _optimize_order_tsp(perm_list, m, time_limit=10.0):
    """
    Try to find a better ordering of perm_list to minimize total length.
    Uses greedy construction + 2-opt local search with time budget.
    """
    start_time = time.time()
    n_perms = len(perm_list)
    
    # Compute all-pairs overlap matrix
    blocks = [_block_for_perm(p, m) for p in perm_list]
    n = n_perms
    ov_matrix = _build_overlap_matrix(blocks)
    
    # Precompute block lengths
    block_lens = [len(b) for b in blocks]
    
    # Transition cost from i to j: block_lens[j] - ov_matrix[i][j]
    # Total cost = block_lens[order[0]] + sum_{k=1}^{n-1} (block_lens[order[k]] - ov_matrix[order[k-1]][order[k]])
    
    def compute_cost(order):
        """Compute total cost for a given order."""
        cost = block_lens[order[0]]
        for k in range(1, len(order)):
            i, j = order[k-1], order[k]
            cost += block_lens[j] - ov_matrix[i][j]
        return cost
    
    def greedy_construct():
        """Greedy construction: always pick the next block with max overlap."""
        used = [False] * n
        order = []
        current = 0  # Start with first permutation
        used[0] = True
        order.append(0)
        
        for _ in range(n - 1):
            best_j = -1
            best_ov = -1
            for j in range(n):
                if not used[j]:
                    ov = ov_matrix[current][j]
                    if ov > best_ov:
                        best_ov = ov
                        best_j = j
            if best_j == -1:
                break
            used[best_j] = True
            order.append(best_j)
            current = best_j
        
        return order
    
    def local_search_2opt(cur_order, cur_cost, t_limit):
        """2-opt local search with O(1) cost delta computation."""
        improved = True
        while improved and (time.time() - start_time) < t_limit:
            improved = False
            for i in range(n - 1):
                if (time.time() - start_time) > t_limit:
                    break
                for j in range(i + 1, n):
                    # Reverse segment [i+1..j]
                    old_edges = 0
                    new_edges = 0
                    # Edge from order[i] to order[i+1]
                    old_edges += block_lens[cur_order[i+1]] - ov_matrix[cur_order[i]][cur_order[i+1]]
                    # Edge from order[j] to order[j+1] (if exists)
                    if j + 1 < n:
                        old_edges += block_lens[cur_order[j+1]] - ov_matrix[cur_order[j]][cur_order[j+1]]
                    # New edge from order[i] to order[j]
                    new_edges += block_lens[cur_order[j]] - ov_matrix[cur_order[i]][cur_order[j]]
                    # New edge from order[i+1] to order[j+1] (if exists)
                    if j + 1 < n:
                        new_edges += block_lens[cur_order[j+1]] - ov_matrix[cur_order[i+1]][cur_order[j+1]]
                    
                    delta = new_edges - old_edges
                    if delta < 0:
                        cur_cost += delta
                        cur_order[i+1:j+1] = cur_order[i+1:j+1][::-1]
                        improved = True
        return cur_order, cur_cost
    
    # Start with greedy construction (often better than classic order)
    best_order = greedy_construct()
    best_cost = compute_cost(best_order)
    
    # Also try classic order
    classic_order = list(range(n))
    classic_cost = compute_cost(classic_order)
    if classic_cost < best_cost:
        best_cost = classic_cost
        best_order = classic_order
    
    # 2-opt from best starting point
    t1 = min(time_limit * 0.7, time_limit - 2.0)
    best_order, best_cost = local_search_2opt(best_order, best_cost, t1)
    
    # Try random restarts if time remains
    if (time.time() - start_time) < time_limit - 1.0:
        import random
        random.seed(42)
        n_restarts = 2
        for r in range(n_restarts):
            if (time.time() - start_time) > time_limit - 0.5:
                break
            rand_order = list(range(n))
            random.shuffle(rand_order)
            cur_order = list(rand_order)
            cur_cost = compute_cost(cur_order)
            cur_order, cur_cost = local_search_2opt(cur_order, cur_cost, time_limit - 0.5)
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_order = cur_order
    
    return best_order, best_cost

def construct(n):
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s kill
    
    if n <= 1:
        return "1"
    
    if n == 2:
        return "121"
    
    if n == 3:
        return "123121321"
    
    # For n >= 4, use optimized construction
    # Build the (n-1)-permutations in classic order
    perm_list = _build_level_perms(n)
    m = n
    n_perms = len(perm_list)
    
    # Get classic length as baseline
    classic_word = _classic_superperm(n)
    classic_len = len(classic_word)
    
    # Try to optimize the ordering at the top level
    if n_perms <= 720 and (time.time() - start_time) < deadline - 20.0:
        time_limit_for_tsp = min(60.0, deadline - (time.time() - start_time) - 20.0)
        if time_limit_for_tsp > 1.0:
            best_order, best_cost = _optimize_order_tsp(perm_list, m, time_limit=time_limit_for_tsp)
            
            # Build the word using the optimized order
            blocks = [_block_for_perm(perm_list[i], m) for i in best_order]
            word = list(blocks[0])
            for k in range(1, len(blocks)):
                b = blocks[k]
                ov = _overlap(word, b)
                word += b[ov:]
            
            if len(word) < classic_len:
                return "".join(map(str, word))
    
    # Fall back to classic
    return "".join(map(str, classic_word))

# EVOLVE-BLOCK-END
