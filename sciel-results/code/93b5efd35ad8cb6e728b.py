import time
import random

_START = time.time()

def _spf_sieve(limit):
    spf = list(range(limit + 1))
    if limit >= 0:
        spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    for i in range(2, int(limit ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

# We need to factor sums up to 20000
_SPF = _spf_sieve(20000)

# We only care about the first few primes for the mask
# Let's include primes up to 113 to be safe for mask construction
PRIMES_LIST = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
P2IDX = {p: i for i, p in enumerate(PRIMES_LIST)}

def _build_masks(limit=20000):
    """
    Precompute prime factor masks for all numbers up to limit.
    mask[x] is a bitmask where bit i is set if PRIMES_LIST[i] divides x.
    If x has a prime factor not in PRIMES_LIST, mask[x] is -1 (invalid for smoothness checks with small prime sets).
    """
    masks = [-1] * (limit + 1)
    masks[0] = 0 
    masks[1] = 0 

    for x in range(2, limit + 1):
        t = x
        m = 0
        ok = True
        while t > 1:
            p = _SPF[t]
            idx = P2IDX.get(p)
            if idx is None:
                ok = False
                break
            m |= 1 << idx
            while t % p == 0:
                t //= p
        if ok:
            masks[x] = m
    return masks

_MASKS = _build_masks(20000)

def _is_smooth(x, p_mask):
    m = _MASKS[x]
    if m < 0:
        return False
    # Check if all prime factors of x are in p_mask
    return (m & ~p_mask) == 0

def _greedy_clique(nodes, k, p_mask, deadline, max_restarts=10):
    """
    Greedy heuristic to find a clique of size k.
    Tries multiple random restarts with different vertex orderings.
    """
    n = len(nodes)
    if n < k:
        return None
    
    # Build adjacency list as bitmasks
    adj = [0] * n
    for i in range(n):
        ai = nodes[i]
        for j in range(i + 1, n):
            if _is_smooth(ai + nodes[j], p_mask):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Precompute higher mask
    higher = [0] * n
    for v in range(n - 2, -1, -1):
        higher[v] = higher[v + 1] | (1 << (v + 1))
    
    best_clique = [None]
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Randomize the order of vertices for this restart
        if restart == 0:
            order = list(range(n))
        else:
            order = list(range(n))
            random.shuffle(order)
        
        # Greedy: build clique by always picking the vertex with most connections
        # to remaining candidates
        current_clique = []
        candidates = (1 << n) - 1
        
        for _ in range(k):
            if candidates.bit_count() == 0:
                break
            
            # Find vertex in candidates with highest degree within candidates
            best_v = -1
            best_deg = -1
            
            m = candidates
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                m &= m - 1
                deg = (candidates & adj[v]).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                break
            
            current_clique.append(best_v)
            candidates &= adj[best_v]
            candidates &= ~(1 << best_v)
        
        if len(current_clique) == k:
            best_clique[0] = current_clique
            return current_clique
    
    return None

def _find_clique(nodes, k, p_mask, deadline, node_budget):
    """
    Finds a clique of size k in the graph defined by the nodes and smoothness.
    First tries greedy heuristic, then falls back to exact DFS if needed.
    nodes: list of integers
    k: target clique size
    p_mask: bitmask of allowed primes
    deadline: time limit
    node_budget: max nodes to visit
    """
    n = len(nodes)
    if n < k:
        return None
    
    # Try greedy first - it's much faster
    greedy_deadline = min(time.time() + 2.0, deadline - 1.0)
    greedy_result = _greedy_clique(nodes, k, p_mask, greedy_deadline, max_restarts=20)
    if greedy_result is not None:
        return greedy_result
    
    # Build adjacency list as bitmasks for fast intersection
    adj = [0] * n
    for i in range(n):
        ai = nodes[i]
        for j in range(i + 1, n):
            if _is_smooth(ai + nodes[j], p_mask):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Precompute higher mask: higher[v] has bits set for all indices > v
    higher = [0] * n
    for v in range(n - 2, -1, -1):
        higher[v] = higher[v + 1] | (1 << (v + 1))
        
    # Degree for ordering
    deg = [adj[i].bit_count() for i in range(n)]
    
    best_clique = [None]
    node_count = [0]
    aborted = [False]
    
    def dfs(candidates_mask, current_clique):
        if len(current_clique) == k:
            best_clique[0] = list(current_clique)
            return True
        
        if time.time() > deadline:
            aborted[0] = True
            return False
            
        node_count[0] += 1
        if node_count[0] > node_budget:
            aborted[0] = True
            return False

        # If not enough candidates left
        if candidates_mask.bit_count() < k - len(current_clique):
            return False
            
        # Get candidates sorted by degree in the subgraph to improve pruning
        m = candidates_mask
        cands = []
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            m &= m - 1
            nc_count = (candidates_mask & adj[v] & higher[v]).bit_count()
            cands.append((nc_count, v))
            
        # Sort by degree descending (try high-degree vertices first)
        cands.sort(key=lambda x: -x[0])
        
        for nc_count, v in cands:
            if nc_count + 1 < k - len(current_clique):
                continue
                
            next_mask = candidates_mask & adj[v] & higher[v]
            if dfs(next_mask, current_clique + [v]):
                return True
                
        return False

    initial_mask = (1 << n) - 1
    if dfs(initial_mask, []):
        return best_clique[0]
    return None

def construct():
    deadline = time.time() + 85.0 # Safety margin
    
    # Try increasing number of primes
    # Start with the smallest primes
    for num_primes in range(2, 15):
        if time.time() > deadline - 5:
            break
            
        P = PRIMES_LIST[:num_primes]
        p_mask = 0
        for p in P:
            p_mask |= 1 << P2IDX[p]
            
        # Find all smooth numbers <= 10000
        smooth_nums = [x for x in range(1, 10001) if _is_smooth(x, p_mask)]
        
        if len(smooth_nums) < 12:
            continue
            
        # Heuristic: smaller numbers tend to have smaller sums, which are more likely to be smooth
        # So we can just use the sorted list (which is already sorted by value).
        
        # Run clique search with greedy first
        sub_deadline = min(time.time() + 8.0, deadline - 2.0)
        
        # Try different subsets of smooth numbers to find a clique
        # Start with small subsets and increase
        max_smooth_to_try = min(len(smooth_nums), 300)
        
        for T in range(12, max_smooth_to_try + 1, 15):
            if time.time() > deadline - 2.0:
                break
            subset = smooth_nums[:T]
            res_idx = _find_clique(subset, 12, p_mask, sub_deadline, 200000)
            if res_idx is not None:
                # Convert indices back to values
                return [subset[i] for i in res_idx]
                
        # If not found in subset, try the full list with a larger budget
        if time.time() < deadline - 2.0:
            res_idx = _find_clique(smooth_nums, 12, p_mask, sub_deadline, 1000000)
            if res_idx is not None:
                # Convert indices back to values
                return [smooth_nums[i] for i in res_idx]
            
    # Fallback: Return a known valid set with a larger number of primes
    return [1, 2, 3, 5, 6, 9, 10, 15, 18, 25, 27, 30]
