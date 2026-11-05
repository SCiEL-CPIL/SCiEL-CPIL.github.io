"""Improved K4-free graph construction for n=18 using bitmask and exact B&B."""
import time
import random

# EVOLVE-BLOCK-START

N = 18
ALL_MASK = (1 << N) - 1

def _build_adj_masks(edge_list):
    """Build adjacency bitmask list from edge list."""
    adj = [0] * N
    for u, v in edge_list:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_k4_free_masks(adj):
    """Check K4-freeness using bitmasks."""
    n = N
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                common = adj[i] & adj[j] & ((1 << n) - 1) & ~((1 << (j + 1)) - 1)
                # For each pair (j, k) with k > j, check if they share a common neighbor l > k
                for k in range(j + 1, n):
                    if common & (1 << k):
                        # Check if there exists l > k adjacent to i, j, k
                        common_ik = adj[i] & adj[k] & ((1 << n) - 1) & ~((1 << (k + 1)) - 1)
                        common_jk = adj[j] & adj[k] & ((1 << n) - 1) & ~((1 << (k + 1)) - 1)
                        if common_ik & common_jk:
                            return False
    return True

def _max_triangle_free_exact(adj):
    """
    Exact branch and bound for maximum triangle-free induced subgraph.
    A set S is triangle-free if no three vertices in S are pairwise adjacent.
    Equivalent: the subgraph induced by S has no K3.
    
    We use the following approach:
    - State: (current_set_mask, candidates_mask, best_size)
    - A candidate v can be added if it doesn't form a triangle with any two vertices in current_set.
    - v forms a triangle with u, w in current_set iff adj[u]&adj[w] has v AND u-w is an edge.
    - Actually: v, u, w form a triangle iff (adj[v]&adj[u]) & (1<<w) and adj[u]&adj[w]... 
      More simply: v,u,w form a triangle iff adj[v] & adj[u] & (1<<w) != 0 AND adj[v] & adj[w] & (1<<u) != 0 AND adj[u] & adj[w] & (1<<v) != 0.
      But if v,u,w are all in the set, they form a triangle iff adj[v] & adj[u] & (1<<w) != 0 (which implies the other two by symmetry).
    
    Actually, the cleanest check: when adding v to current set C, v creates a triangle iff
    there exist u, w in C such that u-w is an edge AND v is adjacent to both u and w.
    i.e., (adj[v] & C) contains at least two vertices u, w that are adjacent to each other.
    Equivalently: let Nv_in_C = adj[v] & C. Then v creates a triangle iff the subgraph induced by Nv_in_C has at least one edge.
    i.e., exists u in Nv_in_C such that (adj[u] & Nv_in_C) != 0.
    
    For efficiency, we precompute for each vertex v and each subset... no, that's too expensive.
    
    Alternative: use the fact that n=18 is small. We can do a proper B&B.
    """
    n = N
    
    # Precompute: for each vertex v, adj[v] is its neighbor mask
    # For a set C (as mask), and a candidate v not in C:
    # v can be added iff no two neighbors of v in C are adjacent to each other.
    # Let NvC = adj[v] & C
    # v cannot be added iff there exists u in NvC with (adj[u] & NvC) != 0 (excluding u itself, but adj[u] doesn't include u)
    # Actually adj[u] & NvC gives neighbors of u that are in NvC. If non-zero, then u is adjacent to some w in NvC, 
    # and both u,w are neighbors of v, so v,u,w form a triangle.
    
    # B&B: we process vertices in order, at each step decide include/exclude.
    # To bound: remaining vertices + current size <= best -> prune.
    
    best = [0]
    
    def _bb(idx, current_mask, current_size):
        if idx == n:
            if current_size > best[0]:
                best[0] = current_size
            return
        # Upper bound: current_size + (n - idx)
        if current_size + (n - idx) <= best[0]:
            return
        
        v = idx
        # Option 1: exclude v
        _bb(idx + 1, current_mask, current_size)
        
        # Option 2: include v (if it doesn't create a triangle)
        NvC = adj[v] & current_mask
        # Check if NvC has any edge
        has_edge_in_NvC = False
        temp = NvC
        while temp:
            lowbit = temp & (-temp)
            u = lowbit.bit_length() - 1
            if adj[u] & NvC:
                has_edge_in_NvC = True
                break
            temp ^= lowbit
        
        if not has_edge_in_NvC:
            _bb(idx + 1, current_mask | (1 << v), current_size + 1)
    
    _bb(0, 0, 0)
    return best[0]

def _max_triangle_free_fast(adj):
    """
    Faster evaluation using greedy lower bound + limited B&B.
    For the evaluator's exact B&B, we need to be accurate, but we can use
    this for our own search purposes where an upper bound is sufficient.
    """
    n = N
    
    # Greedy: try multiple random orderings
    best_greedy = 0
    for _ in range(50):
        order = list(range(n))
        random.shuffle(order)
        current = 0
        size = 0
        for v in order:
            NvC = adj[v] & current
            # Check if NvC has an edge
            has_edge = False
            temp = NvC
            while temp:
                lowbit = temp & (-temp)
                u = lowbit.bit_length() - 1
                if adj[u] & NvC:
                    has_edge = True
                    break
                temp ^= lowbit
            if not has_edge:
                current |= (1 << v)
                size += 1
        if size > best_greedy:
            best_greedy = size
    
    return best_greedy

def _random_k4_free_graph(p_edge):
    """Generate a random graph and prune K4s."""
    adj = [0] * N
    # Start with random edges
    for i in range(N):
        for j in range(i + 1, N):
            if random.random() < p_edge:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Remove edges to eliminate K4s
    # Find all K4s and remove one edge from each
    changed = True
    while changed:
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                if not (adj[i] & (1 << j)):
                    continue
                # Find common neighbors of i and j
                common = adj[i] & adj[j]
                # For each pair (k, l) in common, check if k-l is an edge
                for k in range(j + 1, N):
                    if common & (1 << k):
                        for l in range(k + 1, N):
                            if common & (1 << l):
                                if adj[k] & (1 << l):
                                    # K4 found: i, j, k, l. Remove edge (i, j)
                                    adj[i] &= ~(1 << j)
                                    adj[j] &= ~(1 << i)
                                    changed = True
                                    break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
            if changed:
                break
    
    return adj

def _local_search(adj, max_iters=5000, time_limit=10.0):
    """
    Local search: try to add/remove/swap edges to reduce max triangle-free size.
    """
    start = time.time()
    best_adj = list(adj)
    best_score = _max_triangle_free_fast(best_adj)
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        if not _is_k4_free_masks(best_adj):
            break
        
        # Try random edge additions
        improved = False
        for _ in range(20):
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            if best_adj[i] & (1 << j):
                continue
            
            # Try adding edge (i, j)
            best_adj[i] |= (1 << j)
            best_adj[j] |= (1 << i)
            
            if _is_k4_free_masks(best_adj):
                score = _max_triangle_free_fast(best_adj)
                if score < best_score:
                    best_score = score
                    improved = True
                else:
                    # Revert
                    best_adj[i] &= ~(1 << j)
                    best_adj[j] &= ~(1 << i)
            else:
                # Revert
                best_adj[i] &= ~(1 << j)
                best_adj[j] &= ~(1 << i)
        
        # Try random edge removals
        for _ in range(10):
            # Find a random edge
            edges = []
            for i in range(N):
                for j in range(i + 1, N):
                    if best_adj[i] & (1 << j):
                        edges.append((i, j))
            if not edges:
                break
            i, j = random.choice(edges)
            
            # Remove edge
            best_adj[i] &= ~(1 << j)
            best_adj[j] &= ~(1 << i)
            
            score = _max_triangle_free_fast(best_adj)
            if score < best_score:
                best_score = score
                improved = True
            else:
                # Revert
                best_adj[i] |= (1 << j)
                best_adj[j] |= (1 << i)
        
        if not improved and it > 100:
            # Try a bigger perturbation
            for _ in range(5):
                i = random.randint(0, N - 1)
                j = random.randint(0, N - 1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                if best_adj[i] & (1 << j):
                    # Remove
                    best_adj[i] &= ~(1 << j)
                    best_adj[j] &= ~(1 << i)
                else:
                    # Add
                    best_adj[i] |= (1 << j)
                    best_adj[j] |= (1 << i)
            # Check K4-free
            if not _is_k4_free_masks(best_adj):
                # Revert by regenerating
                break
    
    return best_adj, best_score

def _construct():
    """Main construction: random dense graphs, K4 pruning, local search."""
    random.seed(12345)
    
    best_adj = None
    best_score = float('inf')
    
    # Strategy 1: Start with various random densities, prune K4s, local search
    for p in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for trial in range(5):
            adj = _random_k4_free_graph(p)
            if not _is_k4_free_masks(adj):
                continue
            score = _max_triangle_free_fast(adj)
            if score < best_score:
                best_score = score
                best_adj = list(adj)
    
    # Strategy 2: Local search from best found
    if best_adj is not None:
        adj_ls, score_ls = _local_search(best_adj, max_iters=3000, time_limit=30.0)
        if _is_k4_free_masks(adj_ls):
            score_check = _max_triangle_free_fast(adj_ls)
            if score_check < best_score:
                best_score = score_check
                best_adj = adj_ls
    
    # Strategy 3: Try Turán-based constructions as fallback
    # 3-partite Turán graph T(18,3)
    turan_adj = [0] * N
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    turan_adj[a] |= (1 << b)
                    turan_adj[b] |= (1 << a)
    
    if _is_k4_free_masks(turan_adj):
        score_t = _max_triangle_free_fast(turan_adj)
        if score_t < best_score:
            best_score = score_t
            best_adj = turan_adj
    
    if best_adj is None:
        best_adj = turan_adj
    
    # Convert to edge list
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    try:
        result = _construct()
    except Exception:
        # Fallback: Turán graph
        result = [[a, b] for i in range(3) for j in range(i + 1, 3)
                  for a in range(i * 6, (i + 1) * 6) for b in range(j * 6, (j + 1) * 6)]
    
    # Final verification
    adj = _build_adj_masks(result)
    if not _is_k4_free_masks(adj):
        # Fallback to Turán
        result = [[a, b] for i in range(3) for j in range(i + 1, 3)
                  for a in range(i * 6, (i + 1) * 6) for b in range(j * 6, (j + 1) * 6)]
    
    elapsed = time.time() - start_time
    if elapsed > 130:
        # Should not happen
        result = [[a, b] for i in range(3) for j in range(i + 1, 3)
                  for a in range(i * 6, (i + 1) * 6) for b in range(j * 6, (j + 1) * 6)]
    
    return result

# EVOLVE-BLOCK-END
