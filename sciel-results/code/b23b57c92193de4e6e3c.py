"""Rewritten program: greedy + min-cut construction of 40 subsets of {0..11} maximizing comparable pairs."""
import time
import random
import itertools
import sys

# EVOLVE-BLOCK-START
N = 12
M = 40
ALL_MASKS = list(range(1 << N))  # 4096 masks
NUM_MASKS = 1 << N
DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _count_comparable(family_masks):
    """Count unordered pairs where one is strict subset of the other."""
    n = len(family_masks)
    count = 0
    for i in range(n):
        mi = family_masks[i]
        for j in range(i + 1, n):
            mj = family_masks[j]
            if (mi & mj) == mi and mi != mj:
                count += 1
            elif (mi & mj) == mj and mi != mj:
                count += 1
    return count

def _build_comparability():
    """Precompute for each mask, the list of masks it's comparable with (strict subset/superset)."""
    # comp[m] = list of masks m' such that m ⊂ m' or m ⊃ m'
    comp = [[] for _ in range(NUM_MASKS)]
    for m in range(NUM_MASKS):
        # m's strict supersets: add any bit not in m
        # m's strict subsets: remove any bit from m
        # Efficient: iterate over all masks and check
        # But that's 4096*4096 = 16M which is feasible
        pass
    # Do it properly
    for m in range(NUM_MASKS):
        for m2 in range(m + 1, NUM_MASKS):
            if (m & m2) == m:
                # m is subset of m2
                if m != m2:
                    comp[m].append(m2)
                    comp[m2].append(m)
            elif (m & m2) == m2:
                # m2 is subset of m
                if m != m2:
                    comp[m].append(m2)
                    comp[m2].append(m)
    return comp

def _max_flow_min_cut(source, sink, adj, deadline):
    """
    Simple max-flow using Edmonds-Karp (BFS-based).
    adj: adjacency list where adj[u] = list of (v, capacity)
    Returns max flow value.
    """
    from collections import deque
    
    n = len(adj)
    # Build residual graph
    # Use dict-based for sparse graph
    # Actually, let's use a different approach: push-relabel might be better for this size
    
    # For 4096+2 nodes with potentially many edges, let's use a simpler approach
    # Actually, the graph is very dense (each node connects to many others)
    # Let me count: for a mask of size k, it has C(12,k) subsets and C(12,12-k) supersets roughly
    # Total edges could be large
    
    # Let's use Dinic's algorithm for better performance
    # Build capacity matrix as list of dicts
    cap = [dict() for _ in range(n)]
    
    for u in range(n):
        for v, c in adj[u]:
            if c > 0:
                cap[u][v] = cap[u].get(v, 0) + c
    
    # Add reverse edges with 0 capacity
    for u in range(n):
        for v in list(cap[u].keys()):
            if v not in cap[v]:
                cap[v][u] = 0
    
    # Dinic's algorithm
    INF = 10**18
    flow = 0
    
    while time.time() < deadline:
        # BFS to find level graph
        level = [-1] * n
        level[source] = 0
        q = deque([source])
        while q:
            u = q.popleft()
            for v in cap[u]:
                if level[v] < 0 and cap[u][v] > 0:
                    level[v] = level[u] + 1
                    q.append(v)
        
        if level[sink] < 0:
            break
        
        # DFS to find blocking flow
        ptr = [0] * n
        
        def dfs(u, pushed):
            if pushed == 0:
                return 0
            if u == sink:
                return pushed
            while ptr[u] < len(cap[u]):
                v = list(cap[u].keys())[ptr[u]]
                if level[v] != level[u] + 1 or cap[u][v] <= 0:
                    ptr[u] += 1
                    continue
                tr = dfs(v, min(pushed, cap[u][v]))
                if tr == 0:
                    ptr[u] += 1
                    continue
                cap[u][v] -= tr
                cap[v][u] += tr
                return tr
            return 0
        
        while True:
            pushed = dfs(source, INF)
            if pushed == 0:
                break
            flow += pushed
    
    return flow

def _min_cut_construct(deadline):
    """
    Use min-cut to find optimal 40-subset.
    
    Formulation: We want to select exactly M=40 nodes to maximize comparable pairs.
    This is a cardinality-constrained optimization.
    
    Approach: Lagrangian relaxation.
    For a parameter λ, maximize: sum of comparable pairs - λ * |S|
    This can be solved as a min-cut problem.
    
    Actually, the standard approach for "maximize edges in induced subgraph of size k"
    is NP-hard in general, but for this specific structure (poset comparability),
    we can use the following:
    
    Key insight: The comparability graph of a poset is a perfect graph (it's the
    complement of a comparability graph... wait, no).
    
    Actually, let me think differently. The problem is:
    Maximize sum_{i<j, comparable(i,j)} x_i * x_j  subject to sum x_i = 40, x_i in {0,1}
    
    This is a quadratic binary program. The Lagrangian relaxation gives:
    Maximize sum_{i<j, comp(i,j)} x_i * x_j - λ * sum x_i
    
    The term sum_{i<j, comp} x_i*x_j can be rewritten. For each comparable pair (i,j),
    the contribution is 1 if both are selected.
    
    This is equivalent to: for each node i, the benefit of selecting i is the number
    of already-selected comparable nodes. This is what the greedy does.
    
    For min-cut: We can formulate this as an s-t cut problem where:
    - Each node i has a weight w_i (from Lagrangian)
    - Each comparable edge (i,j) has weight 1
    - We want to maximize: sum of edge weights in S - sum of node weights in S
    
    This can be converted to a min-cut by:
    - Source to i: capacity = max(0, w_i) if w_i > 0
    - i to sink: capacity = max(0, -w_i) if w_i < 0  
    - For each edge (i,j): add undirected edge with capacity 1
    
    Actually, the standard reduction for "maximize cut with node weights" is:
    Minimize: sum of node weights not in S + sum of edge weights crossing the cut
    
    Let me use a different approach. Since the greedy is already quite good,
    let me implement a local search / simulated annealing as the main improvement,
    and keep the min-cut as a secondary method.
    
    For now, let me implement a more sophisticated greedy with local search polish.
    """
    return None  # Placeholder - we'll implement local search instead

def _local_search_polish(family, deadline, rng):
    """
    Improve a family by trying to swap one element for another.
    A swap: remove element i, add element j. Accept if it improves the count.
    """
    family_set = set(family)
    current_count = _count_comparable(family)
    
    # Precompute comparability for all masks with current family
    # For each mask m not in family, compute how many family members it's comparable with
    scores = {}
    for m in range(NUM_MASKS):
        if m in family_set:
            continue
        cnt = 0
        for f in family:
            if (m & f) == m and m != f:
                cnt += 1
            elif (m & f) == f and m != f:
                cnt += 1
        scores[m] = cnt
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try each element in family for removal
        for i in range(len(family)):
            if time.time() > deadline:
                break
            removed = family[i]
            # For each candidate to add
            # The change in count = (new comparable pairs with added) - (old comparable pairs with removed)
            # new comparable pairs = scores[added] - (1 if added comparable with removed else 0)
            # old comparable pairs = number of family members comparable with removed
            #   (excluding removed itself)
            
            # Compute old comparable count for removed
            old_comp = 0
            for j in range(len(family)):
                if j == i:
                    continue
                f = family[j]
                if (removed & f) == removed and removed != f:
                    old_comp += 1
                elif (removed & f) == f and removed != f:
                    old_comp += 1
            
            # Find best candidate to add
            best_gain = -1
            best_add = -1
            for m, sc in scores.items():
                # If we remove 'removed' and add 'm':
                # New pairs involving m: sc (but sc was computed with 'removed' in family)
                # If m is comparable with removed, then sc includes 1 for removed
                # After removing, the comparable count for m becomes sc - (1 if comp(m, removed) else 0)
                # Pairs lost: old_comp
                # Pairs gained: sc - (1 if comp(m, removed) else 0)
                # Net change: sc - (1 if comp(m, removed) else 0) - old_comp
                
                is_comp_with_removed = False
                if (m & removed) == m and m != removed:
                    is_comp_with_removed = True
                elif (m & removed) == removed and m != removed:
                    is_comp_with_removed = True
                
                gain = sc - (1 if is_comp_with_removed else 0) - old_comp
                if gain > best_gain:
                    best_gain = gain
                    best_add = m
            
            if best_gain > 0 and best_add != -1:
                # Perform the swap
                family[i] = best_add
                family_set.remove(removed)
                family_set.add(best_add)
                
                # Update scores
                # For all masks not in family, update their scores
                # When we remove 'removed' and add 'best_add':
                # For each mask m not in {removed, best_add} and not in family:
                #   score[m] changes by:
                #     -1 if m comparable with removed
                #     +1 if m comparable with best_add
                
                for m in list(scores.keys()):
                    if m == removed or m == best_add:
                        continue
                    if m in family_set:
                        continue
                    # Check if m comparable with removed
                    if (m & removed) == m and m != removed:
                        scores[m] -= 1
                    elif (m & removed) == removed and m != removed:
                        scores[m] -= 1
                    # Check if m comparable with best_add
                    if (m & best_add) == m and m != best_add:
                        scores[m] += 1
                    elif (m & best_add) == best_add and m != best_add:
                        scores[m] += 1
                
                # Add best_add to scores (it's now in family, so remove from scores)
                del scores[best_add]
                # removed is no longer in family, but we don't track its score
                
                current_count += best_gain
                improved = True
    
    return family, current_count

def _greedy_construct(seed, deadline):
    """Greedy construction with random restart, returns (count, family)."""
    rng = random.Random(seed)
    
    # Initialize scores to 0
    scores = [0] * NUM_MASKS
    in_family = [False] * NUM_MASKS
    family = []
    
    # First element: pick randomly
    first = rng.choice(ALL_MASKS)
    family.append(first)
    in_family[first] = True
    
    # Update scores for all other masks
    for y in ALL_MASKS:
        if y == first:
            continue
        if (first & y) == first and first != y:
            scores[y] += 1
        elif (first & y) == y and first != y:
            scores[y] += 1
    
    for _ in range(M - 1):
        if time.time() > deadline:
            break
        
        # Find the mask with highest score that's not in family
        best_mask = -1
        best_score = -1
        candidates = []
        for y in ALL_MASKS:
            if not in_family[y]:
                if scores[y] > best_score:
                    best_score = scores[y]
                    best_mask = y
                    candidates = [y]
                elif scores[y] == best_score:
                    candidates.append(y)
        
        if best_mask == -1:
            break
        
        # Pick randomly among top candidates for diversity
        chosen = rng.choice(candidates)
        family.append(chosen)
        in_family[chosen] = True
        
        # Update scores for all other masks
        for y in ALL_MASKS:
            if in_family[y]:
                continue
            if (chosen & y) == chosen and chosen != y:
                scores[y] += 1
            elif (chosen & y) == y and chosen != y:
                scores[y] += 1
    
    if len(family) < M:
        remaining = [m for m in ALL_MASKS if not in_family[m]]
        rng.shuffle(remaining)
        family.extend(remaining[:M - len(family)])
    
    count = _count_comparable(family)
    return count, family

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_count = 0
    best_family = None
    
    # Phase 1: Multiple greedy restarts
    num_restarts = 300
    
    for i in range(num_restarts):
        if time.time() > deadline - 30.0:  # Leave 30s for local search
            break
        
        count, family = _greedy_construct(i, deadline)
        
        if count > best_count:
            best_count = count
            best_family = family
    
    # Phase 2: Local search polish on best family
    if best_family is not None:
        rng = random.Random(12345)
        polish_deadline = deadline - 5.0  # Leave 5s margin
        best_family, best_count = _local_search_polish(best_family, polish_deadline, rng)
    
    # If we didn't find anything (shouldn't happen), fallback to simple construction
    if best_family is None:
        vertices = list(range(6))
        all_sets = []
        for r in range(7):
            for combo in itertools.combinations(vertices, r):
                all_sets.append(list(combo))
        all_sets.sort(key=len)
        best_family = all_sets[:M]
    
    # Convert to lists of ints
    result = []
    for mask in best_family:
        result.append([b for b in range(N) if (mask >> b) & 1])
    
    return result
# EVOLVE-BLOCK-END
