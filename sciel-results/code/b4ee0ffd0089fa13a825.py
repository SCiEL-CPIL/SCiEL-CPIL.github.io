import time
import random
import sys
from collections import deque

def _check_girth_nonbacktracking(adj, n, target_girth):
    """Check girth >= target_girth using non-backtracking walks from every vertex."""
    for start in range(n):
        dist = [float('inf')] * n
        prev = [-1] * n
        dist[start] = 0
        queue = deque([(start, -1)])
        
        while queue:
            v, p = queue.popleft()
            for u in adj[v]:
                if u == p:
                    continue
                new_dist = dist[v] + 1
                if new_dist < dist[u]:
                    dist[u] = new_dist
                    prev[u] = v
                    queue.append((u, v))
                elif new_dist == dist[u] and u != prev[v]:
                    cycle_len = new_dist + 1
                    if cycle_len < target_girth:
                        return False
    
    for v in range(n):
        for u in adj[v]:
            if v < u:
                if dist[v] + dist[u] + 1 < target_girth:
                    if prev[v] != u and prev[u] != v:
                        return False
    return True


def _bfs_girth(adj, n):
    """Compute exact girth using BFS from each vertex."""
    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            v = queue.popleft()
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = dist[v] + 1
                    parent[u] = v
                    queue.append(u)
                elif u != parent[v] and v != parent[u]:
                    cycle_len = dist[v] + dist[u] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    if min_girth <= 2:
                        return min_girth
    return min_girth


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction (fallback)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    rule_list = [rule[c] for c in range(2, d + 1)]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for idx, c in enumerate(range(2, d + 1)):
                a, b = rule_list[idx]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    i = 0
    while i < len(order):
        vertex = order[i]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        i += 1
    return len(order), edges


def _cayley_symmetric_group_construct(k, g, deadline):
    """
    Construct a k-regular graph with girth >= g using a Cayley graph of a
    finite symmetric group with involutory generators.
    
    Strategy: Work in S_m for some m. Pick k involutions (products of disjoint
    transpositions) that generate a large subgroup. The Cayley graph has one vertex
    per group element. We need the girth to be at least g.
    
    For k=3, g=16: We'll try S_m for various m, picking 3 involutory generators
    that are "sufficiently far apart" in terms of word length.
    
    The key insight: if the generators have no short relations, the Cayley graph
    will have large girth. We verify girth explicitly.
    """
    if k != 3:
        return None
    
    # For girth 16, we need a large group. Let's try S_m for m around 10-15.
    # The order of S_m is m!, so S_10 has ~3.6M elements, S_9 has ~362K.
    # We want n as small as possible but >= Moore bound (512 for (3,16)).
    
    # Actually, let's think differently. The LUW construction already gives
    # a valid graph. The Cayley graph approach should give a SMALLER graph.
    # 
    # For a 3-regular Cayley graph of S_m, n = m!.
    # S_6 = 720, S_7 = 5040, S_8 = 40320...
    # S_6 = 720 vertices is close to the known cage size of 936.
    # But we need girth >= 16, which requires careful generator selection.
    
    # Let's try S_6 with carefully chosen involutions.
    # An involution in S_6 can be:
    # - A single transposition: (ab)
    # - Product of 2 disjoint transpositions: (ab)(cd)
    # - Product of 3 disjoint transpositions: (ab)(cd)(ef)
    
    # For high girth, we want generators that are "far" from each other.
    # Let's try products of 3 disjoint transpositions (maximal involutions).
    
    # S_6 has 720 elements. Let's enumerate them and build the Cayley graph.
    
    m = 6
    # Generate all permutations of S_m
    import itertools
    
    perms = list(itertools.permutations(range(m)))
    n = len(perms)  # 720 for m=6
    
    if n < 512:
        return None
    
    # Index permutations
    perm_index = {p: i for i, p in enumerate(perms)}
    
    # Define 3 involutory generators (products of 3 disjoint transpositions)
    # Generator 1: (0,1)(2,3)(4,5) -> swap pairs
    # In one-line notation: p[i] = where i goes
    # (0,1) means 0->1, 1->0; (2,3) means 2->3, 3->2; (4,5) means 4->5, 5->4
    # So the permutation is: 0->1, 1->0, 2->3, 3->2, 4->5, 5->4
    # One-line: [1, 0, 3, 2, 5, 4]
    
    # Generator 2: A different product of 3 disjoint transpositions
    # (0,2)(1,4)(3,5): 0->2, 2->0, 1->4, 4->1, 3->5, 5->3
    # One-line: [2, 4, 0, 5, 1, 3]
    
    # Generator 3: Another
    # (0,3)(1,5)(2,4): 0->3, 3->0, 1->5, 5->1, 2->4, 4->2
    # One-line: [3, 5, 4, 0, 2, 1]
    
    gens = [
        (1, 0, 3, 2, 5, 4),  # (0,1)(2,3)(4,5)
        (2, 4, 0, 5, 1, 3),  # (0,2)(1,4)(3,5)
        (3, 5, 4, 0, 2, 1),  # (0,3)(1,5)(2,4)
    ]
    
    # Verify each generator is an involution (g^2 = identity)
    for g in gens:
        # Compose g with itself
        composed = tuple(g[g[i]] for i in range(m))
        if composed != tuple(range(m)):
            return None  # Not an involution, try different generators
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for i, p in enumerate(perms):
        for g in gens:
            # Compose g * p (apply p first, then g)
            # (g*p)[i] = g[p[i]]
            composed = tuple(g[p[j]] for j in range(m))
            j = perm_index.get(composed)
            if j is not None and j != i:
                adj[i].append(j)
    
    # Verify 3-regular
    for i in range(n):
        if len(adj[i]) != k:
            return None
    
    # Check for simple graph (no multi-edges)
    for i in range(n):
        if len(set(adj[i])) != len(adj[i]):
            return None
    
    # Check girth
    girth = _bfs_girth(adj, n)
    if girth >= g:
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    return None


def _cayley_symmetric_group_search(k, g, deadline):
    """
    Search for good involutory generators in S_m to maximize girth.
    """
    if k != 3:
        return None
    
    import itertools
    
    m = 6
    perms = list(itertools.permutations(range(m)))
    n = len(perms)
    
    if n < 512:
        return None
    
    perm_index = {p: i for i, p in enumerate(perms)}
    
    # Generate all involutions that are products of 3 disjoint transpositions
    # These are the "maximal" involutions in S_6
    # A product of 3 disjoint transpositions in S_6
    # We need to partition {0,1,2,3,4,5} into 3 pairs
    
    # All ways to partition 6 elements into 3 unordered pairs:
    # (0,1)(2,3)(4,5), (0,1)(2,4)(3,5), (0,1)(2,5)(3,4),
    # (0,2)(1,3)(4,5), (0,2)(1,4)(3,5), (0,2)(1,5)(3,4),
    # (0,3)(1,2)(4,5), (0,3)(1,4)(2,5), (0,3)(1,5)(2,4),
    # (0,4)(1,2)(3,5), (0,4)(1,3)(2,5), (0,4)(1,5)(2,3),
    # (0,5)(1,2)(3,4), (0,5)(1,3)(2,4), (0,5)(1,4)(2,3)
    # Total: 15 such involutions
    
    # Let's generate them systematically
    def make_involution(pairs):
        """Create a permutation from a list of 3 disjoint transpositions."""
        p = list(range(m))
        for a, b in pairs:
            p[a], p[b] = b, a
        return tuple(p)
    
    # Generate all partitions of {0,1,2,3,4,5} into 3 pairs
    all_invols = []
    used = set()
    for i in range(m):
        for j in range(i+1, m):
            for k in range(j+1, m):
                for l in range(k+1, m):
                    # Check if {i,j,k,l} uses all 6 elements
                    if len({i,j,k,l}) == 4:
                        # Remaining two elements
                        remaining = [x for x in range(m) if x not in {i,j,k,l}]
                        if len(remaining) == 2:
                            a, b = remaining
                            pairs = [(i,j), (k,l), (a,b)]
                            # Sort pairs for canonical form
                            pairs = sorted(pairs)
                            key = tuple(tuple(p) for p in pairs)
                            if key not in used:
                                used.add(key)
                                all_invols.append(make_involution(pairs))
    
    # We have 15 involutions. Pick 3 of them: C(15,3) = 455 combinations
    # For each, check if they generate S_6 and what the girth is.
    
    # Precompute composition table for speed
    # comp[a][b] = index of perm a composed with perm b
    # Actually, let's just compute on the fly but cache
    
    # For efficiency, let's precompute the action of each involution
    # on each permutation
    
    inv_actions = []
    for inv in all_invols:
        action = []
        for p in perms:
            composed = tuple(inv[p[j]] for j in range(m))
            action.append(perm_index[composed])
        inv_actions.append(action)
    
    # Now search over triples of involutions
    best_result = None
    best_girth = 0
    
    num_invols = len(all_invols)
    if num_invols < 3:
        return None
    
    for a_idx in range(num_invols):
        if time.time() > deadline:
            break
        for b_idx in range(a_idx + 1, num_invols):
            if time.time() > deadline:
                break
            for c_idx in range(b_idx + 1, num_invols):
                if time.time() > deadline:
                    break
                
                # Check if these 3 involutions generate S_6
                # Quick check: the subgroup generated should have order 720
                # For now, just build the Cayley graph and check its size
                # and girth
                
                action_a = inv_actions[a_idx]
                action_b = inv_actions[b_idx]
                action_c = inv_actions[c_idx]
                
                # Build adjacency from vertex 0 (identity)
                # BFS to find connected component
                visited = [False] * n
                visited[0] = True
                queue = deque([0])
                component_size = 1
                
                while queue:
                    v = queue.popleft()
                    for action in (action_a, action_b, action_c):
                        u = action[v]
                        if not visited[u]:
                            visited[u] = True
                            component_size += 1
                            queue.append(u)
                
                if component_size < n:
                    continue  # Doesn't generate the full group
                
                # Build full adjacency
                adj = [[] for _ in range(n)]
                for i in range(n):
                    adj[i].append(action_a[i])
                    adj[i].append(action_b[i])
                    adj[i].append(action_c[i])
                
                # Verify 3-regular and simple
                valid = True
                for i in range(n):
                    if len(adj[i]) != 3:
                        valid = False
                        break
                    if len(set(adj[i])) != 3:
                        valid = False
                        break
                    if i in adj[i]:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check girth
                girth = _bfs_girth(adj, n)
                if girth > best_girth:
                    best_girth = girth
                    if girth >= g:
                        edges = []
                        for i in range(n):
                            for j in adj[i]:
                                if i < j:
                                    edges.append((i, j))
                        best_result = (n, edges)
                        return best_result
    
    return best_result


def _construct_cage_3_16():
    """
    Construct a 3-regular graph with girth >= 16.
    """
    deadline = time.time() + 110.0
    
    # Try Cayley graph construction first
    result = _cayley_symmetric_group_search(3, 16, deadline)
    if result is not None:
        return result
    
    # Fallback: random construction
    n = 936
    half = n // 2
    
    for attempt in range(5000):
        if time.time() > deadline:
            break
        
        stubs = []
        for j in range(half):
            stubs.extend([j, j, j])
        random.shuffle(stubs)
        
        left_adj = [set() for _ in range(half)]
        valid = True
        
        for i in range(half):
            for offset in range(3):
                j = stubs[i * 3 + offset]
                left_adj[i].add(j)
            if len(left_adj[i]) != 3:
                valid = False
                break
        
        if not valid:
            continue
        
        right_degree = [0] * half
        for i in range(half):
            for j in left_adj[i]:
                right_degree[j] += 1
        
        if any(d != 3 for d in right_degree):
            continue
        
        adj = [[] for _ in range(n)]
        for i in range(half):
            for j in left_adj[i]:
                adj[i].append(half + j)
                adj[half + j].append(i)
        
        girth = _bfs_girth(adj, n)
        if girth >= 16:
            edges = []
            for i in range(half):
                for j in left_adj[i]:
                    edges.append((i, half + j))
            return n, edges
    
    return None


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130.0
    
    if k == 3 and g == 16:
        result = _construct_cage_3_16()
        if result is not None:
            return result
        return _luw_construct(k, g)
    
    return _luw_construct(k, g)
