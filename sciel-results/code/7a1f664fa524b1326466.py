import random
import time
import sys
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Combines cyclic (Z_n-regular) construction with proper non-backtracking
    girth verification. Falls back to LW graph if needed.
    """
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s kill

    def check_girth_nb(adj, n, g):
        """
        Check girth using non-backtracking walks from every vertex.
        Returns True if no cycle shorter than g exists.
        """
        for start in range(n):
            for first_neighbor in adj[start]:
                # Non-backtracking walk state: (current_node, prev_node)
                # We start with edge (start, first_neighbor)
                visited = set()
                queue = deque()
                queue.append((first_neighbor, start))
                visited.add((first_neighbor, start))
                
                found_short_cycle = False
                while queue:
                    curr, prev = queue.popleft()
                    # The walk so far is: start -> ... -> prev -> curr
                    # Length of walk = number of edges = depth (number of transitions from initial edge)
                    # Initial edge is depth 1. Each step adds 1.
                    # If we can go curr -> start and that's not backtracking (start != prev),
                    # we have a cycle. The cycle length is the path length from start to curr + 1.
                    # Path length from start to curr: if curr is at distance d from start in the non-backtracking tree
                    # starting with (start, first_neighbor), then d is the number of edges.
                    # Actually, let's track depth explicitly.
                    
                    # Let's restructure to track depth to make the logic clearer.
                    pass
                
                # Let's redo with depth tracking for clarity and correctness
                visited = set()
                queue = deque()
                # State: (node, parent, depth)
                # Start: (first_neighbor, start, 1) -- 1 edge used
                queue.append((first_neighbor, start, 1))
                visited.add((first_neighbor, start))
                
                found_short_cycle = False
                while queue:
                    curr, prev, depth = queue.popleft()
                    
                    # Check if we can close the cycle to 'start'
                    # Edge (curr, start) is valid if start != prev (non-backtracking)
                    if start != prev:
                        # The cycle would be: start -> ... -> curr -> start
                        # Length = depth + 1
                        if depth + 1 < g:
                            found_short_cycle = True
                            break
                    
                    # Expand
                    for nxt in adj[curr]:
                        if nxt == prev:
                            continue  # Non-backtracking
                        if (nxt, curr) not in visited:
                            visited.add((nxt, curr))
                            queue.append((nxt, curr, depth + 1))
                    
                    # Prune: if depth + 1 >= g, we don't need to go deeper
                    # because any cycle closed from a deeper node would be even longer
                    # Actually, if depth+1 >= g, then any cycle formed by adding one more edge
                    # would have length >= g+1, which is fine.
                    # But we should still check if nxt == start?
                    # If nxt == start, cycle length is depth + 2.
                    # If depth + 2 < g, it's a short cycle.
                    # My check above `if start != prev` handles the case where we are AT 'curr'
                    # and look for an edge to 'start'.
                    # Wait, the check `if start != prev` is inside the loop for 'curr'.
                    # This checks if the edge (curr, start) exists and is non-backtracking.
                    # Is it sufficient?
                    # Yes, because any non-backtracking walk from start that returns to start
                    # must end with an edge (curr, start) where curr is the previous node.
                    # The condition `start != prev` ensures non-backtracking at the final step.
                    # The length is `depth` (edges from start to curr) + 1 (edge curr to start).
                    # So `depth + 1` is the cycle length.
                    
                    if found_short_cycle:
                        break
                
                if found_short_cycle:
                    return False
        
        return True

    def make_cyclic_graph(n, offsets):
        """
        Create a cyclic Z_n-regular graph with given offsets.
        For k=4, offsets should be 4 distinct non-zero values modulo n,
        such that no offset is the negative of another (to avoid multiedges).
        """
        edges = []
        edge_set = set()
        
        # Verify offsets validity for simplicity
        # 1. No zero
        if 0 in offsets:
            return None
        # 2. Distinct modulo n
        if len(set(o % n for o in offsets)) < len(offsets):
            return None
        # 3. No pair sums to 0 mod n (which would create multiedges i->j via +a and j->i via +b where b=-a)
        # Actually, if we have offset a and -a mod n, then edge (i, i+a) is the same as (i+a, i) = (i+a, (i+a)-a).
        # In our construction, we add edges (i, (i+off)%n) for all i.
        # If we have offsets a and b where b == -a mod n:
        # Edge from i with offset a: (i, i+a)
        # Edge from i+a with offset b: (i+a, i+a-b) = (i+a, i+a+a) ?? No.
        # Let's trace:
        # Vertex i connects to (i+a)%n.
        # Vertex j connects to (j+b)%n.
        # If we pick j = i+a, then j+b = i+a-b. If b = -a, then j+b = i+a+a = i+2a.
        # This is not i.
        # Wait, the multiedge issue is:
        # Edge (i, i+a) is undirected {i, i+a}.
        # Edge (i+a, i) is undirected {i+a, i}.
        # When do we generate edge (i+a, i)?
        # We generate (v, v+off) for all v.
        # So we generate (i+a, i+a+off).
        # For this to be {i+a, i}, we need i+a+off = i => off = -a.
        # So if offsets contain a and -a mod n, we generate the same undirected edge twice.
        # Thus, we must ensure no offset is the negative of another mod n.
        
        for i in range(len(offsets)):
            for j in range(i+1, len(offsets)):
                if (offsets[i] + offsets[j]) % n == 0:
                    return None
        
        for i in range(n):
            for off in offsets:
                j = (i + off) % n
                if i == j:
                    return None
                key = (min(i, j), max(i, j))
                if key in edge_set:
                    return None
                edge_set.add(key)
                edges.append((i, j))
        
        # Check degree
        degree = [0] * n
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        if any(d != k for d in degree):
            return None
            
        return edges

    def check_girth_from_edges(edges, n, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return check_girth_nb(adj, n, g)

    best_n = float('inf')
    best_edges = None

    # Moore bound for (4,10) is 240 (approx). The problem states lower bound 243.
    # We search for cyclic graphs.
    # For k=4, we need 4 offsets.
    # Common strategy: try small n first.
    # Cyclic graphs often work for n > some threshold.
    
    # Candidate sizes: start from lower bound, go up.
    # But cyclic graphs require n to be large enough to accommodate distinct offsets.
    # For g=10, n needs to be reasonably large.
    # Let's try n from 243 to 400, step 1? Too slow for full search.
    # Let's try specific promising n values or ranges.
    
    # Since the known best is 320, we should focus around there.
    # But we want to minimize n.
    # Let's try n from 243 to 320.
    
    # Optimization: Randomly search for offset sets.
    # For a given n, pick 4 random offsets from 1..n-1.
    
    max_n_search = 350  # Search up to 350
    min_n_search = 243
    
    # We'll iterate n from min to max.
    # For each n, try a limited number of random offset sets.
    # If we find a valid graph, update best.
    
    # To speed up, we can pre-filter offsets.
    # But random selection is usually fast enough if we limit attempts.
    
    for n in range(min_n_search, max_n_search + 1):
        if time.time() > deadline - 5:
            break
        
        # Limit attempts per n
        # If n is close to lower bound, it's harder.
        # If n is larger, it's easier.
        # Let's do 1000 attempts for small n, 100 for larger.
        attempts = 2000 if n < 260 else 500
        
        found = False
        for _ in range(attempts):
            if time.time() > deadline - 2:
                break
                
            # Generate 4 random distinct offsets from 1 to n-1
            # Ensure no pair sums to 0 mod n
            offsets = []
            while len(offsets) < 4:
                o = random.randint(1, n-1)
                if o in offsets:
                    continue
                # Check against existing offsets
                valid = True
                for existing in offsets:
                    if (o + existing) % n == 0:
                        valid = False
                        break
                if valid:
                    offsets.append(o)
            
            edges = make_cyclic_graph(n, offsets)
            if edges is not None:
                if check_girth_from_edges(edges, n, g):
                    if n < best_n:
                        best_n = n
                        best_edges = edges
                    found = True
                    break  # Found a valid graph for this n, move to next n (smaller is better, but we go up)
                    # Actually, we want the SMALLEST n.
                    # Since we iterate n from min to max, the first valid n is the best.
                    # So we can return immediately?
                    # Yes, if we find a valid graph at n, and we are searching upwards,
                    # this is the best possible cyclic graph of this type for this n range.
                    # However, maybe a smaller n exists that we missed due to random search?
                    # We started from min_n_search. If we didn't find it in min_n_search,
                    # it might not exist or we didn't try hard enough.
                    # Given time constraints, finding the first valid n in ascending order is a good strategy.
                    # But we should be careful: if we fail to find a valid graph for n=250,
                    # we move to 251. If 250 had a solution, we missed it.
                    # To mitigate, we can do more attempts for smaller n.
                    # Or, we can stop as soon as we find ANY valid graph, since we're going up.
                    # This guarantees we return the smallest n for which we found a solution.
                    return n, edges

    # If no cyclic graph found in the range, fall back to LW graph
    # The LW graph construction from the seed program
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
