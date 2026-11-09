"""Stochastic local search (Simulated Annealing) for k-regular graphs of large girth."""
import random
import math
import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Uses Simulated Annealing with 2-switches to maximize girth.
    
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j).
    """
    # Moore bound: minimum possible vertices
    if g % 2 == 0:
        moore = 2 * sum((k - 1) ** i for i in range(g // 2))
    else:
        moore = 1 + k * sum((k - 1) ** i for i in range((g - 3) // 2 + 1))
    
    # Start with a reasonable number of vertices above the Moore bound
    n = max(moore, k * (g - 1) // 2 + 2)
    
    def generate_k_regular(n, k):
        """Generate a random simple k-regular graph using configuration model."""
        if k == 0:
            return []
        if n * k % 2 != 0:
            raise ValueError("n*k must be even")
        
        max_attempts = 500
        for attempt in range(max_attempts):
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            
            random.shuffle(stubs)
            
            edges = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                edge = (min(u, v), max(u, v))
                if edge in edges:
                    valid = False
                    break
                edges.add(edge)
            
            if valid:
                return list(edges)
        
        # Fallback: use a different randomization strategy
        # Try multiple times with different seeds
        for seed in range(100):
            random.seed(seed)
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            edges = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                edge = (min(u, v), max(u, v))
                if edge in edges:
                    valid = False
                    break
                edges.add(edge)
            if valid:
                return list(edges)
        
        # Last resort: return a simple cycle (not k-regular, but search can fix)
        edges = []
        for i in range(n):
            edges.append((i, (i + 1) % n))
        return edges
    
    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj
    
    def count_short_cycles_fast(n, adj, max_len):
        """Count distinct cycles of length < max_len using non-backtracking walks."""
        # Use a set to track unique cycles (by their sorted vertex sequence)
        # For efficiency, we'll count walks and divide by appropriate factors
        count = 0
        for start in range(n):
            # BFS with state (vertex, prev) to avoid backtracking
            visited = set()
            queue = deque()
            for neighbor in adj[start]:
                queue.append((neighbor, start, 1))
                visited.add((neighbor, start))
            while queue:
                u, prev, dist = queue.popleft()
                if dist + 1 >= max_len:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    if v == start and dist + 1 >= 3:
                        count += 1
                        continue
                    state = (v, u)
                    if state not in visited:
                        visited.add(state)
                        queue.append((v, u, dist + 1))
        # Each cycle of length L is counted 2L times (L starting points * 2 directions)
        # But we want to count each cycle once. For our purposes, the raw count
        # is a good proxy for the number of short cycles.
        return count
    
    def sa_search(n, k, g, time_limit=110):
        """Run simulated annealing to maximize girth."""
        edges = generate_k_regular(n, k)
        adj = build_adj(n, edges)
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        
        start_time = time.time()
        T = 1.0
        T_min = 0.001
        cooling = 0.999
        
        best_edges = edges[:]
        best_count = count_short_cycles_fast(n, adj, g)
        current_count = best_count
        
        # Precompute adjacency for fast lookup
        adj_set = set()
        for u, v in edges:
            adj_set.add((min(u, v), max(u, v)))
        
        while time.time() - start_time < time_limit:
            # Choose a random 2-switch
            m = len(edges)
            if m < 2:
                break
            
            i = random.randrange(m)
            j = random.randrange(m)
            if i == j:
                continue
            
            a, b = edges[i]
            c, d = edges[j]
            
            # Check if the four vertices are distinct
            if a == c or a == d or b == c or b == d:
                continue
            
            # Try both reconnection options
            new_e1 = (min(a, c), max(a, c))
            new_e2 = (min(b, d), max(b, d))
            new_e3 = (min(a, d), max(a, d))
            new_e4 = (min(b, c), max(b, c))
            
            valid_options = []
            if new_e1 not in adj_set and new_e2 not in adj_set and new_e1 != new_e2:
                valid_options.append((new_e1, new_e2))
            if new_e3 not in adj_set and new_e4 not in adj_set and new_e3 != new_e4:
                valid_options.append((new_e3, new_e4))
            
            if not valid_options:
                continue
            
            new_e1, new_e2 = random.choice(valid_options)
            
            # Apply the switch
            edges[i] = new_e1
            edges[j] = new_e2
            adj_set.discard((min(a, b), max(a, b)))
            adj_set.discard((min(c, d), max(c, d)))
            adj_set.add(new_e1)
            adj_set.add(new_e2)
            
            # Rebuild adjacency
            new_adj = build_adj(n, edges)
            
            # Evaluate
            new_count = count_short_cycles_fast(n, new_adj, g)
            
            if new_count < current_count:
                # Improvement - always accept
                current_count = new_count
                if new_count < best_count:
                    best_count = new_count
                    best_edges = edges[:]
            elif new_count == current_count:
                # Neutral - accept with some probability
                if random.random() < 0.5:
                    current_count = new_count
            else:
                # Worse - accept with probability exp(-delta/T)
                delta = new_count - current_count
                if random.random() < math.exp(-delta / max(T, 1e-10)):
                    current_count = new_count
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best_edges, best_count
    
    # Run SA
    best_edges, best_count = sa_search(n, k, g, time_limit=110)
    
    # Verify the result
    final_adj = build_adj(n, best_edges)
    
    # Check degree
    for i in range(n):
        if len(final_adj[i]) != k:
            # Try to fix by regenerating
            edges = generate_k_regular(n, k)
            best_edges, best_count = sa_search(n, k, g, time_limit=100)
            final_adj = build_adj(n, best_edges)
            break
    
    # Check for short cycles
    if count_short_cycles_fast(n, final_adj, g) > 0:
        # The result is invalid; fallback to a known construction
        # Use the Lazebnik-Ustimenko-Woldar style construction as fallback
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
        index, order, fallback_edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    fallback_edges.append((index[vertex], index[other]))
        
        return len(order), fallback_edges
    
    return n, best_edges
