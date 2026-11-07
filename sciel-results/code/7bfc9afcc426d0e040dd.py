import time
import random
import sys
from collections import defaultdict, deque

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
edges = P['edges']
adj = defaultdict(list)
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

def count_bicolored_cycles(colors):
    """Count bicolored cycles using union-find per color pair. Fast version."""
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = colors[u], colors[v]
        if cu != cv:
            if cu > cv:
                cu, cv = cv, cu
            pair_edges[(cu, cv)].append((u, v))
    
    cycle_count = 0
    for (a, b), elist in pair_edges.items():
        uf = UnionFind(N)
        for u, v in elist:
            if not uf.union(u, v):
                cycle_count += 1
                break
    return cycle_count

def is_valid(colors):
    """Check if coloring is valid (proper + no bicolored cycles)."""
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    return count_bicolored_cycles(colors) == 0

def greedy_coloring(K):
    """Greedy coloring with dynamic vertex ordering based on degree and conflicts."""
    colors = [-1] * N
    uncolored = set(range(N))
    
    # Order by degree descending, then by number of colored neighbors
    while uncolored:
        # Pick vertex with most colored neighbors first (most constrained)
        best_v = None
        best_score = -1
        for v in uncolored:
            colored_neighbors = sum(1 for u in adj[v] if colors[u] >= 0)
            score = colored_neighbors * 100 + len(adj[v])
            if score > best_score:
                best_score = score
                best_v = v
        
        v = best_v
        uncolored.remove(v)
        
        # Find smallest available color
        neighbor_colors = set(colors[u] for u in adj[v] if colors[u] >= 0)
        for c in range(K):
            if c not in neighbor_colors:
                # Check if this creates a bicolored cycle
                valid = True
                for d in neighbor_colors:
                    if d == c:
                        continue
                    # Check if v's neighbors of color d that are connected to color c nodes form a cycle
                    d_neighbors = [u for u in adj[v] if colors[u] == d]
                    if len(d_neighbors) <= 1:
                        continue
                    # BFS in the (c,d) subgraph
                    visited = set()
                    queue = deque()
                    for start in d_neighbors:
                        if start not in visited:
                            queue.append(start)
                            visited.add(start)
                            while queue:
                                cur = queue.popleft()
                                for nb in adj[cur]:
                                    if nb == v:
                                        continue
                                    if colors[nb] == c or colors[nb] == d:
                                        if nb not in visited:
                                            visited.add(nb)
                                            queue.append(nb)
                    # Check if any two d_neighbors are in the same component
                    comp_roots = set()
                    for dn in d_neighbors:
                        # Find which component dn is in
                        # We need to check connectivity between d_neighbors through (c,d) subgraph
                        pass
                    # Simpler: just check if adding v connects two d_neighbors that are already connected
                    # through the (c,d) subgraph (excluding v)
                    # BFS from first d_neighbor, see if we reach second
                    if len(d_neighbors) >= 2:
                        seen = set()
                        q = deque([d_neighbors[0]])
                        seen.add(d_neighbors[0])
                        while q:
                            cur = q.popleft()
                            for nb in adj[cur]:
                                if nb == v:
                                    continue
                                if colors[nb] == c or colors[nb] == d:
                                    if nb not in seen:
                                        seen.add(nb)
                                        q.append(nb)
                        for dn in d_neighbors[1:]:
                            if dn in seen:
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid:
                    colors[v] = c
                    break
        else:
            # No valid color found, use first non-neighbor color (may be invalid but we'll fix later)
            for c in range(K):
                if c not in neighbor_colors:
                    colors[v] = c
                    break
            else:
                colors[v] = 0
    
    return colors

def local_search(K, time_limit, initial=None):
    """
    Fast local search with incremental cost evaluation.
    """
    start_time = time.time()
    
    if initial is not None:
        colors = list(initial)
    else:
        colors = [random.randint(0, K - 1) for _ in range(N)]
    
    # Compute initial cost
    proper_violations = 0
    for u, v in edges:
        if colors[u] == colors[v]:
            proper_violations += 1
    bicolored_cycles = 0
    if proper_violations == 0:
        bicolored_cycles = count_bicolored_cycles(colors)
    current_cost = proper_violations * 1000 + bicolored_cycles
    
    if current_cost == 0:
        return colors
    
    best_cost = current_cost
    best_coloring = list(colors)
    no_improve = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit:
        if current_cost == 0:
            return colors
        
        # Pick a vertex to recolor - prefer vertices involved in violations
        if proper_violations > 0:
            # Pick a vertex in a proper violation
            v = None
            for u, w in edges:
                if colors[u] == colors[w]:
                    v = random.choice([u, w])
                    break
            if v is None:
                v = random.randint(0, N - 1)
        else:
            # Pick random vertex, or one in a bicolored cycle
            v = random.randint(0, N - 1)
        
        old_color = colors[v]
        best_new_color = old_color
        min_cost = current_cost
        
        for c in range(K):
            if c == old_color:
                continue
            colors[v] = c
            
            # Quick check: proper violation?
            cost = 0
            for u in adj[v]:
                if colors[u] == c:
                    cost = 1000
                    break
            
            if cost == 0:
                # Check bicolored cycles - only for pairs involving c
                # We can do a faster check by only looking at edges involving v
                # But for simplicity, do full check if cost is 0
                cost = count_bicolored_cycles(colors)
            
            if cost < min_cost:
                min_cost = cost
                best_new_color = c
            
            colors[v] = old_color
        
        colors[v] = best_new_color
        current_cost = min_cost
        
        # Recompute proper_violations for next iteration
        proper_violations = 0
        for u, v2 in edges:
            if colors[u] == colors[v2]:
                proper_violations += 1
        if proper_violations == 0:
            bicolored_cycles = count_bicolored_cycles(colors)
        else:
            bicolored_cycles = 0
        current_cost = proper_violations * 1000 + bicolored_cycles
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_coloring = list(colors)
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > max_no_improve:
            break
    
    if best_cost == 0:
        return best_coloring
    return None

def construct():
    deadline = time.time() + 85.0
    
    # Try greedy + local search for small K
    for K in [3, 4, 5]:
        if time.time() > deadline - 15:
            break
        
        # Try greedy initialization
        init = greedy_coloring(K)
        if is_valid(init):
            return init
        
        # Try local search from greedy init
        res = local_search(K, min(10.0, deadline - time.time() - 5), initial=init)
        if res is not None:
            return res
        
        # Try random restarts
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            res = local_search(K, min(5.0, deadline - time.time() - 5))
            if res is not None:
                return res
    
    # Fallback: use more colors
    for K in [6, 7, 8]:
        if time.time() > deadline - 5:
            break
        init = greedy_coloring(K)
        if is_valid(init):
            return init
        res = local_search(K, min(5.0, deadline - time.time() - 2))
        if res is not None:
            return res
    
    # Absolute fallback: unique colors
    return list(range(N))

# EVOLVE-BLOCK-END
