import time
import sys

# Increase recursion limit if needed, but iterative or careful recursion is better
sys.setrecursionlimit(10000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

# Build adjacency list
adj = [[] for _ in range(P['n'])]
for u, v in P['edges']:
    adj[u].append(v)
    adj[v].append(u)

# DSU class
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    
    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px == py:
            return False # Cycle detected
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin for 60s limit
    
    # Try increasing number of colors
    for k in range(3, 10):
        if time.time() > deadline:
            break
            
        colors = [-1] * P['n']
        # DSU for each pair of colors
        # We only need DSUs for pairs that have been assigned at least one vertex
        # To simplify, we can maintain a dict of (c1, c2) -> DSU
        # But DSU needs to be persistent and support undo? 
        # Backtracking with DSU is tricky because we need to undo unions.
        # Alternative: Just check for cycles at the end of a complete assignment?
        # With k=3 or 4, the branching factor is small. 
        # However, checking at the end might be too late for pruning.
        
        # Let's try a simpler approach: 
        # Assign colors one by one.
        # For pruning, we can check if the current partial assignment violates the "forest" property for any pair.
        # A partial assignment violates the forest property for pair (c1, c2) if there is a cycle in the induced subgraph of assigned vertices.
        # We can maintain a DSU for each pair (c1, c2) and undo changes.
        # Or, since n=50, we can just do a DFS/BFS cycle detection on the induced subgraph of assigned vertices for each pair whenever we make a move.
        # This is O(E) per check. 50 moves * 50 edges * k^2 pairs. 50*50*9 = 22500 ops per step? No, k is small.
        # Actually, checking cycle in induced subgraph of assigned vertices:
        # We only care about edges between assigned vertices.
        # Let's implement a function `has_cycle_in_induced_subgraph(assigned_colors, c1, c2)`
        # This might be slow if called at every step.
        
        # Better: Use the DSU with undo.
        # We can store the changes made to DSU and revert them.
        
        # Let's try the "check at end" first, with some pruning based on adjacent colors.
        # If that's too slow, we'll optimize.
        
        # Vertex ordering: 
        # A good heuristic is to order vertices by the number of uncolored neighbors (dynamic) or static degree.
        # All degrees are 4. 
        # Let's just use index order or a static order.
        
        # Let's implement a recursive backtracker.
        
        # To speed up, we can pre-compute the list of pairs.
        
        def is_valid_partial(assigned_colors, v, c):
            # Check adjacent constraint
            for u in adj[v]:
                if assigned_colors[u] != -1 and assigned_colors[u] == c:
                    return False
            
            # Check bicolored cycle constraint
            # For each color c' != c, check if adding v (color c) creates a cycle in pair (c, c')
            # The cycle would involve v and some neighbors of v with color c'.
            # Specifically, if two neighbors u1, u2 of v with color c' are already connected in the induced subgraph of colors {c, c'}, then adding v creates a cycle.
            # Wait, the induced subgraph includes vertices of color c and c'.
            # If we add v (color c), we connect it to all its neighbors of color c'.
            # If any two such neighbors are connected via a path using only vertices of color c and c', a cycle is formed.
            # We need to know connectivity in the current partial assignment for pair (c, c').
            
            # This requires maintaining DSU state.
            pass

        # Let's use a global DSU state that is updated and reverted.
        # This is complex. 
        # Given the small size, maybe a simpler check is enough.
        # Let's try to find a solution with k=4 or 5 using a randomized greedy approach first?
        # Or just run the backtracker with the "check at end" and hope it's fast enough for k=3,4.
        
        # Let's implement the backtracker with a helper to check if a full assignment is valid.
        # And prune only on adjacent colors.
        # If k=3, 3^50 is huge. We need pruning.
        
        # Let's use the DSU with undo.
        # We'll maintain a list of DSUs for all pairs (0,1), (0,2), ..., (k-2, k-1).
        # When we assign v color c, we update DSU for (c, c') for all c' in 0..k-1, c' != c.
        # We record the changes (unions) to revert later.
        
        # This is doable.
        
        # Let's write the solver.
        
        def solve_k(k):
            colors = [-1] * P['n']
            # DSUs for each pair
            # Index pairs by (c1, c2) with c1 < c2
            dsus = {}
            for c1 in range(k):
                for c2 in range(c1+1, k):
                    dsus[(c1, c2)] = DSU(P['n'])
            
            # History for undo
            # We can store the state of DSU parents? No, that's expensive.
            # Store the list of unions performed.
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                
                if idx == P['n']:
                    return True
                
                v = idx
                # Try colors
                # Heuristic: try colors that appear in neighbors first?
                # Or just 0..k-1
                for c in range(k):
                    # Check adjacent
                    valid = True
                    for u in adj[v]:
                        if colors[u] == c:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Check bicolored cycles
                    # For each c' != c, check if adding v to pair (c, c') creates a cycle
                    # We need to find neighbors of v with color c'.
                    # If two such neighbors are connected in DSU[(c, c')], cycle.
                    
                    changes = []
                    ok = True
                    for c_prime in range(k):
                        if c_prime == c:
                            continue
                        key = (min(c, c_prime), max(c, c_prime))
                        dsu = dsus[key]
                        
                        # Find neighbors of v with color c_prime
                        neighbors_cp = [u for u in adj[v] if colors[u] == c_prime]
                        
                        # If there are >= 2 neighbors, check if they are connected
                        # Actually, we just need to union v with each neighbor.
                        # If union fails, cycle.
                        # But we need to revert if we fail later.
                        
                        # Let's simulate the unions.
                        # We can just check connectivity before unioning.
                        # If any two neighbors are connected, cycle.
                        # Also, if a neighbor is connected to another neighbor via a path that goes through v? No, v is new.
                        # So we just check if any two neighbors of v with color c_prime are in the same component.
                        
                        if len(neighbors_cp) > 1:
                            # Check if all neighbors are in distinct components?
                            # No, if any two are in the same component, adding v connects them, forming a cycle.
                            # So we need all neighbors to be in distinct components.
                            roots = set()
                            for u in neighbors_cp:
                                r = dsu.find(u)
                                if r in roots:
                                    ok = False
                                    break
                                roots.add(r)
                            if not ok:
                                break
                        
                        # If ok so far, we need to actually perform the unions to update DSU for deeper recursion.
                        # We will record the unions.
                        for u in neighbors_cp:
                            if dsu.union(v, u):
                                changes.append((key, v, u))
                            else:
                                # This shouldn't happen if we checked roots above, 
                                # unless v was already in the set? No, v is new.
                                # Wait, if v is already unioned with some u1, and we try to union with u2,
                                # and u1, u2 were in different components, it's fine.
                                # But if u1, u2 were in the same component, we would have caught it above.
                                pass
                    
                    if not ok:
                        continue
                    
                    # Assign
                    colors[v] = c
                    
                    if backtrack(idx + 1):
                        return True
                    
                    # Undo
                    colors[v] = -1
                    for key, u1, u2 in reversed(changes):
                        # Undo union
                        # DSU doesn't have easy undo.
                        # We need to save parent/rank or use a different structure.
                        # Given the complexity, let's avoid DSU undo.
                        pass
            
            return False
        
        # The DSU undo is the problem.
        # Let's just do the "check at end" but with a stronger pruning:
        # We can check if the partial assignment already contains a cycle for any pair.
        # We can do this by maintaining the induced subgraph edges for each pair.
        # But that's essentially the same as DSU.
        
        # Alternative: Since n=50 is small, we can use a SAT solver? No, standard lib only.
        
        # Let's try a different approach:
        # Randomized greedy with backtracking.
        # Or just try a few specific colorings.
        
        # Let's look at the structure again.
        # It looks like a grid.
        # Maybe a periodic coloring works?
        # If we use 4 colors in a 2x2 pattern:
        # 0 1 0 1 ...
        # 2 3 2 3 ...
        # 0 1 0 1 ...
        # 2 3 2 3 ...
        # Let's test this pattern on the graph.
        
        def check_pattern(pattern_func):
            colors = [pattern_func(i) for i in range(P['n'])]
            # Check adjacent
            for u, v in P['edges']:
                if colors[u] == colors[v]:
                    return None
            # Check bicolored cycles
            # For each pair (c1, c2), check if induced subgraph is forest
            # We can use DSU on the induced subgraph
            used_colors = set(colors)
            for c1 in used_colors:
                for c2 in used_colors:
                    if c1 >= c2:
                        continue
                    # Build induced subgraph
                    # Use DSU
                    dsu = DSU(P['n'])
                    for u, v in P['edges']:
                        cu = colors[u]
                        cv = colors[v]
                        if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                            if not dsu.union(u, v):
                                return None # Cycle found
            return colors
        
        # Try 4-color pattern
        def pattern_4(i):
            # 2x2 grid
            r = i // 7 # Assuming 7 columns?
            c = i % 7
            return (r % 2) * 2 + (c % 2)
        
        res = check_pattern(pattern_4)
        if res:
            return res
            
        # Try other patterns or backtracking
        # If pattern fails, we need a robust solver.
        
        # Let's implement a simple backtracker that checks validity at the end of each complete assignment.
        # To make it fast, we use a good variable ordering and value ordering.
        # And we prune if adjacent colors are the same.
        # We hope that for k=3 or 4, the solution space is small enough.
        
        # Actually, let's try k=4 with a simple randomized restarts.
        
        import random
        random.seed(42)
        
        for attempt in range(1000):
            if time.time() > deadline:
                break
            colors = [-1] * P['n']
            # Randomly assign some vertices?
            # Or just backtracking with random value order.
            
            def backtrack_random(idx):
                if time.time() > deadline:
                    return False
                if idx == P['n']:
                    # Final check for bicolored cycles
                    used_colors = set(colors)
                    for c1 in used_colors:
                        for c2 in used_colors:
                            if c1 >= c2:
                                continue
                            dsu = DSU(P['n'])
                            for u, v in P['edges']:
                                cu = colors[u]
                                cv = colors[v]
                                if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                                    if not dsu.union(u, v):
                                        return False
                    return True
                
                v = idx
                vals = list(range(k))
                random.shuffle(vals)
                
                for c in vals:
                    # Adjacent check
                    valid = True
                    for u in adj[v]:
                        if colors[u] == c:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    colors[v] = c
                    if backtrack_random(idx + 1):
                        return True
                    colors[v] = -1
                return False
            
            if backtrack_random(0):
                return colors
        
        # If all fails, return a trivial valid solution (50 colors)
        return list(range(P['n']))

# The above might be too slow if k=3 is infeasible and k=4 is hard.
# But the pattern check is fast.
# Let's refine the pattern check to cover more possibilities.
# The graph is 4-regular.
# Let's just return the pattern if it works.
# If not, the backtracker will try.

# To ensure we don't timeout, we keep the deadline guard.
