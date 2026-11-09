"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin

    n = P['n']
    edges = P['edges']

    # Build adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    class UndoableUnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
            self.log = []

        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x

        def union(self, x, y):
            px = self.find(x)
            py = self.find(y)
            if px == py:
                return False
            
            # Log state for undo
            # We need to restore parent[py] and potentially rank[px]
            self.log.append((py, px, self.parent[py], self.rank[px]))
            
            if self.rank[px] < self.rank[py]:
                px, py = py, px
                # Note: if we swapped, the log entry is still valid for the final state
                # But we need to be careful. 
                # Actually, simpler log:
                # Store: (child_root, parent_root, old_parent_of_child, old_rank_of_parent, old_rank_of_child)
                # Let's redo log structure to be safer.
                self.log.pop() # Remove the one we just added, we'll add a better one
                self.log.append((py, px, self.parent[py], self.rank[px], self.rank[py]))
                
                # Perform union with swap
                self.parent[py] = px
                if self.rank[px] == self.rank[py]:
                    self.rank[px] += 1
            else:
                self.parent[py] = px
                if self.rank[px] == self.rank[py]:
                    self.rank[px] += 1
            return True

        def undo(self):
            if not self.log:
                return
            py, px, old_parent_py, old_rank_px, old_rank_py = self.log.pop()
            self.parent[py] = old_parent_py
            self.rank[px] = old_rank_px
            self.rank[py] = old_rank_py

    def solve(max_colors, timeout=10.0):
        """
        Backtracking search with MRV and Undoable DSUs.
        """
        end_time = start_time + timeout
        coloring = [-1] * n
        dsus = {} # key: (c1, c2) -> UndoableUnionFind
        uncolored = set(range(n))
        
        # For MRV, we need to quickly find the vertex with fewest options.
        # Since n=50, we can just scan all uncolored vertices each time.
        
        def get_available(v):
            used = set()
            for u in adj[v]:
                if coloring[u] != -1:
                    used.add(coloring[u])
            return [c for c in range(max_colors) if c not in used]

        def backtrack():
            if time.time() > end_time:
                return False
            
            if not uncolored:
                return True
            
            # MRV: Find vertex with fewest available colors
            best_v = None
            best_avail = None
            min_count = max_colors + 1
            
            for v in list(uncolored):
                avail = get_available(v)
                if len(avail) < min_count:
                    min_count = len(avail)
                    best_v = v
                    best_avail = avail
                    if min_count == 0:
                        return False
                    if min_count == 1:
                        break
            
            if best_v is None:
                return False

            v = best_v
            uncolored.remove(v)
            
            # Value Ordering: Try colors that appear least often in neighbors? 
            # Or just 0..max_colors-1.
            # Let's try to prefer colors that are already used to keep count low? 
            # No, we want to minimize distinct colors, so reusing is good.
            # But we don't know which ones are used yet.
            # Just iterate.
            
            for c in best_avail:
                coloring[v] = c
                conflict = False
                actions = [] # List of (key, success)
                
                for u in adj[v]:
                    d = coloring[u]
                    if d != -1:
                        if d == c:
                            conflict = True
                            break
                        key = (min(c, d), max(c, d))
                        if key not in dsus:
                            dsus[key] = UndoableUnionFind(n)
                        uf = dsus[key]
                        success = uf.union(u, v)
                        actions.append((key, success))
                        if not success:
                            conflict = True
                            break
                
                if not conflict:
                    if backtrack():
                        return True
                
                # Undo
                coloring[v] = -1
                for key, success in reversed(actions):
                    if success:
                        dsus[key].undo()
                        
            uncolored.add(v)
            return False

        if backtrack():
            return coloring[:]
        return None

    best_coloring = None
    best_count = n + 1

    # Try 3 colors first with a generous timeout
    # If 3-coloring exists, it's likely found quickly if it's "easy".
    # If it's hard, we might waste time. But 50 nodes is small.
    coloring = solve(3, timeout=20.0)
    if coloring is not None:
        # Verify
        valid = True
        for u, v in edges:
            if coloring[u] == coloring[v]:
                valid = False
                break
        if valid:
            # Check bicolored forests
            color_pairs = set()
            for u, v in edges:
                c1, c2 = coloring[u], coloring[v]
                if c1 != c2:
                    color_pairs.add((min(c1, c2), max(c1, c2)))
            
            for c1, c2 in color_pairs:
                uf = UndoableUnionFind(n)
                for u, v in edges:
                    cu, cv = coloring[u], coloring[v]
                    if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                        if not uf.union(u, v):
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                used = len(set(coloring))
                if used < best_count:
                    best_count = used
                    best_coloring = coloring[:]

    # If 3-coloring failed or not found, try 4 colors
    if best_count > 4 and time.time() < deadline:
        coloring = solve(4, timeout=15.0)
        if coloring is not None:
            valid = True
            for u, v in edges:
                if coloring[u] == coloring[v]:
                    valid = False
                    break
            if valid:
                color_pairs = set()
                for u, v in edges:
                    c1, c2 = coloring[u], coloring[v]
                    if c1 != c2:
                        color_pairs.add((min(c1, c2), max(c1, c2)))
                
                for c1, c2 in color_pairs:
                    uf = UndoableUnionFind(n)
                    for u, v in edges:
                        cu, cv = coloring[u], coloring[v]
                        if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                            if not uf.union(u, v):
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid:
                    used = len(set(coloring))
                    if used < best_count:
                        best_count = used
                        best_coloring = coloring[:]

    # Fallback: Greedy 4-coloring (should always work for 4-regular? No, but 4 colors is plenty for proper coloring)
    # The current program guarantees a 12.5 score (4 colors). We must not do worse.
    if best_coloring is None or best_count > 4:
        # Simple greedy to get a valid 4-coloring
        # We know 4 colors are sufficient for proper coloring of 4-regular graphs?
        # Brooks' theorem: max degree 4, so 4 colors suffice unless it's a complete graph K5 or odd cycle.
        # K5 has degree 4. If graph is K5, 4 colors not enough for proper coloring? No, K5 needs 5.
        # But this graph is not K5. It's 4-regular on 50 nodes.
        # So 4 colors are sufficient for proper coloring.
        # But we need bicolored forests.
        
        # Let's try a specific 4-coloring pattern or just use the previous logic with a very tight timeout
        # Actually, the previous program found a 4-coloring. 
        # Let's try random restarts for 4 colors if the solver failed.
        
        for attempt in range(100):
            if time.time() > deadline:
                break
            random.seed(1000 + attempt)
            order = list(range(n))
            random.shuffle(order)
            coloring = [-1] * n
            valid = True
            for v in order:
                neighbor_colors = set(coloring[u] for u in adj[v] if coloring[u] != -1)
                available = [c for c in range(4) if c not in neighbor_colors]
                if not available:
                    valid = False
                    break
                coloring[v] = random.choice(available)
            
            if valid:
                # Check bicolored forests
                color_pairs = set()
                for u, v in edges:
                    c1, c2 = coloring[u], coloring[v]
                    if c1 != c2:
                        color_pairs.add((min(c1, c2), max(c1, c2)))
                
                ok = True
                for c1, c2 in color_pairs:
                    uf = UndoableUnionFind(n)
                    for u, v in edges:
                        cu, cv = coloring[u], coloring[v]
                        if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                            if not uf.union(u, v):
                                ok = False
                                break
                    if not ok:
                        break
                
                if ok:
                    used = len(set(coloring))
                    if used < best_count:
                        best_count = used
                        best_coloring = coloring[:]
                    break

    # Final fallback: 5 colors
    if best_coloring is None:
        coloring = list(range(n))
        best_coloring = coloring
        best_count = n

    result = [int(c) for c in best_coloring]
    return result

# EVOLVE-BLOCK-END
