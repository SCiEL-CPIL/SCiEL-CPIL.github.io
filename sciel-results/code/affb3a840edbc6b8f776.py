"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _create_union_find(n):
    parent = list(range(n))
    rank = [0] * n
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True
            
    return find, union, parent, rank

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Sort vertices by degree descending to constrain early
    vertices = sorted(range(n), key=lambda x: -len(adj[x]))
    vertex_to_index = {v: i for i, v in enumerate(vertices)}
    
    best_colors = None
    best_num = n + 1
    
    # Try increasing number of colors
    for max_colors in range(2, 10):
        colors = [-1] * n
        # We need to track Union-Find for each color pair
        # Since max_colors is small, we can pre-allocate or create on the fly.
        # Key: (color_a, color_b) with a < b
        # Value: [find, union, parent, rank]
        # But we need to backtrack, so we need to save/restore state or use a persistent structure.
        # Simpler: Use a recursive function that passes the state down, or uses a global stack of changes.
        # Given n=50, recursion depth is 50. We can just copy the UF states if we are careful, 
        # but copying 10 UFs of size 50 is cheap.
        
        # Let's use a simpler approach: 
        # Maintain a list of "active" UFs.
        # To allow backtracking, we can store the parent and rank arrays and copy them.
        
        def solve(idx):
            if idx == n:
                return True
            
            v = vertices[idx]
            
            for c in range(max_colors):
                # Check if color c is valid for v
                valid = True
                
                # 1. Check adjacent vertices don't have color c
                for u in adj[v]:
                    if colors[u] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                # 2. Check bicolored cycles
                # We need to see if adding v with color c creates a cycle in any pair {c, d}
                # We will simulate the addition.
                # We need to know the current connectivity for pairs involving c.
                # Since we are backtracking, we can maintain the UF states in the recursion stack.
                # For efficiency, let's assume we maintain a global state and undo it.
                # However, implementing undo for Union-Find is tricky (need to store history).
                # Given small n, let's just copy the UF states for the relevant pairs.
                
                # Actually, we can maintain a single "constraint check" that is incremental.
                # But the most robust way without complex undo logic is to pass the UF states as arguments or use a global variable with snapshots.
                
                # Let's use a global dictionary `uf_states` that maps (c1, c2) to (parent, rank).
                # We will snapshot the states for pairs involving c before trying, and restore if we fail.
                
                # Pairs involving c: (c, d) for all d != c.
                # We only need to check pairs where the other color d is present in neighbors.
                # Actually, we need to check all pairs (c, d) because v might connect to multiple neighbors of different colors.
                
                # Optimization: Only check pairs (c, d) where d is the color of a neighbor u.
                # If no neighbor has color d, adding v (color c) doesn't affect the subgraph of (c, d).
                
                affected_pairs = set()
                for u in adj[v]:
                    if colors[u] != -1 and colors[u] != c:
                        d = colors[u]
                        pair = (min(c, d), max(c, d))
                        affected_pairs.add(pair)
                
                # Snapshot the UF states for affected pairs
                snapshots = {}
                for p in affected_pairs:
                    if p in uf_states:
                        snapshots[p] = (uf_states[p][0][:], uf_states[p][1][:])
                    else:
                        # If no UF exists, we create a new one, so snapshot is effectively "None" or empty
                        snapshots[p] = None

                # Try to assign c
                colors[v] = c
                
                # Update UFs
                failed = False
                for u in adj[v]:
                    if colors[u] != -1:
                        d = colors[u]
                        if d == c:
                            failed = True
                            break
                        pair = (min(c, d), max(c, d))
                        if pair not in uf_states:
                            # Create new UF for this pair
                            find, union, parent, rank = _create_union_find(n)
                            uf_states[pair] = (parent, rank, find, union)
                        
                        parent, rank, find, union = uf_states[pair]
                        if not union(u, v):
                            failed = True
                            break
                        
                if not failed:
                    if solve(idx + 1):
                        return True
                
                # Restore state
                colors[v] = -1
                for p in affected_pairs:
                    if snapshots[p] is None:
                        if p in uf_states:
                            del uf_states[p]
                    else:
                        uf_states[p] = (snapshots[p][0], snapshots[p][1], uf_states[p][2], uf_states[p][3])
                        # Note: find and union functions close over parent and rank. 
                        # If we restore parent and rank, the functions still point to the old lists? 
                        # No, in Python, closures capture the variable reference. 
                        # _create_union_find returns functions that close over the local variables `parent` and `rank`.
                        # If we replace `uf_states[p]` with a new tuple, the `find` and `union` functions in the tuple still refer to the OLD parent/rank lists if we don't recreate them.
                        # This is a bug in my logic above. The functions `find` and `union` are bound to the specific `parent` and `rank` lists created in `_create_union_find`.
                        # If I modify `parent` in place, the functions see the change.
                        # If I restore `parent` to a snapshot, I need the functions to use that restored list.
                        # Since `find` and `union` close over the *names* `parent` and `rank` in the scope of `_create_union_find`, and those names point to the lists, if I replace the list in `uf_states`, the functions still point to the original list object unless I rebind them.
                        
                        # Correct approach: Don't store functions. Store parent and rank. Implement find/union inline or pass parent/rank to them.
                        
                # Let's refactor to avoid closure issues.

        # Refactored solve with explicit UF management
        uf_states = {} # pair -> (parent, rank)
        
        def _find(parent, x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(parent, rank, x, y):
            px, py = _find(parent, x), _find(parent, y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True

        def solve2(idx):
            if idx == n:
                return True
            
            v = vertices[idx]
            
            for c in range(max_colors):
                valid = True
                for u in adj[v]:
                    if colors[u] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                affected_pairs = set()
                for u in adj[v]:
                    if colors[u] != -1 and colors[u] != c:
                        d = colors[u]
                        pair = (min(c, d), max(c, d))
                        affected_pairs.add(pair)
                
                # Snapshot
                snapshots = {}
                for p in affected_pairs:
                    if p in uf_states:
                        parent, rank = uf_states[p]
                        snapshots[p] = (parent[:], rank[:])
                    else:
                        snapshots[p] = None

                colors[v] = c
                failed = False
                for u in adj[v]:
                    if colors[u] != -1:
                        d = colors[u]
                        if d == c:
                            failed = True
                            break
                        pair = (min(c, d), max(c, d))
                        if pair not in uf_states:
                            uf_states[pair] = (list(range(n)), [0]*n)
                        
                        parent, rank = uf_states[pair]
                        if not _union(parent, rank, u, v):
                            failed = True
                            break
                
                if not failed:
                    if solve2(idx + 1):
                        return True
                
                colors[v] = -1
                for p in affected_pairs:
                    if snapshots[p] is None:
                        if p in uf_states:
                            del uf_states[p]
                    else:
                        uf_states[p] = (snapshots[p][0], snapshots[p][1])

        # Run solve2
        # Reset uf_states for each max_colors attempt
        uf_states = {}
        if solve2(0):
            best_colors = colors
            best_num = max_colors
            break

    if best_colors is None:
        # Fallback
        best_colors = list(range(n))
        best_num = n
        
    return [int(c) for c in best_colors]
# EVOLVE-BLOCK-END
