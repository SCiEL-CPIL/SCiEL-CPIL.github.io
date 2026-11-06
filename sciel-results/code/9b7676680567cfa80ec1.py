import time
import random

def construct():
    start_time = time.time()
    hard_deadline = start_time + 115.0

    def verify(n, colors):
        """Verify that no color class contains a C6."""
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        for c in range(4):
            adj_list = [[] for _ in range(n)]
            for i in range(n):
                mask = adj[c][i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    adj_list[i].append(j)
                    mask -= lsb
            
            for u in range(n):
                for v in adj_list[u]:
                    if v <= u: continue
                    for w in adj_list[v]:
                        if w == u: continue
                        for x in adj_list[w]:
                            if x == u or x == v: continue
                            for y in adj_list[x]:
                                if y == u or y == v or y == w: continue
                                for z in adj_list[y]:
                                    if z == u or z == v or z == w or z == x: continue
                                    if (adj[c][z] >> u) & 1:
                                        return False
        return True

    def has_path_len_5(c, u, v, adj):
        """
        Check if there is a simple path of length 5 from u to v in graph c.
        Path: u -> x1 -> x2 -> x3 -> x4 -> v
        adj is list of bitmasks for color c.
        """
        # We search for path u -> ... -> v of length 5.
        # This is equivalent to finding x4 such that x4 is neighbor of v,
        # and there is a path of length 4 from u to x4 avoiding v.
        
        # To speed up, we can do BFS from u up to depth 4 and see if any node at depth 4 is neighbor of v.
        # Or BFS from both ends?
        # Let's do iterative deepening or just BFS with depth limit.
        
        # BFS from u
        # Level 0: {u}
        # Level 1: N(u)
        # ...
        # Level 4: Nodes reachable in exactly 4 steps.
        # Check if any node in Level 4 is connected to v.
        
        # Using bitmasks for efficiency.
        # visited[0] = 1 << u
        # visited[k] = nodes reached in exactly k steps.
        
        # Actually, we just need *any* path.
        # Let's use a set of (node, depth) but bitmasks are faster.
        
        # We need to ensure the path is simple (no repeated vertices).
        # BFS with bitmask visited is standard but might be slow in Python if not careful.
        
        # Alternative: Recursive DFS with bitmask.
        # Given the small n (<=17), recursion depth 5 is fine.
        
        def dfs(cur, depth, visited_mask):
            if depth == 0:
                return (adj[c][cur] >> v) & 1 != 0
            if depth == 1:
                # cur must be neighbor of v
                return (adj[c][cur] >> v) & 1 != 0
            
            # Try all neighbors of cur not in visited
            mask = adj[c][cur] & ~visited_mask
            while mask:
                lsb = mask & -mask
                nxt = lsb.bit_length() - 1
                if dfs(nxt, depth - 1, visited_mask | lsb):
                    return True
                mask -= lsb
            return False

        # Start from u, need to reach v in 5 steps.
        # So we take 4 steps from u, then check if we are neighbor of v.
        # Wait, path length 5 means 5 edges.
        # u - e1 - x1 - e2 - x2 - e3 - x3 - e4 - x4 - e5 - v
        # So we need to find x4 such that dist(u, x4) = 4 and x4 is neighbor of v.
        # My dfs above checks if from `cur` we can reach `v` in `depth` steps.
        # So call dfs(u, 5, 1 << u).
        
        return dfs(u, 5, 1 << u)

    def solve(n):
        """Backtracking solver with symmetry breaking."""
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)
        colors = [-1] * num_edges
        adj = [[0] * n for _ in range(4)]
        
        # Symmetry breaking: Fix first 4 edges
        # (0,1) -> 0, (0,2) -> 1, (0,3) -> 2, (0,4) -> 3
        # Indices:
        # idx(0,1) = 0
        # idx(0,2) = 1
        # idx(0,3) = 2
        # idx(0,4) = 3
        
        def get_idx(u, v):
            if u > v: u, v = v, u
            return u * (n - 1) - u * (u + 1) // 2 + (v - u - 1)

        # Apply symmetry breaking
        colors[0] = 0
        adj[0][0] |= (1 << 1)
        adj[0][1] |= (1 << 0)
        
        colors[1] = 1
        adj[1][0] |= (1 << 2)
        adj[1][2] |= (1 << 0)
        
        colors[2] = 2
        adj[2][0] |= (1 << 3)
        adj[2][3] |= (1 << 0)
        
        colors[3] = 3
        adj[3][0] |= (1 << 4)
        adj[3][4] |= (1 << 0)
        
        # We start searching from edge index 4
        stack = [(4, 0)] # (edge_idx, tried_mask)
        
        while stack:
            if time.time() > hard_deadline:
                return None
            
            idx, tried_mask = stack[-1]
            
            if idx == num_edges:
                return colors[:]
            
            u, v = edges[idx]
            found = False
            for c in range(4):
                cbit = 1 << c
                if tried_mask & cbit:
                    continue
                
                # Check if adding (u,v) with color c creates a C6
                # This happens if there is a path of length 5 from u to v in adj[c]
                if not has_path_len_5(c, u, v, adj):
                    # Valid
                    colors[idx] = c
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)
                    
                    stack[-1] = (idx, tried_mask | cbit)
                    stack.append((idx + 1, 0))
                    found = True
                    break
            
            if not found:
                stack.pop()
                if idx > 3: # If we are backtracking past the fixed edges
                    prev_idx = idx - 1
                    prev_u, prev_v = edges[prev_idx]
                    prev_c = colors[prev_idx]
                    if prev_c != -1:
                        adj[prev_c][prev_u] &= ~(1 << prev_v)
                        adj[prev_c][prev_v] &= ~(1 << prev_u)
                        colors[prev_idx] = -1
                # If idx <= 3, we shouldn't be popping the fixed edges in this simple implementation
                # but the stack logic handles it by just popping and moving to parent.
                # The fixed edges are already in adj, so we don't remove them.
                # Wait, the stack starts at 4. So idx will never be < 4 in the loop above?
                # Yes, stack starts with (4, 0).
                # When we pop, we return to the previous state which was (idx-1, mask).
                # We need to undo the assignment at idx-1?
                # No, the stack holds the state *before* trying the next edge.
                # When we push (idx+1, 0), we have assigned edge idx.
                # When we pop (idx+1, ...), we are at state where edge idx is assigned.
                # We need to un-assign edge idx to try the next color for it?
                # No, the `tried_mask` in the parent frame handles which colors were tried.
                # But we must remove the edge from `adj` and `colors` before the parent tries the next color.
                
                # Let's re-verify the stack logic.
                # State: stack[-1] = (idx, mask). This means edges 0..idx-1 are assigned.
                # We are trying to assign edge idx.
                # If we find a color c, we assign it, update adj, and push (idx+1, 0).
                # If we fail (no color works), we pop.
                # When we pop, we are back at (idx, mask).
                # But we have already tried some colors and updated adj.
                # We need to revert the adj changes for the last color tried?
                # Actually, the loop `for c in range(4)` tries colors.
                # If it finds a valid one, it breaks and pushes.
                # If it doesn't find one, it means all 4 colors failed.
                # In that case, we haven't left `adj` in a "assigned" state for edge idx?
                # Wait, inside the loop, we check `has_path_len_5`. We do NOT modify `adj` during the check.
                # We only modify `adj` if the color is valid.
                # So if no color is valid, `adj` is unchanged from the state at the start of the loop.
                # So we just pop, and the parent state is intact.
                
                # However, if we *did* find a valid color, we modified `adj` and pushed.
                # If the deeper search fails, we pop back to this frame.
                # At this point, `adj` still contains the assignment for edge idx.
                # We need to remove it so we can try the next color for edge idx?
                # No, the `tried_mask` in this frame remembers which colors we tried.
                # But we only push if we *succeed* in assigning.
                # So if we push, we are done with this frame for now.
                # If we pop back here, it means the child failed.
                # We need to try the *next* color for edge idx.
                # But we already updated `tried_mask` in the stack entry?
                # `stack[-1] = (idx, tried_mask | cbit)`.
                # So when we pop, we get `(idx, tried_mask | cbit)`.
                # But we must revert `adj` and `colors` for edge idx first!
                
                # Let's fix the logic:
                # When popping, we must undo the assignment of the edge at the current top's index?
                # No.
                # Let's trace:
                # 1. Stack: [(4, 0)]. Top is (4,0). Edge 4 is unassigned.
                # 2. Try c=0 for edge 4. Valid.
                # 3. Assign edge 4 to c=0. Update adj.
                # 4. Update stack top to (4, 1).
                # 5. Push (5, 0). Stack: [(4, 1), (5, 0)].
                # 6. Process (5, 0). Suppose it fails completely.
                # 7. Pop (5, 0). Stack: [(4, 1)].
                # 8. Top is (4, 1). Edge 4 is assigned to c=0 in adj.
                # 9. Loop for c in range(4). c=0 is in tried_mask (1). Skip.
                # 10. Try c=1. Check valid.
                # 11. If valid:
                #     We need to remove edge 4 from c=0 in adj.
                #     We need to add edge 4 to c=1 in adj.
                #     Update colors[4] = 1.
                #     Update stack top to (4, 1 | 2) = (4, 3).
                #     Push (5, 0).
                # 12. If not valid, try c=2, etc.
                # 13. If all fail, pop (4, 1).
                
                # So, we need to handle the "undo" when we are at a node and want to try the next color.
                # The current structure does this implicitly? No.
                # The `adj` reflects the assignment of the *current* top's edge?
                # No, `adj` reflects the assignments of all edges 0..idx-1.
                # Wait, if stack top is (idx, mask), it means edges 0..idx-1 are assigned.
                # Edge idx is NOT assigned in `adj` yet?
                # Let's re-read my code snippet.
                
                # In the snippet:
                # `if not has_path_len_5...`:
                #   `colors[idx] = c`
                #   `adj[c][u] |= ...`
                #   `stack[-1] = (idx, tried_mask | cbit)`
                #   `stack.append((idx + 1, 0))`
                
                # So when we are at (idx, mask), edge idx is NOT in adj.
                # When we pick a color, we add it to adj and push (idx+1, 0).
                # So when we pop (idx+1, 0), we are back at (idx, mask_with_bit).
                # Edge idx IS in adj.
                # We need to remove it before trying the next color.
                
                # So, after popping, if the top is (idx, mask), and we are about to try the next color:
                # We must check if edge idx is currently in adj.
                # How do we know?
                # If we just popped, it means the previous color's subtree failed.
                # So edge idx is in adj with the previous color.
                # We need to find which color it was.
                # `colors[idx]` holds it.
                
                # So, in the `while stack` loop:
                # 1. Get top (idx, mask).
                # 2. If idx == num_edges, return.
                # 3. If we are resuming (i.e., we just popped or this is the first time), 
                #    we might need to clean up?
                #    Actually, the first time we enter (idx, 0), edge idx is not in adj.
                #    If we enter (idx, mask) after a pop, edge idx IS in adj.
                
                # How to distinguish?
                # We can check if `colors[idx] != -1`.
                # If it is not -1, we need to remove it from adj before trying new colors.
                
                # Let's add this check.
                
                pass # Placeholder for logic fix

        return None

    # Implementing the fixed logic in a clean function
    def solve_fixed(n):
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)
        colors = [-1] * num_edges
        adj = [[0] * n for _ in range(4)]
        
        def get_idx(u, v):
            if u > v: u, v = v, u
            return u * (n - 1) - u * (u + 1) // 2 + (v - u - 1)

        # Symmetry breaking
        # (0,1)->0, (0,2)->1, (0,3)->2, (0,4)->3
        colors[0] = 0; adj[0][0] |= (1 << 1); adj[0][1] |= (1 << 0)
        colors[1] = 1; adj[1][0] |= (1 << 2); adj[1][2] |= (1 << 0)
        colors[2] = 2; adj[2][0] |= (1 << 3); adj[2][3] |= (1 << 0)
        colors[3] = 3; adj[3][0] |= (1 << 4); adj[3][4] |= (1 << 0)
        
        stack = [(4, 0)]
        
        while stack:
            if time.time() > hard_deadline:
                return None
            
            idx, tried_mask = stack[-1]
            
            if idx == num_edges:
                return colors[:]
            
            # If we are resuming from a child failure, edge idx is currently assigned.
            # We need to remove it from adj and colors before trying the next color.
            if colors[idx] != -1:
                old_c = colors[idx]
                u, v = edges[idx]
                adj[old_c][u] &= ~(1 << v)
                adj[old_c][v] &= ~(1 << u)
                colors[idx] = -1
            
            u, v = edges[idx]
            found = False
            for c in range(4):
                cbit = 1 << c
                if tried_mask & cbit:
                    continue
                
                if not has_path_len_5(c, u, v, adj):
                    # Valid
                    colors[idx] = c
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)
                    
                    stack[-1] = (idx, tried_mask | cbit)
                    stack.append((idx + 1, 0))
                    found = True
                    break
            
            if not found:
                stack.pop()
                # If we pop, and idx > 3, the parent will handle cleanup of its edge?
                # No, the parent's edge is already in adj.
                # When we pop to parent (idx-1, mask), the parent's edge idx-1 is in adj.
                # The loop will start, see colors[idx-1] != -1, and remove it.
                # This logic holds.
        
        return None

    # Try n=17
    for n in [17, 16, 15, 14, 13]:
        if time.time() > hard_deadline - 5:
            break
        
        res = solve_fixed(n)
        if res is not None:
            if verify(n, res):
                return {'n': n, 'colors': res}
        
        # If n=17 fails, we might want to try again with different symmetry breaking?
        # Or just move to n=16.
        # Given time constraints, moving to n=16 is safer if 17 fails quickly.
        # But 17 is the target.
        # Let's give 17 most of the time.
        if n == 17:
             # If 17 failed, try n=16
             pass

    # Fallback
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    if verify(n, colors):
        return {'n': n, 'colors': colors}
    
    return {'n': 1, 'colors': []}

if __name__ == '__main__':
    construct()
