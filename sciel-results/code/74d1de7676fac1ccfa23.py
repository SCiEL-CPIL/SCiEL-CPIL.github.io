import time
import sys
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 50  # 50s deadline to be safe
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Find connected components
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            components.append(comp)
            
    largest_comp = max(components, key=len)
    comp_set = set(largest_comp)
    
    # Filter adjacency to only include edges within the largest component
    # This helps with the "remaining degree" calculation
    adj_comp = [set() for _ in range(n)]
    for u in largest_comp:
        for v in adj[u]:
            if v in comp_set:
                adj_comp[u].add(v)

    def find_hamiltonian_path():
        """
        Finds a Hamiltonian path in the largest component using DFS with pruning.
        Returns the path if found, else None.
        """
        N = len(largest_comp)
        # Map component vertices to 0..N-1 for faster bitset ops? 
        # Or just use sets. N=100 is small.
        
        # Pre-check: Count odd degree vertices
        odd_deg_count = sum(1 for u in largest_comp if len(adj_comp[u]) % 2 == 1)
        if odd_deg_count > 2:
            # Cannot have a Hamiltonian path
            return None
            
        best_path = []
        
        def dfs(start):
            nonlocal best_path
            if time.time() > deadline:
                return False
            
            path = [start]
            visited = {start}
            # Remaining degrees of unvisited nodes
            # We don't need to track explicitly if we compute on the fly, 
            # but for pruning, it's better to know if a node is isolated.
            
            def is_valid_next(curr, next_node):
                # next_node is unvisited
                # Check if this move isolates any unvisited node
                # A node u is isolated if it has no unvisited neighbors.
                # We only care about nodes that are adjacent to the path or to each other.
                # Actually, just check if any unvisited node has 0 unvisited neighbors.
                # This is O(N^2) per step. 100^2 * 100 steps = 1e6 ops. Fast.
                
                # Optimization: Only check nodes in the "frontier" or neighbors of visited nodes?
                # No, a node far away might become isolated if its only bridge is cut.
                # But in a sparse graph, "far away" nodes are usually in different components (already filtered).
                # Within the component, if we visit a node, we remove it from unvisited.
                # The unvisited graph is the original graph minus visited nodes.
                # We need to ensure the unvisited graph is connected (or has the right number of components for the remaining path ends).
                # Actually, for a simple path, the unvisited nodes must form a single connected component 
                # (if we are building from one end) OR two components if we are building from both ends?
                # No, we are building a simple path from one end.
                # So the unvisited nodes must form a connected subgraph?
                # Not necessarily. The path can jump? No, it's a simple path.
                # The path is a sequence. The unvisited nodes are not yet in the sequence.
                # The sequence must be contiguous.
                # So the unvisited nodes don't need to be connected to each other, 
                # BUT the next node must be adjacent to the current end.
                # And we must not "cut off" any unvisited node such that it can never be reached.
                # A node is "cut off" if it has no path to the current end through unvisited nodes.
                # This is equivalent to: the unvisited node must be in the same connected component 
                # as the current end in the graph induced by unvisited nodes + current end?
                # Actually, simpler: 
                # If we add `next_node` to visited, does any unvisited node become isolated 
                # from the rest of the unvisited nodes?
                # Or does it become isolated from the path?
                # The path is linear. The "future" of the path must be able to visit all remaining nodes.
                # This requires that the remaining nodes form a single connected component 
                # that is adjacent to the current end?
                # No, the remaining nodes just need to be connectable in a path starting from the current end.
                # This is the Hamiltonian Path problem on the remaining graph with a fixed start.
                # Pruning rule:
                # 1. If the unvisited graph is disconnected, and we haven't finished, we are stuck?
                #    Not if we are not required to visit all nodes? But we are maximizing length.
                #    If we want a Hamiltonian path of the component, we MUST visit all nodes.
                #    So the unvisited nodes must form a connected component that contains the next node?
                #    Actually, the unvisited nodes must form a connected component.
                #    If they are disconnected, we can't visit all of them in a single path.
                #    So: Check if the subgraph induced by unvisited nodes is connected.
                #    If not, fail.
                
                return True # Placeholder
                
            # Better pruning:
            # 1. If an unvisited node has degree 0 in the unvisited graph, fail.
            # 2. If an unvisited node has degree 1 in the unvisited graph, and it's not adjacent to the current end, 
            #    then it must be the *last* node of the path? No, it must be visited, but if it's not adjacent to the current end,
            #    we can't reach it unless we go through its only neighbor.
            #    If its only neighbor is also unvisited, we are fine.
            #    If its only neighbor is visited, and it's not the current end, we are stuck.
            
            def check_connectivity_and_forced(unvisited_set, curr):
                # Check if unvisited_set is connected
                if not unvisited_set:
                    return True, None
                
                # BFS from any node in unvisited_set
                start_node = next(iter(unvisited_set))
                visited_bfs = {start_node}
                queue = deque([start_node])
                while queue:
                    u = queue.popleft()
                    for v in adj_comp[u]:
                        if v in unvisited_set and v not in visited_bfs:
                            visited_bfs.add(v)
                            queue.append(v)
                            
                if len(visited_bfs) != len(unvisited_set):
                    return False, None # Disconnected
                
                # Check for forced moves
                # If any unvisited node u has only 1 unvisited neighbor v:
                # If v == curr, then we MUST go to u.
                # If v != curr, then u must be visited *after* v. 
                # If v is not curr, we can still go elsewhere, but we must ensure we don't cut off u.
                # Actually, if u has 1 neighbor v in unvisited, and v is not curr, 
                # then the path must enter u from v and leave u? No, u is a leaf in the unvisited graph.
                # So the path must end at u or start at u.
                # Since we are extending from curr, the path cannot start at u.
                # So the path must END at u.
                # This means u must be the LAST node visited.
                # If there is more than one such "leaf" node, we can only visit one at the end.
                # So if there are >= 2 leaves in the unvisited graph, and neither is adjacent to curr, 
                # we can only visit one of them at the very end. The other one will be cut off.
                # So:
                # Count leaves (unvisited nodes with unvisited-degree 1).
                leaves = []
                for u in unvisited_set:
                    deg = sum(1 for v in adj_comp[u] if v in unvisited_set)
                    if deg == 1:
                        leaves.append(u)
                        
                if len(leaves) > 1:
                    # We can only visit one leaf at the end.
                    # So we must visit the other leaves? No, we can't.
                    # So this is a dead end unless one of the leaves is adjacent to curr and we go there NOW?
                    # If we go to a leaf now, it becomes visited. The other leaves remain.
                    # If there are 2 leaves, and we don't visit one now, we can't visit it later 
                    # (because it will become isolated or we won't be able to end there if we end elsewhere).
                    # Actually, if we visit one leaf now, it's removed.
                    # The other leaves might become connected?
                    # This is getting complex.
                    # Simple rule: If len(leaves) > 1, and curr is not adjacent to any of them, FAIL.
                    # If curr is adjacent to one, we MUST go to that one?
                    # Not necessarily, but it's a strong hint.
                    # Let's just return the forced move if there's only 1 leaf and it's adjacent to curr.
                    pass

                # Specific forced move detection:
                # If curr has only 1 unvisited neighbor, we MUST go there.
                curr_unvisited_neighbors = [v for v in adj_comp[curr] if v in unvisited_set]
                if len(curr_unvisited_neighbors) == 1:
                    return True, curr_unvisited_neighbors[0]
                    
                # If there is an unvisited node u with 0 unvisited neighbors, FAIL.
                # (Already covered by connectivity check? No, isolated node is a component of size 1.
                # If unvisited_set has >1 nodes and is disconnected, we fail.
                # If unvisited_set has 1 node, it's connected.)
                
                return True, None

            def dfs_step(curr, path, visited):
                if time.time() > deadline:
                    return False
                
                if len(path) == N:
                    nonlocal best_path
                    if len(path) > len(best_path):
                        best_path = path[:]
                    return True
                    
                unvisited = comp_set - visited
                
                ok, forced_next = check_connectivity_and_forced(unvisited, curr)
                if not ok:
                    return False
                    
                if forced_next is not None:
                    nexts = [forced_next]
                else:
                    nexts = [v for v in adj_comp[curr] if v in unvisited]
                    
                # Heuristic: Sort by degree (Warnsdorff)
                # Lower degree first
                nexts.sort(key=lambda v: sum(1 for u in adj_comp[v] if u in unvisited))
                
                for v in nexts:
                    path.append(v)
                    visited.add(v)
                    if dfs_step(v, path, visited):
                        return True
                    path.pop()
                    visited.remove(v)
                    
                return False

            # Try starting from each vertex?
            # Or just the endpoints (degree 1 or 2)?
            # In a Hamiltonian path, endpoints have degree 1 in the path.
            # In the graph, they have degree >= 1.
            # Let's try all vertices as start, but prioritize low degree.
            
            starts = sorted(largest_comp, key=lambda u: len(adj_comp[u]))
            
            for start in starts:
                if time.time() > deadline:
                    break
                if dfs_step(start, [start], {start}):
                    if len(best_path) == N:
                        return best_path
                # If we found a full path, return it
                if best_path and len(best_path) == N:
                    return best_path

            return best_path if best_path else None

    path = find_hamiltonian_path()
    
    if not path:
        # Fallback: Greedy
        path = []
        if largest_comp:
            start = largest_comp[0]
            path.append(start)
            visited = {start}
            while True:
                curr = path[-1]
                nexts = [v for v in adj_comp[curr] if v not in visited]
                if not nexts:
                    break
                # Pick neighbor with fewest options
                nexts.sort(key=lambda v: len(adj_comp[v] - visited))
                nxt = nexts[0]
                path.append(nxt)
                visited.add(nxt)
                
    # Verify
    for i in range(len(path) - 1):
        if path[i+1] not in adj[path[i]]:
            return [0]
            
    return path
