import time
import sys
import random
from collections import deque

# Increase recursion limit for safety, though N=100 is small
sys.setrecursionlimit(10000)

def construct():
    n = 100
    # Pinned graph edges from the prompt's P dictionary
    edges = [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Find largest connected component
    visited_comp = [False] * n
    components = []
    for i in range(n):
        if not visited_comp[i]:
            comp = []
            stack = [i]
            visited_comp[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in adj[node]:
                    if not visited_comp[nb]:
                        visited_comp[nb] = True
                        stack.append(nb)
            components.append(comp)
            
    components.sort(key=len, reverse=True)
    largest_comp = components[0]
    max_possible_len = len(largest_comp)
    
    # If the largest component is small, we can solve it exactly or quickly
    # If it's large, we need good heuristics.
    
    best_path = []
    best_len = 0
    start_time = time.time()
    time_limit = 55.0 # Safety margin
    
    # Helper: Get size of the connected component of unvisited nodes reachable from 'start_node'
    # 'unvisited' is a set of nodes not yet in the path
    # 'start_node' is the current endpoint
    def get_reachable_unvisited_size(start_node, unvisited):
        if start_node not in unvisited:
            # The start node itself is visited, so we look at its neighbors
            # The "reachable unvisited component" is the union of components of unvisited neighbors
            # But we can only go into ONE component (the one we pick first).
            # So the bound is the size of the LARGEST connected component among the unvisited neighbors of start_node.
            pass
        
        # Actually, the path continues from start_node to an unvisited neighbor.
        # So the next step is to pick a neighbor $v \in adj[start_node] \cap unvisited$.
        # Once we pick $v$, we are locked into the connected component of $v$ in the subgraph induced by $unvisited$.
        # So the max extension is the size of the largest such component.
        
        max_size = 0
        # To avoid recomputing components for every neighbor, we can just do a BFS from each neighbor?
        # That might be slow if degree is high.
        # But N=100 is small.
        
        # Optimization: If we just want the bound, we can find the component sizes of unvisited neighbors.
        # Let's cache or compute on the fly.
        
        # For each neighbor of start_node that is unvisited, find the size of its component in 'unvisited'.
        # We can do a single BFS/DFS on the unvisited subgraph to find all components and their sizes, then check which ones are adjacent to start_node.
        
        # 1. Find all connected components in the subgraph induced by 'unvisited'
        comp_sizes = {}
        unvisited_list = list(unvisited)
        visited_unvisited = set()
        
        for node in unvisited_list:
            if node not in visited_unvisited:
                # BFS to find component
                comp = []
                queue = deque([node])
                visited_unvisited.add(node)
                while queue:
                    curr = queue.popleft()
                    comp.append(curr)
                    for nb in adj[curr]:
                        if nb in unvisited and nb not in visited_unvisited:
                            visited_unvisited.add(nb)
                            queue.append(nb)
                comp_sizes[tuple(sorted(comp))] = len(comp) # Using tuple as key is expensive, just store size for the set?
                # Better: store a dict mapping node -> component_id or just size
                # Let's just store the size for each node in the component
                for c in comp:
                    comp_sizes[c] = len(comp)
        
        # 2. Check neighbors of start_node
        max_bound = 0
        for nb in adj[start_node]:
            if nb in unvisited:
                size = comp_sizes.get(nb, 0)
                if size > max_bound:
                    max_bound = size
        
        return max_bound

    def dfs(path, unvisited, current_node):
        nonlocal best_path, best_len
        
        # Time check
        if time.time() - start_time > time_limit:
            return False # Signal to stop
        
        cur_len = len(path)
        
        # Update best
        if cur_len > best_len:
            best_len = cur_len
            best_path = path[:]
            
        # Pruning: Check if we can beat best_len
        # The maximum number of nodes we can add is the size of the largest reachable unvisited component from current_node
        if cur_len > 0:
            max_ext = get_reachable_unvisited_size(current_node, unvisited)
            if cur_len + max_ext <= best_len:
                return True # Pruned, continue searching elsewhere
        
        # Get neighbors
        neighbors = [nb for nb in adj[current_node] if nb in unvisited]
        
        if not neighbors:
            return True
        
        # Heuristic ordering:
        # 1. Prefer neighbors that lead to larger components (greedy)
        # 2. Or prefer neighbors with fewer options (dead-end first) to satisfy constraints early?
        # Usually, for longest path, "fail first" (fewest options) is good for finding valid paths, but "greedy" (largest component) is good for maximizing length.
        # Let's try sorting by the size of the component they lead to, descending.
        
        # We need the component sizes for sorting. We already computed them in get_reachable... but that was for the bound.
        # Let's reuse the logic or just sort by degree as a proxy?
        # Let's compute component sizes for sorting.
        
        # To save time, we can just sort by degree in unvisited subgraph.
        # Or, we can compute the component size for each neighbor once.
        
        # Let's compute component sizes for all unvisited nodes once per call?
        # That's O(N+E) per node. N=100. 100*100 = 10k ops per level. Depth 100. 1M ops. Very fast.
        
        # Let's refactor get_reachable... to also return the component size for each neighbor.
        
        # Actually, let's just do a simple sort by degree for now, or random if tie.
        # Better: Sort by the size of the component the neighbor belongs to in 'unvisited'.
        
        # We can calculate this by doing a BFS from each neighbor? No, that's O(deg * N).
        # Let's do the global component calculation for 'unvisited' once here.
        
        unvisited_comp_size = {}
        visited_unvisited = set()
        for node in unvisited:
            if node not in visited_unvisited:
                comp_size = 0
                stack = [node]
                visited_unvisited.add(node)
                while stack:
                    curr = stack.pop()
                    comp_size += 1
                    for nb in adj[curr]:
                        if nb in unvisited and nb not in visited_unvisited:
                            visited_unvisited.add(nb)
                            stack.append(nb)
                for c in visited_unvisited:
                    # This is wrong, visited_unvisited accumulates.
                    pass
        
        # Correct component calculation:
        unvisited_comp_size = {}
        visited_unvisited = set()
        for node in unvisited:
            if node not in visited_unvisited:
                comp_nodes = []
                stack = [node]
                visited_unvisited.add(node)
                while stack:
                    curr = stack.pop()
                    comp_nodes.append(curr)
                    for nb in adj[curr]:
                        if nb in unvisited and nb not in visited_unvisited:
                            visited_unvisited.add(nb)
                            stack.append(nb)
                for c in comp_nodes:
                    unvisited_comp_size[c] = len(comp_nodes)
        
        # Now sort neighbors by unvisited_comp_size[neighbor] descending
        neighbors.sort(key=lambda x: unvisited_comp_size.get(x, 0), reverse=True)
        
        for nb in neighbors:
            path.append(nb)
            unvisited.remove(nb)
            
            # Recurse
            # If we hit time limit, we need to unwind.
            if not dfs(path, unvisited, nb):
                # Time limit reached
                path.pop()
                unvisited.add(nb)
                return False
            
            path.pop()
            unvisited.add(nb)
            
            if time.time() - start_time > time_limit:
                return False
                
        return True

    # Start search from each node in the largest component
    # Order by degree ascending (likely endpoints)
    start_nodes = sorted(largest_comp, key=lambda x: len(adj[x]))
    
    for start in start_nodes:
        if time.time() - start_time > time_limit:
            break
            
        path = [start]
        unvisited = set(largest_comp)
        unvisited.remove(start)
        
        dfs(path, unvisited, start)
        
        # If we found a Hamiltonian path (length == max_possible_len), we can stop
        if best_len == max_possible_len:
            break

    return best_path
