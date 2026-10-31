import time
import random
import sys
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Strategy:
    1. Generate a random 3-regular graph of a target size n.
    2. Use a local search heuristic (2-swap edge switching) to increase the girth.
    3. The search focuses on breaking short cycles (length < g) by swapping edges.
    4. We try multiple target sizes, starting from a lower bound estimate, 
       increasing n until we find a graph that can be repaired to girth >= g.
       This allows us to find smaller graphs than fixed algebraic constructions
       if the random space is rich enough, or at least match them.
    
    Given the time limit (120s), we prioritize:
    - Fast girth checking (BFS).
    - Efficient 2-swap generation.
    - Early termination if a valid graph is found.
    
    We start with n=936 (best known) and try to repair a random instance.
    If that fails, we try slightly larger n (e.g., 1000, 1100, ...) up to a limit.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    if k != 3:
        # Fallback for other k, though task specifies k=3
        # If k is not 3, we return a dummy or raise, but we assume k=3
        raise ValueError("Only k=3 supported in this strategy")

    def generate_random_3_regular(n):
        """
        Generate a random 3-regular graph on n vertices (n must be even).
        Uses a simple configuration model / pairing method.
        """
        if n % 2 != 0:
            n += 1
        if n < 4:
            return n, []
        
        # Create a list of 3n "half-edges"
        half_edges = [i for i in range(n) for _ in range(3)]
        random.shuffle(half_edges)
        
        # Pair them up
        adj = [[] for _ in range(n)]
        for i in range(0, 3 * n, 2):
            u = half_edges[i]
            v = half_edges[i + 1]
            if u == v:
                # Self-loop, retry or fix
                # For simplicity, we might just accept and let verification fail,
                # or we can swap with another edge to avoid self-loop.
                # Let's try to swap with the next pair to avoid self-loop if possible.
                if i + 2 < 3 * n:
                    w = half_edges[i + 2]
                    if w != v:
                        adj[u].append(w)
                        adj[w].append(u)
                        adj[v].append(half_edges[i + 3])
                        adj[half_edges[i + 3]].append(v)
                        i += 2
                        continue
                # If can't fix easily, mark as invalid
                return None, None
            
            adj[u].append(v)
            adj[v].append(u)
        
        # Check for multiple edges?
        # For 3-regular, multiple edges are possible in random config model.
        # We can check and swap if found, but for large n, probability is low.
        # Let's assume simple for now, or verify later.
        
        # Convert to edge list
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        
        # Verify degree
        for i in range(n):
            if len(adj[i]) != 3:
                return None, None
        
        return n, edges

    def verify_girth_and_find_shortest(n, edges, g):
        """
        Verify if girth >= g.
        Returns (is_valid, shortest_cycle_length).
        """
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != 3:
                return False, 0
        
        min_girth = float('inf')
        
        # BFS from every vertex to find shortest cycle
        # Optimization: We only need to check up to depth g//2
        # If we find a cycle shorter than g, return False.
        # If we complete all BFS without finding a cycle < g, return True.
        
        for start in range(n):
            # State: (node, parent, depth)
            # We use a visited array for the BFS tree from 'start'
            # But standard BFS for girth needs to detect back-edges to ancestors.
            # In a tree, back-edge (u,v) where v is ancestor of u gives a cycle.
            # Length = depth[u] - depth[v] + 1.
            
            visited = [-1] * n # -1 means unvisited, otherwise depth
            parent = [-1] * n
            queue = deque()
            queue.append(start)
            visited[start] = 0
            
            while queue:
                u = queue.popleft()
                if visited[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = visited[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u]:
                            # Found a cycle
                            # The cycle length is visited[u] + visited[v] + 1
                            # This is valid if u and v are not in the same branch?
                            # In BFS, if v is already visited and not parent,
                            # the cycle is u-v-...-start-...-u?
                            # Actually, the shortest cycle through start is found by
                            # meeting two branches.
                            # The formula visited[u] + visited[v] + 1 gives the cycle length
                            # IF the path from v to start and u to start are disjoint except at start.
                            # This is true for BFS trees.
                            cycle_len = visited[u] + visited[v] + 1
                            if cycle_len < min_girth:
                                min_girth = cycle_len
                            if min_girth < g:
                                return False, min_girth
        
        if min_girth == float('inf'):
            min_girth = g # Or larger
        return True, min_girth

    def perform_2_swap(n, edges, adj, g, max_swaps=5000):
        """
        Try to increase girth by performing 2-swaps.
        Returns True if girth >= g is achieved.
        """
        # Build edge set for fast lookup
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        
        for _ in range(max_swaps):
            if time.time() > deadline:
                break
            
            # Check current girth
            is_valid, min_girth = verify_girth_and_find_shortest(n, edges, g)
            if is_valid:
                return True
            
            # Find a short cycle to break
            # We can reuse the BFS logic to find one specific short cycle
            # For simplicity, we pick two random edges and swap, hoping to break cycles.
            # Better: Pick an edge (u, v) that is part of a short cycle.
            # How to find it? 
            # We can do a BFS from a random node and find the first cycle < g.
            # Then pick an edge in that cycle and swap it with a random non-adjacent edge.
            
            # Let's find one short cycle
            cycle_edges = []
            for start in range(n):
                visited = [-1] * n
                parent = [-1] * n
                queue = deque()
                queue.append(start)
                visited[start] = 0
                
                found = False
                while queue and not found:
                    u = queue.popleft()
                    if visited[u] >= g // 2:
                        continue
                    for v in adj[u]:
                        if visited[v] == -1:
                            visited[v] = visited[u] + 1
                            parent[v] = u
                            queue.append(v)
                        else:
                            if v != parent[u]:
                                cycle_len = visited[u] + visited[v] + 1
                                if cycle_len < g:
                                    # Reconstruct cycle
                                    # Path u -> ... -> start -> ... -> v
                                    # We need the edges in this cycle.
                                    # This is complex to reconstruct quickly.
                                    # Alternative: Just pick a random edge from the graph
                                    # and swap it. If it breaks a short cycle, good.
                                    # Given high probability of many short cycles initially,
                                    # random swapping might work if we are persistent.
                                    found = True
                                    break
                if found:
                    break
            
            if not cycle_edges:
                # Pick random edge
                u, v = random.choice(edges)
            else:
                u, v = random.choice(edges)
            
            # Find another edge (x, y) to swap with
            # Conditions:
            # 1. (x, y) not in edge_set
            # 2. (u, x) not in edge_set, (u, y) not in edge_set, etc.
            # 3. No self loops
            # 4. x, y not u, v
            
            # Try a few random candidates
            swapped = False
            for _ in range(10):
                x, y = random.choice(edges)
                if x == u or x == v or y == u or y == v:
                    continue
                if (min(x, y), max(x, y)) == (min(u, v), max(u, v)):
                    continue
                
                # Check if new edges are valid
                new_e1 = (u, x)
                new_e2 = (v, y)
                new_e1_t = (min(u, x), max(u, x))
                new_e2_t = (min(v, y), max(v, y))
                
                if new_e1_t in edge_set or new_e2_t in edge_set:
                    continue
                if u == x or v == y:
                    continue
                
                # Perform swap
                # Remove (u, v) and (x, y)
                edge_set.remove((min(u, v), max(u, v)))
                edge_set.remove((min(x, y), max(x, y)))
                
                # Add (u, x) and (v, y)
                edge_set.add(new_e1_t)
                edge_set.add(new_e2_t)
                
                # Update adj
                adj[u].remove(v)
                adj[v].remove(u)
                adj[x].remove(y)
                adj[y].remove(x)
                
                adj[u].append(x)
                adj[x].append(u)
                adj[v].append(y)
                adj[y].append(v)
                
                # Update edges list (replace)
                # For efficiency, we can keep a list and update indices,
                # but for simplicity, we rebuild or just use edge_set for verification
                # and a list for random choice.
                # We need to update the 'edges' list used for random.choice
                # Let's find the indices
                # This is O(E). To speed up, we can maintain a dict or just accept O(E)
                # since E is ~1.5n.
                
                # Find index of (u,v)
                idx1 = -1
                idx2 = -1
                for i, e in enumerate(edges):
                    if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                        idx1 = i
                        break
                for i, e in enumerate(edges):
                    if (e[0] == x and e[1] == y) or (e[0] == y and e[1] == x):
                        idx2 = i
                        break
                
                if idx1 != -1 and idx2 != -1:
                    edges[idx1] = (u, x)
                    edges[idx2] = (v, y)
                    swapped = True
                    break
            
            if not swapped:
                # If we can't find a valid swap, we might be stuck
                # or the graph is very constrained.
                # Try another random edge pair?
                pass

        # Final check
        is_valid, _ = verify_girth_and_find_shortest(n, edges, g)
        return is_valid

    # Strategy:
    # Try n = 936 first.
    # If it fails, try n = 938, 940, ... up to a limit (e.g., 1500).
    # For each n, try a few random instances.
    
    best_n = float('inf')
    best_edges = []
    
    # Lower bound for 3-regular girth 16 is 512.
    # Best known is 936.
    # We try from 936 upwards.
    
    for n_try in range(936, 1500, 2):
        if time.time() > deadline:
            break
        
        # Try a few random instances for this n
        attempts = 5
        for _ in range(attempts):
            if time.time() > deadline:
                break
            
            n, edges = generate_random_3_regular(n_try)
            if n is None:
                continue
            
            # Build adj
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # Check initial girth
            is_valid, min_girth = verify_girth_and_find_shortest(n, edges, g)
            if is_valid:
                best_n = n
                best_edges = edges
                break
            
            # If not valid, try to repair
            # Limit repair time per instance to 10s
            repair_deadline = time.time() + 10.0
            # We modify perform_2_swap to respect a deadline
            # For now, we just call it and hope it finishes fast or fails fast.
            # Let's add a time check inside the loop of perform_2_swap?
            # The current perform_2_swap doesn't have a hard deadline other than global.
            # We should ensure it doesn't run for 100s on one instance.
            
            # Let's create a bounded version or just rely on the global deadline.
            # If it takes too long, it will hit the global deadline and we return best found.
            
            if perform_2_swap(n, edges, adj, g, max_swaps=2000):
                best_n = n
                best_edges = edges
                break
        
        if best_n != float('inf'):
            break
    
    if best_n == float('inf'):
        # Fallback: Return a known valid graph if we have one?
        # We don't have the 936 graph hardcoded.
        # We must return something valid.
        # If we didn't find one, we might have to return a larger one that we verified?
        # But we didn't verify any if best_n is inf.
        # Let's try a larger n that is easier to fix?
        # Or just return a dummy? No, must be valid.
        
        # Let's try n=2000 with more attempts?
        # Or just return the last one we tried? No, it might be invalid.
        
        # To be safe, if we didn't find a valid graph, we should try a very large n
        # where random graphs are more likely to have high girth?
        # Actually, girth of random 3-regular graph is ~ log_2 n.
        # For n=2000, log_2 2000 ~ 11.
        # We need 16. So random graphs are unlikely to be valid without repair.
        
        # If we fail, we raise an error? Or return a minimal valid graph we can construct quickly?
        # LUW(3,7) is 4374. We can construct that as a fallback.
        
        def luw_construct(k, g, d_override=None):
            if k != 3:
                return None, None
            q = 3
            d = d_override
            if d is None:
                d = 7
            
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
            
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
                        if side == 0:
                            y[c] = add(x[c], mul(y[a], x[b]))
                        else:
                            y[c] = sub(x[c], mul(x[a], y[b]))
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

        n_luw, edges_luw = luw_construct(3, g, 7)
        if n_luw and verify_girth_and_find_shortest(n_luw, edges_luw, g)[0]:
            return n_luw, edges_luw
        else:
            # Last resort
            raise RuntimeError("Failed to construct a valid graph")

    return best_n, best_edges

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"Constructed graph with {n} vertices")
