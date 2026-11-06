import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Known valid graph: 8 disjoint K4's -> n=32, alpha=8 < 9, no C5
    known_32 = {
        'n': 32,
        'edges': [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    }
    
    best_result = known_32

    def has_c5(adj, n):
        """
        Check if graph contains a 5-cycle.
        adj: list of sets of neighbors
        n: number of vertices
        """
        # Optimization: If any vertex has degree < 2, it can't be in a C5, 
        # but its neighbors might. 
        # We iterate over edges (u, v) with u < v.
        # For each edge, we look for a path of length 3 from u to v 
        # that doesn't use the edge (u,v) directly? No, the cycle is u-v-x-y-z-u.
        # So we fix edge (u,v). We need x, y, z such that:
        # v-x, x-y, y-z, z-u are edges.
        # x != u, v
        # y != u, v, x
        # z != u, v, x, y
        
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                # Neighbors of v, excluding u
                # We need to find a path v-x-y-z-u
                # x in N(v) \ {u}
                # y in N(x) \ {u, v}
                # z in N(y) \ {u, v, x}
                # Check if z in N(u)
                
                # To speed up, we can pre-compute or use sets.
                # adj is a list of sets.
                
                # Iterate x
                for x in adj[v]:
                    if x == u:
                        continue
                    # Iterate y
                    # y must be in N(x) and not u, v
                    for y in adj[x]:
                        if y == u or y == v:
                            continue
                        # Iterate z
                        # z must be in N(y) and not u, v, x
                        # And z must be in N(u)
                        # So z in N(y) & N(u) \ {v, x}
                        # Let's compute common neighbors of y and u, then filter.
                        common_y_u = adj[y] & adj[u]
                        # Remove v and x
                        if v in common_y_u:
                            common_y_u.discard(v)
                        if x in common_y_u:
                            common_y_u.discard(x)
                        if common_y_u:
                            return True
        return False

    def has_independent_set_9(adj, n):
        """
        Check if graph has an independent set of size 9.
        adj: list of sets of neighbors
        n: number of vertices
        """
        # Backtracking search
        # candidates: set of vertices that can still be added to the independent set
        # current_size: size of current independent set
        # target: 9
        
        def search(candidates, current_size):
            if current_size == 9:
                return True
            if not candidates:
                return False
            if len(candidates) < 9 - current_size:
                return False
            
            # Heuristic: pick vertex with minimum degree within candidates?
            # Or just pick the first one.
            # Let's pick the one with the fewest neighbors in candidates to prune faster.
            min_deg = float('inf')
            best_v = -1
            for v in candidates:
                deg = len(adj[v] & candidates)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if min_deg == 0:
                        break
            
            v = best_v
            
            # Option 1: Include v
            # New candidates are those in candidates that are not neighbors of v and not v itself
            new_candidates = candidates - adj[v] - {v}
            if search(new_candidates, current_size + 1):
                return True
            
            # Option 2: Exclude v
            new_candidates = candidates - {v}
            if search(new_candidates, current_size):
                return True
            
            return False
        
        return search(set(range(n)), 0)

    def is_valid(n, adj):
        if has_c5(adj, n):
            return False
        if has_independent_set_9(adj, n):
            return False
        return True

    def adj_to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append([i, j])
        return edges

    # Search for n = 33, 34, 35
    # We use a randomized local search.
    # Objective: Minimize (num_c5 * 100 + num_is9)
    # But we don't have a fast count. We just have existence.
    # So we do:
    # 1. Generate random graph.
    # 2. Check if valid. If yes, return.
    # 3. If not, try to fix violations?
    #    If has C5, remove an edge from a C5.
    #    If has IS9, add an edge between two vertices in the IS9.
    
    for target_n in [33, 34, 35]:
        if time.time() > deadline:
            break
        
        found = False
        attempts = 0
        max_attempts = 5000  # Adjust based on time
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Initialize random graph
            # Probability of edge p. 
            # For C5-free, graphs should be somewhat sparse or structured.
            # For alpha < 9, graphs should be dense enough.
            # Let's try p around 0.3 to 0.5.
            p = random.uniform(0.3, 0.5)
            
            adj = [set() for _ in range(target_n)]
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < p:
                        adj[i].add(j)
                        adj[j].add(i)
            
            # Local search
            for step in range(100):
                if time.time() > deadline:
                    break
                
                # Check validity
                if is_valid(target_n, adj):
                    found = True
                    break
                
                # If not valid, try to improve
                # Get a C5 or IS9?
                # has_c5 returns bool. We need the actual cycle to remove an edge.
                # This is expensive to find.
                # Alternative: Randomly flip an edge.
                # Or: If has_c5, pick a random edge and remove it.
                # If has_is9, pick a random non-edge and add it.
                
                has_c5_flag = has_c5(adj, target_n)
                has_is9_flag = has_independent_set_9(adj, target_n)
                
                if has_c5_flag and has_is9_flag:
                    # 50% remove edge, 50% add edge
                    if random.random() < 0.5:
                        # Remove random edge
                        all_edges = [(i, j) for i in range(target_n) for j in adj[i] if i < j]
                        if all_edges:
                            u, v = random.choice(all_edges)
                            adj[u].remove(v)
                            adj[v].remove(u)
                    else:
                        # Add random non-edge
                        non_edges = [(i, j) for i in range(target_n) for j in range(i+1, target_n) if j not in adj[i]]
                        if non_edges:
                            u, v = random.choice(non_edges)
                            adj[u].add(v)
                            adj[v].add(u)
                elif has_c5_flag:
                    # Remove random edge
                    all_edges = [(i, j) for i in range(target_n) for j in adj[i] if i < j]
                    if all_edges:
                        u, v = random.choice(all_edges)
                        adj[u].remove(v)
                        adj[v].remove(u)
                elif has_is9_flag:
                    # Add random non-edge
                    non_edges = [(i, j) for i in range(target_n) for j in range(i+1, target_n) if j not in adj[i]]
                    if non_edges:
                        u, v = random.choice(non_edges)
                        adj[u].add(v)
                        adj[v].add(u)
                else:
                    # Should have returned True above
                    break
            
            if found:
                edges = adj_to_edges(adj, target_n)
                best_result = {'n': target_n, 'edges': edges}
                # Since we found a larger n, we can return immediately or check for even larger?
                # The loop goes 33, 34, 35. If we find 33, we should try 34?
                # Yes, the outer loop continues.
                break
        
        # If we found a solution for this target_n, we keep it as best_result
        # and continue to the next target_n to see if we can do better.
        # But the `found` flag is local to the while loop.
        # We need to track if we found one for this target_n.
        # The `break` inside the while loop exits the while loop.
        # If `found` is True, we update best_result.
        # Then the outer for loop continues to the next target_n.

    return best_result

# Helper to run construct
if __name__ == "__main__":
    res = construct()
    print(res['n'])
