import time
import math
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s margin
    
    N = 256
    nodes = list(range(1, N + 1))
    
    # Build adjacency list
    # adj[u] is a set of v such that u,v conflict
    adj = defaultdict(set)
    
    for a in nodes:
        # Find divisors of a^2
        a2 = a * a
        # Iterate d from 1 to sqrt(a2)
        limit = int(math.isqrt(a2))
        for d in range(1, limit + 1):
            if a2 % d == 0:
                # d is a divisor
                # b = d - a
                b = d - a
                if b > 0 and b != a and b <= N:
                    if b > a: # Add edge only once
                        adj[a].add(b)
                        adj[b].add(a)
                # Other divisor
                d2 = a2 // d
                if d2 != d:
                    b2 = d2 - a
                    if b2 > 0 and b2 != a and b2 <= N:
                        if b2 > a:
                            adj[a].add(b2)
                            adj[b2].add(a)
                            
    # Check if graph is bipartite
    # Color 0, 1
    color = [-1] * (N + 1)
    is_bipartite = True
    for u in nodes:
        if color[u] == -1:
            color[u] = 0
            q = deque([u])
            while q:
                curr = q.popleft()
                for nei in adj[curr]:
                    if color[nei] == -1:
                        color[nei] = color[curr] ^ 1
                        q.append(nei)
                    elif color[nei] == color[curr]:
                        is_bipartite = False
                        break
                if not is_bipartite:
                    break
        if not is_bipartite:
            break
            
    if is_bipartite:
        # Find Maximum Matching in Bipartite Graph
        # Left side: color 0, Right side: color 1
        # But we can just use standard Hopcroft-Karp on the whole graph
        # Or simpler: since N=256, even O(VE) is fine.
        # Let's use a simple DFS-based matching (Kuhn's algorithm)
        # It's O(VE) which is 256 * 1300 ~ 300k ops. Very fast.
        
        match = [-1] * (N + 1) # match[v] = u means v is matched with u
        
        def try_kuhn(v):
            for u in adj[v]:
                if match[u] == -1 or try_kuhn(match[u]):
                    match[u] = v
                    return True
            return False
        
        matching_size = 0
        for v in nodes:
            # To make it faster, we can reset visited or use a global visited set per iteration
            # Standard Kuhn requires clearing visited or using timestamps
            visited = [False] * (N + 1)
            def try_kuhn_with_vis(v):
                for u in adj[v]:
                    if not visited[u]:
                        visited[u] = True
                        if match[u] == -1 or try_kuhn_with_vis(match[u]):
                            match[u] = v
                            return True
                return False
            
            if try_kuhn_with_vis(v):
                matching_size += 1
                
        # MIS size = N - matching_size
        # We need to construct the set.
        # The set of vertices not covered by the matching is NOT the MIS.
        # The MIS is the complement of the Minimum Vertex Cover.
        # By Konig's Theorem, Min Vertex Cover size = Max Matching size.
        # We can find the Min Vertex Cover from the Max Matching.
        # Then MIS = All Nodes - Min Vertex Cover.
        
        # Find Min Vertex Cover
        # 1. Find set U of unmatched vertices on the Left side (color 0).
        # 2. Find set Z of vertices reachable from U via alternating paths.
        #    (Alternating path: unmatched edge, matched edge, unmatched edge...)
        #    Actually, standard algorithm:
        #    Let L be left nodes, R be right nodes.
        #    Find all unmatched nodes in L.
        #    BFS/DFS from them using:
        #    - Unmatched edges to go from L to R
        #    - Matched edges to go from R to L
        #    Let Z be the set of visited nodes.
        #    Min Vertex Cover = (L \ Z) U (R \cap Z)
        
        # We need to identify L and R.
        # In a bipartite graph, we can assign colors.
        # We already have `color`.
        # Let L = {v | color[v] == 0}, R = {v | color[v] == 1}
        
        # Identify unmatched nodes in L
        unmatched_in_L = []
        for v in nodes:
            if color[v] == 0:
                # Check if v is matched
                # match[u] = v means u (in R) is matched with v (in L)
                # We need to check if there exists u in R such that match[u] == v
                # Or we can maintain match_L[v] = u
                # Let's just check if v is in the values of match
                pass
        
        # Better to maintain match_L and match_R
        match_L = [-1] * (N + 1)
        match_R = [-1] * (N + 1)
        
        # Re-run matching to populate these? Or just derive from match?
        # match[u] = v where u in R, v in L?
        # In my try_kuhn, I iterated v (any node) and matched u (neighbor).
        # This doesn't strictly enforce L/R sides in the match array logic,
        # but the graph is bipartite, so any edge goes L-R.
        # So if match[u] = v, then u and v are in different partitions.
        
        # Let's re-implement matching with explicit L/R to be safe.
        # Left nodes: color 0. Right nodes: color 1.
        
        # Reset match
        match_L = [-1] * (N + 1)
        match_R = [-1] * (N + 1)
        
        def try_kuhn_l(v):
            # v is in L
            for u in adj[v]:
                # u is in R
                if match_R[u] == -1 or try_kuhn_l(match_R[u]):
                    match_R[u] = v
                    match_L[v] = u
                    return True
            return False
        
        matching_size = 0
        for v in nodes:
            if color[v] == 0:
                visited = [False] * (N + 1)
                def try_kuhn_l_with_vis(v):
                    for u in adj[v]:
                        if not visited[u]:
                            visited[u] = True
                            if match_R[u] == -1 or try_kuhn_l_with_vis(match_R[u]):
                                match_R[u] = v
                                match_L[v] = u
                                return True
                    return False
                
                if try_kuhn_l_with_vis(v):
                    matching_size += 1
                    
        # Find Unmatched in L
        unmatched_L = [v for v in nodes if color[v] == 0 and match_L[v] == -1]
        
        # BFS from unmatched_L
        # We need to traverse:
        # From L node, follow UNMATCHED edges to R
        # From R node, follow MATCHED edges to L
        
        visited_nodes = [False] * (N + 1)
        q = deque(unmatched_L)
        for v in unmatched_L:
            visited_nodes[v] = True
            
        while q:
            curr = q.popleft()
            if color[curr] == 0: # curr is in L
                # Go to neighbors in R via UNMATCHED edges
                for nei in adj[curr]:
                    if color[nei] == 1:
                        if match_L[curr] != nei: # Edge is unmatched
                            if not visited_nodes[nei]:
                                visited_nodes[nei] = True
                                q.append(nei)
            else: # curr is in R
                # Go to neighbors in L via MATCHED edges
                for nei in adj[curr]:
                    if color[nei] == 0:
                        if match_R[curr] == nei: # Edge is matched
                            if not visited_nodes[nei]:
                                visited_nodes[nei] = True
                                q.append(nei)
                            
        # Min Vertex Cover = (L \ Z) U (R \cap Z)
        min_vc = set()
        for v in nodes:
            if color[v] == 0 and not visited_nodes[v]:
                min_vc.add(v)
            elif color[v] == 1 and visited_nodes[v]:
                min_vc.add(v)
                
        # MIS = All Nodes - Min VC
        mis = [v for v in nodes if v not in min_vc]
        
        # Verify
        # Check validity
        for i in range(len(mis)):
            for j in range(i+1, len(mis)):
                a, b = mis[i], mis[j]
                if (a*b) % (a+b) == 0:
                    return list(range(1, 257, 2)) # Fallback
                    
        return mis
        
    else:
        # Fallback: Greedy + Local Search
        # Simple Greedy: Pick node with lowest degree, remove it and neighbors
        remaining = set(nodes)
        selected = []
        
        # Calculate initial degrees
        deg = {v: len(adj[v]) for v in nodes}
        
        while remaining and time.time() < deadline:
            # Find node with min degree
            # Sort is O(N log N). N=256. Fast.
            min_node = min(remaining, key=lambda x: deg[x])
            selected.append(min_node)
            
            # Remove min_node and its neighbors
            to_remove = {min_node} | adj[min_node]
            for v in to_remove:
                if v in remaining:
                    remaining.remove(v)
                    # Update degrees of remaining neighbors?
                    # Not strictly necessary for the greedy choice if we recalculate, 
                    # but updating is better.
            
            # Update degrees
            for v in remaining:
                # deg[v] decreases by 1 if min_node was neighbor
                # This is O(N * E) total if done naively. 
                # But N is small.
                pass 
            # Actually, just recalculating deg for remaining nodes in the next iteration is fine.
            # Or we can just use the static degree? No, dynamic is better.
            # Let's just recalculate deg for all remaining nodes each step? 
            # 256 steps * 256 nodes * avg_deg (5) = 300k ops. Fine.
            
            # Wait, the loop above removes from remaining. 
            # We need to update deg for the next min calculation.
            # Let's just recompute deg dict for remaining nodes.
            for v in remaining:
                deg[v] = len([u for u in adj[v] if u in remaining])
            
        # Local Search: Try to improve
        # Current set S = selected
        # Try to add a node v not in S by removing 1 neighbor in S? No, size must increase.
        # Try to find a node v not in S such that |N(v) \cap S| == 0? 
        # If so, add it. (Greedy should have added it if it was low degree, but maybe not).
        
        S = set(selected)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for v in nodes:
                if v not in S:
                    # Check if v can be added
                    # Conflict if any neighbor is in S
                    conflicts = [u for u in adj[v] if u in S]
                    if not conflicts:
                        S.add(v)
                        improved = True
                    elif len(conflicts) == 1:
                        # If we remove 1 conflict, we add 1. Size same.
                        # But maybe the removed node allows adding others?
                        # This is complex. Skip for now.
                        pass
                    else:
                        # Try to remove conflicts and add v if it leads to net gain?
                        # Hard to check locally.
                        pass
                        
        # One more pass: Swap
        # If we can swap 1 node in S with 2 nodes not in S, we gain 1.
        # This requires checking subsets.
        
        # Verify
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                a, b = list(S)[i], list(S)[j]
                if (a*b) % (a+b) == 0:
                    return list(range(1, 257, 2))
                    
        return list(S)

# EVOLVE-BLOCK-END
