import time
import random
import numpy as np
import sys
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin for 150s limit
    
    n = 20
    num_colours = 20
    
    # Precompute all 4-cycles (u, v, w, x) with u < v < w < x? No, 4-cycles are sets of 4 vertices.
    # A 4-cycle is a set of 4 distinct vertices {a,b,c,d} and a specific ordering of edges.
    # Actually, the condition "every 4-cycle is rainbow" applies to any subgraph isomorphic to C4.
    # In a simple graph, a C4 is defined by 4 distinct vertices a,b,c,d and edges (a,b), (b,c), (c,d), (d,a).
    # Note that (a,c) and (b,d) are diagonals and don't matter for the cycle itself, 
    # but they exist in the graph. The constraint is only on the 4 edges forming the cycle.
    
    # There are C(20,4) * 3 ways to form a cycle on 4 vertices?
    # Number of 4-cycles in K_n is n(n-1)(n-2)(n-3)/24.
    # For n=20: 20*19*18*17/24 = 20475. This is small enough to check.
    
    # Let's precompute the list of 4-cycles as tuples of 4 edge indices or vertex tuples.
    # A cycle is (u, v, w, x) where u,v,w,x are distinct. The edges are (u,v), (v,w), (w,x), (x,u).
    # To avoid duplicates, we can fix u < v and w < x? No, that's for pairs.
    # Standard enumeration: for all distinct u,v,w,x, the cycle u-v-w-x-u is the same as x-u-v-w-x.
    # We can just generate all permutations of 4 distinct vertices and check the specific cycle edges.
    # Or simpler: for each set of 4 vertices {a,b,c,d}, there are 3 distinct 4-cycles (up to rotation/reflection):
    # 1. a-b-c-d-a
    # 2. a-b-d-c-a
    # 3. a-c-b-d-a
    # So total cycles = C(20,4) * 3 = 4845 * 3 = 14535.
    
    cycles = []
    vertices = list(range(n))
    from itertools import combinations
    for combo in combinations(vertices, 4):
        a, b, c, d = combo
        # Cycle 1: a-b-c-d-a
        cycles.append(((a,b), (b,c), (c,d), (d,a)))
        # Cycle 2: a-b-d-c-a
        cycles.append(((a,b), (b,d), (d,c), (c,a)))
        # Cycle 3: a-c-b-d-a
        cycles.append(((a,c), (c,b), (b,d), (d,a)))
        
    # Total ~14535 cycles.
    
    def can_colour(adj_matrix, new_edge_u, new_edge_v):
        """
        Checks if adding edge (u,v) to the graph with current colours in adj_matrix
        allows for a valid colouring.
        adj_matrix: n x n, -1 for no edge, else colour.
        We try to assign a colour to (u,v) such that:
        1. It doesn't conflict with adjacent edges (proper edge colouring).
        2. It doesn't create a non-rainbow 4-cycle.
        
        If a valid colour exists, we return it. Otherwise None.
        Note: This is a local check. It assumes existing edges are already fixed and valid.
        Is it sufficient? 
        Adding an edge (u,v) only creates new 4-cycles that include (u,v).
        Any 4-cycle not including (u,v) was already valid.
        So we only need to check cycles involving (u,v).
        """
        u, v = new_edge_u, new_edge_v
        
        # 1. Proper edge colouring constraint:
        # Colour of (u,v) must be different from colours of all edges incident to u or v.
        forbidden_proper = set()
        for w in range(n):
            if w != u and adj_matrix[u, w] != -1:
                forbidden_proper.add(adj_matrix[u, w])
            if w != v and adj_matrix[w, v] != -1:
                forbidden_proper.add(adj_matrix[w, v])
                
        # 2. 4-cycle rainbow constraint:
        # A 4-cycle containing (u,v) looks like u-v-w-x-u or u-v-x-w-u.
        # The other 3 edges must have distinct colours, and none equal to the new colour.
        # Let the new colour be c.
        # For each potential 4-cycle involving (u,v):
        #   Edges: (u,v), (v,w), (w,x), (x,u).
        #   Colours: c, c_vw, c_wx, c_xu.
        #   Condition: c, c_vw, c_wx, c_xu are all distinct.
        #   This implies:
        #   a) c != c_vw, c != c_wx, c != c_xu
        #   b) c_vw != c_wx, c_vw != c_xu, c_wx != c_xu (This is already true if the graph was valid before?
        #      Wait. Before adding (u,v), the edges (v,w), (w,x), (x,u) existed.
        #      Did they form a 4-cycle? No, because (u,v) was missing.
        #      So the previous validity does NOT guarantee that c_vw, c_wx, c_xu are distinct.
        #      However, they are part of the graph. If they had same colours, was that a violation?
        #      Only if they formed a 4-cycle. They don't form a 4-cycle without (u,v).
        #      So we MUST check that the 3 existing edges in the new cycle also have distinct colours.
        #      If they don't, then NO colour c can fix it? 
        #      Wait. If c_vw == c_wx, then the cycle u-v-w-x-u has two adjacent edges with same colour.
        #      Can we change the colours of existing edges? 
        #      The strategy is "add edge and assign colour". It does NOT re-colour existing edges.
        #      So if the 3 existing edges in a potential 4-cycle involving (u,v) already have a colour conflict,
        #      we cannot add (u,v) with ANY colour, because the cycle will still be non-rainbow.
        #      Therefore, we must check:
        #      For each 4-cycle containing (u,v), the 3 other edges must have distinct colours.
        #      AND the new colour c must be distinct from those 3.
        
        forbidden_cycle = set()
        
        # Find all 4-cycles containing edge (u,v).
        # The cycle is u-v-w-x-u.
        # w and x must be distinct from u,v and each other.
        # Edges (v,w), (w,x), (x,u) must exist.
        
        # Iterate w in neighbours of v
        for w in range(n):
            if w == u or w == v: continue
            if adj_matrix[v, w] == -1: continue
            c_vw = adj_matrix[v, w]
            
            # Iterate x in neighbours of w AND u
            # x must be neighbour of w and x must be neighbour of u
            # And x != u, x != v, x != w
            for x in range(n):
                if x == u or x == v or x == w: continue
                if adj_matrix[w, x] == -1: continue
                if adj_matrix[x, u] == -1: continue
                
                c_wx = adj_matrix[w, x]
                c_xu = adj_matrix[x, u]
                
                # Check if the 3 existing edges have distinct colours
                if c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                    # This 4-cycle is "poisoned" by existing colour conflicts.
                    # We cannot add (u,v) if this cycle exists in the graph structure.
                    # Wait, if the edges exist, the cycle exists.
                    # If the colours are not distinct, the graph is INVALID for this cycle.
                    # But we assumed the current graph is valid.
                    # If the current graph is valid, does it imply that any 3 edges in a potential 4-cycle have distinct colours?
                    # No. The 3 edges (v,w), (w,x), (x,u) do not form a 4-cycle in the current graph because (u,v) is missing.
                    # So they could have same colours.
                    # If they do, adding (u,v) creates a 4-cycle with non-distinct colours.
                    # So we must forbid adding (u,v) if ANY such triple has non-distinct colours.
                    # Actually, we just need to find ONE valid colour c.
                    # If c_vw == c_wx, then for ANY c, the set {c, c_vw, c_wx, c_xu} has a duplicate (c_vw, c_wx).
                    # So the cycle will never be rainbow.
                    # Thus, if we encounter such a triple, we can immediately return None?
                    # No, we just need to ensure that for ALL cycles, the condition holds.
                    # If even ONE cycle has a conflict among the 3 fixed edges, we can't fix it by choosing c.
                    # So we return None.
                    return None
                
                # The 3 fixed colours are distinct.
                # Now c must be different from c_vw, c_wx, c_xu.
                forbidden_cycle.add(c_vw)
                forbidden_cycle.add(c_wx)
                forbidden_cycle.add(c_xu)
                
        # Combine forbidden sets
        all_forbidden = forbidden_proper.union(forbidden_cycle)
        
        # Find a colour not in all_forbidden
        for c in range(num_colours):
            if c not in all_forbidden:
                return c
                
        return None

    # Strategy:
    # 1. Start with empty graph.
    # 2. Generate list of all potential edges.
    # 3. Shuffle them.
    # 4. For each edge, try to add it.
    #    - Check if there is a valid colour.
    #    - If yes, add it with that colour.
    #    - If no, skip it.
    # 5. This produces a maximal valid graph (with respect to the order).
    # 6. To maximize density, we can try multiple random orders (restarts).
    # 7. Additionally, after the greedy pass, we can try to "swap" or "re-order"? 
    #    No, let's stick to multiple random restarts first.
    #    The "maximal" property depends on order. Different orders yield different densities.
    
    best_edges = 0
    best_adj = None
    
    # Number of restarts limited by time
    # Each restart: O(E * Cost_Check)
    # E ~ 190. Cost_Check: iterating neighbours. 
    # Neighbours of v: up to 19. Neighbours of w: up to 19.
    # Inner loop is ~ 19*19 = 361 iterations per edge.
    # Total ~ 190 * 361 = 68k operations per restart. Very fast.
    # We can do many restarts.
    
    # To improve results, we can also try a "second pass" where we try to add skipped edges
    # after the first pass, maybe with different colour choices? 
    # But the greedy check is deterministic given the current state.
    # So we just need to vary the edge order.
    
    # Heuristic for order: 
    # Random order is good.
    # Maybe "high degree potential" first? 
    # Let's try random shuffle first.
    
    for attempt in range(500): # Try up to 500 restarts
        if time.time() > deadline:
            break
            
        # Initialize empty graph
        adj = np.full((n, n), -1, dtype=int)
        
        # Create list of edges
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                edges.append((u, v))
                
        random.shuffle(edges)
        
        # Greedy Add
        for u, v in edges:
            c = can_colour(adj, u, v)
            if c is not None:
                adj[u, v] = c
                adj[v, u] = c
                
        # Count edges
        edge_count = int(np.sum(adj != -1)) // 2
        
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = adj.copy()
            
            # If we reached a very high density, we can stop?
            # Max edges in C4-free graph is ~ n^2/4 = 100? 
            # But we have colour constraints.
            # If we get > 80, it's very good.
            if best_edges >= 90:
                # Might be hard to beat, but let's run a few more
                pass

    # If best_adj is None, fallback
    if best_adj is None:
        adj = np.full((n, n), -1, dtype=int)
        # Add a star
        for v in range(1, n):
            adj[0, v] = 0
            adj[v, 0] = 0
        best_adj = adj
        best_edges = 19

    # Final verification (optional but good for safety)
    # Check all 4-cycles in best_adj
    # This is O(14535) checks. Fast.
    valid = True
    for cycle in cycles:
        e1, e2, e3, e4 = cycle
        c1 = best_adj[e1[0], e1[1]]
        c2 = best_adj[e2[0], e2[1]]
        c3 = best_adj[e3[0], e3[1]]
        c4 = best_adj[e4[0], e4[1]]
        
        # All must be present (not -1)
        if c1 == -1 or c2 == -1 or c3 == -1 or c4 == -1:
            continue # Not a cycle in the graph
            
        # Check distinctness
        if len({c1, c2, c3, c4}) < 4:
            valid = False
            break
            
    if not valid:
        # This should not happen if can_colour is correct.
        # But if it does, we need to return a valid graph.
        # We can just return the star (which is valid).
        adj = np.full((n, n), -1, dtype=int)
        for v in range(1, n):
            adj[0, v] = 0
            adj[v, 0] = 0
        best_adj = adj
        best_edges = 19

    # Format output
    edges_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v] != -1:
                edges_list.append([u, v, int(best_adj[u, v])])
                
    return edges_list

if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res)}")
    if res:
        print(f"Sample: {res[0]}")
