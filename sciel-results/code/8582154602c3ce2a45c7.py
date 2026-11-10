import time
import random
import itertools

N = 16
ALL_PAIRS = list(itertools.combinations(range(N), 2))
NUM_PAIRS = len(ALL_PAIRS)

def _creates_cube(u, v, adj):
    """Return True if adding edge (u,v) to adjacency `adj` (list of int bitsets)
    creates a 3-cube (Q3) as a subgraph.
    """
    Nu = adj[u] & ~(1 << v)
    Nv = adj[v] & ~(1 << u)
    
    L_u = [i for i in range(N) if (Nu >> i) & 1]
    L_v = [i for i in range(N) if (Nv >> i) & 1]
    
    if len(L_u) < 2 or len(L_v) < 2:
        return False
        
    for i in range(len(L_u)):
        x = L_u[i]
        for j in range(i+1, len(L_u)):
            w = L_u[j]
            mask_y = adj[x] & Nv & ~(1 << x) & ~(1 << w)
            mask_z = adj[w] & Nv & ~(1 << x) & ~(1 << w)
            
            if not mask_y or not mask_z:
                continue
            
            if mask_y == mask_z and (mask_y & (mask_y - 1)) == 0:
                continue
                
            return True
    return False


def _find_q3_edges(adj):
    """Find one Q3 in the graph. Returns list of 12 (u,v) edges or None."""
    # For each pair of vertices u,v that are adjacent, try to build a Q3 containing edge (u,v)
    # A Q3 containing edge (u,v): u and v are adjacent in Q3.
    # In Q3, each edge is in exactly 2 squares (4-cycles).
    # So we need two 4-cycles sharing edge (u,v).
    # The two other vertices of the first square: x, w where x~u, x~v, w~u, w~v, x!=w
    # Actually, a 4-cycle containing (u,v) has form u-x-v-w-u or u-w-v-x-u
    # Let's find all 4-cycles containing (u,v)
    
    for u, v in ALL_PAIRS:
        if not ((adj[u] >> v) & 1):
            continue
        # Find common neighbors of u and v
        common_uv = adj[u] & adj[v] & ~(1 << u) & ~(1 << v)
        common_uv_list = [i for i in range(N) if (common_uv >> i) & 1]
        
        if len(common_uv_list) < 2:
            continue
        
        # Each pair of common neighbors forms a 4-cycle with u,v
        # 4-cycle: u - x - v - y - u where x,y are common neighbors
        # For Q3, we need two such 4-cycles sharing edge (u,v), and the remaining structure
        # The two 4-cycles give us 4 other vertices: x1, y1, x2, y2
        # In Q3, the two squares share only the edge (u,v)
        # So {x1,y1} and {x2,y2} should be disjoint
        
        for i in range(len(common_uv_list)):
            x1 = common_uv_list[i]
            for j in range(i+1, len(common_uv_list)):
                y1 = common_uv_list[j]
                # First square: u-x1-v-y1-u
                # Now find second square: need x2, y2 common neighbors of u,v, disjoint from {x1,y1}
                # Also need x2~x1 or x2~y1 (they share an edge in Q3)
                # In Q3, the two squares sharing an edge: the other 4 vertices form a 4-cycle
                # x1-x2-y1-y2-x1 (or similar)
                
                for k in range(len(common_uv_list)):
                    x2 = common_uv_list[k]
                    if x2 in (x1, y1):
                        continue
                    for l in range(k+1, len(common_uv_list)):
                        y2 = common_uv_list[l]
                        if y2 in (x1, y1, x2):
                            continue
                        
                        # Check if x1,x2,y1,y2 form a 4-cycle
                        # Possible: x1-x2, x2-y1, y1-y2, y2-x1
                        # Or x1-y2, y2-x2, x2-y1, y1-x1
                        # Check both
                        def _check_4cyc(a,b,c,d):
                            return ((adj[a]>>b)&1) and ((adj[b]>>c)&1) and ((adj[c]>>d)&1) and ((adj[d]>>a)&1)
                        
                        if _check_4cyc(x1,x2,y1,y2) or _check_4cyc(x1,y2,x2,y1):
                            # We have a candidate Q3: vertices {u,v,x1,y1,x2,y2,?,?}
                            # Wait, that's only 6 vertices. Q3 has 8.
                            # I made an error. Let me reconsider.
                            # In Q3, an edge is in 2 squares. The two squares share the edge.
                            # Square 1: u, v, a, b (4 vertices)
                            # Square 2: u, v, c, d (4 vertices)
                            # The 8 vertices are {u,v,a,b,c,d} — that's only 6!
                            # No wait: square 1 has vertices {u,v,a,b}, square 2 has {u,v,c,d}
                            # Union is {u,v,a,b,c,d} = 6 vertices. But Q3 has 8 vertices.
                            # I'm confused. Let me look at Q3 structure.
                            # Q3: vertices 0,1,2,3,4,5,6,7
                            # Edges: 0-1,1-3,3-2,2-0 (square 1)
                            #        4-5,5-7,7-6,6-4 (square 2)
                            #        0-4,1-5,2-6,3-7 (verticals)
                            # Edge 0-1 is in square 1 (0-1-3-2-0) and... is it in another square?
                            # 0-1-5-4-0 is another square! So edge 0-1 is in squares {0,1,3,2} and {0,1,5,4}.
                            # Union of these two squares: {0,1,2,3,4,5} = 6 vertices.
                            # But Q3 has 8 vertices. So two squares sharing an edge only give 6 vertices.
                            # The remaining 2 vertices (6,7) are connected via verticals to 2,3 and 6,7.
                            # So to find Q3, I need to find 2 squares sharing an edge, then check if the remaining 2 vertices complete the cube.
                            
                            # Let me abandon this approach and use a simpler one.
                            pass
    return None


def construct():
    start = time.time()
    deadline = start + 110.0

    best_edges = None
    best_count = -1

    def time_left():
        return deadline - time.time()

    rng = random.Random(12345)

    # Local search: start from dense graphs, remove edges to break Q3s, then add back
    for trial in range(50):
        if time_left() < 10.0:
            break
        
        # Start with a random dense graph
        density = 0.6 + 0.3 * rng.random()
        adj = [0] * N
        edges = []
        for u, v in ALL_PAIRS:
            if rng.random() < density:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
        
        # Remove edges to break all Q3s
        # Iteratively find Q3s and remove an edge
        max_iter = 200
        for _ in range(max_iter):
            if time_left() < 5.0:
                break
            # Find a Q3 by checking all 8-vertex subsets
            found = False
            for vert_set in itertools.combinations(range(N), 8):
                vs = set(vert_set)
                # Check if induced subgraph on vs contains a Q3
                # Build induced subgraph
                ind_edges = [(u,v) for u,v in edges if u in vs and v in vs]
                if len(ind_edges) < 12:
                    continue
                # Check if there's a 3-regular spanning subgraph isomorphic to Q3
                # Simple check: each vertex has degree >= 3 in induced subgraph
                degs = {}
                for v in vert_set:
                    degs[v] = sum(1 for u in vert_set if u != v and (adj[v] >> u) & 1)
                if any(d < 3 for d in degs.values()):
                    continue
                # Try to find a Q3: use backtracking on 8 vertices
                # This is expensive but 8 is small
                # For now, just remove a random edge from the induced subgraph
                if ind_edges:
                    rem_u, rem_v = rng.choice(ind_edges)
                    adj[rem_u] &= ~(1 << rem_v)
                    adj[rem_v] &= ~(1 << rem_u)
                    edges.remove((rem_u, rem_v))
                    found = True
                    break
            if not found:
                break
        
        # Now try to add edges back
        count = len(edges)
        for u, v in ALL_PAIRS:
            if time_left() < 5.0:
                break
            if (adj[u] >> v) & 1:
                continue
            if not _creates_cube(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
                count += 1
        
        if count > best_count:
            best_count = count
            best_edges = [[u, v] for u, v in edges]

    if best_edges is None:
        best_edges = [[a, b] for a in range(3) for b in range(3, 16)]
    return best_edges
