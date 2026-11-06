import random
import time
import sys

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # State representation
    # adj[u][v] = 1 if edge exists, 0 otherwise
    # color[u][v] = color of edge
    # neighbors[u] = set of neighbors
    
    def is_valid_edges(edges):
        """
        Verify the entire graph is valid.
        Returns True if all 4-cycles are rainbow.
        """
        if not edges:
            return True
        adj = [[0]*n for _ in range(n)]
        color_mat = [[-1]*n for _ in range(n)]
        for u, v, c in edges:
            adj[u][v] = 1
            adj[v][u] = 1
            color_mat[u][v] = c
            color_mat[v][u] = c
        
        # Check all 4-cycles
        # Optimization: Only check cycles u-v-w-x where u < v < w < x or similar canonical form?
        # For n=20, O(n^4) is 160,000 checks. Each check is O(1).
        # 160k * constant is fast enough.
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]: continue
                for w in range(v+1, n):
                    if not adj[v][w]: continue
                    # x must be different from u, v, w
                    # Cycle u-v-w-x-u
                    # We need adj[w][x] and adj[x][u]
                    for x in range(w+1, n):
                        if x == u: continue # u is fixed, x > w > v > u, so x != u
                        if not adj[w][x] or not adj[x][u]: continue
                        c1 = color_mat[u][v]
                        c2 = color_mat[v][w]
                        c3 = color_mat[w][x]
                        c4 = color_mat[x][u]
                        if len({c1, c2, c3, c4}) != 4:
                            return False
        return True

    def get_forbidden_colors(u, v, adj, color_mat, neighbors):
        """
        Determine forbidden colors for edge (u,v).
        Returns a set of forbidden colors.
        If the set is all colors, edge cannot be added.
        """
        forbidden = set()
        # Iterate over paths v-w-x-u
        # w in neighbors[v], x in neighbors[u]
        # w != u, x != v, w != x
        # adj[w][x] must be 1
        for w in neighbors[v]:
            if w == u: continue
            for x in neighbors[u]:
                if x == v or x == w: continue
                if not adj[w][x]: continue
                
                c_vw = color_mat[v][w]
                c_wx = color_mat[w][x]
                c_xu = color_mat[x][u]
                
                # If any two of these are the same, the path is "tainted"
                # and no color for (u,v) can make the cycle rainbow.
                if c_vw == c_wx or c_wx == c_xu or c_vw == c_xu:
                    return set(range(num_colors))
                
                forbidden.add(c_vw)
                forbidden.add(c_wx)
                forbidden.add(c_xu)
                
        return forbidden

    def run_greedy_with_repair(seed):
        random.seed(seed)
        adj = [[0]*n for _ in range(n)]
        color_mat = [[-1]*n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        # Heuristic: Sort edges by "difficulty"
        # Initially, difficulty is low for all.
        # We can just shuffle.
        random.shuffle(all_edges)
        
        # Phase 1: Greedy Addition
        for u, v in all_edges:
            forbidden = get_forbidden_colors(u, v, adj, color_mat, neighbors)
            valid_colors = set(range(num_colors)) - forbidden
            
            if not valid_colors:
                continue
            
            # Pick a color. 
            # Strategy: Pick the color that appears least often in the graph?
            # Or just random.
            c = random.choice(list(valid_colors))
            
            adj[u][v] = 1
            adj[v][u] = 1
            color_mat[u][v] = c
            color_mat[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
            edges.append([u, v, c])
            
        # Phase 2: Local Search - Try to add skipped edges
        # Keep a list of edges that failed
        # We can just iterate through all non-edges and try to add them.
        # If we add one, we might break validity? No, get_forbidden_colors ensures validity.
        # But adding an edge might make *other* previously skipped edges addable? No, adding edges only *increases* constraints.
        # So if an edge couldn't be added, it still can't be added later unless we *remove* or *recolor* edges.
        
        # Therefore, simple greedy is monotonic. To improve, we need RECOLORING.
        
        # Let's implement a simple Recoloring Local Search.
        # 1. Identify a non-edge (u,v) that is blocked.
        # 2. Identify a 4-cycle u-v-w-x-u that is blocking it (i.e., colors conflict).
        # 3. Try to recolor one of the edges in that cycle to free up a color for (u,v).
        
        # This is complex. Let's try a simpler structural change first:
        # **Bipartite Initialization**
        
        # Let's restart the function to use Bipartite Init.
        return edges

    # --- Revised Strategy: Bipartite Core + Greedy Intra ---
    
    def construct_bipartite_core():
        """
        Construct K_{10,10} with a valid coloring.
        Vertices 0-9 in A, 10-19 in B.
        """
        adj = [[0]*n for _ in range(n)]
        color_mat = [[-1]*n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        # Color scheme for bipartite edges (a in 0..9, b in 10..19)
        # c(a, b) = (a * 2 + (b - 10) * 3) % 20
        # Let's verify if this works for 4-cycles.
        # Cycle a1-b1-a2-b2-a1
        # c1 = 2a1 + 3b1
        # c2 = 2a2 + 3b1
        # c3 = 2a2 + 3b2
        # c4 = 2a1 + 3b2
        # Are they distinct?
        # c1 == c2 => 2a1 = 2a2 => a1=a2 (contradiction)
        # c1 == c3 => 2a1+3b1 = 2a2+3b2. Possible?
        # c1 == c4 => 3b1 = 3b2 => b1=b2 (contradiction)
        # c2 == c3 => 3b1 = 3b2 => b1=b2 (contradiction)
        # c2 == c4 => 2a2+3b1 = 2a1+3b2.
        # So we only need to worry about c1==c3 and c2==c4.
        # c1==c3 => 2(a1-a2) = 3(b2-b1).
        # Since a1,a2 in 0..9, diff in -9..9.
        # b1,b2 in 0..9 (offset), diff in -9..9.
        # 2*da = 3*db.
        # If da=3, db=2. 2*3=6, 3*2=6. Match!
        # So if a1-a2=3 and b2-b1=2, colors collide.
        # This simple linear scheme is NOT valid.
        
        # We need a better scheme or just use random + repair for the bipartite part too.
        # Or use a Latin Square based coloring which is known to be orthogonal.
        # Let's just use the greedy method on the bipartite edges first.
        
        bip_edges = [(a, b) for a in range(10) for b in range(10, 20)]
        random.shuffle(bip_edges)
        
        for u, v in bip_edges:
            forbidden = get_forbidden_colors(u, v, adj, color_mat, neighbors)
            valid_colors = set(range(num_colors)) - forbidden
            if not valid_colors:
                # If bipartite edge fails, it's a problem.
                # But K_{10,10} should be easy to color with 20 colors.
                continue
            c = random.choice(list(valid_colors))
            adj[u][v] = 1
            adj[v][u] = 1
            color_mat[u][v] = c
            color_mat[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
            edges.append([u, v, c])
            
        return adj, color_mat, neighbors, edges

    def run_full_search(seed):
        random.seed(seed)
        
        # 1. Build Bipartite Core
        adj, color_mat, neighbors, edges = construct_bipartite_core()
        
        # 2. Add Intra-Part Edges
        intra_edges = []
        for i in range(10):
            for j in range(i+1, 10):
                intra_edges.append((i, j))
        for i in range(10):
            for j in range(i+1, 10):
                intra_edges.append((10+i, 10+j))
        
        random.shuffle(intra_edges)
        
        for u, v in intra_edges:
            forbidden = get_forbidden_colors(u, v, adj, color_mat, neighbors)
            valid_colors = set(range(num_colors)) - forbidden
            if not valid_colors:
                continue
            c = random.choice(list(valid_colors))
            adj[u][v] = 1
            adj[v][u] = 1
            color_mat[u][v] = c
            color_mat[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
            edges.append([u, v, c])
            
        # 3. Local Search: Try to add remaining edges
        # We can try to add edges that were skipped in bip or intra.
        # But as noted, adding edges only restricts.
        # So we are stuck with the greedy result.
        
        # To improve, we need to allow **edge swaps** or **recoloring**.
        # Let's try a simple hill climbing: 
        # Pick a random edge, remove it, try to add 2 new edges?
        # Or just return the current best.
        
        return edges

    best_edges = []
    best_count = 0
    
    restarts = 0
    while time.time() < deadline:
        edges = run_full_search(seed=restarts)
        
        if is_valid_edges(edges):
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges
                if best_count > 150:
                    break
        restarts += 1
        
    return best_edges
