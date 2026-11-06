import time
import random
import sys

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 145.0

    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # We will maintain:
    # 1. edge_colors: dict {(u,v): color}
    # 2. color_graphs: list of sets of edges per color, to check C4-freeness efficiently
    #    Actually, checking C4-freeness on the fly for 20 vertices is cheap.
    #    A graph is C4-free if no 4 vertices form a cycle.
    #    For a specific color c, the subgraph G_c must be C4-free.
    #    
    #    The condition "Every 4-cycle is rainbow" is equivalent to:
    #    1. For every color c, the subgraph induced by edges of color c is C4-free.
    #       (Because if a 4-cycle used a color twice, it wouldn't be rainbow. 
    #        Specifically, if edges (u,v) and (w,x) have same color c, and they are in a 4-cycle, 
    #        the cycle is not rainbow. 
    #        Wait, the condition is that EVERY 4-cycle in the *union* graph must be rainbow.
    #        This implies that no 4-cycle can have two edges of the same color.
    #        This is exactly equivalent to saying that for each color c, the subgraph G_c 
    #        contains no 4-cycle.
    #        Why? If G_c contains a 4-cycle, that 4-cycle in the main graph has 4 edges of color c, 
    #        so it is not rainbow.
    #        Conversely, if G_c is C4-free for all c, can a 4-cycle in the main graph be non-rainbow?
    #        A 4-cycle is non-rainbow if it has at least two edges of the same color.
    #        Let the 4-cycle be u-v-w-x-u. 
    #        If (u,v) and (w,x) have the same color c, then u,v,w,x form a 4-cycle in G_c? 
    #        Not necessarily. (u,v) is in G_c, (w,x) is in G_c. But (v,w) and (x,u) might not be in G_c.
    #        So G_c does NOT necessarily contain the 4-cycle u-v-w-x-u.
    #        
    #        Let's re-read carefully: "every 4-cycle is rainbow".
    #        This means for any 4 distinct vertices a,b,c,d forming a cycle a-b-c-d-a in the main graph,
    #        the colors of (a,b), (b,c), (c,d), (d,a) are all distinct.
    #        
    #        This does NOT imply that each color class is C4-free in the sense of containing a 4-cycle 
    #        as a subgraph. It implies that no 4-cycle in the main graph can have repeated colors.
    #        
    #        However, if we restrict each color class to be a forest (which is C4-free), 
    #        does that help?
    #        If G_c is a forest, it has no cycles at all. 
    #        If G_c has no cycles, it certainly has no 4-cycles.
    #        But the violation condition is about 4-cycles in the MAIN graph.
    #        
    #        Let's test: Can we have a non-rainbow 4-cycle if all G_c are forests?
    #        Yes. Consider a 4-cycle a-b-c-d-a. 
    #        Colors: c_ab = 1, c_bc = 2, c_cd = 1, c_da = 3.
    #        Is this allowed? No, because c_ab == c_cd.
    #        Does G_1 contain a cycle? G_1 has edges (a,b) and (c,d). This is not a cycle.
    #        So G_1 is a forest.
    #        But the 4-cycle a-b-c-d-a is not rainbow.
    #        
    #        So "Color classes are forests" is a SUFFICIENT condition for validity?
    #        No, the example above shows it's NOT sufficient.
    #        The example has a non-rainbow 4-cycle, but all color classes are forests (single edges).
    #        
    #        Wait, the condition is: "Every 4-cycle (as a subgraph) must receive four distinct colours".
    #        My example: a-b (1), b-c (2), c-d (1), d-a (3).
    #        Colors are {1, 2, 1, 3}. Not distinct.
    #        So this configuration is INVALID.
    #        
    #        So, requiring color classes to be forests is NOT sufficient to guarantee validity.
    #        It prevents 4-cycles WITHIN a color class, but not 4-cycles in the main graph 
    #        that use a color twice on non-adjacent edges of the 4-cycle.
    #        
    #        However, if we require color classes to be C4-free, is that sufficient?
    #        If G_c has no 4-cycle, does it prevent the pattern above?
    #        In the pattern above, G_1 has edges (a,b) and (c,d). No 4-cycle in G_1.
    #        So C4-free color classes are also NOT sufficient.
    #        
    #        The correct necessary and sufficient condition is:
    #        For any 4 vertices a,b,c,d forming a cycle in G, the colors must be distinct.
    #        
    #        Let's stick to the direct check but optimize it.
    #        The previous program's check_conflict was O(deg(u)*deg(v)).
    #        With n=20, max degree is 19. So O(19*19) = 361 checks per edge addition.
    #        This is fast enough.
    #        
    #        Why did the previous program score 4.26?
    #        It used Simulated Annealing.
    #        The inspiration says a program with score 4.36 used "itertools, random, time".
    #        The diff for p73 (4.47) shows it restructured the SA loop with "principled conflict-counting".
    #        
    #        Let's try a different structural approach:
    #        Instead of random edge addition, let's try to construct a specific known dense structure.
    #        Or, use a smarter initialization: 
    #        Start with a complete graph K_20? No, K_20 has many 4-cycles.
    #        Start with a Turan graph?
    #        
    #        Let's try a "Greedy with Lookahead" or "Backtracking" approach.
    #        Given the small n=20, we can afford a more expensive check if it leads to higher density.
    #        
    #        Alternative Idea:
    #        The constraint is local to 4-cycles.
    #        We can model this as a Constraint Satisfaction Problem (CSP).
    #        Variables: Edges of the complete graph.
    #        Domain: {0, 1, ..., 19, None} (None means edge not present).
    #        Constraints: For every 4-cycle, if all 4 edges are present, their colors must be distinct.
    #        
    #        We want to maximize the number of edges not None.
    #        
    #        Let's use a local search that focuses on "Unblocking" moves.
    #        1. Start with a valid sparse graph.
    #        2. Identify "conflict-free" missing edges.
    #        3. Add them.
    #        4. If stuck, try to recolor existing edges to free up space for new edges.
    #        
    #        Key Insight for Speed:
    #        Precompute all 4-cycles in K_20.
    #        There are C(20,4) * 3 = 4845 * 3 = 14535 4-cycles.
    #        This is a small number.
    #        We can maintain a count of "violations" for each potential edge addition?
    #        No, because adding an edge affects many 4-cycles.
    #        
    #        Let's stick to the direct check but make it faster using bitmasks or sets.
    #        The current check is:
    #        for w in adj[v]:
    #            for x in adj[u]:
    #                if (w,x) in edge_colors:
    #                    check colors
    #        
    #        This is already quite fast for n=20.
    #        
    #        What if we use a different heuristic for SA?
    #        Current SA accepts worsening moves.
    #        Let's try "Iterated Local Search" (ILS).
    #        1. Greedy construction.
    #        2. Local Search (Hill Climbing) to maximize edges.
    #           - Try adding any missing edge with any color. If valid, add it.
    #           - Try changing color of any edge. If it allows adding a new edge later? 
    #             (This is hard to evaluate locally).
    #           - Just maximize edges: If we can add an edge, do it.
    #        3. Perturbation: Remove k random edges.
    #        4. Repeat Local Search.
    #        
    #        This is a standard approach for Max-Cut and similar problems.
    #        
    #        Let's implement ILS.
        
    def get_cycle_conflicts(u, v, c, edge_colors, adj):
        """
        Returns True if adding (u,v) with color c creates a non-rainbow 4-cycle.
        """
        # A 4-cycle involving (u,v) is u-v-w-x-u
        # Edges: (u,v), (v,w), (w,x), (x,u)
        # Colors: c, col(v,w), col(w,x), col(x,u)
        # Must be all distinct.
        
        # Optimization: 
        # Iterate over w in adj[v]
        # For each w, iterate over x in adj[u]
        # Check if (w,x) is an edge.
        
        for w in adj[v]:
            if w == u: continue
            cw = edge_colors.get((min(v, w), max(v, w)))
            if cw is None: continue
            if cw == c: return True # Duplicate color in cycle
            
            for x in adj[u]:
                if x == v or x == w: continue
                cx = edge_colors.get((min(w, x), max(w, x)))
                if cx is None: continue
                if cx == c or cx == cw: return True # Duplicate color in cycle
                
                cu = edge_colors.get((min(x, u), max(x, u)))
                if cu is None: continue
                if cu == c or cu == cw or cu == cx: return True
                
        return False

    def greedy_construct():
        adj = [[] for _ in range(n)]
        ec = {}
        edges = all_edges[:]
        random.shuffle(edges)
        
        for u, v in edges:
            # Try to find a valid color
            for c in range(num_colors):
                if not get_cycle_conflicts(u, v, c, ec, adj):
                    ec[(u, v)] = c
                    adj[u].append(v)
                    adj[v].append(u)
                    break
        return ec, adj

    def local_search(ec, adj):
        """
        Hill climbing: try to add any missing edge.
        """
        improved = True
        while improved:
            improved = False
            missing = [e for e in all_edges if e not in ec]
            random.shuffle(missing)
            
            for u, v in missing:
                if time.time() > deadline - 5:
                    return ec, adj
                
                for c in range(num_colors):
                    if not get_cycle_conflicts(u, v, c, ec, adj):
                        ec[(u, v)] = c
                        adj[u].append(v)
                        adj[v].append(u)
                        improved = True
                        break
                        # Note: We break out of color loop, but continue checking other missing edges
                        # This might be suboptimal as we don't re-evaluate other edges after each addition.
                        # But it's fast.
                        
        return ec, adj

    def perturb(ec, adj, k):
        """
        Remove k random edges.
        """
        if not ec:
            return ec, adj
        keys = list(ec.keys())
        random.shuffle(keys)
        for _ in range(min(k, len(keys))):
            e = keys[_]
            u, v = e
            del ec[e]
            adj[u].remove(v)
            adj[v].remove(u)
        return ec, adj

    best_ec = {}
    best_adj = [[] for _ in range(n)]
    best_count = 0
    
    # Initial construction
    ec, adj = greedy_construct()
    ec, adj = local_search(ec, adj)
    if len(ec) > best_count:
        best_count = len(ec)
        best_ec = dict(ec)
        best_adj = [list(l) for l in adj]
        
    # ILS Loop
    while time.time() < deadline - 10:
        # Perturb
        k = random.randint(5, 15)
        ec, adj = perturb(best_ec, best_adj, k)
        
        # Local Search
        ec, adj = local_search(ec, adj)
        
        if len(ec) > best_count:
            best_count = len(ec)
            best_ec = dict(ec)
            best_adj = [list(l) for l in adj]
            
        # Optional: Try to improve current state further with more aggressive local search?
        # The current local search just adds edges. It doesn't recolor.
        # Recoloring can sometimes unblock edge additions.
        # Let's add a "Recolor" step to the local search.

    # Enhanced Local Search with Recoloring
    def local_search_with_recolor(ec, adj, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Try adding edges
            missing = [e for e in all_edges if e not in ec]
            random.shuffle(missing)
            for u, v in missing:
                for c in range(num_colors):
                    if not get_cycle_conflicts(u, v, c, ec, adj):
                        ec[(u, v)] = c
                        adj[u].append(v)
                        adj[v].append(u)
                        improved = True
                        break

            # 2. Try recoloring edges to potentially enable future additions
            # This is heuristic: pick a random edge, try a random new color.
            # If it's valid, accept it (even if it doesn't immediately add an edge).
            # This explores the color space.
            if ec:
                e = random.choice(list(ec.keys()))
                u, v = e
                old_c = ec[e]
                new_c = random.randint(0, num_colors - 1)
                if new_c != old_c:
                    # Check if new color is valid
                    # We need to check conflicts for (u,v) with new_c
                    # But (u,v) is already in the graph.
                    # The check is: does changing (u,v) to new_c create a non-rainbow 4-cycle?
                    # This is similar to get_cycle_conflicts but we must exclude the edge (u,v) itself
                    # from the "existing" set during the check, or handle it carefully.
                    
                    # Let's write a specific check for recoloring
                    valid = True
                    # 4-cycle u-v-w-x-u. Edges: (u,v), (v,w), (w,x), (x,u).
                    # If we change (u,v) to new_c, we need {new_c, col(v,w), col(w,x), col(x,u)} to be distinct.
                    # Note: (u,v) is not in the "other" edges.
                    
                    for w in adj[v]:
                        if w == u: continue
                        cw = ec.get((min(v, w), max(v, w)))
                        if cw is None: continue
                        if cw == new_c:
                            valid = False
                            break
                            
                        for x in adj[u]:
                            if x == v or x == w: continue
                            cx = ec.get((min(w, x), max(w, x)))
                            if cx is None: continue
                            if cx == new_c or cx == cw:
                                valid = False
                                break
                            cu = ec.get((min(x, u), max(x, u)))
                            if cu is None: continue
                            if cu == new_c or cu == cw or cu == cx:
                                valid = False
                                break
                        if not valid:
                            break
                            
                    if valid:
                        ec[e] = new_c
                        improved = True # We consider a valid recolor as an improvement in search progress

        return ec, adj

    # Restart ILS with enhanced local search
    best_ec = {}
    best_adj = [[] for _ in range(n)]
    best_count = 0
    
    num_restarts = 10
    time_per_restart = (deadline - time.time() - 5) / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # 1. Greedy
        ec, adj = greedy_construct()
        
        # 2. Enhanced Local Search
        ec, adj = local_search_with_recolor(ec, adj, time.time() + time_per_restart - 2)
        
        if len(ec) > best_count:
            best_count = len(ec)
            best_ec = dict(ec)
            best_adj = [list(l) for l in adj]

    result = []
    for (u, v), c in best_ec.items():
        result.append([u, v, c])
        
    return result

