import time
import math
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin
    
    N = 256
    
    # Squares up to 2*256 = 512
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Identify valid nodes (no self-conflict: 2v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
    
    n = len(valid_nodes)
    node_index = {v: i for i, v in enumerate(valid_nodes)}
    
    # Build conflict edges (MIS on this graph)
    conflicts = [[] for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if valid_nodes[i] + valid_nodes[j] in squares:
                conflicts[i].append(j)
                conflicts[j].append(i)
                edges.append((i, j))
    
    if not edges:
        return valid_nodes
    
    # Build complement graph adjacency as bitmasks
    # In complement graph, edge (i,j) exists if i,j are NOT in conflict
    # MIS on conflict graph = Max Clique on complement graph
    comp_adj = [0] * n
    for i in range(n):
        # Start with all nodes except self
        mask = 0
        for j in range(n):
            if j != i:
                mask |= (1 << j)
        # Remove conflict neighbors (they are NOT adjacent in complement)
        for j in conflicts[i]:
            mask &= ~(1 << j)
        comp_adj[i] = mask
    
    # Tomita's Maximum Clique algorithm with greedy coloring
    best_clique = [0]  # size
    best_clique_mask = [0]  # bitmask of the clique
    
    def greedy_color_order(P, color_bound):
        """
        Greedily color vertices in P. Returns list of (vertex, color) sorted by color.
        The last color number is the upper bound on clique size in P.
        """
        uncolored = list(P)
        colors = {}
        color_num = 0
        
        while uncolored:
            color_num += 1
            colored_this_round = []
            still_uncolored = []
            
            for v in uncolored:
                # Check if v conflicts with any already colored in this round
                can_color = True
                for u in colored_this_round:
                    if (comp_adj[v] & (1 << u)) == 0:
                        can_color = False
                        break
                if can_color:
                    colors[v] = color_num
                    colored_this_round.append(v)
                else:
                    still_uncolored.append(v)
            uncolored = still_uncolored
        
        # Sort by color (ascending), then by index
        sorted_list = sorted(colors.items(), key=lambda x: (x[1], x[0]))
        return sorted_list, color_num
    
    def max_clique(P, R):
        """
        Branch and bound for maximum clique.
        P: set of candidate vertices (list of ints)
        R: current clique (list of ints)
        """
        if time.time() > deadline:
            return
        
        if not P:
            if len(R) > best_clique[0]:
                best_clique[0] = len(R)
                mask = 0
                for v in R:
                    mask |= (1 << v)
                best_clique_mask[0] = mask
            return
        
        # Upper bound: |R| + |P|
        if len(R) + len(P) <= best_clique[0]:
            return
        
        # Greedy coloring for tighter bound
        ordered, color_bound = greedy_color_order(P, color_bound)
        
        if len(R) + color_bound <= best_clique[0]:
            return
        
        # Process vertices in reverse color order (highest color first)
        for v, c in reversed(ordered):
            if time.time() > deadline:
                return
            
            # Upper bound check with this vertex's color
            if len(R) + c <= best_clique[0]:
                return
            
            # Branch: include v
            new_R = R + [v]
            # New P: vertices in P that are adjacent to v and come after v in ordering
            new_P = []
            for u, cu in ordered:
                if cu < c:  # only vertices with lower color (processed later)
                    if (comp_adj[v] & (1 << u)) != 0:
                        new_P.append(u)
            
            max_clique(new_P, new_R)
    
    # Initial greedy clique for lower bound
    def greedy_clique():
        """Greedy construction of a large clique."""
        P = list(range(n))
        R = []
        while P:
            # Pick vertex with max degree in P
            best_v = -1
            best_deg = -1
            for v in P:
                deg = (comp_adj[v] & sum((1 << u) for u in P)).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            R.append(best_v)
            P = [u for u in P if u != best_v and (comp_adj[best_v] & (1 << u)) != 0]
        return R
    
    initial = greedy_clique()
    best_clique[0] = len(initial)
    mask = 0
    for v in initial:
        mask |= (1 << v)
    best_clique_mask[0] = mask
    
    # Run Tomita's algorithm
    max_clique(list(range(n)), [])
    
    # Extract result
    result_mask = best_clique_mask[0]
    result_nodes = []
    for i in range(n):
        if result_mask & (1 << i):
            result_nodes.append(valid_nodes[i])
    
    if not result_nodes:
        result_nodes = [valid_nodes[0]]
    
    # Verify
    valid = True
    for i in range(len(result_nodes)):
        for j in range(i, len(result_nodes)):
            s = result_nodes[i] + result_nodes[j]
            r = math.isqrt(s)
            if r * r == s:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: greedy MIS
        result_set = set()
        for v in valid_nodes:
            can_add = True
            for u in result_set:
                if u + v in squares or 2 * v in squares:
                    can_add = False
                    break
            if can_add and 2 * v not in squares:
                result_set.add(v)
        result_nodes = list(result_set)
        if not result_nodes:
            result_nodes = [valid_nodes[0]]
    
    return result_nodes
