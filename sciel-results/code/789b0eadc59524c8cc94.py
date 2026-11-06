"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin
    
    N = 512
    numbers = list(range(1, N + 1))
    n = len(numbers)
    
    # Build conflict graph: edge between i,j if lcm(numbers[i], numbers[j]) > 512
    # We want maximum independent set in this graph (no two adjacent)
    # Equivalently, maximum clique in the complement graph
    
    # For Tomita's algorithm, we work with the graph where we want max clique.
    # Let's define: in the "compatibility" graph, two nodes are connected if they CAN coexist (lcm <= 512).
    # Then we want maximum clique in the compatibility graph.
    
    # Build adjacency as bitmasks for efficiency
    # adj[i] is a bitmask of nodes compatible with node i
    
    adj = [0] * n
    for i in range(n):
        ai = numbers[i]
        for j in range(i + 1, n):
            bj = numbers[j]
            g = gcd(ai, bj)
            lcm_val = ai * bj // g
            if lcm_val <= 512:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Now find maximum clique in this graph using Tomita's algorithm
    # Use recursive branch and bound with coloring heuristic
    
    best_clique = []
    best_size = 0
    
    # Convert to Python integers for bitmask operations (Python ints are arbitrary precision)
    # For n=512, we need 512-bit integers, which Python handles natively
    
    def color_sort(candidates):
        """Color sort for greedy coloring heuristic. Returns (sorted_list, colors)"""
        # candidates is a set or list of node indices
        if not candidates:
            return [], []
        
        cand_list = list(candidates)
        # Greedy coloring: assign smallest possible color
        color = {}
        used_colors = [set() for _ in range(64)]  # max color estimate
        
        # Order by degree in candidates subgraph (descending) for better coloring
        deg = {}
        for v in cand_list:
            deg[v] = 0
        for v in cand_list:
            for u in cand_list:
                if v != u and (adj[v] >> u) & 1:
                    deg[v] += 1
        
        # Sort by degree descending
        cand_sorted = sorted(cand_list, key=lambda x: -deg[x])
        
        color_assignment = []
        for v in cand_sorted:
            # Find smallest color not used by neighbors of v that are already colored
            neighbor_colors = set()
            for u in color:
                if (adj[v] >> u) & 1:
                    neighbor_colors.add(color[u])
            c = 0
            while c in neighbor_colors:
                c += 1
            color[v] = c
            color_assignment.append((v, c))
        
        # Sort by color ascending (nodes with lower color first)
        color_assignment.sort(key=lambda x: x[1])
        sorted_nodes = [v for v, c in color_assignment]
        colors = [c for v, c in color_assignment]
        
        return sorted_nodes, colors
    
    def tomita(R, P, size_R):
        """
        R: current clique (list of nodes)
        P: candidates (list of nodes)
        size_R: current size of R
        """
        nonlocal best_clique, best_size, start_time, deadline
        
        if time.time() > deadline:
            return
        
        if not P:
            if size_R > best_size:
                best_size = size_R
                best_clique = list(R)
            return
        
        # Color sort P
        sorted_P, colors = color_sort(set(P))
        
        # Process in reverse order (highest color first)
        for idx in range(len(sorted_P) - 1, -1, -1):
            v = sorted_P[idx]
            c = colors[idx]
            
            # Upper bound: size_R + c
            if size_R + c <= best_size:
                return  # Prune
            
            if time.time() > deadline:
                return
            
            # Add v to clique
            R.append(v)
            
            # New candidates: intersection of P[0:idx] with neighbors of v
            new_P = []
            for i in range(idx):
                u = sorted_P[i]
                if (adj[v] >> u) & 1:
                    new_P.append(u)
            
            tomita(R, new_P, size_R + 1)
            
            R.pop()
            P.remove(v)  # This is O(n), but necessary
    
    # Start Tomita's algorithm
    # Initial candidates: all nodes
    # To improve, we can order nodes by degree
    all_nodes = list(range(n))
    # Sort by degree descending
    all_nodes.sort(key=lambda x: -bin(adj[x]).count('1'))
    
    tomita([], all_nodes, 0)
    
    # If we found a clique, return those numbers
    if best_clique:
        return [numbers[i] for i in best_clique]
    
    # Fallback: return seed
    return list(range(1, 23))
# EVOLVE-BLOCK-END
