"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    
    # Compute all perfect squares up to 2*N
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Filter self-conflicts: 2*v must not be a perfect square
    valid_nodes = [v for v in range(1, N + 1) if 2 * v not in squares]
    node_set = set(valid_nodes)
    
    # Build adjacency list: adj[v] = list of u in valid_nodes such that u != v and u+v in squares
    adj = {v: [] for v in valid_nodes}
    for idx_i, v in enumerate(valid_nodes):
        for u in valid_nodes[idx_i + 1:]:
            if v + u in squares:
                adj[v].append(u)
                adj[u].append(v)
    
    best_set = set()
    best_size = 0
    
    def greedy_min_degree(node_list, adj_map):
        """Greedy Minimum Degree Independent Set"""
        remaining = set(node_list)
        chosen = []
        # Use a copy of adjacency for efficiency
        while remaining:
            # Find node with minimum degree within remaining
            min_deg = float('inf')
            min_node = None
            for v in remaining:
                deg = sum(1 for u in adj_map[v] if u in remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_node = v
                    if deg == 0:
                        break
            chosen.append(min_node)
            remaining.discard(min_node)
            # Remove neighbors from consideration (they can't be in independent set)
            for u in adj_map[min_node]:
                remaining.discard(u)
        return set(chosen)
    
    def local_search(initial_set, adj_map, node_list):
        """Local search: try to add nodes, then swap to improve"""
        current = set(initial_set)
        current_list = list(current)
        
        # Try to add any valid node that doesn't conflict
        for v in node_list:
            if v not in current:
                if all(u not in current for u in adj_map[v]):
                    current.add(v)
        
        # Try swaps: remove one, add two
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_list = list(current)
            for i, v_remove in enumerate(current_list):
                # Find nodes that conflict with v_remove
                conflicting = [u for u in adj_map[v_remove] if u in current]
                # Try removing v_remove and adding some conflicting nodes
                for u1 in conflicting:
                    new_set = current - {v_remove}
                    new_set.add(u1)
                    # Try to add another node
                    for v2 in node_list:
                        if v2 not in new_set and v2 != v_remove:
                            if all(u not in new_set for u in adj_map[v2]):
                                if len(new_set + {v2}) > len(current):
                                    current = new_set | {v2}
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    break
        return current
    
    # Multiple restarts with randomized greedy
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Randomize the order of nodes
        shuffled = valid_nodes[:]
        random.shuffle(shuffled)
        
        # Greedy MIS on shuffled order
        remaining = set(shuffled)
        chosen = []
        while remaining:
            # Pick node with minimum degree in remaining
            min_deg = float('inf')
            min_node = None
            for v in remaining:
                deg = sum(1 for u in adj[v] if u in remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_node = v
                    if deg == 0:
                        break
            chosen.append(min_node)
            remaining.discard(min_node)
            for u in adj[min_node]:
                remaining.discard(u)
        
        current_set = set(chosen)
        
        # Local search improvement
        current_set = local_search(current_set, adj, valid_nodes)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
