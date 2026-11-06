"""Dense 4-cycle-rainbow edge-coloured graph on 20 vertices.
Uses conflict-graph based maximum independent set with branch-and-bound."""
import time
import random
import math
from itertools import combinations

N = 20
DEADLINE_MARGIN = 8.0

def _precompute_cycles():
    """Precompute all 4-cycles and map each edge to the cycles it belongs to."""
    cycles = []
    edge_to_cycles = {e: [] for e in combinations(range(N), 2)}
    
    for a, b, c, d in combinations(range(N), 4):
        perms = [
            ((a,b), (b,c), (c,d), (d,a)),
            ((a,b), (b,d), (d,c), (c,a)),
            ((a,c), (c,b), (b,d), (d,a))
        ]
        for edges in perms:
            norm_edges = tuple(sorted((min(u,v), max(u,v)) for u,v in edges))
            cyc_id = len(cycles)
            cycles.append(norm_edges)
            for e in norm_edges:
                edge_to_cycles[e].append(cyc_id)
    return cycles, edge_to_cycles

def _build_conflict_graph(all_edges):
    """Build conflict graph where nodes are (edge, color) pairs.
    Two nodes conflict if they share a vertex or would create a non-rainbow 4-cycle."""
    # Node: (edge_idx, color) where edge_idx is index into all_edges
    num_nodes = len(all_edges) * N
    # Conflict adjacency as bitsets for speed
    conflicts = [0] * num_nodes
    
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    # Conflict 1: Two edges sharing a vertex cannot both be present (simple graph constraint)
    # Actually in a simple graph, we can have multiple edges from same vertex with different colors
    # The constraint is just that we pick at most one (edge, color) per edge
    # And that for every 4-cycle, all 4 edges have distinct colors
    
    # Let's think differently. The constraint is:
    # 1. For each edge e, we pick at most one color
    # 2. For every 4-cycle (e1, e2, e3, e4), the 4 colors must be distinct
    
    # This is a maximum independent set in a graph where:
    # - Nodes are (edge, color) pairs
    # - Two nodes conflict if:
    #   a) They are for the same edge (can only pick one color per edge)
    #   b) They are for two edges in some 4-cycle and have the same color
    
    # Build conflicts
    # For each edge, conflict all colors with each other
    for i in range(len(all_edges)):
        base = i * N
        for c1 in range(N):
            for c2 in range(c1 + 1, N):
                n1 = base + c1
                n2 = base + c2
                # Set bit in conflicts
                conflicts[n1] |= (1 << n2)
                conflicts[n2] |= (1 << n1)
    
    # For each 4-cycle, conflict same-color pairs
    cycles, edge_to_cycles = _precompute_cycles()
    for cyc_edges in cycles:
        for i in range(4):
            for j in range(i + 1, 4):
                e1 = cyc_edges[i]
                e2 = cyc_edges[j]
                idx1 = edge_to_idx[e1]
                idx2 = edge_to_idx[e2]
                for c in range(N):
                    n1 = idx1 * N + c
                    n2 = idx2 * N + c
                    conflicts[n1] |= (1 << n2)
                    conflicts[n2] |= (1 << n1)
    
    return conflicts, edge_to_idx

def _max_independent_set_bnb(conflicts, num_nodes, time_limit, initial_solution=None):
    """Maximum independent set using branch-and-bound with greedy upper bound."""
    t_start = time.time()
    deadline = t_start + time_limit
    
    # Greedy initial solution
    if initial_solution is None:
        initial_solution = _greedy_independent_set(conflicts, num_nodes)
    
    best_set = set(initial_solution)
    best_size = len(best_set)
    
    # Sort nodes by degree (ascending) for better branching
    degrees = [bin(conflicts[i]).count('1') for i in range(num_nodes)]
    node_order = sorted(range(num_nodes), key=lambda x: degrees[x])
    
    # Branch and bound
    def _branch(node_idx, current_set, current_size, remaining):
        nonlocal best_set, best_size
        nonlocal deadline
        
        if time.time() > deadline:
            return
        
        if node_idx == num_nodes:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return
        
        node = node_order[node_idx]
        
        # Upper bound: current_size + number of remaining nodes that don't conflict with current_set
        # Simple bound: current_size + (num_nodes - node_idx)
        if current_size + (num_nodes - node_idx) <= best_size:
            return
        
        # Try including node
        new_current = list(current_set)
        new_current.append(node)
        _branch(node_idx + 1, new_current, current_size + 1, remaining - 1)
        
        # Try excluding node
        _branch(node_idx + 1, current_set, current_size, remaining)
    
    # This naive BnB will be too slow. Let's use a better approach.
    # Use local search instead.
    return best_set

def _greedy_independent_set(conflicts, num_nodes):
    """Greedy: pick nodes in order of lowest degree, skip if conflicts with already picked."""
    picked = set()
    remaining = set(range(num_nodes))
    
    while remaining:
        # Find node with minimum conflicts to remaining
        best_node = None
        min_conf = float('inf')
        for node in list(remaining)[:100]:  # Sample for speed
            conf_count = 0
            mask = conflicts[node]
            for r in remaining:
                if mask >> r & 1:
                    conf_count += 1
            if conf_count < min_conf:
                min_conf = conf_count
                best_node = node
        
        if best_node is not None:
            picked.add(best_node)
            # Remove best_node and all its neighbors from remaining
            mask = conflicts[best_node]
            to_remove = set()
            for r in remaining:
                if r == best_node or (mask >> r & 1):
                    to_remove.add(r)
            remaining -= to_remove
    
    return list(picked)

def _local_search_improve(conflicts, num_nodes, initial_set, time_limit):
    """Improve independent set via local search: add nodes, remove conflicts."""
    t_start = time.time()
    deadline = t_start + time_limit
    
    current_set = set(initial_set)
    best_set = set(current_set)
    best_size = len(current_set)
    
    # Build reverse conflict map
    reverse_conflicts = [set() for _ in range(num_nodes)]
    for i in range(num_nodes):
        mask = conflicts[i]
        for j in range(num_nodes):
            if mask >> j & 1:
                reverse_conflicts[i].add(j)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding each unpicked node
        for node in range(num_nodes):
            if node in current_set:
                continue
            
            # Check if node conflicts with any in current_set
            conflicts_with_current = []
            mask = conflicts[node]
            for cn in current_set:
                if mask >> cn & 1:
                    conflicts_with_current.append(cn)
            
            # If we can add node and remove fewer than 1 node, it's a win
            # Actually, we need to remove all conflicts. So gain = 1 - len(conflicts_with_current)
            if len(conflicts_with_current) == 0:
                current_set.add(node)
                improved = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
        
        # Try removing a node to allow adding multiple others
        for node in list(current_set):
            # If we remove node, which new nodes can we add?
            potential_adds = []
            for other in range(num_nodes):
                if other in current_set or other == node:
                    continue
                # Check if other only conflicts with node in current_set
                mask = conflicts[other]
                only_conflicts_with_node = True
                for cn in current_set:
                    if cn != node and (mask >> cn & 1):
                        only_conflicts_with_node = False
                        break
                if only_conflicts_with_node:
                    potential_adds.append(other)
            
            # If we remove 1 and add > 1, it's a win
            if len(potential_adds) > 1:
                current_set.discard(node)
                for add in potential_adds:
                    current_set.add(add)
                improved = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
    
    return best_set

def _extract_edges_from_set(node_set, all_edges):
    """Extract (edge, color) pairs from independent set node indices."""
    result = {}
    for node in node_set:
        edge_idx = node // N
        color = node % N
        e = all_edges[edge_idx]
        result[e] = color
    return result

def construct():
    t0 = time.time()
    deadline = 150.0 - DEADLINE_MARGIN
    
    all_edges = list(combinations(range(N), 2))
    cycles, edge_to_cycles = _precompute_cycles()
    
    # Build conflict graph
    conflicts, edge_to_idx = _build_conflict_graph(all_edges)
    num_nodes = len(all_edges) * N
    
    # Greedy initial solution
    initial_set = _greedy_independent_set(conflicts, num_nodes)
    
    # Local search improvement
    best_set = _local_search_improve(conflicts, num_nodes, initial_set, deadline - time.time() - 2.0)
    
    # Extract result
    result_colors = _extract_edges_from_set(best_set, all_edges)
    result = [[u, v, c] for (u, v), c in sorted(result_colors.items())]
    
    # Verify and try to improve with random restarts
    best_count = len(result)
    
    while time.time() < deadline - 2.0:
        # Random perturbation
        perturbed = set(best_set)
        for _ in range(random.randint(5, 20)):
            if perturbed:
                perturbed.discard(random.choice(tuple(perturbed)))
            # Add a random node
            n = random.randint(0, num_nodes - 1)
            perturbed.add(n)
        
        # Local search from perturbed
        improved_set = _local_search_improve(conflicts, num_nodes, perturbed, 2.0)
        
        if len(improved_set) > best_count:
            best_count = len(improved_set)
            best_set = improved_set
            result_colors = _extract_edges_from_set(best_set, all_edges)
            result = [[u, v, c] for (u, v), c in sorted(result_colors.items())]
    
    return result if result else [[0, v, 0] for v in range(1, N)]

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} edges")
