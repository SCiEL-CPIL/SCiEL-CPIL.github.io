"""Solve maximum weight clique in complement graph where edge exists if lcm(a,b)>256."""
# EVOLVE-BLOCK-START
import time
from math import lcm, gcd
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit
    
    # Nodes are 2..256
    nodes = list(range(2, 257))
    n = len(nodes)  # 255 nodes
    
    # Weight for each node: 1/x
    weights = [Fraction(1, x) for x in nodes]
    
    # Build adjacency: edge between i,j if lcm(nodes[i], nodes[j]) > 256
    # We want a clique in this graph (all pairs have lcm > 256)
    # Equivalently, in the complement, no two selected nodes have lcm <= 256
    
    # Precompute incompatibility: incompatible[i][j] = True if lcm(nodes[i],nodes[j]) <= 256
    # We need a set with NO incompatible pairs -> independent set in incompatibility graph
    # Or equivalently, clique in the "compatible" graph
    
    # For efficiency, use bitsets
    # incompatible_mask[i] is a bitset where bit j is set if nodes[i] and nodes[j] are incompatible (lcm <= 256)
    incompatible_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if lcm(nodes[i], nodes[j]) <= 256:
                incompatible_mask[i] |= (1 << j)
                incompatible_mask[j] |= (1 << i)
    
    # Greedy coloring for upper bound in branch and bound
    # We solve max weight independent set in the incompatibility graph
    # Using branch and bound with greedy coloring
    
    # Order nodes by weight descending for better pruning
    order = sorted(range(n), key=lambda i: -weights[i])
    
    # Reindex
    # new_index[i] = position of node i in the ordered list
    # We'll work with the ordered list
    
    # Let's use a different approach: branch and bound for maximum weight independent set
    # State: (candidates, current_set, current_weight)
    # candidates: list of node indices (in original indexing) that can still be added
    # We use bitset for candidates
    
    # Actually, let's use a recursive branch and bound with the following:
    # - candidates is a bitset of nodes that are still available
    # - We process nodes in order of decreasing weight
    # - At each step, we either include the current node or exclude it
    # - Upper bound: current_weight + sum of weights of all remaining candidates (optimistic)
    # - Better upper bound: use greedy coloring on the subgraph induced by candidates
    
    # For the coloring-based bound, we need to color the incompatibility subgraph
    # The max weight independent set <= sum over colors of (max weight in that color class)
    # Actually for unweighted, it's the number of color classes. For weighted, we need:
    # bound = sum over color classes c of (sum of weights of nodes in c that are in candidates)
    # No wait, that's not right either. The standard bound for MWIS with coloring is:
    # If we color the graph such that nodes in the same color class are mutually adjacent (incompatibility),
    # then at most one node from each color class can be in the independent set.
    # So the upper bound is sum over color classes of max weight in that class.
    
    # Let me implement a cleaner branch and bound
    
    best_weight = Fraction(0)
    best_set = set()
    
    # Precompute for each node, the set of compatible nodes (not incompatible)
    # compatible_mask[i]: bit j set if nodes i and j are compatible (lcm > 256) or i==j
    compatible_mask = [0] * n
    for i in range(n):
        compatible_mask[i] = ((1 << n) - 1) ^ incompatible_mask[i]
        # Don't set bit for itself if we don't want self-loops, but for independent set we just need no edges
    
    # Actually for independent set, we need: if we select node i, we cannot select any node j where incompatible_mask[i] has bit j set.
    
    # Let's do branch and bound with the following strategy:
    # Process nodes in order of decreasing weight
    # State: (idx, available_mask, current_set, current_weight)
    # available_mask: bitset of nodes that are still available to be selected
    # We consider nodes in the order list
    
    # For the upper bound, we use a greedy coloring heuristic on the subgraph of available nodes
    # This gives us an upper bound on the additional weight we can get
    
    def greedy_color_bound(available_mask):
        """Compute an upper bound on the weight of an independent set from available_mask.
        Use greedy coloring: color the incompatibility subgraph induced by available nodes.
        Upper bound = sum over color classes of max weight in that class."""
        # Get list of available nodes
        avail_nodes = []
        for i in range(n):
            if available_mask & (1 << i):
                avail_nodes.append(i)
        
        if not avail_nodes:
            return Fraction(0)
        
        # Greedy coloring: assign colors greedily
        color_of = {}
        color_weights = {}  # color -> max weight in that color class
        
        for node in avail_nodes:
            # Find a color that doesn't conflict
            # A color conflicts if there's a node in that color class that is incompatible with this node
            used_colors = set()
            for other_node, c in color_of.items():
                if incompatible_mask[node] & (1 << other_node):
                    used_colors.add(c)
            
            # Find smallest available color
            c = 0
            while c in used_colors:
                c += 1
            
            color_of[node] = c
            w = weights[node]
            if c not in color_weights or w > color_weights[c]:
                color_weights[c] = w
        
        # Upper bound is sum of max weights per color class
        bound = Fraction(0)
        for c in color_weights:
            bound += color_weights[c]
        
        return bound
    
    # Branch and bound
    def branch_and_bound(order_idx, available_mask, current_set, current_weight):
        nonlocal best_weight, best_set
        
        if time.time() > deadline:
            return
        
        # If no more nodes to consider
        if order_idx >= n:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = set(current_set)
            return
        
        # Upper bound: current_weight + greedy_color_bound(available_mask)
        ub = current_weight + greedy_color_bound(available_mask)
        if ub <= best_weight:
            return
        
        node = order[order_idx]
        
        # Option 1: Include node if it's available
        if available_mask & (1 << node):
            new_available = available_mask & compatible_mask[node]
            # Remove node itself from available (we're processing in order, so it won't be considered again)
            # Actually, we need to also remove nodes that are incompatible with node
            # compatible_mask[node] has bits set for nodes that are compatible (including self)
            # So new_available = available_mask & compatible_mask[node] removes incompatible nodes
            # But we also need to make sure we don't re-add node, which is handled by the order
            new_available &= ~(1 << node)  # remove node itself
            
            branch_and_bound(order_idx + 1, new_available, current_set | {node}, current_weight + weights[node])
        
        # Option 2: Exclude node
        branch_and_bound(order_idx + 1, available_mask, current_set, current_weight)
    
    # Initial call
    initial_mask = (1 << n) - 1
    branch_and_bound(0, initial_mask, set(), Fraction(0))
    
    # Convert best_set back to actual values
    result = [nodes[i] for i in best_set]
    result.sort()
    
    # Verify
    for a, b in combinations(result, 2):
        if lcm(a, b) <= 256:
            # Fallback: return a simple valid solution
            return [256]
    
    return result if result else [256]
# EVOLVE-BLOCK-END
