import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin against 90s limit

    N = 256
    all_nodes = list(range(2, N + 1))
    n_total = len(all_nodes)
    node_to_idx = {v: i for i, v in enumerate(all_nodes)}
    idx_to_node = all_nodes

    # Precompute conflict masks: conflict if lcm(a,b) <= 256
    # lcm(a,b) = a*b/gcd(a,b) <= 256  <=>  a*b <= 256*gcd(a,b)
    conflict_mask = [0] * n_total
    for i in range(n_total):
        a = all_nodes[i]
        mask = 0
        for j in range(n_total):
            if i == j:
                continue
            b = all_nodes[j]
            if a * b <= 256 * gcd(a, b):
                mask |= (1 << j)
        conflict_mask[i] = mask

    # Complement: compatible if lcm(a,b) > 256
    # In the clique problem, we want a set where every pair is compatible
    # So the "graph" for max clique has edge (i,j) iff NOT (conflict)
    # i.e., compatible_mask[i] has bit j set if lcm(all_nodes[i], all_nodes[j]) > 256

    compatible_mask = [0] * n_total
    for i in range(n_total):
        mask = 0
        for j in range(n_total):
            if i == j:
                continue
            if not (conflict_mask[i] >> j) & 1:
                mask |= (1 << j)
        compatible_mask[i] = mask

    # Weight of each node
    weights = [1.0 / all_nodes[i] for i in range(n_total)]

    # Greedy coloring for upper bound: color the candidate set so that
    # nodes with same color are mutually conflicting (independent set in compat graph)
    # This gives an upper bound on the max clique weight in the subgraph.
    # Actually for weighted max clique, we use a greedy coloring where each color class
    # is an independent set in the compatibility graph (i.e., a clique in conflict graph).
    # The number of colors gives an upper bound on the cardinality of a clique.
    # For weighted, we use a weighted coloring bound.

    # Let's implement a branch-and-bound for maximum weight clique in the compatibility graph.
    # The compatibility graph: edge (i,j) exists iff lcm(all_nodes[i], all_nodes[j]) > 256.
    # We want the maximum weight clique.

    # Sort nodes by weight descending for better branching
    order_by_weight = sorted(range(n_total), key=lambda i: weights[i], reverse=True)

    # Precompute for each node, the set of compatible nodes with higher weight (for branching)
    # Actually, let's use the standard maximum clique B&B approach.

    # For the B&B, we work with the candidate set P (nodes that can be added to the current clique)
    # and the current clique C. The bound is the sum of weights of nodes in P (crude) or
    # a coloring-based bound.

    # Let's implement a proper weighted maximum clique solver.
    # We'll use a recursive branch-and-bound with:
    # 1. Greedy coloring for upper bound
    # 2. Branch on nodes in P ordered by weight

    # For coloring: we color nodes in P such that adjacent nodes (compatible) get different colors.
    # Wait, no. In maximum clique, we want to find a set of mutually compatible nodes.
    # The standard approach: we build a clique incrementally. At each step, P is the set of
    # nodes in the original graph that are compatible with all nodes in the current clique C.
    # We color P using a greedy algorithm where we assign colors such that nodes with the same
    # color are NOT mutually compatible (i.e., they form an independent set in the compat graph,
    # or a clique in the conflict graph). Then the maximum weight clique in P has weight at most
    # the sum over colors of the max weight in each color class... 

    # Actually, the standard bound for unweighted max clique is: color P greedily (each color class
    # is an independent set in the graph = a set of mutually incompatible nodes). Then the max clique
    # in P can have at most one node from each color class, so the clique size <= number of colors.
    # For weighted: the max weight clique in P has weight <= sum over color classes of (max weight in that class).
    # This is because a clique can pick at most one node from each independent set (color class).

    # Let's implement this.

    # Greedy coloring of a set of nodes (given as a bitmask or list)
    def greedy_color_bound(nodes_list):
        """
        nodes_list: list of node indices
        Returns an upper bound on the max weight clique in the subgraph induced by nodes_list.
        Uses greedy coloring where each color class is an independent set in the compatibility graph.
        """
        if not nodes_list:
            return 0.0
        
        # Sort by weight descending for better coloring
        sorted_nodes = sorted(nodes_list, key=lambda i: weights[i], reverse=True)
        
        # colors: list of lists, each inner list is a color class (independent set in compat graph)
        # A color class can contain nodes that are mutually incompatible (no compat edge between them)
        colors = []  # each element is a set of node indices
        
        for node in sorted_nodes:
            # Try to assign node to an existing color class where it's compatible with NO node in that class
            # Wait, I have it backwards. Let me re-think.
            
            # In max clique B&B: we have a graph G (compatibility graph). We want max weight clique.
            # P is the candidate set. We color P such that each color class is an independent set in G.
            # Independent set in G = set of nodes with no edges between them = mutually incompatible.
            # So a color class is a set of mutually incompatible nodes.
            # A clique in G can have at most one node from each color class.
            # Therefore, max weight clique weight <= sum over color classes of max weight in that class.
            
            # To assign node to a color class: node must be INCOMPATIBLE with all nodes already in that class.
            # i.e., for all v in color_class, NOT compatible(node, v), i.e., (compatible_mask[node] >> v) & 1 == 0
            
            placed = False
            for c_idx, color_class in enumerate(colors):
                can_place = True
                for v in color_class:
                    if (compatible_mask[node] >> v) & 1:
                        can_place = False
                        break
                if can_place:
                    color_class.add(node)
                    placed = True
                    break
            
            if not placed:
                colors.append({node})
        
        # Upper bound: sum of max weight in each color class
        bound = 0.0
        for color_class in colors:
            max_w = max(weights[v] for v in color_class)
            bound += max_w
        
        return bound

    # Branch and bound for max weight clique
    best_clique = []
    best_weight = 0.0
    best_weight_exact = Fraction(0, 1)

    # We'll use a recursive approach with a global best
    # To avoid recursion depth issues, we'll use an explicit stack or increase recursion limit
    import sys
    sys.setrecursionlimit(10000)

    def bnb(clique, P, current_weight, current_weight_frac):
        """
        clique: list of nodes in current clique
        P: set of candidate nodes (compatible with all in clique)
        current_weight: float weight of current clique
        current_weight_frac: Fraction exact weight
        """
        nonlocal best_clique, best_weight, best_weight_exact
        
        # Update best if current clique is better
        if current_weight > best_weight:
            best_weight = current_weight
            best_clique = clique[:]
            best_weight_exact = current_weight_frac

        if not P:
            return

        # Bound: if current_weight + bound(P) <= best_weight, prune
        # Use a quick bound first: sum of all weights in P
        # Actually, let's use the coloring bound
        P_list = list(P)
        if not P_list:
            return
        
        # Quick bound: sum of weights
        quick_bound = current_weight + sum(weights[v] for v in P_list)
        if quick_bound <= best_weight + 1e-15:
            return

        # Coloring bound
        color_bound = current_weight + greedy_color_bound(P_list)
        if color_bound <= best_weight + 1e-15:
            return

        # Time check
        if (time.time() - start_time) > deadline - 2:
            return

        # Branch: pick a node from P to include
        # Sort P by weight descending
        P_sorted = sorted(P_list, key=lambda i: weights[i], reverse=True)
        
        for node in P_sorted:
            # Include node in clique
            new_clique = clique + [node]
            new_weight = current_weight + weights[node]
            new_weight_frac = current_weight_frac + Fraction(1, all_nodes[node])
            
            # New P: nodes in P that are compatible with node
            new_P = set()
            for v in P:
                if v != node and (compatible_mask[node] >> v) & 1:
                    new_P.add(v)
            
            bnb(new_clique, new_P, new_weight, new_weight_frac)
            
            # Remove node from P for next iterations
            P.discard(node)
            
            # If P is empty, break
            if not P:
                break

    # Initial call: empty clique, P = all nodes
    # But we should be smarter: start with a good greedy solution to set initial best

    # Greedy initial solution
    def greedy_clique():
        """Greedy max weight clique: repeatedly pick the highest weight node compatible with all selected."""
        selected = []
        remaining = set(range(n_total))
        
        while remaining:
            # Pick the highest weight node
            best_node = max(remaining, key=lambda i: weights[i])
            selected.append(best_node)
            remaining.discard(best_node)
            # Remove all nodes incompatible with best_node
            to_remove = set()
            for v in remaining:
                if not ((compatible_mask[best_node] >> v) & 1):
                    to_remove.add(v)
            remaining -= to_remove
        
        return selected

    # Get initial greedy solution
    greedy_sol = greedy_clique()
    greedy_weight = sum(weights[i] for i in greedy_sol)
    greedy_weight_frac = sum(Fraction(1, all_nodes[i]) for i in greedy_sol)
    best_clique = greedy_sol
    best_weight = greedy_weight
    best_weight_exact = greedy_weight_frac

    # Also try a few different greedy orderings
    for _ in range(5):
        # Randomized greedy
        import random
        rng = random.Random(_ * 12345 + 67890)
        remaining = set(range(n_total))
        selected = []
        while remaining:
            # Pick with some randomness among top candidates
            sorted_rem = sorted(remaining, key=lambda i: weights[i] * rng.uniform(0.8, 1.2), reverse=True)
            best_node = sorted_rem[0]
            selected.append(best_node)
            remaining.discard(best_node)
            to_remove = set()
            for v in remaining:
                if not ((compatible_mask[best_node] >> v) & 1):
                    to_remove.add(v)
            remaining -= to_remove
        
        w = sum(weights[i] for i in selected)
        if w > best_weight:
            best_weight = w
            best_clique = selected
            best_weight_exact = sum(Fraction(1, all_nodes[i]) for i in selected)

    # Now run B&B
    # Start with P = all nodes, clique = empty
    # But to make it more efficient, we can start from the greedy solution
    # and try to improve by swapping

    # Actually, let's just run B&B from scratch with the initial bound from greedy
    # We need to be careful about the size of P initially (255 nodes)
    # The coloring bound should help prune

    # Let's run B&B
    P_initial = set(range(n_total))
    bnb([], P_initial, 0.0, Fraction(0, 1))

    # Local search polish: try to add nodes or swap
    def local_search_polish(selected_nodes):
        """Try to improve the solution via local search."""
        selected = selected_nodes[:]
        selected_set = set(selected)
        
        improved = True
        while improved and (time.time() - start_time) < deadline - 3:
            improved = False
            
            # Try adding nodes
            for i in range(n_total):
                if i in selected_set:
                    continue
                # Check if i is compatible with all in selected
                compatible_with_all = True
                for v in selected:
                    if not ((compatible_mask[i] >> v) & 1):
                        compatible_with_all = False
                        break
                if compatible_with_all:
                    selected.append(i)
                    selected_set.add(i)
                    improved = True
            
            # Try 1-swap: remove one, add one
            non_selected = [i for i in range(n_total) if i not in selected_set]
            for s_idx in range(len(selected)):
                s_node = selected[s_idx]
                # Check if we can replace s_node with a better non-selected node
                for n_node in non_selected:
                    if weights[n_node] <= weights[s_node]:
                        continue
                    # Check if n_node is compatible with all other selected nodes
                    compatible_with_others = True
                    for v in selected:
                        if v == s_node:
                            continue
                        if not ((compatible_mask[n_node] >> v) & 1):
                            compatible_with_others = False
                            break
                    if compatible_with_others:
                        selected[s_idx] = n_node
                        selected_set.discard(s_node)
                        selected_set.add(n_node)
                        improved = True
                        break
                if improved:
                    break
        
        return selected

    # Polish the best solution
    polished = local_search_polish(best_clique)
    polished_weight = sum(weights[i] for i in polished)
    if polished_weight > best_weight:
        best_weight = polished_weight
        best_clique = polished
        best_weight_exact = sum(Fraction(1, all_nodes[i]) for i in polished)

    # Convert to values
    result = [all_nodes[i] for i in best_clique]
    
    if not result:
        return [256]
    
    return result
