import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    N = 256
    # Build conflict graph: conflict[a][b] = True if a+b is a perfect square
    # We only need a <= b, but let's build full adjacency for efficiency
    
    # Perfect squares up to 512
    squares = set()
    s = 1
    while s * s <= 512:
        squares.add(s * s)
        s += 1
    
    # Build adjacency: for each a, which b values conflict
    # conflict[a] = set of b where a+b is a perfect square
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for sq in squares:
            b = sq - a
            if 1 <= b <= N:
                conflict[a].add(b)
    
    # Diagonal check: 2a must not be a perfect square
    # If 2a is a perfect square, a cannot be in the set
    valid_nodes = set(range(1, N + 1))
    for a in range(1, N + 1):
        if 2 * a in squares:
            valid_nodes.discard(a)
    
    # Remove nodes that conflict with themselves (already handled by 2a check)
    # Now build the graph on valid_nodes
    
    # Greedy independent set: order nodes by degree (ascending), pick if no conflict with already picked
    def greedy_is(nodes):
        # Sort by degree in the induced subgraph
        node_list = list(nodes)
        # Compute degree within the set
        degree = {}
        for a in node_list:
            deg = 0
            for b in conflict[a]:
                if b in nodes and b >= a:  # count each edge once
                    deg += 1
            degree[a] = deg
        # Sort by degree ascending
        node_list.sort(key=lambda x: degree[x])
        chosen = set()
        for a in node_list:
            # Check if a conflicts with any chosen node
            can_add = True
            for b in chosen:
                if b in conflict[a]:
                    can_add = False
                    break
            if can_add:
                chosen.add(a)
        return chosen
    
    # Initial greedy
    best_set = greedy_is(valid_nodes)
    best_size = len(best_set)
    
    # Local search: try to improve by swapping
    # For each node not in best_set, try to add it and remove conflicting nodes
    # If net gain is positive, accept
    
    def is_conflict(a, b):
        return b in conflict[a]
    
    # Local search iterations
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # Try adding each unselected node
        unselected = valid_nodes - best_set
        best_gain = 0
        best_swap = None  # (node_to_add, nodes_to_remove)
        
        for a in unselected:
            # Find all nodes in best_set that conflict with a
            to_remove = set()
            for b in best_set:
                if is_conflict(a, b):
                    to_remove.add(b)
            gain = 1 - len(to_remove)
            if gain > best_gain:
                best_gain = gain
                best_swap = (a, to_remove)
        
        if best_swap is not None and best_gain > 0:
            a, to_remove = best_swap
            best_set = (best_set - to_remove) | {a}
            best_size = len(best_set)
            improved = True
        else:
            # Try 2-swap: remove two, add one, or remove one, add two
            # Try: for each pair to remove, can we add a node that conflicts with both?
            # This is more expensive, so only do if we have time
            if (time.time() - start_time) < deadline - 10:
                # Try removing one node and adding two
                for b in best_set:
                    # Nodes that conflict with b
                    conflicting = conflict[b] & best_set
                    # If we remove b, we might be able to add some nodes
                    # Try adding each unselected node that was blocked by b
                    for a in unselected:
                        if b in conflict[a]:
                            # If we remove b, can we add a?
                            # Check if a conflicts with any other node in best_set - {b}
                            new_set = best_set - {b}
                            can_add_a = True
                            for c in new_set:
                                if is_conflict(a, c):
                                    can_add_a = False
                                    break
                            if can_add_a:
                                # Now check if we can also add another node
                                # This gets complex, let's just try adding a
                                new_best = new_set | {a}
                                if len(new_best) > best_size:
                                    best_set = new_best
                                    best_size = len(best_set)
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    continue
                
                # Try removing two nodes and adding one
                best_list = list(best_set)
                for i in range(len(best_list)):
                    for j in range(i + 1, len(best_list)):
                        b1, b2 = best_list[i], best_list[j]
                        remove_set = {b1, b2}
                        # Find nodes that conflict with both b1 and b2
                        for a in unselected:
                            if is_conflict(a, b1) and is_conflict(a, b2):
                                new_set = best_set - remove_set
                                can_add_a = True
                                for c in new_set:
                                    if is_conflict(a, c):
                                        can_add_a = False
                                        break
                                if can_add_a:
                                    new_best = new_set | {a}
                                    if len(new_best) > best_size:
                                        best_set = new_best
                                        best_size = len(best_set)
                                        improved = True
                                        break
                        if improved:
                            break
                    if improved:
                        break
    
    return list(best_set)
