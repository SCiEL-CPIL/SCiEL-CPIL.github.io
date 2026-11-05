import random
import time
import sys

def construct():
    """
    Constructs a C4-free subgraph of Q7 maximizing edge count.
    Uses randomized greedy with multiple restarts and local search improvement.
    """
    start_time = time.time()
    time_limit = 105.0  # Safety margin before 120s kill
    
    d = 7
    n = 2 ** d  # 128 vertices
    
    # Precompute all edges with their bit dimension
    # Edge encoded as (u, v) where u < v
    all_edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y, i))
    
    # For each edge, precompute the 4-cycle it would complete if added
    # A 4-cycle in Q7 is formed by two parallel edges (same dimension)
    # Edge (u,v) with dim i: for each j != i, consider u^bit_j, v^bit_j
    # The cycle is: u - v (dim i), u - u^bit_j (dim j), v - v^bit_j (dim i), u^bit_j - v^bit_j (dim j)
    # So edge (u,v,i) would complete a cycle if the other 3 edges are present:
    #   (u, u^bit_j), (v, v^bit_j), (u^bit_j, v^bit_j)
    
    # For fast lookup, we'll track which edges are selected in a set
    # We need to quickly check if adding edge (u,v) would create a C4
    
    # Precompute for each edge the list of "conflict sets" - sets of 3 edges that would complete a cycle
    # Actually, let's think differently. When we add edge (u,v) with dim i,
    # for each j != i, we check if edges (u, u^bit_j), (v, v^bit_j), and (u^bit_j, v^bit_j) are all present.
    
    # Let's precompute for each edge (u,v,i) the list of triples that would complete a cycle
    edge_conflicts = {}
    for idx, (u, v, i) in enumerate(all_edges):
        conflicts = []
        for j in range(d):
            if j == i:
                continue
            bit_j = 1 << j
            u2 = u ^ bit_j
            v2 = v ^ bit_j
            # The three other edges in the cycle:
            # (min(u,u2), max(u,u2)) - dim j
            # (min(v,v2), max(v,v2)) - dim j  
            # (min(u2,v2), max(u2,v2)) - dim i
            e1 = (min(u, u2), max(u, u2))
            e2 = (min(v, v2), max(v, v2))
            e3 = (min(u2, v2), max(u2, v2))
            conflicts.append((e1, e2, e3))
        edge_conflicts[idx] = conflicts
    
    # Build a mapping from edge tuple to index
    edge_to_idx = {}
    for idx, (u, v, i) in enumerate(all_edges):
        edge_to_idx[(u, v)] = idx
    
    def is_valid_add(edge_idx, selected):
        """Check if adding edge edge_idx would create a C4."""
        u, v, i = all_edges[edge_idx]
        for e1, e2, e3 in edge_conflicts[edge_idx]:
            if e1 in selected and e2 in selected and e3 in selected:
                return False
        return True
    
    def greedy_randomized(seed):
        """Randomized greedy: shuffle edges, add if valid."""
        rng = random.Random(seed)
        order = list(range(len(all_edges)))
        rng.shuffle(order)
        selected = set()
        result = []
        for idx in order:
            if is_valid_add(idx, selected):
                selected.add(idx)
                u, v, i = all_edges[idx]
                result.append([u, v])
        return result, selected
    
    def local_search(result, selected, rng, time_budget):
        """
        Local search: try to swap edges to increase count.
        Actually, since we're maximizing, and greedy already gives a maximal set,
        we can try to find a set of edges that can be removed and more added.
        
        Alternative: try to remove one edge and see if we can add more than one.
        """
        if time.time() - start_time > time_limit - 5:
            return result, selected
        
        best_result = result
        best_selected = selected
        best_len = len(result)
        
        # Try removing each edge and see if we can add back more
        for attempt in range(min(200, len(result))):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Pick a random edge to remove
            remove_idx = rng.randint(0, len(all_edges) - 1)
            if remove_idx not in selected:
                continue
            
            # Remove it
            selected.remove(remove_idx)
            
            # Try to add edges that were previously blocked
            added = 0
            for idx in range(len(all_edges)):
                if idx in selected:
                    continue
                if is_valid_add(idx, selected):
                    selected.add(idx)
                    added += 1
            
            if added > 1:  # Net gain
                new_result = []
                for idx in selected:
                    u, v, i = all_edges[idx]
                    new_result.append([u, v])
                if len(new_result) > best_len:
                    best_result = new_result
                    best_selected = set(selected)
                    best_len = len(new_result)
            else:
                # Put back the removed edge
                selected.add(remove_idx)
        
        return best_result, best_selected
    
    best_edges = []
    best_count = 0
    
    # Multiple restarts with different seeds
    seed = 42
    while time.time() - start_time < time_limit - 10:
        result, selected = greedy_randomized(seed)
        count = len(result)
        
        if count > best_count:
            best_count = count
            best_edges = result
            # Try local search
            result_ls, selected_ls = local_search(result, selected, random.Random(seed), time_limit - 10)
            if len(result_ls) > best_count:
                best_count = len(result_ls)
                best_edges = result_ls
        
        seed += 1
        # Use different randomization strategies occasionally
        if seed % 5 == 0:
            # Try a different approach: prioritize edges by some heuristic
            pass
    
    # Final verification
    if len(best_edges) == 0:
        # Fallback: deterministic greedy
        selected = set()
        for idx in range(len(all_edges)):
            if is_valid_add(idx, selected):
                selected.add(idx)
        for idx in selected:
            u, v, i = all_edges[idx]
            best_edges.append([u, v])
    
    return best_edges
