import time
import random
from itertools import combinations

def is_non_2_colorable(n, edges):
    """
    Returns True if the hypergraph has no proper 2-coloring.
    Uses backtracking with unit propagation.
    """
    # Preprocess edges into a format suitable for fast lookup
    # For each vertex, list of edges containing it
    vertex_edges = [[] for _ in range(n)]
    for i, edge in enumerate(edges):
        for v in edge:
            vertex_edges[v].append(i)
            
    # We need to track the state of each edge
    # edge_state[i] = (count0, count1, unknowns_list)
    # But updating this is expensive.
    
    # Alternative: Just use the list of edges and check constraints on the fly.
    # With unit propagation, we need to know which edges are "close" to being monochromatic.
    
    # Let's use a simpler backtracking without heavy propagation first, 
    # or a very simple one.
    
    colors = [-1] * n
    
    def propagate(idx, color):
        # Assign colors[idx] = color
        # Check all edges containing idx
        # If an edge has 4 of one color, the 5th is forced.
        # If an edge has 5 of one color, conflict.
        pass # Too complex to implement quickly in a tight loop without optimization.
        
    # Let's use a standard backtracking with a simple conflict check.
    # For n=10, depth is 10. Branching factor 2. 2^10 = 1024 leaves.
    # This is very fast.
    
    def backtrack(v):
        if v == n:
            # All assigned. Check if any edge is monochromatic.
            for edge in edges:
                c0 = colors[edge[0]]
                # Check if all same
                mono = True
                for u in edge[1:]:
                    if colors[u] != c0:
                        mono = False
                        break
                if mono:
                    return False # This coloring is invalid (has monochromatic edge)
            return True # Valid coloring found (all edges bichromatic)
            
        # Try color 0
        colors[v] = 0
        if backtrack(v + 1):
            return True
            
        # Try color 1
        colors[v] = 1
        if backtrack(v + 1):
            return True
            
        colors[v] = -1
        return False

    return not backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Try to find a small non-2-colorable hypergraph
    # Start with n=10
    n = 10
    all_edges = list(combinations(range(n), 5))
    
    best_edges = None
    best_n = None
    
    # 1. Generate a random dense hypergraph
    # Shuffle all edges
    shuffled_edges = all_edges[:]
    random.shuffle(shuffled_edges)
    
    # Start with a large subset
    # Let's try to find one with ~100 edges first, then prune
    # Or just try random subsets of size k
    
    # Strategy: 
    # 1. Find ANY non-2-colorable hypergraph on n=10.
    # 2. Prune it.
    
    # Step 1: Find a non-2-colorable hypergraph
    # Try increasing edge counts
    # Actually, random hypergraphs with ~50% density are likely non-2-colorable for n=10?
    # Let's try k=100.
    
    found = False
    current_edges = []
    
    # Try random sets of edges
    for attempt in range(1000):
        if time.time() > deadline:
            break
            
        # Try a random subset of size 100
        k = 100
        random.shuffle(shuffled_edges)
        candidate = shuffled_edges[:k]
        
        if is_non_2_colorable(n, candidate):
            current_edges = candidate
            found = True
            break
            
    if not found:
        # Try larger k
        for k in [120, 140, 160, 180, 200, 220, 240, 252]:
            if time.time() > deadline:
                break
            for attempt in range(100):
                if time.time() > deadline:
                    break
                random.shuffle(shuffled_edges)
                candidate = shuffled_edges[:k]
                if is_non_2_colorable(n, candidate):
                    current_edges = candidate
                    found = True
                    break
            if found:
                break
                
    if not found:
        # Fallback to seed
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    # Step 2: Prune
    # Try to remove edges
    # Sort edges? Random order?
    # To be efficient, we can try removing random edges.
    
    edges_to_check = list(range(len(current_edges)))
    random.shuffle(edges_to_check)
    
    pruned_edges = current_edges[:]
    
    # We need to check if removing an edge keeps it non-2-colorable.
    # This is expensive.
    # We can do a greedy deletion:
    # Iterate through edges. For each edge, check if removing it results in a 2-colorable graph.
    # If yes, remove it. If no, keep it.
    # But checking is O(2^n * m).
    
    # Let's limit the pruning steps to fit in time.
    
    pruned_count = 0
    for i in range(len(pruned_edges)):
        if time.time() > deadline:
            break
            
        # Try to remove edge i
        # Create a copy without edge i
        # To avoid copying the whole list, we can use a set or just pass a modified list
        # For n=10, m~100, copying is cheap.
        
        test_edges = pruned_edges[:i] + pruned_edges[i+1:]
        
        if is_non_2_colorable(n, test_edges):
            # It is still non-2-colorable without this edge
            pruned_edges = test_edges
            pruned_count += 1
            # Restart the loop? Or just continue?
            # If we continue, we might miss other removals because the list changed.
            # But this is a greedy heuristic.
            # Let's restart to ensure we try all edges again in the new configuration.
            # Actually, just restarting the whole loop is safer but slower.
            # Let's just continue for now, or break and restart.
            # Given time limits, let's just do one pass or a few passes.
            pass # We can't easily restart the outer loop structure here without a while loop.
            
    # Let's use a while loop for pruning
    progress = True
    while progress and time.time() < deadline:
        progress = False
        # Shuffle edges to try different orders
        random.shuffle(pruned_edges)
        
        for i in range(len(pruned_edges)):
            if time.time() > deadline:
                break
            
            test_edges = pruned_edges[:i] + pruned_edges[i+1:]
            if is_non_2_colorable(n, test_edges):
                pruned_edges = test_edges
                progress = True
                break # Restart the loop with the new list
                
    if time.time() > deadline:
        # Return best found so far
        if len(pruned_edges) < 126:
            return {'n': n, 'edges': [list(e) for e in pruned_edges]}
        else:
            return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
            
    if len(pruned_edges) < 126:
        return {'n': n, 'edges': [list(e) for e in pruned_edges]}
    else:
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
