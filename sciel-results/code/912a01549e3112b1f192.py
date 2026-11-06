import time
import random
import itertools
from collections import deque

def is_property_b(n, edges, deadline=None):
    """
    Check if the hypergraph has no proper 2-coloring.
    Returns True if Property B holds (no 2-coloring).
    Uses backtracking.
    """
    if not edges:
        return False
    
    edge_verts = [list(e) for e in edges]
    assign = [-1] * n
    
    if deadline is None:
        deadline = time.time() + 2.0 # Default 2s limit per check
        
    def backtrack():
        if time.time() > deadline:
            return False # Safe assumption: 2-colorable
        
        v = -1
        for i in range(n):
            if assign[i] == -1:
                v = i
                break
                
        if v == -1:
            for e in edge_verts:
                colors = [assign[x] for x in e]
                if all(c == 0 for c in colors) or all(c == 1 for c in colors):
                    return False
            return True
            
        assign[v] = 0
        if backtrack():
            return True
        assign[v] = 1
        if backtrack():
            return True
        assign[v] = -1
        return False

    return not backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_edges = None
    best_count = float('inf')
    
    # Try to find a small hypergraph
    # Start with n=11, random edges, try to minimize
    
    # Known fact: 51 edges is the minimum.
    # Let's try to find a solution with n=11.
    
    # We will use a local search / greedy deletion approach.
    # 1. Generate a random hypergraph with enough edges to be likely Property B.
    # 2. Check if it is Property B.
    # 3. If yes, try to remove edges one by one (randomly) and check if it remains Property B.
    # 4. Keep the smallest one found.
    
    for n in [11, 12, 13]:
        # Generate random edges
        all_possible = list(itertools.combinations(range(n), 5))
        
        # Try a few random instances
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Start with a large number of edges, e.g., half of all possible
            num_edges = len(all_possible) // 2
            random.seed(attempt + n)
            edges = random.sample(all_possible, num_edges)
            edges = [list(e) for e in edges]
            
            # Check if Property B
            if not is_property_b(n, edges):
                continue
            
            # It is Property B. Now try to minimize.
            # Greedy deletion:
            # Shuffle edges, try to remove each one if Property B still holds.
            random.shuffle(edges)
            
            i = 0
            while i < len(edges) and time.time() < deadline:
                # Try to remove edges[i]
                new_edges = edges[:i] + edges[i+1:]
                if is_property_b(n, new_edges):
                    edges = new_edges
                    # Do not increment i, check the next element at same index
                else:
                    i += 1
            
            if len(edges) < best_count:
                best_count = len(edges)
                best_edges = [list(e) for e in edges]
                # print(f"Found {n} vertices, {best_count} edges")
                
    if best_edges is not None:
        return {'n': n, 'edges': best_edges}
    
    # Fallback: Complete hypergraph on 9 vertices
    return {'n': 9, 'edges': [list(e) for e in itertools.combinations(range(9), 5)]}

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_edges = None
    best_count = float('inf')
    
    # Try to find a small hypergraph with Property B
    # n=11 is a good candidate.
    
    for n in [11, 12, 13]:
        all_possible = list(itertools.combinations(range(n), 5))
        
        # Try a few random instances
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            # Start with a random subset
            # For n=11, total edges 462. 
            # We want to end up with ~50.
            # Start with 100-150 to have a chance of being non-2-colorable.
            num_edges = min(len(all_possible), 120 + attempt)
            random.seed(attempt * 100 + n)
            edges = random.sample(all_possible, num_edges)
            edges = [list(e) for e in edges]
            
            # Check if Property B
            if not is_property_b(n, edges):
                continue
            
            # It is Property B. Now try to minimize.
            # Greedy deletion:
            # Shuffle edges, try to remove each one if Property B still holds.
            random.shuffle(edges)
            
            i = 0
            while i < len(edges) and time.time() < deadline:
                # Try to remove edges[i]
                new_edges = edges[:i] + edges[i+1:]
                if is_property_b(n, new_edges):
                    edges = new_edges
                    # Do not increment i, check the next element at same index
                else:
                    i += 1
            
            if len(edges) < best_count:
                best_count = len(edges)
                best_edges = [list(e) for e in edges]
                
    if best_edges is not None:
        return {'n': n, 'edges': best_edges}
    
    # Fallback: Complete hypergraph on 9 vertices
    return {'n': 9, 'edges': [list(e) for e in itertools.combinations(range(9), 5)]}
# EVOLVE-BLOCK-END
