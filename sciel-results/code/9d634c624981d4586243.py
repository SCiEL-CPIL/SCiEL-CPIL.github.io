"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def _is_3_colorable(adj):
    """Check if graph is 3-colorable using backtracking."""
    n = len(adj)
    color = [-1] * n
    
    def backtrack(v):
        if v == n:
            return True
        for c in range(3):
            if all(color[nei] != c for nei in adj[v]):
                color[v] = c
                if backtrack(v + 1):
                    return True
                color[v] = -1
        return False
    
    # Heuristic: order vertices by degree descending to prune earlier
    # For n=16, simple order is fine, but let's sort for robustness
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    # Map original index to new order index for coloring? 
    # Easier: just permute the adjacency matrix or use the order in backtrack.
    # Let's stick to simple backtrack on 0..n-1 as n is small.
    
    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe within 150s limit
    
    # 1. Initialize with Groetzsch graph (11 edges)
    # Vertices 0-4 form a cycle, 5-9 are "copies", 10 is "center"
    # Edges:
    # Cycle: (0,1), (1,2), (2,3), (3,4), (4,0)
    # Copies: (0,5), (1,5), (1,6), (2,6), (2,7), (3,7), (3,8), (4,8), (4,9), (0,9) ? 
    # Wait, Mycielski construction of C5:
    # V = {0,1,2,3,4, 0',1',2',3',4', *}
    # E(C5) + {(i, i') for i in 0..4} + {(i, j') for (i,j) in E(C5)} + {(i, *) for i in 0..4}
    # Let's map: 0->0, 1->1, 2->2, 3->3, 4->4
    # 0'->5, 1'->6, 2'->7, 3'->8, 4'->9
    # *->10
    # Edges:
    # (0,1), (1,2), (2,3), (3,4), (4,0)
    # (0,5), (1,6), (2,7), (3,8), (4,9)
    # (0,6), (1,5), (1,7), (2,6), (2,8), (3,7), (3,9), (4,8), (4,5) -- Wait, (i, j') where (i,j) in E.
    # If (0,1) in E, then (0, 1')=(0,6) and (1, 0')=(1,5).
    # If (1,2) in E, then (1,7) and (2,6).
    # If (2,3) in E, then (2,8) and (3,7).
    # If (3,4) in E, then (3,9) and (4,8).
    # If (4,0) in E, then (4,5) and (0,9).
    # Center edges: (0,10), (1,10), (2,10), (3,10), (4,10).
    
    edges = []
    # Cycle
    edges.append([0, 1])
    edges.append([1, 2])
    edges.append([2, 3])
    edges.append([3, 4])
    edges.append([4, 0])
    # Copies
    edges.append([0, 5])
    edges.append([1, 6])
    edges.append([2, 7])
    edges.append([3, 8])
    edges.append([4, 9])
    # Cross
    edges.append([0, 6])
    edges.append([1, 5])
    edges.append([1, 7])
    edges.append([2, 6])
    edges.append([2, 8])
    edges.append([3, 7])
    edges.append([3, 9])
    edges.append([4, 8])
    edges.append([4, 5])
    # Center
    edges.append([0, 10])
    edges.append([1, 10])
    edges.append([2, 10])
    edges.append([3, 10])
    edges.append([4, 10])
    
    # Vertices 11-15 are isolated initially
    
    n = 16
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    edge_set = set((min(u,v), max(u,v)) for u,v in edges)
    
    # Local Search
    while True:
        if time.time() > deadline:
            break
            
        # Find candidates: pairs (u,v) not in edge_set, such that adj[u] and adj[v] are disjoint
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in edge_set:
                    if not (adj[u] & adj[v]):
                        candidates.append((u, v))
                        
        if not candidates:
            break
            
        random.shuffle(candidates)
        
        added = False
        for u, v in candidates:
            if time.time() > deadline:
                break
                
            # Check if adding (u,v) keeps graph non-3-colorable
            # Temporarily add
            adj[u].add(v)
            adj[v].add(u)
            
            if not _is_3_colorable(adj):
                # Valid addition
                edge_set.add((u, v))
                edges.append([u, v])
                added = True
                break
            else:
                # Revert
                adj[u].remove(v)
                adj[v].remove(u)
                
        if not added:
            # Try a few random restarts/shuffles if stuck? 
            # For now, if no edge added in a full pass of candidates, we are stuck.
            # Given the small size, we might just break.
            # However, we might be in a local optimum where we need to remove an edge to add two?
            # The task is to maximize edges. Greedy addition is standard for this size.
            # If we can't add any, we stop.
            break
            
    return edges
# EVOLVE-BLOCK-END
