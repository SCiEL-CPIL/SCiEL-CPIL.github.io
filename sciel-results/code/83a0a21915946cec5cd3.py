"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    # Strategy: Time-bounded search for a non-2-colorable 5-uniform hypergraph
    # with fewer edges than the seed (126).
    # We try random instances on n=26 with e decreasing from 60 to 50.
    # We verify with backtracking.
    # If we find one, we return it.
    # If time runs out, we return the seed.
    
    start_time = time.time()
    deadline = start_time + 10.0 # 10s budget for search
    
    # Helper to check 2-colorability
    def is_2_colorable(n, edges):
        edge_masks = []
        for e in edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
            
        # Precompute adjacency: for each vertex, list of edge indices
        # This helps in propagation
        
        # Simple backtracking
        # assigned: list of -1, 0, 1
        assigned = [-1] * n
        
        def propagate(edge_idx, assigned):
            # This is a simplified propagation. 
            # For each edge, if 4 are assigned, the 5th is forced.
            # We do this globally? No, usually local.
            # For simplicity and speed in Python, we might just rely on branching.
            pass
            
        def backtrack(edge_idx):
            if time.time() > deadline:
                return None # Timeout
            
            if edge_idx == len(edges):
                return True # All edges satisfied (non-mono)
                
            mask = edge_masks[edge_idx]
            # Get vertices
            vs = []
            for v in range(n):
                if mask & (1 << v):
                    vs.append(v)
                    
            c0 = 0
            c1 = 0
            unassigned = []
            for v in vs:
                if assigned[v] == 0:
                    c0 += 1
                elif assigned[v] == 1:
                    c1 += 1
                else:
                    unassigned.append(v)
                    
            if c0 == 5 or c1 == 5:
                return False
                
            if not unassigned:
                return backtrack(edge_idx + 1)
                
            # Heuristic: Pick vertex with most constraints? 
            # For now, pick first unassigned.
            v = unassigned[0]
            
            # Try 0
            assigned[v] = 0
            res = backtrack(edge_idx) # Re-check same edge? 
            # If we assign v, the edge might become forced or satisfied.
            # If we just recurse to same edge_idx, we re-evaluate it.
            if res is not None:
                return res
            assigned[v] = -1
            
            # Try 1
            assigned[v] = 1
            res = backtrack(edge_idx)
            assigned[v] = -1
            
            if res is not None:
                return res
                
            return False # No solution found
            
        res = backtrack(0)
        if res is None:
            return None # Timeout
        return res # True if colorable, False if not
        
    # Search
    # Try n=26
    # Start with e=60, go down to 50
    for e in range(60, 49, -1):
        if time.time() > deadline:
            break
            
        # Try a few random instances
        for _ in range(5):
            if time.time() > deadline:
                break
                
            edges = []
            seen = set()
            while len(edges) < e:
                ed = tuple(sorted(random.sample(range(26), 5)))
                if ed not in seen:
                    seen.add(ed)
                    edges.append(list(ed))
                    
            # Check
            # Note: is_2_colorable returns True if colorable, False if not.
            # We want NOT colorable.
            res = is_2_colorable(26, edges)
            if res is False:
                # Found a non-2-colorable one!
                return {'n': 26, 'edges': edges}
            elif res is None:
                # Timeout in checker, skip
                continue
            else:
                # Colorable, try next
                continue
                
    # Fallback
    from itertools import combinations
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}

# EVOLVE-BLOCK-END
