import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable, False otherwise.
    Uses brute force for n <= 20, backtracking for larger n.
    """
    if not edges:
        return True

    # Precompute edge masks
    edge_masks = []
    for e in edges:
        m = 0
        for v in e:
            m |= (1 << v)
        edge_masks.append(m)

    if n <= 20:
        # Brute force all 2^n colorings
        # Optimization: If n is small, this is very fast.
        # 2^20 is 1,048,576. 
        # If edges is large, checking each coloring might be slow.
        # But for n=9, 512 colorings * ~100 edges = 51,200 ops.
        # For n=20, 1M colorings * 200 edges = 200M ops. Might be slow in Python.
        # Let's limit brute force to n <= 15.
        if n <= 15:
            for c in range(1 << n):
                is_bad = False
                for m in edge_masks:
                    if (c & m) == 0 or (c & m) == m:
                        is_bad = True
                        break
                if not is_bad:
                    return True
            return False
        else:
            # Backtracking
            # ...
            pass
    else:
        # Backtracking
        # ...
        pass

    # Fallback backtracking for n > 15
    # Simple backtracking with unit propagation is complex to implement correctly in short time.
    # But we likely won't need n > 15.
    # Let's implement a basic backtracking.
    
    # Assign colors to vertices 0..n-1
    # colors: list of -1 (unassigned), 0, 1
    colors = [-1] * n
    
    def propagate(colors, edges, edge_masks):
        # Check for conflicts and propagate
        # For each edge, if 4 vertices are same color, the 5th must be different? No.
        # If 4 are 0, the 5th MUST be 1 to avoid monochromatic.
        # If 4 are 1, the 5th MUST be 0.
        # If 3 are 0, 1 is 1, 1 is unassigned: no constraint?
        #   If unassigned becomes 0, edge is mono. So unassigned MUST be 1.
        #   Wait. Edge {0,0,0,1,x}. If x=0, mono. So x must be 1.
        #   So if an edge has 3 of one color and 1 of the other, the last MUST be the other.
        #   If an edge has 4 of one color, the last MUST be the other.
        
        changed = True
        while changed:
            changed = False
            for m in edge_masks:
                # Count 0s and 1s in this edge
                count_0 = 0
                count_1 = 0
                unassigned = -1
                for v in range(n):
                    if (m >> v) & 1:
                        if colors[v] == 0:
                            count_0 += 1
                        elif colors[v] == 1:
                            count_1 += 1
                        else:
                            unassigned = v
                
                # If there are no unassigned vertices, check if mono
                if unassigned == -1:
                    if count_0 == 5 or count_1 == 5:
                        return False # Conflict
                
                # If there is an unassigned vertex
                if unassigned != -1:
                    if count_0 == 4:
                        # Must be 1
                        if colors[unassigned] == 0:
                            return False
                        colors[unassigned] = 1
                        changed = True
                    elif count_1 == 4:
                        # Must be 0
                        if colors[unassigned] == 1:
                            return False
                        colors[unassigned] = 0
                        changed = True
                    elif count_0 == 3 and count_1 == 1:
                        # Must be 1
                        if colors[unassigned] == 0:
                            return False
                        colors[unassigned] = 1
                        changed = True
                    elif count_1 == 3 and count_0 == 1:
                        # Must be 0
                        if colors[unassigned] == 1:
                            return False
                        colors[unassigned] = 0
                        changed = True
        return True

    def backtrack(idx):
        if idx == n:
            return True # All assigned, no conflicts found
        
        # Try 0
        colors[idx] = 0
        if propagate(colors.copy(), edges, edge_masks):
            if backtrack(idx + 1):
                return True
        
        # Try 1
        colors[idx] = 1
        if propagate(colors.copy(), edges, edge_masks):
            if backtrack(idx + 1):
                return True
        
        colors[idx] = -1
        return False

    return backtrack(0)


def greedy_reduce(n, edges, deadline):
    """
    Greedily remove edges while preserving non-2-colorability.
    """
    curr = edges[:]
    
    # Check initial state
    if is_2_colorable(n, curr):
        return curr # Should not happen if we start with non-2-colorable
    
    while True:
        if time.time() > deadline:
            break
        
        changed = False
        indices = list(range(len(curr)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            # Try removing edge i
            test_edges = curr[:i] + curr[i+1:]
            
            # If removing it makes it 2-colorable, we MUST keep it.
            # If removing it leaves it non-2-colorable, we CAN remove it.
            if not is_2_colorable(n, test_edges):
                curr.pop(i)
                changed = True
                # Restart the loop to re-shuffle
                break
        
        if not changed:
            break
            
    return curr


def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_edges = None
    best_count = float('inf')
    best_n = 9
    
    # Try different n
    # Start with small n where non-2-colorable hypergraphs are known to exist
    # K_9^5 is a good candidate.
    
    for n in [9, 10, 11]:
        if time.time() > deadline:
            break
            
        # Generate all edges
        all_edges = [list(e) for e in combinations(range(n), 5)]
        
        # Check if the complete hypergraph is non-2-colorable
        # For n=9, K_9^5 is non-2-colorable.
        # For n=10, K_10^5 is non-2-colorable.
        # For n=11, K_11^5 is non-2-colorable.
        
        # We need to verify this.
        # Actually, for large enough n, K_n^k is non-2-colorable.
        # For k=5, n=9 is sufficient?
        # Let's assume it is. If not, greedy_reduce will return the full set (or fail).
        
        # To be safe, we can check.
        # But checking K_9^5 (126 edges) is fast.
        
        # Run greedy reduction
        # We might want to try multiple restarts with different random seeds/shuffles
        # But greedy_reduce already shuffles internally in each pass.
        
        # Let's run it once for each n.
        # If we have time, we can run multiple times.
        
        # Actually, let's just run it.
        reduced = greedy_reduce(n, all_edges, deadline)
        
        if len(reduced) < best_count:
            best_count = len(reduced)
            best_edges = reduced
            best_n = n
            
        # If we found 51, we can stop
        if best_count <= 51:
            break

    if best_edges is None:
        # Fallback: return K_9^5
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return {'n': best_n, 'edges': best_edges}
