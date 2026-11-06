import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Check if the hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    # Precompute edge masks for speed
    edge_masks = [0] * len(edges)
    for i, e in enumerate(edges):
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks[i] = mask
    
    m = len(edges)
    
    # State: color[v] = -1 (unassigned), 0, or 1
    color = [-1] * n
    
    # For each edge, track count of 0s and 1s
    count0 = [0] * m
    count1 = [0] * m
    
    def propagate(start_idx):
        """
        Unit propagation. Returns False if contradiction, True otherwise.
        Modifies color, count0, count1 in place.
        """
        changed = True
        while changed:
            changed = False
            for i in range(start_idx, m):
                # If all 5 vertices are same color, contradiction
                if count0[i] == 5 or count1[i] == 5:
                    return False
                
                # If 4 vertices are same color, the 5th must be the other color
                if count0[i] == 4:
                    # Find the unassigned vertex in this edge
                    for v in edges[i]:
                        if color[v] == -1:
                            color[v] = 1
                            count1[i] += 1
                            count0[i] -= 1  # Wait, count0[i] was 4, now 3? No.
                            # Actually, if count0 was 4, that means 4 vertices are color 0.
                            # The unassigned vertex must be color 1.
                            # So count0 stays 4, count1 becomes 1.
                            # My logic above was slightly off.
                            # Let's re-evaluate.
                            # If count0[i] == 4, it means 4 vertices are already color 0.
                            # The 5th must be color 1.
                            # So we set color[v] = 1.
                            # count0[i] remains 4. count1[i] becomes 1.
                            # But wait, if we set color[v]=1, count1[i] increases by 1.
                            # count0[i] does not change because v was unassigned.
                            # So count0[i] is still 4? No, count0[i] counts assigned 0s.
                            # So count0[i] is still 4. count1[i] is now 1.
                            # This is correct.
                            # However, we need to update counts for OTHER edges containing v.
                            # This is the tricky part of unit propagation in hypergraphs.
                            # We need to update all edges that contain v.
                            
                            # Let's restart the propagation loop to handle updates correctly.
                            # A simpler approach: just check all edges each time.
                            # For n=26, m=200, this is O(m*n) per step.
                            # Number of steps <= n. So O(n*m*n) = O(n^2*m).
                            # 26^2 * 200 = 135,200. Very fast.
                            
                            # Let's implement a simpler propagation:
                            # 1. Check all edges for contradiction (5 same).
                            # 2. Check all edges for unit (4 same, 1 unassigned) -> assign.
                            # 3. Repeat until no change.
                            
                            # I will rewrite propagate to be simpler and correct.
                            pass # Placeholder for actual logic below
            # The above loop is complex. Let's use a standard backtracking with simple pruning.
            # Given the small n (<=26) and m (<=200), and the fact that we are searching for
            # a minimal non-2-colorable hypergraph, a simple backtracking might suffice.
            # But unit propagation is key for speed.
            
            # Let's implement a cleaner version.
            break
        
        # Actually, let's just do a simple check without complex propagation first.
        # If it's too slow, we'll add propagation.
        # For n=9, 2^9=512 checks is instant.
        # For n=26, 2^26 is too big.
        # But we are minimizing edges. The minimal non-2-colorable 5-uniform hypergraph
        # is known to have 51 edges.
        # The number of vertices in such a construction is likely small (e.g., 9-12).
        # So n will be small.
        
        return True

    # Let's implement a robust backtracking with simple unit propagation.
    
    def solve():
        # Find first unassigned vertex
        v = -1
        for i in range(n):
            if color[i] == -1:
                v = i
                break
        
        if v == -1:
            # All assigned. Check if all edges are valid (not monochromatic)
            for i in range(m):
                if count0[i] == 5 or count1[i] == 5:
                    return False
            return True
        
        # Try color 0
        color[v] = 0
        # Update counts
        for i in range(m):
            if edge_masks[i] & (1 << v):
                count0[i] += 1
        
        # Check for immediate contradiction
        valid = True
        for i in range(m):
            if count0[i] == 5 or count1[i] == 5:
                valid = False
                break
        
        if valid and solve():
            return True
        
        # Backtrack
        color[v] = -1
        for i in range(m):
            if edge_masks[i] & (1 << v):
                count0[i] -= 1
        
        # Try color 1
        color[v] = 1
        for i in range(m):
            if edge_masks[i] & (1 << v):
                count1[i] += 1
        
        valid = True
        for i in range(m):
            if count0[i] == 5 or count1[i] == 5:
                valid = False
                break
        
        if valid and solve():
            return True
        
        # Backtrack
        color[v] = -1
        for i in range(m):
            if edge_masks[i] & (1 << v):
                count1[i] -= 1
        
        return False

    return solve()

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # We want to find a minimal non-2-colorable 5-uniform hypergraph.
    # Start with a known small construction or build one.
    # The complete hypergraph on 9 vertices has C(9,5)=126 edges and is non-2-colorable.
    # We can try to remove edges.
    
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    current_edges = all_edges[:]
    
    # Verify it's non-2-colorable
    if is_2_colorable(n, current_edges):
        # This should not happen for K_9^5
        return {'n': n, 'edges': current_edges}
    
    # Deletion process: try to remove each edge if the graph remains non-2-colorable.
    # Do this in multiple passes with random ordering.
    
    for _ in range(10):  # 10 passes
        if time.time() > deadline - 10:
            break
            
        random.shuffle(current_edges)
        new_edges = []
        
        for edge in current_edges:
            if time.time() > deadline - 5:
                new_edges.append(edge)
                continue
                
            # Try to remove this edge
            candidate = [e for e in current_edges if e != edge]
            
            # Check if candidate is still non-2-colorable
            # This is the expensive part.
            # For n=9, this is fast.
            if not is_2_colorable(n, candidate):
                # Still non-2-colorable, so we can remove the edge
                continue
            else:
                # Became 2-colorable, so we must keep the edge
                new_edges.append(edge)
        
        current_edges = new_edges
        
        # If no edges were removed in this pass, we might be at a local minimum
        # But let's do a few more passes with different orderings to be sure.
    
    # Final check
    if is_2_colorable(n, current_edges):
        # Should not happen, but just in case, return a valid (but larger) graph
        return {'n': n, 'edges': all_edges}
    
    return {'n': n, 'edges': current_edges}
