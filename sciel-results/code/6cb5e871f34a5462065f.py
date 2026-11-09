import numpy as np
import time

def ring_routing() -> np.ndarray:
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 500  # 500s safety margin
    
    n_nodes = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute edge indices for each demand for both directions
    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        # Clockwise: s, s+1, ..., t-1 (mod n)
        edges_cw = []
        curr = s
        while curr != t:
            edges_cw.append(curr)
            curr = (curr + 1) % n_nodes
        cw_edges.append(edges_cw)
        
        # Counter-clockwise: all other edges
        # This is equivalent to going t, t+1, ..., s-1 (mod n) in reverse, 
        # but the edges are the same set: {e | e is not in cw_edges}
        # Actually, easier: the path is t -> t-1 -> ... -> s.
        # Edges are (t-1)%n, (t-2)%n, ... s%n? No.
        # Edge e connects e and (e+1)%n.
        # Counter-clockwise from s to t means going s-1, s-2, ... t+1.
        # The edges traversed are (s-1)%n, (s-2)%n, ..., (t+1)%n.
        edges_ccw = []
        curr = (s - 1) % n_nodes
        while curr != (t + 1) % n_nodes:
            edges_ccw.append(curr)
            curr = (curr - 1) % n_nodes
        # Wait, if s=0, t=5. CCW is 0->9->8->7->6->5.
        # Edges: 9, 8, 7, 6.
        # My loop: curr = 9. while curr != 6: append 9, curr=8; append 8, curr=7; append 7, curr=6. Stop.
        # Edges: [9, 8, 7]. Missing 6?
        # Edge 6 connects 6 and 7. So traversing 7->6 uses edge 6.
        # Let's re-verify edge definition.
        # "edge e joins nodes e and (e + 1) mod 10"
        # Clockwise from s to t:
        # Step 1: s -> s+1. Uses edge s.
        # Step 2: s+1 -> s+2. Uses edge s+1.
        # ...
        # Last step: t-1 -> t. Uses edge t-1.
        # So edges are s, s+1, ..., t-1.
        
        # Counter-clockwise from s to t:
        # Step 1: s -> s-1. Uses edge (s-1)%n.
        # Step 2: s-1 -> s-2. Uses edge (s-2)%n.
        # ...
        # Last step: t+1 -> t. Uses edge (t+1)%n.
        # So edges are (s-1)%n, (s-2)%n, ..., (t+1)%n.
        
        # Let's re-check the loop for s=0, t=5.
        # Edges should be 9, 8, 7, 6.
        # My loop:
        # curr = 9. target = (5+1)%10 = 6.
        # while curr != 6:
        #   append 9. curr = 8.
        #   append 8. curr = 7.
        #   append 7. curr = 6.
        # Loop ends. Edges: [9, 8, 7].
        # It missed 6. Why?
        # The last edge is (t+1)%n = 6.
        # The loop condition `while curr != (t+1)%n` stops BEFORE appending the last edge.
        # It should append until it reaches the target, but the target edge IS the last one.
        # So the condition should be `while True` with a break, or adjust logic.
        # Actually, the set of edges for CCW is simply the complement of CW edges.
        # Since n=10, we can just compute complement.
        pass
        
    # Let's just use the complement logic, it's safer and faster.
    for i in range(D):
        cw_set = set(cw_edges[i])
        ccw_set = set(range(n_nodes)) - cw_set
        ccw_edges.append(list(ccw_set))
        
    # Brute force search
    best_max_load = float('inf')
    best_routing = np.zeros(D)
    
    # We can prune: if current load exceeds best_max_load, stop.
    # But with 2^15, even without pruning it's fast.
    # 32768 iterations * 15 demands * 10 edges = ~5M ops. Very fast.
    
    for mask in range(1 << D):
        if time.time() > deadline:
            break
            
        loads = np.zeros(n_nodes)
        
        for i in range(D):
            if (mask >> i) & 1:
                # Counter-clockwise
                for e in ccw_edges[i]:
                    loads[e] += demands[i][2]
            else:
                # Clockwise
                for e in cw_edges[i]:
                    loads[e] += demands[i][2]
                    
        current_max = np.max(loads)
        
        if current_max < best_max_load:
            best_max_load = current_max
            # Reconstruct routing from mask
            routing = np.zeros(D)
            for i in range(D):
                if (mask >> i) & 1:
                    routing[i] = 1.0
                else:
                    routing[i] = 0.0
            best_routing = routing
            
    return best_routing
