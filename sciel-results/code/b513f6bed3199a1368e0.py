import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Try n from 34 downwards
    for n in range(34, 1, -1):
        if time.time() > deadline:
            break
            
        # Precomputation for this n
        edges = list(combinations(range(n), 3))
        m = len(edges)
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        # Map each edge to the quads and pents it belongs to
        # Quad is a set of 4 vertices. It contains 4 edges.
        # Pent is a set of 5 vertices. It contains 10 edges.
        
        quads = list(combinations(range(n), 4))
        pents = list(combinations(range(n), 5))
        
        # For each edge, which quads contain it?
        # An edge (u,v,w) is in a quad (u,v,w,x) for any x != u,v,w.
        edge_to_quads = [[] for _ in range(m)]
        for q_idx, q in enumerate(quads):
            # Generate 4 edges from this quad
            q_edges = list(combinations(q, 3))
            for e in q_edges:
                e_idx = edge_to_idx[e]
                edge_to_quads[e_idx].append(q_idx)
                
        # For each edge, which pents contain it?
        # An edge (u,v,w) is in a pent (u,v,w,x,y) for any x,y != u,v,w, x<y.
        edge_to_pents = [[] for _ in range(m)]
        for p_idx, p in enumerate(pents):
            # Generate 10 edges from this pent
            p_edges = list(combinations(p, 3))
            for e in p_edges:
                e_idx = edge_to_idx[e]
                edge_to_pents[e_idx].append(p_idx)

        # Initialize state
        # Let's try a random initialization first, or a heuristic one.
        # Heuristic: color based on sum % 3? Or just random.
        # Random is often better for escape.
        
        best_violations = float('inf')
        best_colors = None
        
        # Run local search for a few seconds
        search_deadline = min(deadline, time.time() + 5.0)
        
        while time.time() < search_deadline:
            colors = [random.randint(0, 1) for _ in range(m)]
            
            # Calculate initial violations
            quad_red_counts = [0] * len(quads)
            pent_blue_counts = [0] * len(pents)
            
            for i, c in enumerate(colors):
                if c == 0: # Red
                    for q_idx in edge_to_quads[i]:
                        quad_red_counts[q_idx] += 1
                else: # Blue
                    for p_idx in edge_to_pents[i]:
                        pent_blue_counts[p_idx] += 1
                        
            violations = sum(1 for c in quad_red_counts if c == 4) + sum(1 for c in pent_blue_counts if c == 10)
            
            if violations == 0:
                return {'n': n, 'colors': colors}
                
            if violations < best_violations:
                best_violations = violations
                best_colors = colors[:]
                
            # Local search loop
            local_deadline = min(search_deadline, time.time() + 0.5)
            while time.time() < local_deadline:
                # Pick a random edge to flip
                i = random.randint(0, m - 1)
                old_c = colors[i]
                new_c = 1 - old_c
                
                # Calculate delta violations
                delta_v = 0
                
                if old_c == 0: # Red -> Blue
                    # Quads: red count decreases
                    for q_idx in edge_to_quads[i]:
                        if quad_red_counts[q_idx] == 4:
                            delta_v -= 1 # Violation removed
                    # Pents: blue count increases
                    for p_idx in edge_to_pents[i]:
                        if pent_blue_counts[p_idx] == 9:
                            delta_v += 1 # Violation added
                else: # Blue -> Red
                    # Quads: red count increases
                    for q_idx in edge_to_quads[i]:
                        if quad_red_counts[q_idx] == 3:
                            delta_v += 1 # Violation added
                    # Pents: blue count decreases
                    for p_idx in edge_to_pents[i]:
                        if pent_blue_counts[p_idx] == 10:
                            delta_v -= 1 # Violation removed
                
                # Accept if delta_v < 0 or with some probability if delta_v > 0
                # Simple hill climbing with random restarts is often sufficient for small n.
                # Let's use a greedy move if possible, otherwise random.
                if delta_v <= 0:
                    # Apply move
                    colors[i] = new_c
                    if new_c == 0: # Now Red
                        for q_idx in edge_to_quads[i]:
                            quad_red_counts[q_idx] += 1
                        for p_idx in edge_to_pents[i]:
                            pent_blue_counts[p_idx] -= 1
                    else: # Now Blue
                        for q_idx in edge_to_quads[i]:
                            quad_red_counts[q_idx] -= 1
                        for p_idx in edge_to_pents[i]:
                            pent_blue_counts[p_idx] += 1
                            
                    violations += delta_v
                    
                    if violations == 0:
                        return {'n': n, 'colors': colors}
                        
                    if violations < best_violations:
                        best_violations = violations
                        best_colors = colors[:]
                else:
                    # If we can't improve, maybe try a random move to escape?
                    # Or just break and restart?
                    # Let's stick to greedy for now to keep it simple and fast.
                    # If stuck, the outer loop restarts with a new random init.
                    pass

        # If we didn't find a solution for this n in the allotted time, try smaller n
        # We don't return best_colors because it's not valid.
        
    # If no n >= 2 found, return n=1 (trivially valid)
    return {'n': 1, 'colors': []}
