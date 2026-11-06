import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # The known maximum is 34. We aim for 34.
    # If we fail on 34 quickly, we can try 33, etc., but 34 is the target.
    
    for n in [34, 33, 32, 31, 30]:
        if time.time() > deadline - 15.0:
            break
            
        # Precomputation
        edges = list(combinations(range(n), 3))
        m = len(edges)
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        quads = list(combinations(range(n), 4))
        pents = list(combinations(range(n), 5))
        n_quads = len(quads)
        n_pents = len(pents)
        
        # Map edge index to list of quad indices and pent indices
        edge_to_quads = [[] for _ in range(m)]
        for q_idx, q in enumerate(quads):
            a,b,c,d = q
            q_edges = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
            for e in q_edges:
                e_idx = edge_to_idx[e]
                edge_to_quads[e_idx].append(q_idx)
                
        edge_to_pents = [[] for _ in range(m)]
        for p_idx, p in enumerate(pents):
            p_edges = list(combinations(p, 3))
            for e in p_edges:
                e_idx = edge_to_idx[e]
                edge_to_pents[e_idx].append(p_idx)

        # Also need reverse: for each quad, which edges it contains
        quad_edges = []
        for q in quads:
            a,b,c,d = q
            q_edges = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
            quad_edges.append([edge_to_idx[e] for e in q_edges])
            
        pent_edges = []
        for p in pents:
            p_edges = list(combinations(p, 3))
            pent_edges.append([edge_to_idx[e] for e in p_edges])

        # Degree-based initialization: count how many quads/pents each edge participates in
        # Edges in more quads should be less likely red (to avoid red K4)
        # Edges in more pents should be less likely blue (to avoid blue K5)
        quad_deg = [len(edge_to_quads[i]) for i in range(m)]
        pent_deg = [len(edge_to_pents[i]) for i in range(m)]
        
        # Normalize
        max_qdeg = max(quad_deg) if quad_deg else 1
        max_pdeg = max(pent_deg) if pent_deg else 1
        
        # Probability of being red: lower for high quad degree, higher for low pent degree
        # P(red) = (1 - quad_deg/max_qdeg) * 0.5 + (pent_deg/max_pdeg) * 0.5
        # This biases toward red when pent degree is high (fewer blue constraints) 
        # and toward blue when quad degree is high (more red constraints)
        
        best_violations = float('inf')
        best_colors = None
        found_solution = False
        
        search_deadline = deadline - 3.0
        
        # Multiple restarts
        n_restarts = 0
        while time.time() < search_deadline and not found_solution:
            n_restarts += 1
            
            # Degree-based initialization with randomness
            colors = []
            for i in range(m):
                p_red = (1.0 - quad_deg[i]/max_qdeg) * 0.5 + (pent_deg[i]/max_pdeg) * 0.5
                # Add some randomness
                p_red = 0.7 * p_red + 0.3 * random.random()
                colors.append(0 if random.random() < p_red else 1)
            
            # Calculate initial violations
            quad_red_counts = [0] * n_quads
            pent_blue_counts = [0] * n_pents
            
            for i, c in enumerate(colors):
                if c == 0:  # Red
                    for q_idx in edge_to_quads[i]:
                        quad_red_counts[q_idx] += 1
                else:  # Blue
                    for p_idx in edge_to_pents[i]:
                        pent_blue_counts[p_idx] += 1
                        
            violations = sum(1 for c in quad_red_counts if c == 4) + sum(1 for c in pent_blue_counts if c == 10)
            
            if violations == 0:
                return {'n': n, 'colors': colors}
                
            if violations < best_violations:
                best_violations = violations
                best_colors = colors[:]
                
            # Conflict-driven repair: iteratively fix violations
            T = 1.0
            cool_rate = 0.9999
            min_T = 0.01
            steps = 0
            stall_count = 0
            last_violations = violations
            
            while T > min_T and time.time() < search_deadline:
                # Find a violating quad or pent and try to fix it
                # Pick a random violating quad or pent
                violating_quads = [i for i, c in enumerate(quad_red_counts) if c == 4]
                violating_pents = [i for i, c in enumerate(pent_blue_counts) if c == 10]
                
                if not violating_quads and not violating_pents:
                    found_solution = True
                    break
                
                # Choose between fixing a quad or pent (prioritize more frequent violation type)
                if violating_quads and (not violating_pents or random.random() < len(violating_quads) / (len(violating_quads) + len(violating_pents))):
                    q_idx = random.choice(violating_quads)
                    # This quad has all 4 edges red. Flip one of them to blue.
                    e_indices = quad_edges[q_idx]
                    # Choose edge to flip: prefer edges with lower quad degree (less impact on other quads)
                    # and higher pent degree (more benefit from being blue)
                    scores = []
                    for e_idx in e_indices:
                        score = -quad_deg[e_idx] + pent_deg[e_idx]
                        scores.append(score)
                    # Weighted random selection
                    total_score = sum(max(s, 0) for s in scores)
                    if total_score == 0:
                        e_idx = random.choice(e_indices)
                    else:
                        r = random.random() * total_score
                        cum = 0
                        e_idx = e_indices[0]
                        for k, s in enumerate(scores):
                            w = max(s, 0)
                            cum += w
                            if r <= cum:
                                e_idx = e_indices[k]
                                break
                else:
                    p_idx = random.choice(violating_pents)
                    # This pent has all 10 edges blue. Flip one of them to red.
                    e_indices = pent_edges[p_idx]
                    # Choose edge to flip: prefer edges with higher quad degree (more impact)
                    # and lower pent degree (less benefit from being blue)
                    scores = []
                    for e_idx in e_indices:
                        score = quad_deg[e_idx] - pent_deg[e_idx]
                        scores.append(score)
                    total_score = sum(max(s, 0) for s in scores)
                    if total_score == 0:
                        e_idx = random.choice(e_indices)
                    else:
                        r = random.random() * total_score
                        cum = 0
                        e_idx = e_indices[0]
                        for k, s in enumerate(scores):
                            w = max(s, 0)
                            cum += w
                            if r <= cum:
                                e_idx = e_indices[k]
                                break
                
                old_c = colors[e_idx]
                new_c = 1 - old_c
                
                # Calculate delta violations
                delta_v = 0
                
                if old_c == 0:  # Red -> Blue
                    for q_idx2 in edge_to_quads[e_idx]:
                        if quad_red_counts[q_idx2] == 4:
                            delta_v -= 1
                else:  # Blue -> Red
                    for q_idx2 in edge_to_quads[e_idx]:
                        if quad_red_counts[q_idx2] == 3:
                            delta_v += 1
                
                if old_c == 1:  # Blue -> Red
                    for p_idx2 in edge_to_pents[e_idx]:
                        if pent_blue_counts[p_idx2] == 10:
                            delta_v -= 1
                else:  # Red -> Blue
                    for p_idx2 in edge_to_pents[e_idx]:
                        if pent_blue_counts[p_idx2] == 9:
                            delta_v += 1
                            
                # Acceptance: always accept if it reduces violations, otherwise probabilistic
                if delta_v < 0:
                    accept = True
                elif delta_v == 0:
                    accept = random.random() < 0.5
                else:
                    if delta_v > 20 * T:
                        accept = False
                    else:
                        accept = random.random() < (2.718281828 ** (-delta_v / T))
                        
                if accept:
                    colors[e_idx] = new_c
                    if new_c == 0:  # Now Red
                        for q_idx2 in edge_to_quads[e_idx]:
                            quad_red_counts[q_idx2] += 1
                        for p_idx2 in edge_to_pents[e_idx]:
                            pent_blue_counts[p_idx2] -= 1
                    else:  # Now Blue
                        for q_idx2 in edge_to_quads[e_idx]:
                            quad_red_counts[q_idx2] -= 1
                        for p_idx2 in edge_to_pents[e_idx]:
                            pent_blue_counts[p_idx2] += 1
                            
                    violations += delta_v
                    
                    if violations == 0:
                        found_solution = True
                        break
                        
                    if violations < best_violations:
                        best_violations = violations
                        best_colors = colors[:]
                    
                    if violations < last_violations:
                        stall_count = 0
                        last_violations = violations
                    else:
                        stall_count += 1
                    
                T *= cool_rate
                steps += 1
                
                # If stuck for too long, do a random perturbation
                if stall_count > 200:
                    for _ in range(3):
                        j = random.randint(0, m - 1)
                        old_cj = colors[j]
                        new_cj = 1 - old_cj
                        colors[j] = new_cj
                        if new_cj == 0:
                            for q_idx2 in edge_to_quads[j]:
                                quad_red_counts[q_idx2] += 1
                            for p_idx2 in edge_to_pents[j]:
                                pent_blue_counts[p_idx2] -= 1
                        else:
                            for q_idx2 in edge_to_quads[j]:
                                quad_red_counts[q_idx2] -= 1
                            for p_idx2 in edge_to_pents[j]:
                                pent_blue_counts[p_idx2] += 1
                    # Recalculate violations
                    violations = sum(1 for c in quad_red_counts if c == 4) + sum(1 for c in pent_blue_counts if c == 10)
                    stall_count = 0
                    last_violations = violations
                    T = max(T, 0.5)

    # If we found a solution, it was already returned
    # Return the best valid n we can guarantee
    # For n <= 3, any coloring is valid
    # For n=4, we need no red K4, so not all 4 edges red
    # For n=5, we need no blue K5, so not all 10 edges blue
    
    # Fallback: return a simple valid solution
    # n=3 is always valid
    return {'n': 3, 'colors': [0, 0, 0]}
