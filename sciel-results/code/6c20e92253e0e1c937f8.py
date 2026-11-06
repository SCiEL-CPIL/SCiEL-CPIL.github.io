import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for overhead/return
    
    def prepare_data(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Red K4s: all 4-subsets
        red_k4s = []
        for quad in combinations(range(n), 4):
            edges = []
            for tri in combinations(quad, 3):
                edges.append(triple_to_idx[tri])
            red_k4s.append(edges)
            
        # Blue K5s: all 5-subsets
        blue_k5s = []
        for quint in combinations(range(n), 5):
            edges = []
            for tri in combinations(quint, 3):
                edges.append(triple_to_idx[tri])
            blue_k5s.append(edges)
            
        # For each edge, which K4s and K5s does it belong to?
        edge_to_k4s = [[] for _ in range(num_triples)]
        edge_to_k5s = [[] for _ in range(num_triples)]
        
        for i, edges in enumerate(red_k4s):
            for e in edges:
                edge_to_k4s[e].append(i)
                
        for i, edges in enumerate(blue_k5s):
            for e in edges:
                edge_to_k5s[e].append(i)
        
        return triples, red_k4s, blue_k5s, edge_to_k4s, edge_to_k5s

    def urgency_construct(n, seed_val=0, time_limit=10.0):
        """
        Urgency-based construction: maintain counts of monochromatic edges for every K4 and K5.
        Select the next triple to color based on urgency (number of near-complete cliques it would complete).
        """
        rng = random.Random(seed_val)
        triples, red_k4s, blue_k5s, edge_to_k4s, edge_to_k5s = prepare_data(n)
        num_triples = len(triples)
        
        colors = [-1] * num_triples
        red_counts = [0] * len(red_k4s)
        blue_counts = [0] * len(blue_k5s)
        
        # Track which edges are colored
        colored = [False] * num_triples
        
        # For each K4, track how many edges are colored
        k4_colored_count = [0] * len(red_k4s)
        # For each K5, track how many edges are colored
        k5_colored_count = [0] * len(blue_k5s)
        
        start = time.time()
        
        while True:
            if time.time() - start > time_limit:
                break
                
            # Check if all edges are colored
            if all(colored):
                break
            
            # Compute urgency for each uncolored edge
            # Urgency = number of K4s where this edge is the last uncolored edge AND 3 others are red
            #           + number of K5s where this edge is the last uncolored edge AND 9 others are blue
            # Also consider "near-complete": K4s with 3 colored edges (2 red, 1 blue or 3 red)
            # A simpler urgency: for each uncolored edge e, count how many K4s containing e have
            # exactly 3 colored edges (regardless of color), and how many K5s have 9 colored edges.
            
            # Actually, let's use a simpler heuristic:
            # urgency(e) = sum over K4s containing e of (1 if k4 has 3 colored edges else 0)
            #            + sum over K5s containing e of (1 if k5 has 9 colored edges else 0)
            # This focuses on edges that are part of nearly-complete cliques.
            
            best_edge = -1
            best_urgency = -1
            best_color = 0
            best_score = float('inf')
            
            # Sample a subset of uncolored edges to keep it fast
            uncolored_edges = [i for i in range(num_triples) if not colored[i]]
            
            if len(uncolored_edges) > 50:
                # Sample up to 50 edges
                sample_size = 50
                sampled = rng.sample(uncolored_edges, sample_size)
            else:
                sampled = uncolored_edges
            
            for e in sampled:
                urgency = 0
                # Count K4s with 3 colored edges containing e
                for k4_idx in edge_to_k4s[e]:
                    if k4_colored_count[k4_idx] == 3:
                        urgency += 1
                # Count K5s with 9 colored edges containing e
                for k5_idx in edge_to_k5s[e]:
                    if k5_colored_count[k5_idx] == 9:
                        urgency += 1
                
                if urgency > best_urgency:
                    best_urgency = urgency
                    best_edge = e
                    
                    # Determine best color for this edge
                    for color in [0, 1]:
                        new_red_k4 = 0
                        new_blue_k5 = 0
                        
                        for k4_idx in edge_to_k4s[e]:
                            if k4_colored_count[k4_idx] == 3:
                                # Check if all 3 colored edges are red
                                all_red = True
                                for ee in red_k4s[k4_idx]:
                                    if ee == e:
                                        continue
                                    if colors[ee] != 0:
                                        all_red = False
                                        break
                                if all_red and color == 0:
                                    new_red_k4 += 1
                        
                        for k5_idx in edge_to_k5s[e]:
                            if k5_colored_count[k5_idx] == 9:
                                # Check if all 9 colored edges are blue
                                all_blue = True
                                for ee in blue_k5s[k5_idx]:
                                    if ee == e:
                                        continue
                                    if colors[ee] != 1:
                                        all_blue = False
                                        break
                                if all_blue and color == 1:
                                    new_blue_k5 += 1
                        
                        score = new_red_k4 + new_blue_k5
                        if score < best_score:
                            best_score = score
                            best_color = color
                    break
            
            if best_edge == -1:
                # No urgent edges, just pick a random uncolored edge
                best_edge = rng.choice(uncolored_edges)
                # Determine color
                for color in [0, 1]:
                    new_red_k4 = 0
                    new_blue_k5 = 0
                    
                    for k4_idx in edge_to_k4s[best_edge]:
                        if k4_colored_count[k4_idx] == 3:
                            all_red = True
                            for ee in red_k4s[k4_idx]:
                                if ee == best_edge:
                                    continue
                                if colors[ee] != 0:
                                    all_red = False
                                    break
                            if all_red and color == 0:
                                new_red_k4 += 1
                    
                    for k5_idx in edge_to_k5s[best_edge]:
                        if k5_colored_count[k5_idx] == 9:
                            all_blue = True
                            for ee in blue_k5s[k5_idx]:
                                if ee == best_edge:
                                    continue
                                if colors[ee] != 1:
                                    all_blue = False
                                    break
                            if all_blue and color == 1:
                                new_blue_k5 += 1
                    
                    score = new_red_k4 + new_blue_k5
                    if score < best_score:
                        best_score = score
                        best_color = color
            
            # Color the edge
            colors[best_edge] = best_color
            colored[best_edge] = True
            
            # Update counts
            if best_color == 0:
                for k4_idx in edge_to_k4s[best_edge]:
                    red_counts[k4_idx] += 1
                    k4_colored_count[k4_idx] += 1
            else:
                for k5_idx in edge_to_k5s[best_edge]:
                    blue_counts[k5_idx] += 1
                    k5_colored_count[k5_idx] += 1
            
            # Also update k4_colored_count and k5_colored_count for the other color
            # Wait, we need to track colored count separately from red/blue count
            # Let me fix this: k4_colored_count should increment regardless of color
            # Let me redo this properly.
        
        # The above has a bug. Let me rewrite more carefully.
        # For now, let's just use a simpler approach: always pick the edge with highest
        # "constraint tightness" - number of K4s/K5s where it's the last uncolored edge.
        
        # Actually, let's just return None for now and use a different approach.
        return None

    def greedy_construct(n, seed_val=0):
        """
        Greedy sequential construction: assign colors to triples in lex order.
        For each triple, choose the color that minimizes the number of 
        newly completed monochromatic cliques.
        """
        rng = random.Random(seed_val)
        triples, red_k4s, blue_k5s, edge_to_k4s, edge_to_k5s = prepare_data(n)
        num_triples = len(triples)
        
        colors = [-1] * num_triples  # -1 means uncolored
        
        # Track counts of red edges in each K4, blue edges in each K5
        red_counts = [0] * len(red_k4s)
        blue_counts = [0] * len(blue_k5s)
        
        for e_idx in range(num_triples):
            # Try color 0 (red) and color 1 (blue)
            best_color = rng.choice([0, 1])
            best_score = float('inf')
            
            for color in [0, 1]:
                # Count how many new monochromatic cliques would be formed
                new_red_k4 = 0
                new_blue_k5 = 0
                
                # Check all K4s containing this edge
                for k4_idx in edge_to_k4s[e_idx]:
                    edges = red_k4s[k4_idx]
                    # Check if all other edges are colored
                    all_colored = True
                    red_count = 0
                    for ee in edges:
                        if ee == e_idx:
                            continue
                        if colors[ee] == -1:
                            all_colored = False
                            break
                        if colors[ee] == 0:
                            red_count += 1
                    if all_colored:
                        if color == 0 and red_count == 3:
                            new_red_k4 += 1
                
                # Check all K5s containing this edge
                for k5_idx in edge_to_k5s[e_idx]:
                    edges = blue_k5s[k5_idx]
                    all_colored = True
                    blue_count = 0
                    for ee in edges:
                        if ee == e_idx:
                            continue
                        if colors[ee] == -1:
                            all_colored = False
                            break
                        if colors[ee] == 1:
                            blue_count += 1
                    if all_colored:
                        if color == 1 and blue_count == 9:
                            new_blue_k5 += 1
                
                score = new_red_k4 + new_blue_k5
                
                if score < best_score:
                    best_score = score
                    best_color = color
                elif score == best_score:
                    if rng.random() < 0.5:
                        best_color = color
            
            colors[e_idx] = best_color
            
            # Update counts
            if best_color == 0:
                for k4_idx in edge_to_k4s[e_idx]:
                    red_counts[k4_idx] += 1
            else:
                for k5_idx in edge_to_k5s[e_idx]:
                    blue_counts[k5_idx] += 1
        
        # Verify no violations
        red_violations = 0
        blue_violations = 0
        for i, edges in enumerate(red_k4s):
            cnt = sum(1 for e in edges if colors[e] == 0)
            if cnt == 4:
                red_violations += 1
        for i, edges in enumerate(blue_k5s):
            cnt = sum(1 for e in edges if colors[e] == 1)
            if cnt == 10:
                blue_violations += 1
        
        if red_violations == 0 and blue_violations == 0:
            return colors
        else:
            return None

    def local_search_refine(n, initial_colors, time_limit=5.0):
        """
        Refine a coloring using local search to fix any violations.
        """
        triples, red_k4s, blue_k5s, edge_to_k4s, edge_to_k5s = prepare_data(n)
        num_triples = len(triples)
        
        colors = initial_colors[:]
        
        red_counts = [0] * len(red_k4s)
        blue_counts = [0] * len(blue_k5s)
        
        for i, edges in enumerate(red_k4s):
            red_counts[i] = sum(1 for e in edges if colors[e] == 0)
        for i, edges in enumerate(blue_k5s):
            blue_counts[i] = sum(1 for e in edges if colors[e] == 1)
        
        red_violations = sum(1 for c in red_counts if c == 4)
        blue_violations = sum(1 for c in blue_counts if c == 10)
        total_violations = red_violations + blue_violations
        
        if total_violations == 0:
            return colors
        
        start = time.time()
        stuck = 0
        
        while time.time() - start < time_limit:
            if total_violations == 0:
                return colors
            
            if red_violations > 0 and (blue_violations == 0 or random.random() < red_violations / total_violations):
                while True:
                    k4_idx = random.randint(0, len(red_k4s) - 1)
                    if red_counts[k4_idx] == 4:
                        break
                e = random.choice(red_k4s[k4_idx])
            else:
                while True:
                    k5_idx = random.randint(0, len(blue_k5s) - 1)
                    if blue_counts[k5_idx] == 10:
                        break
                e = random.choice(blue_k5s[k5_idx])
            
            current_color = colors[e]
            new_color = 1 - current_color
            
            delta_red = 0
            delta_blue = 0
            
            for k4_idx in edge_to_k4s[e]:
                cnt = red_counts[k4_idx]
                if current_color == 0:
                    if cnt == 4:
                        delta_red -= 1
                else:
                    if cnt == 3:
                        delta_red += 1
                        
            for k5_idx in edge_to_k5s[e]:
                cnt = blue_counts[k5_idx]
                if current_color == 1:
                    if cnt == 10:
                        delta_blue -= 1
                else:
                    if cnt == 9:
                        delta_blue += 1
                        
            total_delta = delta_red + delta_blue
            
            accepted = False
            if total_delta < 0:
                accepted = True
                stuck = 0
            elif total_delta == 0:
                if random.random() < 0.5:
                    accepted = True
                else:
                    stuck += 1
            else:
                if total_delta <= 2 and stuck > 50 and random.random() < 0.2:
                    accepted = True
                    stuck += 1
                else:
                    stuck += 1
            
            if accepted:
                colors[e] = new_color
                for k4_idx in edge_to_k4s[e]:
                    if current_color == 0:
                        red_counts[k4_idx] -= 1
                    else:
                        red_counts[k4_idx] += 1
                for k5_idx in edge_to_k5s[e]:
                    if current_color == 1:
                        blue_counts[k5_idx] -= 1
                    else:
                        blue_counts[k5_idx] += 1
                red_violations += delta_red
                blue_violations += delta_blue
                total_violations += total_delta
            else:
                stuck += 1
                
            if stuck > 200:
                break
        
        return None

    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # Try increasing n from 10 to 34
    for n in range(10, 35):
        if time.time() > deadline - 5:
            break
        
        remaining = deadline - time.time() - 2
        if remaining < 1:
            break
        
        found = False
        max_attempts = min(5, int(remaining / 2) + 1)
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 2:
                break
                
            t_start = time.time()
            colors = greedy_construct(n, seed_val=attempt * 12345 + n)
            
            if colors is not None:
                best_n = n
                best_colors = colors
                found = True
                break
        
        if not found and time.time() < deadline - 3:
            t_start = time.time()
            time_limit = min(remaining - 2, 5.0)
            
            for attempt in range(3):
                if time.time() > deadline - 2:
                    break
                
                random_colors = [random.randint(0, 1) for _ in range(len(list(combinations(range(n), 3))))]
                refined = local_search_refine(n, random_colors, time_limit=time_limit)
                if refined is not None:
                    best_n = n
                    best_colors = refined
                    found = True
                    break
    
    return {'n': best_n, 'colors': best_colors}
