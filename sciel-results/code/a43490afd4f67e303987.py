import random
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # We will try to find valid colorings for increasing n
    # Use a local search approach with numpy for speed
    
    def try_find_coloring(n, time_budget):
        """Try to find a valid coloring for n vertices within time_budget seconds."""
        if time.time() - start_time > deadline - 5:
            return None
        
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        
        if num_triples == 0:
            return []
        
        # Precompute the structure
        # For each triple (edge), which K4s (red forbidden) and K5s (blue forbidden) does it belong to?
        # A K4 is a set of 4 vertices. The 4 triples within it are the edges of the K4.
        # A K5 is a set of 5 vertices. The 10 triples within it are the edges of the K5.
        
        # Map triple to index
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Build list of K4s and K5s as lists of edge indices
        red_k4s = []
        for quad in combinations(range(n), 4):
            edges = [triple_to_idx[tri] for tri in combinations(quad, 3)]
            red_k4s.append(edges)
            
        blue_k5s = []
        for quint in combinations(range(n), 5):
            edges = [triple_to_idx[tri] for tri in combinations(quint, 3)]
            blue_k5s.append(edges)
        
        # For each edge, list of K4 indices and K5 indices it belongs to
        edge_to_red_k4s = [[] for _ in range(num_triples)]
        for k4_idx, edges in enumerate(red_k4s):
            for e in edges:
                edge_to_red_k4s[e].append(k4_idx)
                
        edge_to_blue_k5s = [[] for _ in range(num_triples)]
        for k5_idx, edges in enumerate(blue_k5s):
            for e in edges:
                edge_to_blue_k5s[e].append(k5_idx)
        
        # Convert to numpy arrays for faster access if possible, but lists of lists are okay for small n
        # Let's stick to lists for simplicity but optimize the inner loop.
        
        # Initialize
        def initialize():
            # Random initialization with bias? 
            # For R(4,5;3), random 0.5 is a good start.
            colors = np.random.randint(0, 2, size=num_triples)
            
            # Calculate initial counts
            red_counts = np.zeros(len(red_k4s), dtype=np.int32)
            blue_counts = np.zeros(len(blue_k5s), dtype=np.int32)
            
            # This initialization is O(num_k4s * 4 + num_k5s * 10)
            # For n=34, num_k4s ~ 46376, num_k5s ~ 278256. 
            # 46376*4 + 278256*10 ~ 3M ops. This might be slow in Python loop.
            # Let's try to vectorize or just accept it for now.
            
            # Actually, let's just loop.
            for i, edges in enumerate(red_k4s):
                cnt = 0
                for e in edges:
                    if colors[e] == 0:
                        cnt += 1
                red_counts[i] = cnt
                
            for i, edges in enumerate(blue_k5s):
                cnt = 0
                for e in edges:
                    if colors[e] == 1:
                        cnt += 1
                blue_counts[i] = cnt
                
            red_violations = np.sum(red_counts == 4)
            blue_violations = np.sum(blue_counts == 10)
            
            return colors, red_counts, blue_counts, red_violations, blue_violations
        
        # Local search loop
        while time.time() - start_time < deadline - 5:
            colors, red_counts, blue_counts, red_violations, blue_violations = initialize()
            total_violations = red_violations + blue_violations
            
            if total_violations == 0:
                return [int(c) for c in colors]
            
            # Greedy repair / Simulated Annealing
            # We want to reduce total_violations.
            # Pick a random edge, calculate delta, accept if delta < 0 or with prob.
            
            # To speed up, we can pick edges involved in violations.
            # But maintaining violation lists is complex.
            # Let's just pick random edges.
            
            steps_since_improvement = 0
            best_total = total_violations
            
            while time.time() - start_time < deadline - 5:
                if total_violations == 0:
                    return [int(c) for c in colors]
                    
                # Pick random edge
                e = np.random.randint(0, num_triples)
                current_color = colors[e]
                new_color = 1 - current_color
                
                delta_red = 0
                delta_blue = 0
                
                # Check red K4s
                for k4_idx in edge_to_red_k4s[e]:
                    cnt = red_counts[k4_idx]
                    if current_color == 0:
                        # Becoming blue, red count decreases
                        if cnt == 4:
                            delta_red -= 1
                    else:
                        # Becoming red, red count increases
                        if cnt == 3:
                            delta_red += 1
                        
                # Check blue K5s
                for k5_idx in edge_to_blue_k5s[e]:
                    cnt = blue_counts[k5_idx]
                    if current_color == 1:
                        # Becoming red, blue count decreases
                        if cnt == 10:
                            delta_blue -= 1
                    else:
                        # Becoming blue, blue count increases
                        if cnt == 9:
                            delta_blue += 1
                            
                total_delta = delta_red + delta_blue
                
                accepted = False
                if total_delta < 0:
                    accepted = True
                    steps_since_improvement = 0
                elif total_delta == 0:
                    if random.random() < 0.5:
                        accepted = True
                else:
                    # Accept worsening moves with small probability to escape local minima
                    # Probability decreases with magnitude of delta
                    # Simple: accept if delta <= 2 and random < 0.1
                    if total_delta <= 2 and random.random() < 0.1:
                        accepted = True
                        steps_since_improvement += 1
                    else:
                        steps_since_improvement += 1
                        
                if accepted:
                    colors[e] = new_color
                    for k4_idx in edge_to_red_k4s[e]:
                        if current_color == 0:
                            red_counts[k4_idx] -= 1
                        else:
                            red_counts[k4_idx] += 1
                    for k5_idx in edge_to_blue_k5s[e]:
                        if current_color == 1:
                            blue_counts[k5_idx] -= 1
                        else:
                            blue_counts[k5_idx] += 1
                            
                    red_violations += delta_red
                    blue_violations += delta_blue
                    total_violations += total_delta
                    
                    if total_violations < best_total:
                        best_total = total_violations
                        steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
                    
                # If stuck for too long, restart this attempt
                if steps_since_improvement > 1000:
                    break
            
        return None

    # Try increasing n
    # Start from a reasonable n and go up.
    # We know n=9 is valid. Let's try 10, 11, ... up to 34.
    # To save time, we can try larger n first if we have a good strategy?
    # No, incremental is safer. But if we fail on n, we might fail on n+1.
    # So we should return the largest n for which we found a valid coloring.
    
    # Let's try n from 10 to 34.
    # Allocate time wisely.
    # If we have 100s left, and we want to try 25 values, that's 4s each.
    # But larger n is harder.
    # Let's just try them in order and break if time is low.
    
    for n in range(10, 35):
        if time.time() - start_time > deadline - 10:
            break
            
        # Give more time to larger n?
        # Or just a fixed chunk?
        # Let's give 2 seconds per n for small n, and more for larger?
        # Actually, let's just run the loop and let it use available time.
        # But we need to make sure we don't spend all time on n=10.
        
        # Let's try to find a coloring for this n.
        # We'll pass a time budget based on remaining time.
        # But the inner loop checks global deadline.
        
        result = try_find_coloring(n, 0) # time_budget is ignored in current impl, uses global deadline
        
        if result is not None:
            best_n = n
            best_colors = result
        else:
            # If we failed for n, it's likely we'll fail for n+1 too?
            # Not necessarily, but usually.
            # However, if we run out of time, we stop.
            # If we just didn't find one in the time allocated (which is dynamic),
            # we should probably stop trying larger n to avoid wasting time.
            # But the inner loop runs until global deadline or stuck.
            # So if it returns None, it means it got stuck or time ran out.
            # If time ran out, the next iteration will break immediately.
            # If it got stuck, we might try again?
            # Let's just break if we didn't find one, to save time for verification or smaller n?
            # No, we want the max n. If we failed n, we can't guarantee n+1.
            # So we should stop.
            break
            
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
