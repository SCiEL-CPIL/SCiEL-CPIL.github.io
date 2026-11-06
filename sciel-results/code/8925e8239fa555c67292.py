import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for overhead/return
    
    # Precompute structures for a given n
    # We will try n from 10 up to 34
    # To save time, we can jump to a higher n if we have time
    # But let's be systematic.
    
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # Try to improve n
    # We will attempt to find a valid coloring for n = 10, 11, ...
    # If we find one, we store it and move to next n.
    # If we run out of time, we return the best found.
    
    # Helper to check validity (for final verification or small n)
    def is_valid(n, colors):
        # colors is a list of 0/1 for triples in lex order
        # Map triple to index
        # This is slow, only use for small n or final check
        # For the search, we use incremental counts.
        pass

    def get_triple_index(a, b, c, n):
        # Lexicographic index of (a,b,c) in combinations(range(n), 3)
        # Formula: C(a, 3) + C(b, 2) + c ... no, that's for combinations of different sizes?
        # Standard combinatorial number system.
        # Index of (a,b,c) with a<b<c in C(n,3)
        # Sum_{i=0}^{a-1} C(n-1-i, 2) + Sum_{j=a+1}^{b-1} C(n-1-j, 1) + (c - b - 1)
        # Actually simpler:
        # idx = 0
        # for i in range(a): idx += C(n-1-i, 2)
        # for j in range(a+1, b): idx += C(n-1-j, 1)
        # idx += c - b - 1
        # This is O(n). For n=34, it's fast enough.
        # Or precompute a dict.
        pass

    # Let's precompute the list of triples and their indices for a given n
    # And for each K4 and K5, the list of edge indices.
    
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
            
        return triples, red_k4s, blue_k5s

    def solve_for_n(n):
        triples, red_k4s, blue_k5s = prepare_data(n)
        num_triples = len(triples)
        
        # Initialize random coloring
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # Initialize counts
        # red_counts[k4_idx] = number of red edges in that K4
        red_counts = [0] * len(red_k4s)
        blue_counts = [0] * len(blue_k5s)
        
        # Precompute which K4s and K5s contain which edge
        edge_to_red_k4s = [[] for _ in range(num_triples)]
        edge_to_blue_k5s = [[] for _ in range(num_triples)]
        
        for i, edges in enumerate(red_k4s):
            for e in edges:
                edge_to_red_k4s[e].append(i)
                
        for i, edges in enumerate(blue_k5s):
            for e in edges:
                edge_to_blue_k5s[e].append(i)
                
        # Initial counts
        red_violations = 0
        blue_violations = 0
        
        for i, edges in enumerate(red_k4s):
            cnt = 0
            for e in edges:
                if colors[e] == 0: # Red
                    cnt += 1
            red_counts[i] = cnt
            if cnt == 4:
                red_violations += 1
                
        for i, edges in enumerate(blue_k5s):
            cnt = 0
            for e in edges:
                if colors[e] == 1: # Blue
                    cnt += 1
            blue_counts[i] = cnt
            if cnt == 10:
                blue_violations += 1
                
        total_violations = red_violations + blue_violations
        
        # Local Search
        # We want to minimize total_violations to 0.
        # Use a greedy approach with some randomness to escape local minima.
        
        # If initial is valid, return immediately
        if total_violations == 0:
            return colors
            
        # Simulated Annealing / Greedy
        # Temperature starts high, decreases.
        # Or just simple hill climbing with random restarts if stuck.
        # Given the small size, simple hill climbing with random flips might work.
        
        # Let's try:
        # While time < deadline:
        #   Pick a random edge e
        #   Calculate delta in violations
        #   If delta < 0, accept
        #   Else if delta == 0, accept with prob 0.5
        #   Else (delta > 0), accept with prob exp(-delta/T) or just small prob if we are stuck
        
        # To be efficient, we can pick an edge involved in a violation.
        # Pick a random violation, pick a random edge in it, flip it.
        # This is a "repair" strategy.
        
        import math
        
        stuck_count = 0
        last_improvement = time.time()
        
        while time.time() < deadline - 2:
            if total_violations == 0:
                return colors
                
            # Pick a random violation
            if red_violations > 0 and (blue_violations == 0 or random.random() < red_violations / total_violations):
                # Pick a random red violation
                # We need a list of indices of red violations to pick from
                # Maintaining this list is overhead. 
                # Instead, pick a random K4 index and check if it's a violation?
                # Or just pick a random edge from a random K4?
                
                # Let's maintain a list of violating K4 indices and K5 indices.
                # Update it when counts change.
                pass
            
            # This "maintain list" approach is better.
            
            # Let's restructure to maintain lists of violating indices.
            # This allows O(1) pick of a violation.
            
            # Re-initialize with lists
            red_viol_idx = [i for i in range(len(red_k4s)) if red_counts[i] == 4]
            blue_viol_idx = [i for i in range(len(blue_k5s)) if blue_counts[i] == 10]
            
            # We need to update these lists when we flip an edge.
            # This is tricky because an edge flip affects multiple K4s/K5s.
            # We can just re-scan the affected ones.
            
            # Let's stick to the simpler "pick random edge, calculate delta" approach.
            # It's O(1) average to pick edge.
            # Delta calculation is O(deg(e)) which is small.
            
            e = random.randint(0, num_triples - 1)
            current_color = colors[e]
            new_color = 1 - current_color
            
            delta_red = 0
            delta_blue = 0
            
            # Check red K4s
            for k4_idx in edge_to_red_k4s[e]:
                cnt = red_counts[k4_idx]
                if current_color == 0: # Was red, becoming blue
                    # cnt decreases by 1
                    if cnt == 4:
                        delta_red -= 1
                else: # Was blue, becoming red
                    # cnt increases by 1
                    if cnt == 3:
                        delta_red += 1
            
            # Check blue K5s
            for k5_idx in edge_to_blue_k5s[e]:
                cnt = blue_counts[k5_idx]
                if current_color == 1: # Was blue, becoming red
                    # cnt decreases by 1
                    if cnt == 10:
                        delta_blue -= 1
                else: # Was red, becoming blue
                    # cnt increases by 1
                    if cnt == 9:
                        delta_blue += 1
            
            total_delta = delta_red + delta_blue
            
            # Acceptance criterion
            accepted = False
            if total_delta < 0:
                accepted = True
            elif total_delta == 0:
                if random.random() < 0.5:
                    accepted = True
            else:
                # Accept worsening moves to escape local minima
                # Probability decreases with delta
                # Simple: accept with prob 0.1 if delta <= 2, else 0
                if total_delta <= 2 and random.random() < 0.1:
                    accepted = True
            
            if accepted:
                colors[e] = new_color
                # Update counts
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
                
                if total_violations < best_violations_found:
                    best_violations_found = total_violations
                    last_improvement = time.time()
                    
                # If stuck for too long, restart?
                # Or just keep going.
            
        return None # Failed to find valid coloring within time

    # Main loop: try increasing n
    # We need to be careful with time.
    # Let's allocate time per n.
    # Start with n=10.
    # If we find a valid one, save it and try n+1.
    # If we don't find one, maybe try again with different random seed?
    # Or just move to next n if we spent enough time?
    
    # Actually, if we can't find a valid coloring for n, we can't guarantee one for n+1.
    # But the search space is connected? Not necessarily.
    # However, usually if you can do n, you can likely do n+1 with more search.
    # But if you fail for n, you might fail for n+1.
    # So it's safer to return the last VALID n.
    
    # Let's try n from 10 to 34.
    # Time budget per n: 5 seconds?
    # 25 n's * 5s = 125s. Too much.
    # Let's do 2s per n. 25 * 2 = 50s. Leaves 60s for the last one or retries.
    
    # Better: 
    # Try n=10. If success, try n=11. 
    # If fail, try n=10 again? Or skip?
    # If we fail n=10, it's likely easy to find, so maybe random restarts help.
    
    # Let's just run a single long search for a high n, say 20?
    # No, incremental is better.
    
    # Strategy:
    # 1. Find valid coloring for n=10 (should be fast).
    # 2. Extend to n=11? No, just start fresh for n=11.
    # 3. Keep going until time is up or n=34.
    
    for n in range(10, 35):
        if time.time() > deadline - 5:
            break
            
        # Allocate time for this n
        # If n is small, give less time. If n is large, give more?
        # Actually, larger n is harder.
        # Let's give 2 seconds for n < 20, 5 seconds for n >= 20.
        
        t_start = time.time()
        t_end = min(deadline - 2, t_start + 2 if n < 20 else t_start + 5)
        
        # Modify solve_for_n to take a deadline
        # Or just run it and check time inside.
        
        # Let's refactor solve_for_n to accept a deadline
        # And return colors or None
        
        # To avoid rewriting, let's just call a version that checks time.
        # I'll rewrite the inner loop to check `time.time() < t_end`.
        
        # Since I can't easily refactor inside the single function without making it messy,
        # I will implement the search directly here for the current n.
        
        triples, red_k4s, blue_k5s = prepare_data(n)
        num_triples = len(triples)
        
        # Random restarts within the time limit for this n
        found = False
        while time.time() < t_end and not found:
            colors = [random.randint(0, 1) for _ in range(num_triples)]
            
            red_counts = [0] * len(red_k4s)
            blue_counts = [0] * len(blue_k5s)
            
            # Precompute edge-to-subgraph maps for this n
            edge_to_red_k4s = [[] for _ in range(num_triples)]
            edge_to_blue_k5s = [[] for _ in range(num_triples)]
            
            for i, edges in enumerate(red_k4s):
                for e in edges:
                    edge_to_red_k4s[e].append(i)
                    
            for i, edges in enumerate(blue_k5s):
                for e in edges:
                    edge_to_blue_k5s[e].append(i)
                    
            red_violations = 0
            blue_violations = 0
            
            for i, edges in enumerate(red_k4s):
                cnt = sum(1 for e in edges if colors[e] == 0)
                red_counts[i] = cnt
                if cnt == 4:
                    red_violations += 1
                    
            for i, edges in enumerate(blue_k5s):
                cnt = sum(1 for e in edges if colors[e] == 1)
                blue_counts[i] = cnt
                if cnt == 10:
                    blue_violations += 1
                    
            total_violations = red_violations + blue_violations
            
            if total_violations == 0:
                best_n = n
                best_colors = colors
                found = True
                break
                
            # Local search
            # Use a simple hill climbing
            # If stuck, restart (break out of local search loop)
            
            stuck = 0
            while time.time() < t_end:
                if total_violations == 0:
                    best_n = n
                    best_colors = colors
                    found = True
                    break
                    
                e = random.randint(0, num_triples - 1)
                current_color = colors[e]
                new_color = 1 - current_color
                
                delta_red = 0
                delta_blue = 0
                
                for k4_idx in edge_to_red_k4s[e]:
                    cnt = red_counts[k4_idx]
                    if current_color == 0:
                        if cnt == 4: delta_red -= 1
                    else:
                        if cnt == 3: delta_red += 1
                        
                for k5_idx in edge_to_blue_k5s[e]:
                    cnt = blue_counts[k5_idx]
                    if current_color == 1:
                        if cnt == 10: delta_blue -= 1
                    else:
                        if cnt == 9: delta_blue += 1
                        
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
                    if total_delta <= 2 and random.random() < 0.1:
                        accepted = True
                        stuck += 1
                    else:
                        stuck += 1
                        
                if accepted:
                    colors[e] = new_color
                    for k4_idx in edge_to_red_k4s[e]:
                        if current_color == 0: red_counts[k4_idx] -= 1
                        else: red_counts[k4_idx] += 1
                    for k5_idx in edge_to_blue_k5s[e]:
                        if current_color == 1: blue_counts[k5_idx] -= 1
                        else: blue_counts[k5_idx] += 1
                    red_violations += delta_red
                    blue_violations += delta_blue
                    total_violations += total_delta
                else:
                    stuck += 1
                    
                # If stuck too long, restart randomization
                if stuck > 100:
                    break
        
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
