import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    n = 16
    num_edges = n * (n - 1) // 2  # 120
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {(i, j): k for k, (i, j) in enumerate(edges)}
    
    # Precompute K4s
    k4s = []
    k4_edges = []
    edge_to_k4s = [[] for _ in range(num_edges)]
    
    k4_idx = 0
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        e_indices = [
            edge_index[(a, b)], edge_index[(a, c)], edge_index[(a, d)],
            edge_index[(b, c)], edge_index[(b, d)], edge_index[(c, d)]
        ]
        k4s.append(quad)
        k4_edges.append(e_indices)
        for e in e_indices:
            edge_to_k4s[e].append(k4_idx)
        k4_idx += 1
        
    num_k4s = len(k4s)
    
    def count_violations(colors):
        """Count total violations across all K4s."""
        total_viol = 0
        for k4_i in range(num_k4s):
            distinct = len(set(colors[e] for e in k4_edges[k4_i]))
            if distinct < 5:
                total_viol += (5 - distinct)
        return total_viol
    
    def is_valid(colors):
        """Check if coloring is valid (every K4 has >= 5 distinct colors)."""
        for k4_i in range(num_k4s):
            if len(set(colors[e] for e in k4_edges[k4_i])) < 5:
                return False
        return True
    
    def colors_used(colors):
        return len(set(colors))
    
    # Baseline: (i+j) mod 17
    baseline = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    best_result = baseline[:]
    best_colors = 17
    
    # Try to improve using greedy + local search
    for target_C in [13, 14, 15, 16]:
        if time.time() > deadline - 10:
            break
        
        # Skip if we already have a solution with fewer colors
        if best_colors <= target_C:
            continue
        
        attempts = 0
        max_attempts = 300 if target_C <= 14 else 100
        
        while attempts < max_attempts and time.time() < deadline - 10:
            attempts += 1
            
            # Greedy construction
            random.seed(random.randint(0, 2**31 - 1))
            edge_order = list(range(num_edges))
            random.shuffle(edge_order)
            
            colors = [-1] * num_edges
            success = True
            
            for e_idx in edge_order:
                # Try each color, prefer ones that help most K4s
                color_scores = []
                for c in range(target_C):
                    # Check feasibility: would this violate any complete K4?
                    feasible = True
                    for k4_i in edge_to_k4s[e_idx]:
                        e_indices = k4_edges[k4_i]
                        k4_colors = []
                        all_colored = True
                        for e in e_indices:
                            if e == e_idx:
                                k4_colors.append(c)
                            else:
                                col = colors[e]
                                if col == -1:
                                    all_colored = False
                                    break
                                k4_colors.append(col)
                        if all_colored:
                            if len(set(k4_colors)) < 5:
                                feasible = False
                                break
                    if feasible:
                        # Score: prefer colors that appear less often in adjacent K4s
                        # to maximize diversity
                        score = 0
                        for k4_i in edge_to_k4s[e_idx]:
                            e_indices = k4_edges[k4_i]
                            for e in e_indices:
                                if e != e_idx and colors[e] != -1:
                                    if colors[e] == c:
                                        score -= 1
                                    else:
                                        score += 0.1
                        color_scores.append((score, c))
                
                if not color_scores:
                    success = False
                    break
                
                # Sort by score descending, pick best with some randomness
                color_scores.sort(reverse=True)
                # Pick from top 3 with some probability
                top_k = min(3, len(color_scores))
                if random.random() < 0.7:
                    chosen_c = color_scores[0][1]
                elif random.random() < 0.5:
                    chosen_c = color_scores[1][1]
                else:
                    chosen_c = color_scores[random.randint(0, top_k - 1)][1]
                
                colors[e_idx] = chosen_c
            
            if not success:
                continue
            
            # Verify
            if not is_valid(colors):
                continue
            
            used = colors_used(colors)
            if used < best_colors:
                best_colors = used
                best_result = colors[:]
                if best_colors <= target_C:
                    if target_C == 13:
                        return best_result
                    break
        
        # Local search repair for this target_C
        if time.time() < deadline - 15:
            # Start from baseline or best found, try to reduce colors
            if best_colors > target_C:
                # Try local search starting from a random valid coloring with target_C+1 colors
                for ls_attempt in range(20):
                    if time.time() > deadline - 10:
                        break
                    
                    # Initialize with target_C colors randomly
                    colors = [random.randint(0, target_C - 1) for _ in range(num_edges)]
                    viol = count_violations(colors)
                    
                    if viol == 0:
                        used = colors_used(colors)
                        if used < best_colors:
                            best_colors = used
                            best_result = colors[:]
                            if best_colors <= target_C:
                                if target_C == 13:
                                    return best_result
                                break
                        continue
                    
                    # Local search
                    no_improve = 0
                    max_no_improve = 2000
                    steps = 0
                    max_steps = 10000
                    
                    while steps < max_steps and no_improve < max_no_improve and time.time() < deadline - 10:
                        steps += 1
                        
                        e_idx = random.randint(0, num_edges - 1)
                        old_c = colors[e_idx]
                        new_c = random.randint(0, target_C - 1)
                        
                        if old_c == new_c:
                            continue
                        
                        # Calculate delta violation
                        delta = 0
                        for k4_i in edge_to_k4s[e_idx]:
                            e_indices = k4_edges[k4_i]
                            # Old distinct count
                            old_set = set()
                            for e in e_indices:
                                if e == e_idx:
                                    old_set.add(old_c)
                                else:
                                    old_set.add(colors[e])
                            old_distinct = len(old_set)
                            
                            # New distinct count
                            new_set = set()
                            for e in e_indices:
                                if e == e_idx:
                                    new_set.add(new_c)
                                else:
                                    new_set.add(colors[e])
                            new_distinct = len(new_set)
                            
                            old_viol = max(0, 5 - old_distinct)
                            new_viol = max(0, 5 - new_distinct)
                            delta += (new_viol - old_viol)
                        
                        new_viol = viol + delta
                        
                        if new_viol < viol:
                            colors[e_idx] = new_c
                            viol = new_viol
                            no_improve = 0
                        elif new_viol == viol and random.random() < 0.3:
                            colors[e_idx] = new_c
                            no_improve += 1
                        else:
                            no_improve += 1
                        
                        if viol == 0:
                            break
                    
                    if viol == 0:
                        used = colors_used(colors)
                        if used < best_colors:
                            best_colors = used
                            best_result = colors[:]
                            if best_colors <= target_C:
                                if target_C == 13:
                                    return best_result
                                break
    
    return best_result
