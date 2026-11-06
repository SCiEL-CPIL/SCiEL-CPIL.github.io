import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    N = 128
    
    # Precompute all 4-term APs
    aps4 = []
    for d in range(1, (N - 3) // 3 + 1):
        for i in range(0, N - 3 * d):
            aps4.append((i, i + d, i + 2 * d, i + 3 * d))
    
    total_aps = len(aps4)
    
    # For each position, precompute which APs it belongs to
    pos_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps4):
        for pos in ap:
            pos_aps[pos].append(idx)
    
    def count_violations(colors):
        """Count violations: 0=valid, 1=one 2-color AP, 2+=multiple or mono"""
        v = 0
        for ap in aps4:
            c = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
            if len(c) < 3:
                v += 1
        return v
    
    def is_valid(colors):
        for ap in aps4:
            if len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3:
                return False
        return True
    
    def local_search_3color(colors, max_steps=5000):
        """Local search to fix violations in a 3-coloring"""
        colors = list(colors)
        v = count_violations(colors)
        
        if v == 0:
            return colors
        
        no_improve = 0
        for step in range(max_steps):
            if time.time() - start_time > DEADLINE:
                break
            
            if v == 0:
                return colors
            
            # Find a violating AP
            viol_ap = None
            for ap in aps4:
                if len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3:
                    viol_ap = ap
                    break
            
            if viol_ap is None:
                return colors
            
            # Try changing one position in the violating AP
            best_move = None
            best_v = v + 1
            
            for pos in viol_ap:
                old_c = colors[pos]
                for new_c in range(3):
                    if new_c == old_c:
                        continue
                    
                    # Calculate delta
                    delta = 0
                    for ap_idx in pos_aps[pos]:
                        ap = aps4[ap_idx]
                        old_set = len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]})
                        # Compute new set
                        new_colors = [colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]]
                        for j, p in enumerate(ap):
                            if p == pos:
                                new_colors[j] = new_c
                        new_set = len(set(new_colors))
                        
                        old_bad = 1 if old_set < 3 else 0
                        new_bad = 1 if new_set < 3 else 0
                        delta += new_bad - old_bad
                    
                    if delta < best_v - v:
                        best_v = v + delta
                        best_move = (pos, new_c)
            
            if best_move is not None:
                pos, new_c = best_move
                colors[pos] = new_c
                v = best_v
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 100:
                    break
        
        return colors if count_violations(colors) == 0 else None
    
    def genetic_algorithm():
        """GA with repair to find valid 3-coloring"""
        pop_size = 30
        max_generations = 500
        elite_size = 5
        
        # Initialize population
        population = []
        for _ in range(pop_size):
            ind = [random.randint(0, 2) for _ in range(N)]
            population.append(ind)
        
        best_ind = None
        best_v = float('inf')
        
        for gen in range(max_generations):
            if time.time() - start_time > DEADLINE:
                break
            
            # Evaluate and sort
            scored = []
            for ind in population:
                v = count_violations(ind)
                scored.append((v, ind))
                if v < best_v:
                    best_v = v
                    best_ind = list(ind)
            
            scored.sort(key=lambda x: x[0])
            
            if best_v == 0:
                return best_ind
            
            # Elitism
            new_pop = [list(scored[i][1]) for i in range(elite_size)]
            
            # Fill rest with offspring
            while len(new_pop) < pop_size:
                if time.time() - start_time > DEADLINE:
                    break
                
                # Tournament selection
                def tournament(k=3):
                    candidates = random.sample(scored, min(k, len(scored)))
                    return min(candidates, key=lambda x: x[0])[1]
                
                p1 = tournament()
                p2 = tournament()
                
                # Crossover
                if random.random() < 0.7:
                    point = random.randint(1, N-1)
                    child = p1[:point] + p2[point:]
                else:
                    child = list(p1)
                
                # Mutation
                for i in range(N):
                    if random.random() < 0.05:
                        child[i] = random.randint(0, 2)
                
                # Repair: try local search on child
                repaired = local_search_3color(child, max_steps=200)
                if repaired is not None:
                    new_pop.append(repaired)
                    if count_violations(repaired) == 0:
                        return repaired
                else:
                    new_pop.append(child)
            
            population = new_pop
        
        return best_ind if best_v == 0 else None
    
    # Strategy:
    # 1. Try targeted local search from structured initializations
    # 2. Try GA with repair
    # 3. Fallback to greedy with more colors
    
    # Phase 1: Try local search from various structured starts
    structured_starts = []
    
    # Start 1: All zeros, then fix
    structured_starts.append([0] * N)
    
    # Start 2: Periodic patterns
    for period in [3, 4, 5, 6]:
        pattern = [i % 3 for i in range(period)]
        structured_starts.append([pattern[i % period] for i in range(N)])
    
    # Start 3: Random with bias
    for _ in range(5):
        structured_starts.append([random.randint(0, 2) for _ in range(N)])
    
    for start in structured_starts:
        if time.time() - start_time > DEADLINE:
            break
        result = local_search_3color(start, max_steps=2000)
        if result is not None:
            return result
    
    # Phase 2: GA
    if time.time() - start_time < DEADLINE:
        result = genetic_algorithm()
        if result is not None:
            return result
    
    # Phase 3: More aggressive local search with restarts
    for restart in range(50):
        if time.time() - start_time > DEADLINE:
            break
        init = [random.randint(0, 2) for _ in range(N)]
        result = local_search_3color(init, max_steps=3000)
        if result is not None:
            return result
    
    # Fallback: greedy with up to 128 colors
    colors = []
    for x in range(1, 129):
        found = False
        for color in range(128):
            valid = True
            # Check 4-APs ending at x
            for d in range(1, (x - 1) // 3 + 1):
                if x - 3 * d < 1:
                    break
                c1 = colors[x - 1 - 3 * d]
                c2 = colors[x - 1 - 2 * d]
                c3 = colors[x - 1 - d]
                if len({color, c1, c2, c3}) < 3:
                    valid = False
                    break
            if not valid:
                continue
            # Check 3-APs ending at x (to prevent future 4-AP violations)
            for d in range(1, (x - 1) // 2 + 1):
                if x - 2 * d < 1:
                    break
                c1 = colors[x - 1 - 2 * d]
                c2 = colors[x - 1 - d]
                if c1 == color and c2 == color:
                    valid = False
                    break
            if valid:
                colors.append(color)
                found = True
                break
        if not found:
            colors.append(0)
    
    return colors
