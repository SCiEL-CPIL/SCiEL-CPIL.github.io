import random
import time

def construct():
    N = 256
    
    # Precompute all 4-APs and neighbor relationships for fast delta calculation
    # This is a standard optimization for local search, distinct from the search strategy itself.
    neighbors = [[] for _ in range(N)]
    ap_list = []
    
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            ap = (i1, i2, i3, i4)
            ap_list.append(ap)
            # For each element in the AP, store the other three
            neighbors[i1].append((i2, i3, i4))
            neighbors[i2].append((i1, i3, i4))
            neighbors[i3].append((i1, i2, i4))
            neighbors[i4].append((i1, i2, i3))

    def calculate_score(colors):
        count = 0
        for ap in ap_list:
            c1, c2, c3, c4 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if c1 == c2 == c3 == c4:
                count += 1
        return count

    def local_search(colors, max_iters=1000):
        """
        First-Improvement Local Search.
        Flips a bit if it strictly reduces the score.
        """
        colors = colors[:]
        current_score = calculate_score(colors)
        
        improved = True
        iterations = 0
        while improved and iterations < max_iters:
            improved = False
            iterations += 1
            # Shuffle indices to avoid bias towards early indices
            indices = list(range(N))
            random.shuffle(indices)
            
            for i in indices:
                # Calculate delta for flipping i
                old_color = colors[i]
                new_color = 1 - old_color
                delta = 0
                
                for j, k, l in neighbors[i]:
                    cj, ck, cl = colors[j], colors[k], colors[l]
                    if cj == ck == cl:
                        if old_color == cj:
                            delta -= 1
                        else:
                            delta += 1
                
                if delta < 0:
                    # Accept the move
                    colors[i] = new_color
                    current_score += delta
                    improved = True
                    # We can continue to the next index in this pass
                    # (First Improvement strategy)
                    
        return colors, current_score

    def perturb(colors, num_flips):
        """
        Perturb the solution by flipping 'num_flips' random bits.
        """
        colors = colors[:]
        indices = random.sample(range(N), num_flips)
        for i in indices:
            colors[i] = 1 - colors[i]
        return colors

    best_overall_score = float('inf')
    best_overall_colors = None
    
    global_start = time.time()
    total_time_limit = 115.0
    
    # Initial random solution
    current_colors = [random.randint(0, 1) for _ in range(N)]
    current_score = calculate_score(current_colors)
    
    if current_score < best_overall_score:
        best_overall_score = current_score
        best_overall_colors = current_colors[:]
        
    # Iterated Local Search
    while (time.time() - global_start) < total_time_limit:
        # 1. Local Search from current state
        ls_colors, ls_score = local_search(current_colors)
        
        if ls_score < best_overall_score:
            best_overall_score = ls_score
            best_overall_colors = ls_colors[:]
            
        # 2. Perturb the best solution found so far (or the current one)
        # Using the best solution ensures we explore around the global best
        # but we might want to explore around the current local minimum too.
        # Standard ILS perturbs the *current* solution after LS.
        # Let's perturb the current local minimum.
        
        # Adaptive perturbation: start with small, increase if stuck?
        # For simplicity, random perturbation size between 5 and 20.
        num_flips = random.randint(5, 20)
        current_colors = perturb(ls_colors, num_flips)
        
        # Check time before next LS
        if (time.time() - global_start) > total_time_limit:
            break

    return best_overall_colors
