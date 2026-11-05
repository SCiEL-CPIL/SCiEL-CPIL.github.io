import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try structured pair-based approach first, then fallback to local search
    for n in range(34, 9, -1):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        
        # Phase 1: Structured pair-based GA
        budget_ga = min(remaining * 0.6, 80.0)
        if budget_ga > 0.5:
            result = solve_structured(n, budget_ga)
            if result is not None:
                return result
        
        # Phase 2: Local search with better initialization
        remaining = deadline - time.time()
        if remaining < 0.5:
            break
        budget_ls = min(remaining * 0.8, 60.0)
        if budget_ls > 0.5:
            result = solve_local(n, budget_ls)
            if result is not None:
                return result
    
    # Fallback
    return {'n': 9, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]}

def solve_structured(n, time_limit):
    """Use pair-based structured representation with genetic algorithm."""
    start = time.time()
    
    # Precompute triples
    triples = list(combinations(range(n), 3))
    M = len(triples)
    
    # For each triple (a,b,c), define color based on pair values
    # Pair values: p[a,b] for a < b, values in {0, 1, 2} (F3)
    # Triple color: f(p[a,b], p[b,c], p[a,c]) -> {0, 1}
    # We'll use: color = 0 if (p[a,b] + p[b,c] + p[a,c]) % 3 == 0 else 1
    # This gives a structured coloring with fewer parameters
    
    # Number of pairs
    pairs = list(combinations(range(n), 2))
    P = len(pairs)
    pair_idx = {p: i for i, p in enumerate(pairs)}
    
    # Map each triple to its three pairs
    triple_pairs = []
    for a, b, c in triples:
        triple_pairs.append((pair_idx[(a,b)], pair_idx[(b,c)], pair_idx[(a,c)]))
    
    # Precompute 4-sets and 5-sets for violation checking
    sets4 = list(combinations(range(n), 4))
    sets5 = list(combinations(range(n), 5))
    N4 = len(sets4)
    N5 = len(sets5)
    
    # For each 4-set, which triples does it contain?
    s4_triple_list = []
    for s4 in sets4:
        ts = list(combinations(s4, 3))
        s4_triple_list.append(ts)
    
    s5_triple_list = []
    for s5 in sets5:
        ts = list(combinations(s5, 3))
        s5_triple_list.append(ts)
    
    def evaluate(pair_vals):
        """Given pair values in {0,1,2}, compute triple colors and count violations."""
        # Compute triple colors
        colors = np.zeros(M, dtype=np.int8)
        for i, (pa, pb, pc) in enumerate(triple_pairs):
            s = (pair_vals[pa] + pair_vals[pb] + pair_vals[pc]) % 3
            colors[i] = 0 if s == 0 else 1
        
        # Count violations
        red_viol = 0
        for ts in s4_triple_list:
            if all(colors[i] == 0 for i in range(M) if False):  # placeholder
                pass
            # Check if all 4 triples in this 4-set are red
            all_red = True
            for t in ts:
                ti = triples.index(t)
                if colors[ti] != 0:
                    all_red = False
                    break
            if all_red:
                red_viol += 1
        
        blue_viol = 0
        for ts in s5_triple_list:
            all_blue = True
            for t in ts:
                ti = triples.index(t)
                if colors[ti] != 1:
                    all_blue = False
                    break
            if all_blue:
                blue_viol += 1
        
        return red_viol + blue_viol, colors
    
    # Faster evaluation using precomputed indices
    # Map triple to index
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    s4_triple_indices = []
    for ts in s4_triple_list:
        s4_triple_indices.append([triple_to_idx[t] for t in ts])
    
    s5_triple_indices = []
    for ts in s5_triple_list:
        s5_triple_indices.append([triple_to_idx[t] for t in ts])
    
    def fast_evaluate(pair_vals):
        """Fast evaluation using precomputed structures."""
        # Compute triple colors
        colors = np.zeros(M, dtype=np.int8)
        for i, (pa, pb, pc) in enumerate(triple_pairs):
            s = (pair_vals[pa] + pair_vals[pb] + pair_vals[pc]) % 3
            colors[i] = 0 if s == 0 else 1
        
        # Count violations
        red_viol = 0
        for tidxs in s4_triple_indices:
            if all(colors[i] == 0 for i in tidxs):
                red_viol += 1
        
        blue_viol = 0
        for tidxs in s5_triple_indices:
            if all(colors[i] == 1 for i in tidxs):
                blue_viol += 1
        
        return red_viol + blue_viol, colors
    
    # Genetic algorithm
    pop_size = 40
    population = []
    for _ in range(pop_size):
        # Random initialization
        pv = np.random.randint(0, 3, P)
        viol, colors = fast_evaluate(pv)
        population.append((pv, viol, colors))
    
    # Also try some structured initializations
    for init_fn in [
        lambda: np.zeros(P, dtype=np.int8),
        lambda: np.ones(P, dtype=np.int8),
        lambda: np.array([i % 3 for i in range(P)], dtype=np.int8),
        lambda: np.array([(i // n) % 3 for i in range(P)], dtype=np.int8),
    ]:
        pv = init_fn()
        viol, colors = fast_evaluate(pv)
        population.append((pv, viol, colors))
    
    population.sort(key=lambda x: x[1])
    
    best_pv, best_viol, best_colors = population[0]
    
    if best_viol == 0:
        return {'n': n, 'colors': best_colors.tolist()}
    
    # GA loop
    gen = 0
    max_gens = 200
    while time.time() - start < time_limit and gen < max_gens:
        gen += 1
        
        # Selection: keep top half
        elite_size = max(5, pop_size // 4)
        elites = population[:elite_size]
        
        # Create new population
        new_pop = list(elites)
        
        while len(new_pop) < pop_size:
            # Crossover
            p1, v1, c1 = random.choice(elites)
            p2, v2, c2 = random.choice(elites)
            
            if random.random() < 0.7:
                # Uniform crossover
                mask = np.random.randint(0, 2, P)
                child = np.where(mask == 0, p1, p2).astype(np.int8)
            else:
                # Single point crossover
                cut = random.randint(1, P - 1)
                child = np.concatenate([p1[:cut], p2[cut:]]).astype(np.int8)
            
            # Mutation
            if random.random() < 0.5:
                n_mut = random.randint(1, max(1, P // 10))
                mut_idx = np.random.choice(P, n_mut, replace=False)
                child[mut_idx] = np.random.randint(0, 3, n_mut)
            
            viol, colors = fast_evaluate(child)
            new_pop.append((child, viol, colors))
        
        population = new_pop
        population.sort(key=lambda x: x[1])
        
        if population[0][1] < best_viol:
            best_pv, best_viol, best_colors = population[0]
            if best_viol == 0:
                return {'n': n, 'colors': best_colors.tolist()}
    
    return None

def solve_local(n, time_limit):
    """Local search with improved initialization."""
    start = time.time()
    
    triples = list(combinations(range(n), 3))
    M = len(triples)
    idx = {t: i for i, t in enumerate(triples)}
    
    sets4 = list(combinations(range(n), 4))
    sets5 = list(combinations(range(n), 5))
    N4 = len(sets4)
    N5 = len(sets5)
    
    s4_triples = []
    for s4 in sets4:
        ts = [idx[tuple(sorted(s4[:j] + s4[j+1:]))] for j in range(4)]
        s4_triples.append(ts)
    
    s5_triples = []
    for s5 in sets5:
        ts = [idx[t] for t in combinations(s5, 3)]
        s5_triples.append(ts)
    
    s4_triples_arr = [np.array(ts) for ts in s4_triples]
    s5_triples_arr = [np.array(ts) for ts in s5_triples]
    
    triple_to_4 = [[] for _ in range(M)]
    for i4, s4 in enumerate(sets4):
        for j in range(4):
            t = tuple(sorted(s4[:j] + s4[j+1:]))
            triple_to_4[idx[t]].append(i4)
    
    triple_to_5 = [[] for _ in range(M)]
    for i5, s5 in enumerate(sets5):
        for j in range(5):
            t = tuple(sorted(s5[:j] + s5[j+1:]))
            triple_to_5[idx[t]].append(i5)
    
    triple_to_4_arr = [np.array(lst) for lst in triple_to_4]
    triple_to_5_arr = [np.array(lst) for lst in triple_to_5]
    
    def init_counts(colors):
        red_count_4 = np.zeros(N4, dtype=np.int8)
        blue_count_5 = np.zeros(N5, dtype=np.int8)
        for i in range(M):
            c = colors[i]
            if c == 0:
                for i4 in triple_to_4_arr[i]:
                    red_count_4[i4] += 1
            else:
                for i5 in triple_to_5_arr[i]:
                    blue_count_5[i5] += 1
        return red_count_4, blue_count_5
    
    def get_violations(red_count_4, blue_count_5):
        v4 = set(np.where(red_count_4 == 4)[0])
        v5 = set(np.where(blue_count_5 == 10)[0])
        return v4, v5
    
    # Try multiple initializations
    inits = [
        lambda: np.random.randint(0, 2, M),
        lambda: np.array([0 if (a+b+c) % 3 == 0 else 1 for a,b,c in triples], dtype=np.int8),
        lambda: np.array([0 if (a*b+c) % 2 == 0 else 1 for a,b,c in triples], dtype=np.int8),
        lambda: np.array([0 if (a^b^c) % 2 == 0 else 1 for a,b,c in triples], dtype=np.int8),
    ]
    
    max_restarts = 8
    for restart in range(max_restarts):
        if time.time() - start > time_limit:
            break
        
        colors = inits[restart % len(inits)]().copy()
        red_count_4, blue_count_5 = init_counts(colors)
        violating_4, violating_5 = get_violations(red_count_4, blue_count_5)
        
        if not violating_4 and not violating_5:
            return {'n': n, 'colors': colors.tolist()}
        
        no_improve = 0
        best_total = len(violating_4) + len(violating_5)
        max_iters = 50000
        
        for it in range(max_iters):
            if time.time() - start > time_limit:
                break
            
            if not violating_4 and not violating_5:
                return {'n': n, 'colors': colors.tolist()}
            
            # Pick violation
            if violating_4 and violating_5:
                if random.random() < len(violating_4) / (len(violating_4) + len(violating_5)):
                    i4 = random.choice(list(violating_4))
                    t_idx = random.choice(s4_triples_arr[i4])
                else:
                    i5 = random.choice(list(violating_5))
                    t_idx = random.choice(s5_triples_arr[i5])
            elif violating_4:
                i4 = random.choice(list(violating_4))
                t_idx = random.choice(s4_triples_arr[i4])
            else:
                i5 = random.choice(list(violating_5))
                t_idx = random.choice(s5_triples_arr[i5])
            
            old_color = colors[t_idx]
            new_color = 1 - old_color
            colors[t_idx] = new_color
            
            for i4 in triple_to_4_arr[t_idx]:
                if old_color == 0:
                    red_count_4[i4] -= 1
                    if red_count_4[i4] == 4:
                        violating_4.add(i4)
                    else:
                        violating_4.discard(i4)
                else:
                    red_count_4[i4] += 1
                    if red_count_4[i4] == 4:
                        violating_4.add(i4)
                    else:
                        violating_4.discard(i4)
            
            for i5 in triple_to_5_arr[t_idx]:
                if old_color == 1:
                    blue_count_5[i5] -= 1
                    if blue_count_5[i5] == 10:
                        violating_5.add(i5)
                    else:
                        violating_5.discard(i5)
                else:
                    blue_count_5[i5] += 1
                    if blue_count_5[i5] == 10:
                        violating_5.add(i5)
                    else:
                        violating_5.discard(i5)
            
            total_viol = len(violating_4) + len(violating_5)
            if total_viol < best_total:
                best_total = total_viol
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > 3000:
                # Perturb
                for _ in range(5):
                    t_idx = random.randint(0, M-1)
                    old_color = colors[t_idx]
                    new_color = 1 - old_color
                    colors[t_idx] = new_color
                    for i4 in triple_to_4_arr[t_idx]:
                        if old_color == 0:
                            red_count_4[i4] -= 1
                            if red_count_4[i4] == 4:
                                violating_4.add(i4)
                            else:
                                violating_4.discard(i4)
                        else:
                            red_count_4[i4] += 1
                            if red_count_4[i4] == 4:
                                violating_4.add(i4)
                            else:
                                violating_4.discard(i4)
                    for i5 in triple_to_5_arr[t_idx]:
                        if old_color == 1:
                            blue_count_5[i5] -= 1
                            if blue_count_5[i5] == 10:
                                violating_5.add(i5)
                            else:
                                violating_5.discard(i5)
                        else:
                            blue_count_5[i5] += 1
                            if blue_count_5[i5] == 10:
                                violating_5.add(i5)
                            else:
                                violating_5.discard(i5)
                no_improve = 0
    
    return None
