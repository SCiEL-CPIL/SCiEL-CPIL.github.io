"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def time_left():
        return deadline - time.time()

    def get_structures(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        
        k4_triples = []
        for k4 in k4s:
            ts = []
            for tri in combinations(k4, 3):
                ts.append(triple_idx[tri])
            k4_triples.append(ts)
            
        k5_triples = []
        for k5 in k5s:
            ts = []
            for tri in combinations(k5, 3):
                ts.append(triple_idx[tri])
            k5_triples.append(ts)
            
        triple_to_k4 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k4_triples):
            for t_idx in ts:
                triple_to_k4[t_idx].append(i)
                
        triple_to_k5 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k5_triples):
            for t_idx in ts:
                triple_to_k5[t_idx].append(i)
                
        return {
            'triples': triples,
            'num_triples': num_triples,
            'triple_idx': triple_idx,
            'k4s': k4s,
            'k5s': k5s,
            'k4_triples': k4_triples,
            'k5_triples': k5_triples,
            'triple_to_k4': triple_to_k4,
            'triple_to_k5': triple_to_k5
        }

    def count_violations(colors, k4_triples, k5_triples):
        red_v = 0
        for ts in k4_triples:
            cnt = 0
            for t in ts:
                if colors[t] == 0:
                    cnt += 1
            if cnt == 4:
                red_v += 1
        
        blue_v = 0
        for ts in k5_triples:
            cnt = 0
            for t in ts:
                if colors[t] == 1:
                    cnt += 1
            if cnt == 10:
                blue_v += 1
        return red_v + blue_v

    def solve_n(n, time_budget):
        if n < 4:
            return None
            
        structs = get_structures(n)
        triples = structs['triples']
        num_triples = structs['num_triples']
        k4_triples = structs['k4_triples']
        k5_triples = structs['k5_triples']
        triple_to_k4 = structs['triple_to_k4']
        triple_to_k5 = structs['triple_to_k5']
        num_k4s = len(structs['k4s'])
        num_k5s = len(structs['k5s'])
        
        def init_colors():
            # Structured init: (a+b+c) % 3 == 0 -> Red (0), else Blue (1)
            # This naturally creates some structure.
            cols = np.zeros(num_triples, dtype=np.int8)
            for i, (a, b, c) in enumerate(triples):
                if (a + b + c) % 3 == 0:
                    cols[i] = 0
                else:
                    cols[i] = 1
            return cols

        def crossover(p1, p2):
            # Uniform crossover
            return np.where(np.random.random(num_triples) < 0.5, p1, p2)

        def mutate(cols, rate):
            # Flip bits with probability rate
            mask = np.random.random(num_triples) < rate
            cols[mask] = 1 - cols[mask]
            return cols

        def ga_solve(budget):
            pop_size = 30
            # Initialize population with diverse starts
            # 1. Structured init
            # 2. Random inits
            # 3. Mutated structured inits
            pop = []
            
            base = init_colors()
            pop.append(base.copy())
            
            for _ in range(pop_size - 1):
                if random.random() < 0.3:
                    c = init_colors()
                    c = mutate(c, 0.2)
                else:
                    c = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
                pop.append(c)
            
            # Evaluate initial population
            scores = [count_violations(c, k4_triples, k5_triples) for c in pop]
            best_idx = np.argmin(scores)
            best_score = scores[best_idx]
            
            if best_score == 0:
                return pop[best_idx].tolist()
            
            end_time = time.time() + budget
            gen = 0
            
            while time.time() < end_time:
                gen += 1
                if time_left() < 5.0:
                    break
                
                # Selection: Tournament
                def select():
                    a = random.randint(0, pop_size - 1)
                    b = random.randint(0, pop_size - 1)
                    return pop[a] if scores[a] < scores[b] else pop[b]
                
                new_pop = []
                
                # Elitism: keep best
                new_pop.append(pop[best_idx].copy())
                
                while len(new_pop) < pop_size:
                    if time.time() > end_time:
                        break
                    p1 = select()
                    p2 = select()
                    child = crossover(p1, p2)
                    # Mutation rate annealing
                    mut_rate = 0.05 * (0.95 ** (gen // 50))
                    if mut_rate < 0.01: mut_rate = 0.01
                    child = mutate(child, mut_rate)
                    new_pop.append(child)
                
                # Evaluate new population
                new_scores = [count_violations(c, k4_triples, k5_triples) for c in new_pop]
                new_best_idx = np.argmin(new_scores)
                new_best_score = new_scores[new_best_idx]
                
                if new_best_score < best_score:
                    best_score = new_best_score
                    best_idx = new_best_idx
                    if best_score == 0:
                        return new_pop[best_idx].tolist()
                
                pop = new_pop
                scores = new_scores
                
                # Inject random immigrants to maintain diversity
                if gen % 10 == 0:
                    idx = random.randint(0, pop_size - 1)
                    pop[idx] = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
                    scores[idx] = count_violations(pop[idx], k4_triples, k5_triples)
                    if scores[idx] < best_score:
                        best_score = scores[idx]
                        best_idx = idx
                        if best_score == 0:
                            return pop[best_idx].tolist()
                
            return None

        def sa_solve(budget):
            # Simulated Annealing with incremental updates
            colors = init_colors()
            
            red_count_k4 = np.zeros(num_k4s, dtype=np.int8)
            blue_count_k5 = np.zeros(num_k5s, dtype=np.int8)
            
            for i, ts in enumerate(k4_triples):
                cnt = 0
                for t_idx in ts:
                    if colors[t_idx] == 0:
                        cnt += 1
                red_count_k4[i] = cnt
                
            for i, ts in enumerate(k5_triples):
                cnt = 0
                for t_idx in ts:
                    if colors[t_idx] == 1:
                        cnt += 1
                blue_count_k5[i] = cnt
                
            red_violations = np.sum(red_count_k4 == 4)
            blue_violations = np.sum(blue_count_k5 == 10)
            
            if red_violations == 0 and blue_violations == 0:
                return colors.tolist()
            
            T = 1.0
            cooling = 0.999
            min_T = 0.01
            
            violating_k4s = [i for i in range(num_k4s) if red_count_k4[i] == 4]
            violating_k5s = [i for i in range(num_k5s) if blue_count_k5[i] == 10]
            
            end_time = time.time() + budget
            steps = 0
            
            while time.time() < end_time:
                if red_violations == 0 and blue_violations == 0:
                    return colors.tolist()
                
                # Select triple
                if red_violations > 0 and (blue_violations == 0 or random.random() < red_violations / (red_violations + blue_violations)):
                    if violating_k4s:
                        k4_idx = random.choice(violating_k4s)
                        t_idx = random.choice(k4_triples[k4_idx])
                    else:
                        t_idx = random.randint(0, num_triples - 1)
                else:
                    if violating_k5s:
                        k5_idx = random.choice(violating_k5s)
                        t_idx = random.choice(k5_triples[k5_idx])
                    else:
                        t_idx = random.randint(0, num_triples - 1)
                        
                old_color = colors[t_idx]
                new_color = 1 - old_color
                
                delta_red = 0
                delta_blue = 0
                
                for k4_idx in triple_to_k4[t_idx]:
                    if old_color == 0:
                        old_cnt = red_count_k4[k4_idx]
                        new_cnt = old_cnt - 1
                        if old_cnt == 4 and new_cnt == 3:
                            delta_red -= 1
                        elif old_cnt == 3 and new_cnt == 4:
                            delta_red += 1
                        red_count_k4[k4_idx] = new_cnt
                    else:
                        old_cnt = red_count_k4[k4_idx]
                        new_cnt = old_cnt + 1
                        if old_cnt == 3 and new_cnt == 4:
                            delta_red += 1
                        elif old_cnt == 4 and new_cnt == 3:
                            delta_red -= 1
                        red_count_k4[k4_idx] = new_cnt
                        
                for k5_idx in triple_to_k5[t_idx]:
                    if old_color == 1:
                        old_cnt = blue_count_k5[k5_idx]
                        new_cnt = old_cnt - 1
                        if old_cnt == 10 and new_cnt == 9:
                            delta_blue -= 1
                        elif old_cnt == 9 and new_cnt == 10:
                            delta_blue += 1
                        blue_count_k5[k5_idx] = new_cnt
                    else:
                        old_cnt = blue_count_k5[k5_idx]
                        new_cnt = old_cnt + 1
                        if old_cnt == 9 and new_cnt == 10:
                            delta_blue += 1
                        elif old_cnt == 10 and new_cnt == 9:
                            delta_blue -= 1
                        blue_count_k5[k5_idx] = new_cnt
                        
                delta_E = delta_red + delta_blue
                
                accepted = False
                if delta_E < 0:
                    accepted = True
                else:
                    if T > min_T:
                        prob = math.exp(-delta_E / T)
                        if random.random() < prob:
                            accepted = True
                            
                if accepted:
                    colors[t_idx] = new_color
                    red_violations += delta_red
                    blue_violations += delta_blue
                    
                    if steps % 100 == 0 or len(violating_k4s) == 0 or len(violating_k5s) == 0:
                        violating_k4s = [i for i in range(num_k4s) if red_count_k4[i] == 4]
                        violating_k5s = [i for i in range(num_k5s) if blue_count_k5[i] == 10]
                else:
                    for k4_idx in triple_to_k4[t_idx]:
                        if old_color == 0:
                            red_count_k4[k4_idx] += 1
                        else:
                            red_count_k4[k4_idx] -= 1
                    for k5_idx in triple_to_k5[t_idx]:
                        if old_color == 1:
                            blue_count_k5[k5_idx] += 1
                        else:
                            blue_count_k5[k5_idx] -= 1
                            
                T *= cooling
                steps += 1
                
            return None

        # Strategy: Try GA first (global search), then SA (local refinement)
        # Or alternate.
        # Let's try GA for 50% of budget, SA for 50%.
        ga_budget = budget * 0.5
        sa_budget = budget * 0.5
        
        result = ga_solve(ga_budget)
        if result is not None:
            return result
            
        if time_left() > 2.0:
            result = sa_solve(sa_budget)
            if result is not None:
                return result
                
        # If both fail, try SA again with different init?
        if time_left() > 2.0:
             result = sa_solve(min(5.0, time_left() - 2.0))
             if result is not None:
                 return result

        return None

    best_n = 9
    best_colors = [0 if (a+b+c) % 3 == 0 else 1 for a,b,c in combinations(range(9), 3)]
    
    # Try to solve for increasing n
    for target_n in range(10, 35):
        if time_left() < 5.0:
            break
        
        # Allocate time budget
        # Give more time to larger n?
        # Let's try to solve with a budget of min(15, time_left()/2)
        budget = min(15.0, time_left() / 2.0)
        
        result = solve_n(target_n, budget)
        if result is not None:
            best_n = target_n
            best_colors = result
        else:
            # If we fail, we might try again with a different seed/init
            if time_left() > 10.0:
                random.seed(random.randint(0, 1000000))
                result = solve_n(target_n, min(10.0, time_left() / 2.0))
                if result is not None:
                    best_n = target_n
                    best_colors = result

    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
