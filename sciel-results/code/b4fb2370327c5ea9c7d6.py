import math
import time
import numpy as np
import random

def construct():
    START = time.time()
    DEADLINE = START + 135
    
    def elapsed():
        return time.time() - START
    
    def time_ok(margin=5):
        return elapsed() < DEADLINE - margin
    
    def compute_score_fast(u):
        u_arr = np.array(u, dtype=np.int64)
        n = len(u_arr)
        if n == 0:
            return -1.0
        if 0 not in u_arr:
            return -1.0
        max_u = int(u_arr.max())
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        sums = u_arr[:, None] + u_arr[None, :]
        sumset_size = len(np.unique(sums))
        diffs = u_arr[:, None] - u_arr[None, :]
        diffset_size = len(np.unique(diffs))
        r = diffset_size / sumset_size
        if r <= 0:
            return -1.0
        return 1.0 + math.log(r) / math.log(m)
    
    def compute_score(u):
        u = sorted(set(u))
        if not u or 0 not in u:
            return -1.0
        if len(u) <= 6:
            max_u = max(u)
            m = 2 * max_u + 1
            if m <= 1:
                return 1.0
            sumset = set()
            diffset = set()
            for a in u:
                for b in u:
                    sumset.add(a + b)
                    diffset.add(a - b)
            r = len(diffset) / len(sumset)
            if r <= 0:
                return -1.0
            return 1.0 + math.log(r) / math.log(m)
        return compute_score_fast(u)
    
    # Known good starting points
    golomb_candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 29, 34],
        [0, 1, 4, 10, 18, 23, 29, 37, 45],
        [0, 1, 4, 10, 18, 23, 29, 37, 45, 52],
    ]
    
    best_set = [0, 1, 3]
    best_score = compute_score(best_set)
    
    for u in golomb_candidates:
        if not time_ok(10):
            break
        s = compute_score(u)
        if s > best_score:
            best_score = s
            best_set = list(u)
    
    # Simulated annealing with structural moves
    def simulated_annealing(start_u, time_budget=60):
        start_t = time.time()
        current_u = sorted(set(start_u))
        current_score = compute_score(current_u)
        best_u = list(current_u)
        best_s = current_score
        
        T = 0.5
        T_min = 1e-6
        cooling = 0.995
        
        while time.time() - start_t < time_budget and time_ok(10):
            if T < T_min:
                break
            
            move_type = random.random()
            candidate = list(current_u)
            
            if move_type < 0.3 and len(candidate) > 2:
                idx = random.randint(1, len(candidate) - 1)
                candidate.pop(idx)
            elif move_type < 0.5:
                max_u = max(candidate)
                new_val = random.randint(max_u + 1, max_u + len(candidate) * len(candidate) + 20)
                candidate.append(new_val)
            elif move_type < 0.8:
                idx = random.randint(1, len(candidate) - 1)
                delta = random.randint(-5, 5)
                new_val = candidate[idx] + delta
                if new_val >= 1:
                    candidate[idx] = new_val
                    candidate = sorted(set(candidate))
                    if len(candidate) != len(current_u):
                        candidate = list(current_u)
                        continue
            else:
                if len(candidate) > 2:
                    idx = random.randint(1, len(candidate) - 1)
                    candidate.pop(idx)
                    max_u = max(candidate) if candidate else 0
                    new_val = random.randint(max_u + 1, max_u + len(candidate) * len(candidate) + 20)
                    candidate.append(new_val)
            
            if not candidate or 0 not in candidate or len(candidate) < 2:
                continue
            
            candidate = sorted(set(candidate))
            cand_score = compute_score(candidate)
            
            if cand_score == -1.0:
                continue
            
            delta_s = cand_score - current_score
            if delta_s > 0 or random.random() < math.exp(delta_s / T):
                current_u = candidate
                current_score = cand_score
                if cand_score > best_s:
                    best_s = cand_score
                    best_u = list(candidate)
            
            T *= cooling
        
        return best_u, best_s
    
    if time_ok(20):
        random.seed(42)
        starting_points = [best_set] + golomb_candidates[:4]
        for sp in starting_points:
            if not time_ok(20):
                break
            sa_u, sa_s = simulated_annealing(sp, time_budget=15)
            if sa_s > best_score:
                best_score = sa_s
                best_set = sa_u
    
    # Genetic algorithm
    if time_ok(30):
        def crossover(u1, u2):
            u1 = sorted(set(u1))
            u2 = sorted(set(u2))
            all_vals = sorted(set(u1 + u2))
            child = [0]
            for v in all_vals:
                if v == 0:
                    continue
                if v in u1 or v in u2:
                    if random.random() < 0.5:
                        child.append(v)
            child = sorted(set(child))
            if len(child) < 2:
                child = [0, 1]
            return child
        
        def mutate(u):
            u = list(u)
            for _ in range(random.randint(1, 3)):
                mtype = random.random()
                if mtype < 0.4 and len(u) > 2:
                    idx = random.randint(1, len(u) - 1)
                    u.pop(idx)
                elif mtype < 0.7:
                    max_u = max(u)
                    u.append(random.randint(max_u + 1, max_u + len(u) * len(u) + 15))
                else:
                    idx = random.randint(1, len(u) - 1)
                    u[idx] = max(1, u[idx] + random.randint(-3, 3))
                    u = sorted(set(u))
                    if len(u) < 2:
                        u = [0, 1]
            return sorted(set(u))
        
        population = [best_set]
        for u in golomb_candidates[:5]:
            population.append(list(u))
        for _ in range(5):
            n = random.randint(3, 10)
            vals = sorted(random.sample(range(1, 100), n - 1))
            population.append([0] + vals)
        
        pop_size = len(population)
        for gen in range(50):
            if not time_ok(30):
                break
            
            scored = []
            for u in population:
                s = compute_score(u)
                scored.append((s, u))
            scored.sort(key=lambda x: -x[0])
            
            if scored[0][0] > best_score:
                best_score = scored[0][0]
                best_set = list(scored[0][1])
            
            survivors = [u for _, u in scored[:pop_size // 2]]
            
            new_pop = list(survivors)
            while len(new_pop) < pop_size:
                p1 = scored[random.randint(0, min(5, pop_size - 1))][1]
                p2 = scored[random.randint(0, min(5, pop_size - 1))][1]
                child = crossover(p1, p2)
                child = mutate(child)
                new_pop.append(child)
            
            population = new_pop[:pop_size]
        
        for u in population:
            s = compute_score(u)
            if s > best_score:
                best_score = s
                best_set = list(u)
    
    # Final polish: local search on the best found
    if time_ok(15):
        current_u = list(best_set)
        current_score = best_score
        for _ in range(100):
            if not time_ok(15):
                break
            improved = False
            max_u = max(current_u)
            for new_val in range(max_u + 1, max_u + len(current_u) ** 2 + 5):
                candidate = sorted(current_u + [new_val])
                s = compute_score(candidate)
                if s > current_score + 1e-12:
                    current_u = candidate
                    current_score = s
                    improved = True
                    break
            if improved:
                continue
            if len(current_u) > 2:
                for i in range(1, len(current_u)):
                    candidate = [x for j, x in enumerate(current_u) if j != i]
                    s = compute_score(candidate)
                    if s > current_score + 1e-12:
                        current_u = candidate
                        current_score = s
                        improved = True
                        break
            if improved:
                continue
            for i in range(1, len(current_u)):
                for delta in [-3, -2, -1, 1, 2, 3]:
                    new_val = current_u[i] + delta
                    if new_val < 1:
                        continue
                    candidate = list(current_u)
                    candidate[i] = new_val
                    candidate = sorted(set(candidate))
                    if len(candidate) != len(current_u) or 0 not in candidate:
                        continue
                    s = compute_score(candidate)
                    if s > current_score + 1e-12:
                        current_u = candidate
                        current_score = s
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        if current_score > best_score:
            best_score = current_score
            best_set = current_u
    
    best_set = sorted(set(best_set))
    if 0 not in best_set:
        best_set = [0] + best_set
    
    return best_set
