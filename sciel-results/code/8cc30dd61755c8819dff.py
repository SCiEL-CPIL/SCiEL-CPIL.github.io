import math
import time
import random

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _is_disjoint(r1, m1, r2, m2):
    g = _gcd(m1, m2)
    return (r1 - r2) % g != 0

def _find_valid_residue(m, existing_pairs):
    for r in range(m):
        valid = True
        for r_e, m_e in existing_pairs:
            if not _is_disjoint(r, m, r_e, m_e):
                valid = False
                break
        if valid:
            return r
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    moduli = list(range(2, 129))
    
    gcd_table = {}
    def get_gcd(m1, m2):
        key = (min(m1,m2), max(m1,m2))
        if key not in gcd_table:
            gcd_table[key] = _gcd(m1, m2)
        return gcd_table[key]
    
    def greedy_init(order):
        solution = {}
        for m in order:
            r = _find_valid_residue(m, list(solution.items()))
            if r is not None:
                solution[m] = r
        return solution
    
    random.seed(42)
    best_solution = {}
    best_size = 0
    
    orderings = [sorted(moduli), sorted(moduli, reverse=True)]
    
    def constraint_degree(m):
        total = 0
        for m2 in moduli:
            if m2 != m:
                total += get_gcd(m, m2)
        return total
    
    orderings.append(sorted(moduli, key=lambda m: -constraint_degree(m)))
    
    for _ in range(15):
        ro = moduli[:]
        random.shuffle(ro)
        orderings.append(ro)
    
    for order in orderings:
        if time.time() > deadline:
            break
        sol = greedy_init(order)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = sol
    
    # SA with incremental scoring
    def simulated_annealing(initial_solution, max_iter=80000):
        current = dict(initial_solution)
        best = dict(current)
        best_size_local = len(current)
        
        # Precompute all pairs for fast violation counting
        def count_violations(sol):
            k = len(sol)
            if k <= 1:
                return 0
            items = list(sol.items())
            viol = 0
            for i in range(k):
                for j in range(i+1, k):
                    r1, m1 = items[i][1], items[i][0]
                    r2, m2 = items[j][1], items[j][0]
                    if not _is_disjoint(r1, m1, r2, m2):
                        viol += 1
            return viol
        
        T = 2.0
        cooling = 0.999
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            if T < 0.005:
                break
            
            prev_viol = count_violations(current)
            
            neighbor = dict(current)
            r_choice = random.random()
            
            if r_choice < 0.4 and len(neighbor) > 0:
                m = random.choice(list(neighbor.keys()))
                neighbor[m] = random.randint(0, m - 1)
            elif r_choice < 0.6:
                missing = [m for m in moduli if m not in neighbor]
                if missing:
                    m = random.choice(missing)
                    neighbor[m] = random.randint(0, m - 1)
            elif r_choice < 0.8 and len(neighbor) > 1:
                items = list(neighbor.items())
                i, j = random.sample(range(len(items)), 2)
                m1, r1 = items[i]
                m2, r2 = items[j]
                if r1 < m2 and r2 < m1:
                    neighbor[m1] = r2
                    neighbor[m2] = r1
            elif len(neighbor) > 1:
                m = random.choice(list(neighbor.keys()))
                del neighbor[m]
            
            new_viol = count_violations(neighbor)
            new_size = len(neighbor)
            
            # Score: size - 3*violations (penalty for invalidity)
            prev_score = len(current) - 3 * prev_viol
            new_score = new_size - 3 * new_viol
            delta = new_score - prev_score
            
            if delta >= 0 or random.random() < math.exp(delta / max(T, 1e-10)):
                current = neighbor
                if new_viol == 0 and new_size > best_size_local:
                    best_size_local = new_size
                    best = dict(current)
            
            T *= cooling
        
        return best
    
    sa_result = simulated_annealing(best_solution, max_iter=100000)
    if len(sa_result) > best_size:
        best_solution = sa_result
        best_size = len(best_solution)
    
    for _ in range(3):
        if time.time() > deadline:
            break
        ro = moduli[:]
        random.shuffle(ro)
        sol = greedy_init(ro)
        sa_res = simulated_annealing(sol, max_iter=60000)
        if len(sa_res) > best_size:
            best_solution = sa_res
            best_size = len(sa_res)
    
    # Final add
    for m in moduli:
        if m in best_solution:
            continue
        r = _find_valid_residue(m, list(best_solution.items()))
        if r is not None:
            best_solution[m] = r
    
    result = [[r, m] for m, r in best_solution.items()]
    if not result:
        result = [[0, 2]]
    
    return result
