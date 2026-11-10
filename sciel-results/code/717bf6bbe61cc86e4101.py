import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 128
    TARGET_MIN = 128
    TARGET_MAX = 256
    
    def get_cost(sums):
        target_sums = sums[TARGET_MIN:TARGET_MAX+1]
        gaps = np.sum(target_sums == 0)
        if gaps > 0:
            return 100000 + gaps * 1000 + np.max(target_sums)
        peak = np.max(target_sums)
        sq = np.sum(target_sums**2)
        return peak * 1000 + sq

    def apply_add(sums, A, x):
        for a in A:
            s = a + x
            if s <= 256:
                sums[s] += 1
        if 2*x <= 256:
            sums[2*x] += 1
        A.add(x)
        
    def apply_remove(sums, A, x):
        for a in A:
            if a == x:
                continue
            s = a + x
            if s <= 256:
                sums[s] -= 1
        if 2*x <= 256:
            sums[2*x] -= 1
        A.remove(x)

    def polish(A, sums, max_iters=500):
        # Greedy local search
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            current_cost = get_cost(sums)
            
            # Try removing an element
            A_list = list(A)
            random.shuffle(A_list)
            for x in A_list:
                apply_remove(sums, A, x)
                new_cost = get_cost(sums)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    break
                else:
                    apply_add(sums, A, x)
            
            if improved:
                continue

            # Try adding an element
            candidates = [x for x in range(N+1) if x not in A]
            random.shuffle(candidates)
            for x in candidates:
                apply_add(sums, A, x)
                new_cost = get_cost(sums)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    break
                else:
                    apply_remove(sums, A, x)
            
            if improved:
                continue

            # Try swapping
            A_list = list(A)
            random.shuffle(A_list)
            for x in A_list:
                candidates = [y for y in range(N+1) if y not in A]
                random.shuffle(candidates)
                for y in candidates:
                    apply_remove(sums, A, x)
                    apply_add(sums, A, y)
                    new_cost = get_cost(sums)
                    if new_cost < current_cost:
                        current_cost = new_cost
                        improved = True
                        break
                    else:
                        apply_remove(sums, A, y)
                        apply_add(sums, A, x)
                if improved:
                    break
        
        return A, get_cost(sums)

    def run_sa(init_A, deadline):
        A = set(init_A)
        sums = np.zeros(257, dtype=np.int32)
        
        A_list = list(A)
        for i in range(len(A_list)):
            for j in range(i, len(A_list)):
                s = A_list[i] + A_list[j]
                if s <= 256:
                    sums[s] += 1
        
        current_cost = get_cost(sums)
        best_cost = current_cost
        best_A = set(A)
        best_sums = sums.copy()
        
        T = 50.0
        T_min = 0.1
        cooling = 0.995
        
        while time.time() < deadline:
            move_type = random.random()
            
            if move_type < 0.8:
                # Swap
                if not A:
                    continue
                x = random.choice(list(A))
                y_candidates = [y for y in range(N+1) if y not in A]
                if not y_candidates:
                    continue
                y = random.choice(y_candidates)
                
                apply_remove(sums, A, x)
                apply_add(sums, A, y)
                
                new_cost = get_cost(sums)
                
                if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                        best_sums = sums.copy()
                else:
                    apply_remove(sums, A, y)
                    apply_add(sums, A, x)
                    
            elif move_type < 0.9:
                # Add
                y_candidates = [y for y in range(N+1) if y not in A]
                if not y_candidates:
                    continue
                y = random.choice(y_candidates)
                
                apply_add(sums, A, y)
                new_cost = get_cost(sums)
                
                if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                        best_sums = sums.copy()
                else:
                    apply_remove(sums, A, y)
                    
            else:
                # Remove
                if len(A) < 10:
                    continue
                x = random.choice(list(A))
                
                apply_remove(sums, A, x)
                new_cost = get_cost(sums)
                
                if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                        best_sums = sums.copy()
                else:
                    apply_add(sums, A, x)
            
            T *= cooling
            if T < T_min:
                break
        
        # Polish the best solution found in this SA run
        best_A, best_cost = polish(best_A, best_sums)
        
        return best_A, best_cost

    best_overall_A = None
    best_overall_cost = float('inf')
    
    num_restarts = 5
    # Allocate time to restarts, leaving 5s for polish/finalization
    sa_deadline = start_time + 50.0
    time_per_restart = (sa_deadline - start_time) / num_restarts
    
    for i in range(num_restarts):
        if time.time() > sa_deadline:
            break
            
        size = random.randint(20, 40)
        init_A = set(random.sample(range(N+1), size))
        
        A, cost = run_sa(init_A, start_time + time_per_restart * (i + 1))
        
        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_A = A

    if best_overall_A is None:
        return list(range(64, 129))
        
    return sorted(best_overall_A)
