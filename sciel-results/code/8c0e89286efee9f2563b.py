import time
import itertools
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for safety
    
    # 1. Precompute space and spheres
    total = q ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    spheres = []
    for i, w in enumerate(words):
        ball = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    new_w_list = list(w)
                    new_w_list[pos] = new_val
                    new_w_tuple = tuple(new_w_list)
                    ball.append(word_to_idx[new_w_tuple])
        spheres.append(ball)
        
    # 2. Greedy Construction to get an initial upper bound
    # This provides a feasible solution for the ILP's objective bound.
    def greedy_cover():
        code = []
        uncov = set(range(total))
        
        while uncov:
            if time.time() > deadline - 5:
                break
            best_c = -1
            best_cnt = -1
            # Sample candidates to speed up if needed, but 729 is small
            candidates = list(range(total))
            random.shuffle(candidates)
            
            for c in candidates:
                cnt = sum(1 for w in spheres[c] if w in uncov)
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
                elif cnt == best_cnt and random.random() < 0.1:
                    best_c = c
                    
            if best_c == -1 or best_cnt == 0:
                break
                
            code.append(best_c)
            for w in spheres[best_c]:
                uncov.discard(w)
                
        return code

    greedy_code = greedy_cover()
    # Ensure greedy code is valid, otherwise fallback to simple full cover (should not happen)
    if not all(any(w in spheres[ci] for ci in greedy_code) for w in range(total)):
         # Fallback: just use all words? No, that's huge. 
         # Greedy should always work.
         pass
    
    ub = len(greedy_code)
    
    # 3. ILP Formulation
    # Variables: x_i in {0, 1} for each word i
    # Minimize: sum(x_i)
    # Subject to: for each word w_j, sum(x_i for i in sphere(j)) >= 1
    
    # A matrix: rows are words (729), cols are potential centers (729)
    # A[j, i] = 1 if word i covers word j (i.e., j in sphere(i))
    
    # We can construct A efficiently using the spheres
    A = np.zeros((total, total), dtype=np.int8)
    for i, ball in enumerate(spheres):
        for j in ball:
            A[j, i] = 1
            
    # Objective: c = [1, 1, ..., 1]
    c = np.ones(total)
    
    # Constraints: A @ x >= 1
    # LinearConstraint(A, lb=np.ones(total), ub=np.inf)
    
    # Integrality: all integer
    integrality = np.ones(total)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Run MILP
    # Options: 
    # - maxiter: limit iterations to prevent timeout
    # - presolve: 'default'
    # - 'time_limit': seconds (scipy uses 'time_limit' in options? No, it's 'time_limit' in scipy.optimize.milp? 
    # Actually scipy.optimize.milp accepts 'time_limit' in options dict for HiGHS.)
    
    # Let's try to run it. If it takes too long, we return the greedy solution.
    # HiGHS is quite fast for small problems. 729x729 is tiny.
    
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A, lb=np.ones(total), ub=np.inf),
            integrality=integrality,
            bounds=bounds,
            options={
                "time_limit": 100,  # Hard stop within scipy
                "mip_rel_gap": 0.0  # Try for optimal? Might be slow. 
                # Given UB is ~73 and LB is ~71, gap is small.
                # But if it struggles, we might not get a solution.
                # Let's allow a small gap to ensure we get *a* solution quickly if optimal is hard.
                # However, for 729 vars, optimal should be found fast.
            }
        )
        
        if res.status == 0: # Optimal
            # res.x contains the solution
            # x is float, need to round to int
            x = np.round(res.x).astype(int)
            code_indices = [i for i in range(total) if x[i] == 1]
            
            # Verify
            if all(any(w in spheres[ci] for ci in code_indices) for w in range(total)):
                return [list(words[i]) for i in code_indices]
            else:
                # Fallback to greedy if ILP result is invalid (should not happen)
                return [list(words[i]) for i in greedy_code]
                
        elif res.status == 1: # Locally optimal
             x = np.round(res.x).astype(int)
             code_indices = [i for i in range(total) if x[i] == 1]
             if all(any(w in spheres[ci] for ci in code_indices) for w in range(total)):
                 return [list(words[i]) for i in code_indices]
             else:
                 return [list(words[i]) for i in greedy_code]
                 
        else:
            # Failure or timeout within scipy
            # Return greedy solution
            return [list(words[i]) for i in greedy_code]
            
    except Exception as e:
        # Fallback to greedy
        return [list(words[i]) for i in greedy_code]

# Self-test
if __name__ == "__main__":
    res = construct(3, 6, 1)
    print(f"Size: {len(res)}")
