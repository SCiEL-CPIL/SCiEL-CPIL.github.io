import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for 120s limit
    
    # Seed solution as fallback
    def get_seed():
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
    
    best_solution = get_seed()
    
    # Try n from 34 down to 10
    for n in range(34, 9, -1):
        if time.time() > deadline - 5.0:
            break
        
        # Time budget per n
        # Give more time to larger n? Or less?
        # Larger n is harder. But we want max n.
        # Let's give 8s to n=34, 5s to n=33, 3s to n=32, 2s to n<=31
        if n >= 33:
            time_budget = 8.0
        elif n == 32:
            time_budget = 5.0
        else:
            time_budget = 3.0
            
        end_time_for_n = time.time() + time_budget
        
        # Precomputation for this n
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Generate all K4s and K5s
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        
        # For each triple, list of K4s and K5s containing it
        k4_containing = [[] for _ in range(num_triples)]
        k5_containing = [[] for _ in range(num_triples)]
        
        # Map K4s to their triples
        k4_triples = []
        for i, k4 in enumerate(k4s):
            tris = []
            for j in range(4):
                t = tuple(k4[k] for k in range(4) if k != j)
                tris.append(triple_to_idx[t])
            k4_triples.append(tris)
            for t_idx in tris:
                k4_containing[t_idx].append(i)
                
        # Map K5s to their triples
        k5_triples = []
        for i, k5 in enumerate(k5s):
            tris = []
            for t in combinations(k5, 3):
                tris.append(triple_to_idx[t])
            k5_triples.append(tris)
            for t_idx in tris:
                k5_containing[t_idx].append(i)
        
        # Initial random coloring
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # Initialize counts
        red_k4_count = [0] * len(k4s)
        blue_k5_count = [0] * len(k5s)
        
        for i, tris in enumerate(k4_triples):
            cnt = 0
            for t_idx in tris:
                if colors[t_idx] == 0: # Red
                    cnt += 1
            red_k4_count[i] = cnt
            
        for i, tris in enumerate(k5_triples):
            cnt = 0
            for t_idx in tris:
                if colors[t_idx] == 1: # Blue
                    cnt += 1
            blue_k5_count[i] = cnt
            
        # Calculate initial cost
        cost = 0
        for cnt in red_k4_count:
            if cnt == 4: cost += 1
        for cnt in blue_k5_count:
            if cnt == 10: cost += 1
            
        # Local Search
        while time.time() < end_time_for_n and cost > 0:
            t_idx = random.randint(0, num_triples - 1)
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            delta = 0
            
            # Calculate delta for K4s
            for k4_idx in k4_containing[t_idx]:
                old_cnt = red_k4_count[k4_idx]
                if old_color == 0: # Red -> Blue
                    new_cnt = old_cnt - 1
                    if old_cnt == 4: delta -= 1
                    if new_cnt == 4: delta += 1
                else: # Blue -> Red
                    new_cnt = old_cnt + 1
                    if old_cnt == 4: delta -= 1
                    if new_cnt == 4: delta += 1
                    
            # Calculate delta for K5s
            for k5_idx in k5_containing[t_idx]:
                old_cnt = blue_k5_count[k5_idx]
                if old_color == 1: # Blue -> Red
                    new_cnt = old_cnt - 1
                    if old_cnt == 10: delta -= 1
                    if new_cnt == 10: delta += 1
                else: # Red -> Blue
                    new_cnt = old_cnt + 1
                    if old_cnt == 10: delta -= 1
                    if new_cnt == 10: delta += 1
                    
            # Acceptance
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.5:
                    accept = True
            else:
                if random.random() < 0.01:
                    accept = True
                    
            if accept:
                colors[t_idx] = new_color
                # Apply changes to counts
                for k4_idx in k4_containing[t_idx]:
                    if old_color == 0:
                        red_k4_count[k4_idx] -= 1
                    else:
                        red_k4_count[k4_idx] += 1
                for k5_idx in k5_containing[t_idx]:
                    if old_color == 1:
                        blue_k5_count[k5_idx] -= 1
                    else:
                        blue_k5_count[k5_idx] += 1
                cost += delta
                
        if cost == 0:
            best_solution = {'n': n, 'colors': colors[:]}
            # Since we are iterating n downwards, the first valid one is the max
            # But we might find a valid one for n=34, then n=33 is not needed.
            # However, if n=34 fails, we try n=33.
            # If n=34 succeeds, we return immediately?
            # Yes, because we want MAX n.
            return best_solution
            
    return best_solution
