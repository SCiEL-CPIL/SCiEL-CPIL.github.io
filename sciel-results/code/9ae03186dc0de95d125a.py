import time
import sys
import random
import math
from itertools import permutations

def construct(n):
    if n <= 1:
        return "1" if n == 1 else ""
    
    def build_standard(m):
        if m == 1:
            return [1]
        S = [1]
        for k in range(2, m + 1):
            seen = set()
            perms_order = []
            for i in range(len(S) - (k - 1) + 1):
                window = tuple(S[i:i + k - 1])
                if len(set(window)) == k - 1 and window not in seen:
                    seen.add(window)
                    perms_order.append(window)
            
            new_S = []
            for P in perms_order:
                block = list(P) + [k] + list(P)
                max_possible_overlap = min(len(new_S), len(block) - 1)
                overlap = 0
                for k_ov in range(max_possible_overlap, -1, -1):
                    if k_ov == 0:
                        overlap = 0
                        break
                    if new_S[-k_ov:] == block[:k_ov]:
                        overlap = k_ov
                        break
                new_S.extend(block[overlap:])
            S = new_S
        return S

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "".join(map(str, build_standard(4)))
    if n == 5:
        return "".join(map(str, build_standard(5)))
    if n == 6:
        return "".join(map(str, build_standard(6)))
    
    if n == 7:
        start_time = time.time()
        deadline = start_time + 110
        
        S6 = build_standard(6)
        
        seen = set()
        perms_order = []
        for i in range(len(S6) - 5):
            window = tuple(S6[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms_order.append(window)
        
        num_perms = len(perms_order)
        
        costs = [[0]*num_perms for _ in range(num_perms)]
        for i in range(num_perms):
            pi = perms_order[i]
            for j in range(num_perms):
                pj = perms_order[j]
                ov = 0
                for k in range(6, -1, -1):
                    if k == 0:
                        ov = 0
                        break
                    if pi[-k:] == pj[:k]:
                        ov = k
                        break
                costs[i][j] = 13 - ov

        # Simulated Annealing
        # Start with standard order
        path = list(range(num_perms))
        
        def get_total_cost(path):
            total = 0
            for i in range(len(path) - 1):
                total += costs[path[i]][path[i+1]]
            return total
        
        current_cost = get_total_cost(path)
        best_cost = current_cost
        best_path = path[:]
        
        # Simulated Annealing Parameters
        T_start = 10.0
        T_end = 0.1
        alpha = 0.999
        T = T_start
        
        # We will run SA for a fixed number of steps or time
        # Estimate steps: 110s / ~10us per step = 11,000,000 steps?
        # Let's just loop until deadline - 5s
        
        while time.time() < deadline - 5:
            # Pick a random move: 2-Opt (reverse segment)
            i = random.randint(0, num_perms - 2)
            j = random.randint(i + 1, num_perms - 1)
            
            # Calculate delta for reversing path[i..j]
            # Edges involved:
            # (i-1, i) and (j, j+1) are broken
            # (i-1, j) and (i, j+1) are created
            # Note: if i=0, no (i-1, i). If j=N-1, no (j, j+1).
            
            old_c1 = costs[path[i-1]][path[i]] if i > 0 else 0
            old_c2 = costs[path[j]][path[j+1]] if j < num_perms - 1 else 0
            old_cost = old_c1 + old_c2
            
            new_c1 = costs[path[i-1]][path[j]] if i > 0 else 0
            new_c2 = costs[path[i]][path[j+1]] if j < num_perms - 1 else 0
            new_cost = new_c1 + new_c2
            
            delta = new_cost - old_cost
            
            # Accept if delta < 0 or with probability exp(-delta/T)
            if delta < 0 or (random.random() < math.exp(-delta / T)):
                # Apply move
                path[i:j+1] = reversed(path[i:j+1])
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_path = path[:]
            
            # Cool down
            T *= alpha
            if T < T_end:
                T = T_end
        
        # Final 2-Opt polish on best_path to ensure local optimality
        path = best_path[:]
        current_cost = get_total_cost(path)
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(num_perms - 1):
                if time.time() > deadline - 1:
                    break
                for j in range(i + 1, num_perms):
                    c1 = costs[path[i-1]][path[i]] if i > 0 else 0
                    c2 = costs[path[j]][path[j+1]] if j < num_perms - 1 else 0
                    new_c1 = costs[path[i-1]][path[j]] if i > 0 else 0
                    new_c2 = costs[path[i]][path[j+1]] if j < num_perms - 1 else 0
                    delta = (new_c1 + new_c2) - (c1 + c2)
                    if delta < 0:
                        path[i:j+1] = reversed(path[i:j+1])
                        current_cost += delta
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_path = path[:]
        
        # Reconstruct S7 using best_path
        S7 = []
        for idx in best_path:
            P = perms_order[idx]
            block = list(P) + [7] + list(P)
            max_possible_overlap = min(len(S7), len(block) - 1)
            overlap = 0
            for k in range(max_possible_overlap, -1, -1):
                if k == 0:
                    overlap = 0
                    break
                if S7[-k:] == block[:k]:
                    overlap = k
                    break
            S7.extend(block[overlap:])
        
        # Verify
        all_perms = set(permutations(range(1, 8)))
        covered = set()
        for i in range(len(S7) - 6):
            w = tuple(S7[i:i+7])
            if w in all_perms:
                covered.add(w)
        
        if len(covered) == len(all_perms):
            return "".join(map(str, S7))
        else:
            S_std = build_standard(7)
            return "".join(map(str, S_std))
    
    S = build_standard(n)
    return "".join(map(str, S))
