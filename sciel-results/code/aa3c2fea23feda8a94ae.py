import time
from math import gcd
import sys
import random

def construct():
    start = time.time()
    deadline = start + 85.0

    def elapsed():
        return time.time() - start

    N = 256
    numbers = list(range(1, N + 1))
    
    # Precompute conflict masks for all pairs
    conflicts = [0] * N
    
    def inc(a, b):
        g = gcd(a, b)
        u = a // g
        v = b // g
        return (u + v) % g == 0

    for i in range(N):
        a = numbers[i]
        mask = 0
        for j in range(N):
            if i == j:
                continue
            b = numbers[j]
            if inc(a, b):
                mask |= (1 << j)
        conflicts[i] = mask

    # 1. Identify odds and evens
    odds_indices = [i for i in range(N) if numbers[i] % 2 == 1]
    evens_indices = [i for i in range(N) if numbers[i] % 2 == 0]

    # 2. Filter evens that are compatible with ALL odds
    odds_ok_indices = []
    for e in evens_indices:
        ok = True
        for o in odds_indices:
            if (conflicts[e] >> o) & 1:
                ok = False
                break
        if ok:
            odds_ok_indices.append(e)

    # 3. Solve MIS on the subgraph induced by odds_ok_indices
    n_ev = len(odds_ok_indices)
    
    if n_ev == 0:
        return [numbers[i] for i in odds_indices]

    # Build local conflict masks
    local_conflicts = [0] * n_ev
    for i in range(n_ev):
        global_i = odds_ok_indices[i]
        mask = 0
        for j in range(n_ev):
            if i == j:
                continue
            global_j = odds_ok_indices[j]
            if (conflicts[global_i] >> global_j) & 1:
                mask |= (1 << j)
        local_conflicts[i] = mask

    def popcount(x):
        return bin(x).count('1')

    # Greedy initialization: repeatedly pick vertex with minimum degree in remaining graph
    def greedy_init():
        selected = 0
        available = (1 << n_ev) - 1
        while available:
            best_v = -1
            best_deg = n_ev + 1
            temp = available
            while temp:
                v = (temp & (-temp)).bit_length() - 1
                deg = popcount(local_conflicts[v] & available)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                temp &= temp - 1
            if best_v >= 0:
                selected |= (1 << best_v)
                available &= ~(1 << best_v) & ~local_conflicts[best_v]
        return selected

    # Local search: try to add vertices, then try 1-for-1 swaps
    def local_search(init_mask):
        selected = init_mask
        improved = True
        while improved:
            improved = False
            # Phase 1: try to add any compatible vertex
            for v in range(n_ev):
                if (selected >> v) & 1:
                    continue
                if (local_conflicts[v] & selected) == 0:
                    selected |= (1 << v)
                    improved = True
            # Phase 2: try 1-for-1 swaps
            for remove_v in range(n_ev):
                if not ((selected >> remove_v) & 1):
                    continue
                for add_v in range(n_ev):
                    if (selected >> add_v) & 1:
                        continue
                    new_sel = (selected & ~(1 << remove_v)) | (1 << add_v)
                    # Check validity: add_v must not conflict with (selected minus remove_v)
                    base = selected & ~(1 << remove_v)
                    if (local_conflicts[add_v] & base) == 0:
                        selected = new_sel
                        improved = True
                        break
                if improved:
                    break
        return selected

    # Simulated annealing with bitmask operations
    def simulated_annealing(init_mask, max_iters=20000):
        selected = init_mask
        best_selected = selected
        best_size = popcount(selected)
        
        temp = 3.0
        cooling = 0.9995
        
        for it in range(max_iters):
            if elapsed() > deadline - 5.0:
                break
            
            move_type = random.random()
            
            if move_type < 0.4:  # add
                unselected = ((1 << n_ev) - 1) & ~selected
                if unselected == 0:
                    continue
                v = (unselected & (-unselected)).bit_length() - 1
                # Random pick from unselected
                bits = bin(unselected).count('1')
                pick = random.randint(0, bits - 1)
                temp_bits = unselected
                for _ in range(pick):
                    temp_bits &= temp_bits - 1
                v = (temp_bits & (-temp_bits)).bit_length() - 1
                new_sel = selected | (1 << v)
                # Check if valid
                if (local_conflicts[v] & selected) == 0:
                    selected = new_sel
                    if popcount(selected) > best_size:
                        best_size = popcount(selected)
                        best_selected = selected
                else:
                    # Invalid move, accept with probability
                    if random.random() < (2.718281828 ** (-1.0 / temp)):
                        selected = new_sel
            
            elif move_type < 0.6:  # remove
                if selected == 0:
                    continue
                bits = popcount(selected)
                pick = random.randint(0, bits - 1)
                temp_bits = selected
                for _ in range(pick):
                    temp_bits &= temp_bits - 1
                v = (temp_bits & (-temp_bits)).bit_length() - 1
                selected &= ~(1 << v)
            
            else:  # swap
                if selected == 0:
                    continue
                # Pick remove
                bits = popcount(selected)
                pick = random.randint(0, bits - 1)
                temp_bits = selected
                for _ in range(pick):
                    temp_bits &= temp_bits - 1
                remove_v = (temp_bits & (-temp_bits)).bit_length() - 1
                
                unselected = ((1 << n_ev) - 1) & ~selected
                if unselected == 0:
                    continue
                bits_u = popcount(unselected)
                pick_u = random.randint(0, bits_u - 1)
                temp_bits_u = unselected
                for _ in range(pick_u):
                    temp_bits_u &= temp_bits_u - 1
                add_v = (temp_bits_u & (-temp_bits_u)).bit_length() - 1
                
                base = selected & ~(1 << remove_v)
                new_sel = base | (1 << add_v)
                if (local_conflicts[add_v] & base) == 0:
                    selected = new_sel
                    if popcount(selected) > best_size:
                        best_size = popcount(selected)
                        best_selected = selected
                else:
                    if random.random() < (2.718281828 ** (-1.0 / temp)):
                        selected = new_sel
            
            temp *= cooling
        
        return best_selected

    # Run multiple strategies
    best_result = 0
    best_size = 0

    # Strategy 1: Greedy + local search
    g_init = greedy_init()
    g_result = local_search(g_init)
    if popcount(g_result) > best_size:
        best_size = popcount(g_result)
        best_result = g_result

    # Strategy 2: Simulated annealing from greedy init
    if elapsed() < deadline - 10.0:
        sa_result = simulated_annealing(g_init, max_iters=15000)
        if popcount(sa_result) > best_size:
            best_size = popcount(sa_result)
            best_result = sa_result

    # Strategy 3: Simulated annealing from random init
    if elapsed() < deadline - 10.0:
        random_mask = 0
        for _ in range(min(n_ev, 15)):
            v = random.randint(0, n_ev - 1)
            random_mask |= (1 << v)
        random_mask = local_search(random_mask)
        sa_result2 = simulated_annealing(random_mask, max_iters=15000)
        if popcount(sa_result2) > best_size:
            best_size = popcount(sa_result2)
            best_result = sa_result2

    # Map back to global indices
    final_indices = set(odds_indices)
    temp = best_result
    while temp:
        idx = (temp & (-temp)).bit_length() - 1
        final_indices.add(odds_ok_indices[idx])
        temp &= temp - 1

    # Convert to values
    final_values = [numbers[i] for i in final_indices]

    # Try to add more evens that weren't in odds_ok_indices
    current_set = set(final_values)
    for e in evens_indices:
        ev = numbers[e]
        if ev in current_set:
            continue
        ok = True
        for cv in current_set:
            if (ev * cv) % (ev + cv) == 0:
                ok = False
                break
        if ok:
            current_set.add(ev)
    
    final_values = list(current_set)
    
    return final_values

if __name__ == "__main__":
    res = construct()
    print(len(res))
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a, b = res[i], res[j]
            if (a*b) % (a+b) == 0:
                valid = False
    print(valid)
