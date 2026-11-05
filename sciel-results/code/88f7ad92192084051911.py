import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin
    
    n = 256
    
    def can_add_general(S, x):
        """Check if x can be added to S (S is a set)."""
        if x in S:
            return False
        S_list = sorted(S)
        m = len(S_list)
        
        # For each a in S where a < x, check if a divides (x + b) or (x + c) for b,c in S
        for i in range(m):
            a = S_list[i]
            if a >= x:
                break
            # x is b: need a < x <= c, check (x + c) % a != 0 for c >= x in S
            for j in range(m):
                c = S_list[j]
                if c >= x and (x + c) % a == 0:
                    return False
            # x is c: need a < b <= x, check (b + x) % a != 0 for b <= x in S
            for j in range(m):
                b = S_list[j]
                if a < b <= x and (b + x) % a == 0:
                    return False
        
        # x is a: need x < b <= c, check (b + c) % x != 0
        for i in range(m):
            b = S_list[i]
            if b > x:
                if (2 * b) % x == 0:
                    return False
                for j in range(i + 1, m):
                    c = S_list[j]
                    if (b + c) % x == 0:
                        return False
        return True
    
    def can_add_greedy(S, x):
        """Check if x can be added to S when x will be the smallest."""
        s_list = list(S)
        m = len(s_list)
        for i in range(m):
            b = s_list[i]
            if (2 * b) % x == 0:
                return False
            for j in range(i + 1, m):
                c = s_list[j]
                if (b + c) % x == 0:
                    return False
        return True
    
    def greedy_from_top():
        """Greedy addition from largest to smallest."""
        S = set()
        for x in range(n, 0, -1):
            if time.time() > deadline - 5.0:
                break
            if can_add_greedy(S, x):
                S.add(x)
        return S
    
    def local_improve(S, max_time):
        """Try to swap larger elements with smaller ones to increase size."""
        S = set(S)
        improved = True
        while improved and time.time() < max_time:
            improved = False
            S_sorted = sorted(S, reverse=True)
            
            for i, large in enumerate(S_sorted):
                if time.time() > max_time:
                    break
                
                S_without = S - {large}
                
                # Try replacing with smaller x
                for x in range(1, large):
                    if x in S:
                        continue
                    if can_add_general(S_without, x):
                        new_S = S_without | {x}
                        if len(new_S) >= len(S):
                            S = new_S
                            improved = True
                            break
        return S
    
    def simulated_annealing(S, max_time):
        """Simulated annealing: try swaps and additions."""
        S = set(S)
        best_S = set(S)
        best_size = len(S)
        
        # Temperature schedule
        T_start = 1.0
        T_end = 0.01
        elapsed = 0.0
        total_time = max_time - time.time()
        if total_time <= 0:
            return S
        
        current_time = time.time()
        
        while time.time() < max_time:
            T = T_start * (T_end / T_start) ** ((time.time() - current_time) / total_time)
            
            # Generate a move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add a random element
                candidates = [x for x in range(1, n + 1) if x not in S]
                if not candidates:
                    continue
                x = random.choice(candidates)
                if can_add_general(S, x):
                    new_S = S | {x}
                    delta = 1
                    if delta > 0 or random.random() < T:
                        S = new_S
                        if len(S) > best_size:
                            best_S = set(S)
                            best_size = len(S)
                else:
                    continue
            elif move_type < 0.7:
                # Try to swap: remove one, add another
                if not S:
                    continue
                x_remove = random.choice(list(S))
                S_temp = S - {x_remove}
                candidates = [x for x in range(1, n + 1) if x not in S_temp]
                if not candidates:
                    continue
                x_add = random.choice(candidates)
                if can_add_general(S_temp, x_add):
                    new_S = S_temp | {x_add}
                    delta = 0  # same size, but different composition
                    if delta > 0 or random.random() < T * 0.5:
                        S = new_S
                        if len(S) > best_size:
                            best_S = set(S)
                            best_size = len(S)
            else:
                # Try double swap: remove two, add two
                if len(S) < 2:
                    continue
                x1, x2 = random.sample(list(S), 2)
                S_temp = S - {x1, x2}
                candidates = [x for x in range(1, n + 1) if x not in S_temp]
                if len(candidates) < 2:
                    continue
                x3, x4 = random.sample(candidates, 2)
                if can_add_general(S_temp, x3) and can_add_general(S_temp | {x3}, x4):
                    new_S = S_temp | {x3, x4}
                    delta = 0
                    if delta > 0 or random.random() < T * 0.3:
                        S = new_S
                        if len(S) > best_size:
                            best_S = set(S)
                            best_size = len(S)
        
        return best_S
    
    def multi_start_greedy(restart_idx):
        """Multi-start greedy with different orderings."""
        rng = random.Random(restart_idx * 7919 + 42)
        S = set()
        
        # Random ordering of numbers
        order = list(range(n, 0, -1))
        rng.shuffle(order)
        
        for x in order:
            if time.time() > deadline - 5.0:
                break
            if can_add_general(S, x):
                S.add(x)
        
        return S
    
    def backtracking_search(S_partial, remaining, max_size, max_time):
        """Backtracking search with forward checking."""
        best = set(S_partial)
        best_size = len(S_partial)
        
        if not remaining or time.time() > max_time:
            return best, best_size
        
        if len(S_partial) + len(remaining) <= best_size:
            return best, best_size
        
        # Try adding the first remaining element
        x = remaining[0]
        rest = remaining[1:]
        
        # Option 1: Add x
        if can_add_general(S_partial, x):
            new_S = S_partial | {x}
            # Filter remaining elements that can still be added
            filtered_rest = [y for y in rest if can_add_general(new_S, y)]
            sub_best, sub_size = backtracking_search(new_S, filtered_rest, max_size, max_time)
            if sub_size > best_size:
                best = sub_best
                best_size = sub_size
        
        # Option 2: Skip x
        sub_best, sub_size = backtracking_search(S_partial, rest, max_size, max_time)
        if sub_size > best_size:
            best = sub_best
            best_size = sub_size
        
        return best, best_size
    
    # Initial greedy solution
    best_S = greedy_from_top()
    best_size = len(best_S)
    
    # Local improvement on greedy result
    if time.time() < deadline - 10.0:
        improved_S = local_improve(best_S, max_time=deadline - time.time() - 10.0)
        if len(improved_S) > best_size:
            best_S = improved_S
            best_size = len(best_S)
    
    # Simulated annealing
    if time.time() < deadline - 15.0:
        sa_S = simulated_annealing(best_S, max_time=deadline - time.time() - 15.0)
        if len(sa_S) > best_size:
            best_S = sa_S
            best_size = len(best_S)
    
    # Multiple restarts with different strategies
    restart_count = 0
    while time.time() < deadline - 5.0 and restart_count < 50:
        restart_count += 1
        
        # Strategy 1: Multi-start greedy
        S = multi_start_greedy(restart_count)
        
        # Local improvement
        if len(S) >= best_size - 3 and time.time() < deadline - 8.0:
            imp_S = local_improve(S, max_time=min(deadline - time.time() - 8.0, 5.0))
            if len(imp_S) > best_size:
                best_S = imp_S
                best_size = len(imp_S)
        
        # Strategy 2: Start with upper half, then add lower
        rng = random.Random(restart_count * 3301 + 7)
        base_start = n - rng.randint(20, 60)
        S2 = set()
        
        for x in range(n, base_start - 1, -1):
            if time.time() > deadline - 3.0:
                break
            if can_add_greedy(S2, x):
                S2.add(x)
        
        candidates = [x for x in range(1, base_start) if x not in S2]
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if can_add_general(S2, x):
                S2.add(x)
        
        if len(S2) > best_size:
            best_S = S2
            best_size = len(S2)
        
        # Quick local improvement if promising
        if len(S2) >= best_size - 2 and time.time() < deadline - 8.0:
            imp_S2 = local_improve(S2, max_time=min(deadline - time.time() - 8.0, 5.0))
            if len(imp_S2) > best_size:
                best_S = imp_S2
                best_size = len(imp_S2)
    
    # Final polish: try to add any remaining elements
    if time.time() < deadline - 2.0:
        S_final = set(best_S)
        for x in range(1, n + 1):
            if x not in S_final and can_add_general(S_final, x):
                S_final.add(x)
        if len(S_final) > best_size:
            best_S = S_final
            best_size = len(best_S)
    
    return list(best_S)
