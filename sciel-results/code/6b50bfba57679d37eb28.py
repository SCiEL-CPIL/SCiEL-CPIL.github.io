import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    target_min = 128
    target_max = 256
    max_sum = 256
    
    best_A = None
    best_score = float('inf') # score = peak + penalty
    
    def get_score(N):
        # Penalty for uncovered
        uncovered = np.sum(N[target_min:target_max+1] == 0)
        if uncovered > 0:
            return 1000 + np.max(N[target_min:target_max+1])
        return float(np.max(N[target_min:target_max+1]))
    
    def calc_histogram(A):
        N = np.zeros(max_sum + 1, dtype=int)
        A_list = list(A)
        for i, a in enumerate(A_list):
            for j in range(i, len(A_list)):
                b = A_list[j]
                N[a + b] += 1
        return N

    # Greedy Initialization
    def greedy_init():
        A = set()
        N = np.zeros(max_sum + 1, dtype=int)
        # Cover targets
        # Prioritize targets with fewest potential pairs?
        # Just iterate targets
        targets = list(range(target_min, target_max + 1))
        random.shuffle(targets)
        
        for t in targets:
            # Check if already covered
            if N[t] > 0:
                continue
            
            # Find best x to add
            best_x = None
            best_gain = -1
            # Candidates x such that t-x is in [0, 128]
            # And x not in A? Or x in A?
            # If x in A and t-x in A, it's already covered.
            # We need to add x such that t-x in A OR t-x not in A (but then we only cover t if we add both? No, adding x covers t only if t-x in A).
            # Wait. If we add x, we cover sums x+y for y in A.
            # So to cover t, we need t-x in A.
            # So candidates are x such that t-x in A.
            # If no such x exists, we can add x and t-x? That's 2 adds.
            # Or add x such that t-x is NOT in A, but hope that later we add t-x?
            # Greedy: prefer x where t-x in A.
            
            candidates = []
            for x in range(n + 1):
                if x in A:
                    continue
                # If t-x in A, adding x covers t immediately.
                if 0 <= t - x <= n and (t - x) in A:
                    candidates.append((x, 1)) # Gain 1 for t, plus other sums
                else:
                    # Adding x doesn't cover t immediately unless t-x is added later.
                    # But it might cover other uncovered targets.
                    pass
            
            if not candidates:
                # No immediate fix. Pick random x to add, hoping to build up density
                x = random.randint(0, n)
                if x not in A:
                    A.add(x)
                    # Update N
                    for y in A:
                        if y != x:
                            N[x + y] += 1
                    N[2 * x] += 1
                continue
            
            # Evaluate candidates by how many *other* uncovered targets they cover
            best_x = random.choice([c[0] for c in candidates]) # Random among immediate fixes
            
            A.add(best_x)
            # Update N
            for y in A:
                if y != best_x:
                    N[best_x + y] += 1
            N[2 * best_x] += 1
            
        return A, N

    # Run multiple restarts with different initializations
    for restart in range(10):
        if time.time() > deadline - 5:
            break
            
        if restart == 0:
            A, N = greedy_init()
        else:
            # Random init
            k = random.randint(20, 30)
            A = set(random.sample(range(n + 1), k))
            N = calc_histogram(A)
            
        current_score = get_score(N)
        
        if current_score < best_score:
            best_score = current_score
            best_A = set(A)
            
        T = 5.0
        T_min = 0.1
        cooling = 0.99
        
        no_improve = 0
        while time.time() < deadline - 5:
            # Generate move
            move_type = random.choice(['add', 'remove', 'swap'])
            accepted = False
            changes = []
            
            old_score = current_score
            
            if move_type == 'add':
                candidates = [x for x in range(n + 1) if x not in A]
                if candidates:
                    x = random.choice(candidates)
                    # Apply
                    for y in A:
                        if y != x:
                            N[x + y] += 1
                            changes.append((x + y, 1))
                    N[2 * x] += 1
                    changes.append((2 * x, 1))
                    A.add(x)
                    
                    new_score = get_score(N)
                    
                    if new_score < old_score:
                        accepted = True
                        no_improve = 0
                    else:
                        # Annealing
                        if random.random() < math.exp(-(new_score - old_score) / T):
                            accepted = True
                            no_improve = 0
                        else:
                            # Revert
                            for s, d in changes:
                                N[s] -= d
                            A.remove(x)
                            no_improve += 1

            elif move_type == 'remove':
                if len(A) > 10:
                    x = random.choice(list(A))
                    # Apply
                    for y in A:
                        if y != x:
                            N[x + y] -= 1
                            changes.append((x + y, -1))
                    N[2 * x] -= 1
                    changes.append((2 * x, -1))
                    A.remove(x)
                    
                    new_score = get_score(N)
                    
                    if new_score < old_score:
                        accepted = True
                        no_improve = 0
                    else:
                        if random.random() < math.exp(-(new_score - old_score) / T):
                            accepted = True
                            no_improve = 0
                        else:
                            for s, d in changes:
                                N[s] -= d
                            A.add(x)
                            no_improve += 1

            elif move_type == 'swap':
                if len(A) > 10:
                    x = random.choice(list(A))
                    candidates = [y for y in range(n + 1) if y not in A]
                    if candidates:
                        y = random.choice(candidates)
                        
                        # Remove x
                        for z in A:
                            if z != x:
                                N[x + z] -= 1
                                changes.append((x + z, -1))
                        N[2 * x] -= 1
                        changes.append((2 * x, -1))
                        A.remove(x)
                        
                        # Add y
                        for z in A:
                            if z != y:
                                N[y + z] += 1
                                changes.append((y + z, 1))
                        N[2 * y] += 1
                        changes.append((2 * y, 1))
                        A.add(y)
                        
                        new_score = get_score(N)
                        
                        if new_score < old_score:
                            accepted = True
                            no_improve = 0
                        else:
                            if random.random() < math.exp(-(new_score - old_score) / T):
                                accepted = True
                                no_improve = 0
                            else:
                                # Revert
                                for s, d in changes:
                                    N[s] -= d
                                A.remove(y)
                                A.add(x)
                                no_improve += 1
            
            if accepted:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_A = set(A)
                    
            T *= cooling
            if T < T_min:
                T = T_min
                
            if no_improve > 200:
                break

    return list(best_A)
