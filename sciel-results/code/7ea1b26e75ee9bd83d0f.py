import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def get_instances():
        instances = []
        for size, seed in ((60, 801), (120, 802), (240, 803)):
            rng = random.Random(seed)
            values = set()
            while len(values) < size:
                x = rng.randint(-10000, 10000)
                if x != 0:
                    values.add(x)
            instances.append(sorted(values))
        return instances

    instances = get_instances()
    results = []

    for S in instances:
        n = len(S)
        S_set = set(S)
        
        # Initial solution: Odd numbers
        A = [v for v in S if v % 2 != 0]
        A_set = set(A)
        
        # Greedy improvement
        # Try to add elements that don't conflict
        changed = True
        while changed:
            changed = False
            for v in S:
                if v in A_set:
                    continue
                # Check if v can be added
                valid = True
                if 2*v in A_set:
                    valid = False
                else:
                    for a in A:
                        if v + a in A_set:
                            valid = False
                            break
                if valid:
                    A.append(v)
                    A_set.add(v)
                    changed = True

        best_A = list(A)
        best_size = len(A)
        
        # Simulated Annealing
        T = 10.0
        cooling = 0.99
        max_iter = 50000
        
        current_size = len(A)
        
        for i in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            # Pick random x not in A
            candidates = [v for v in S if v not in A_set]
            if not candidates:
                # All in A, try removing? No, max size is n.
                break
            x = random.choice(candidates)
            
            # Find conflicts
            conflicts = []
            if 2*x in A_set:
                conflicts.append(2*x)
            for a in A:
                if a != x: # x not in A, so a!=x always
                    s_val = x + a
                    if s_val in A_set:
                        conflicts.append(s_val)
            
            # Unique conflicts
            conflicts = list(set(conflicts))
            
            num_conflicts = len(conflicts)
            
            if num_conflicts == 0:
                # Add x
                A.append(x)
                A_set.add(x)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_A = list(A)
            elif num_conflicts == 1:
                # Swap: remove conflict, add x
                y = conflicts[0]
                # Check if removing y and adding x is valid
                # We know x conflicts with y.
                # Are there other conflicts? No, num_conflicts=1.
                # So (A - {y}) U {x} is valid.
                A.remove(y)
                A_set.remove(y)
                A.append(x)
                A_set.add(x)
                # Size unchanged
                # Accept? Yes, to move in search space.
            else:
                # Remove multiple conflicts
                # Gain = 1 - num_conflicts
                # Delta E = - (1 - num_conflicts) = num_conflicts - 1 (cost)
                # Accept prob = exp(-(num_conflicts - 1) / T)
                if random.random() < math.exp(-(num_conflicts - 1) / T):
                    # Remove all conflicts, add x
                    for y in conflicts:
                        A.remove(y)
                        A_set.remove(y)
                    A.append(x)
                    A_set.add(x)
                    current_size = current_size - num_conflicts + 1
                    if current_size > best_size:
                        best_size = current_size
                        best_A = list(A)
            
            T *= cooling
            if T < 0.01:
                # Restart
                A = list(best_A)
                A_set = set(A)
                current_size = best_size
                T = 10.0

        results.append(best_A)

    return results
