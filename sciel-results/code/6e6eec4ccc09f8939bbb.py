import time
import random
import math

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Precompute all forbidden triples (a, b, c) with a < b < c, 1/a = 1/b + 1/c
    # This means (b-a)(c-a) = a^2
    forbidden_triples = []
    triple_index = {}  # Maps element to list of triple indices it's in
    
    for a in range(1, 91):
        a2 = a * a
        for d in range(1, a):
            if a2 % d == 0:
                b = a + d
                c = a + a2 // d
                if b < c and b <= 180 and c <= 180:
                    idx = len(forbidden_triples)
                    forbidden_triples.append((a, b, c))
                    triple_index.setdefault(a, []).append(idx)
                    triple_index.setdefault(b, []).append(idx)
                    triple_index.setdefault(c, []).append(idx)
    
    n_triples = len(forbidden_triples)
    
    # Simulated annealing approach
    # State: binary vector x[1..180], x[i]=1 means i is in the set
    # Objective: maximize sum(x) - penalty * sum(violated_triples)
    # A triple is violated if all three elements are present
    
    # Start with a greedy solution
    S = set(range(91, 181))
    
    def get_energy(x, penalty):
        """Compute energy = -|S| + penalty * (number of violated triples)"""
        count = sum(x)
        violated = 0
        for triple in forbidden_triples:
            if x[triple[0]] and x[triple[1]] and x[triple[2]]:
                violated += 1
        return -count + penalty * violated
    
    def count_violations(x):
        """Count number of violated triples"""
        violated = 0
        for triple in forbidden_triples:
            if x[triple[0]] and x[triple[1]] and x[triple[2]]:
                violated += 1
        return violated
    
    # Convert set to array
    def set_to_array(S):
        x = [0] * 181
        for i in S:
            x[i] = 1
        return x
    
    def array_to_set(x):
        return {i for i in range(1, 181) if x[i]}
    
    # Start with greedy solution
    x = set_to_array(S)
    current_energy = get_energy(x, 1.0)
    best_x = x[:]
    best_energy = current_energy
    best_count = sum(x)
    
    # Simulated annealing parameters
    T_start = 50.0
    T_end = 0.1
    cooling_rate = 0.999
    
    # Dynamic penalty: start high, decrease as we find better solutions
    penalty = 10.0
    
    T = T_start
    iteration = 0
    max_iterations = 500000
    
    while T > T_end and iteration < max_iterations:
        if time.time() > deadline - 5:
            break
        
        # Choose a random element to flip
        i = random.randint(1, 180)
        
        # Compute energy change
        old_x = x[i]
        x[i] = 1 - old_x
        
        new_energy = get_energy(x, penalty)
        delta_e = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta_e/T) if worse
        if delta_e < 0 or random.random() < math.exp(-delta_e / T):
            current_energy = new_energy
            if sum(x) > best_count and count_violations(x) == 0:
                best_x = x[:]
                best_count = sum(x)
                best_energy = current_energy
                # Decrease penalty as we find better solutions
                penalty = max(1.0, penalty * 0.95)
        else:
            x[i] = old_x  # Revert
        
        # Cool down
        T *= cooling_rate
        iteration += 1
    
    # Final polish: try to add any elements that don't create violations
    final_S = array_to_set(best_x)
    
    # Try to add remaining elements greedily
    for a in range(1, 181):
        if a not in final_S:
            test_S = final_S | {a}
            x_test = set_to_array(test_S)
            if count_violations(x_test) == 0:
                final_S = test_S
    
    # One more pass of local search on the final solution
    x = set_to_array(final_S)
    for _ in range(100000):
        if time.time() > deadline - 2:
            break
        
        i = random.randint(1, 180)
        old_x = x[i]
        x[i] = 1 - old_x
        
        if count_violations(x) == 0:
            if sum(x) > best_count:
                best_x = x[:]
                best_count = sum(x)
        else:
            x[i] = old_x  # Revert if violation
    
    return sorted(array_to_set(best_x))
