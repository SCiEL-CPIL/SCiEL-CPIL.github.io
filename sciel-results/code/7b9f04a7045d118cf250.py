import random
import math
import time

def construct():
    def score(S):
        present = [False] * 513
        for x in S:
            present[x] = True
            
        diffs = set()
        list_S = list(S)
        n = len(list_S)
        
        for i in range(n):
            a = list_S[i]
            for j in range(i+1, n):
                c = list_S[j]
                s = a + c
                if s % 2 == 0:
                    b = s // 2
                    if present[b]:
                        d = c - a
                        diffs.add(d)
        return len(diffs)

    def random_set():
        return set(random.sample(range(1, 513), 32))

    # Initialize with a set that might have better structure
    # e.g., multiples of a small number? Or just random.
    # Let's try a few random starts and pick the best, then optimize.
    
    start_time = time.time()
    time_limit = 110
    
    # Initial population of a few candidates to find a good starting point
    candidates = [random_set() for _ in range(10)]
    best_start = max(candidates, key=score)
    best_score = score(best_start)
    
    current = best_start
    current_score = best_score
    
    # Simulated Annealing / Local Search
    T = 10.0 # Temperature
    cooling = 0.999
    min_T = 0.1
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            # Restart with a slightly perturbed version of best or new random
            # Or just reset T and perturb current
            T = 5.0
            # Perturb current by swapping 2-5 elements
            for _ in range(random.randint(2, 5)):
                items = list(current)
                x = random.choice(items)
                current.remove(x)
                current.add(random.randint(1, 512))
            current_score = score(current)
            if current_score > best_score:
                best_score = current_score
                best_start = set(current)
            continue
            
        # Generate a neighbor: swap one element in with one out
        items_in = list(current)
        x = random.choice(items_in)
        
        # Find a random y not in current
        # To avoid infinite loop if set is full (it's not, 32 << 512)
        y = random.randint(1, 512)
        attempts = 0
        while y in current and attempts < 100:
            y = random.randint(1, 512)
            attempts += 1
        
        if y in current:
            continue
            
        new_set = set(current)
        new_set.remove(x)
        new_set.add(y)
        
        new_score = score(new_set)
        
        # Acceptance criteria
        if new_score >= current_score:
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_start = set(current)
        else:
            # Accept with probability exp(-(new - old) / T)
            delta = current_score - new_score
            if random.random() < math.exp(-delta / T):
                current = new_set
                current_score = new_score
                
        T *= cooling

    return list(best_start)
