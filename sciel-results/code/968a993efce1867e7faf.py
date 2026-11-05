import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # The optimal strategy identified is to include 2 and all odd numbers in [129, 255].
    # Let's verify if any other combination beats this.
    # Sum = 0.5 + sum(1/x for x in odds in 129..255)
    
    # Construct the candidate set
    candidate = [2]
    for x in range(129, 256, 2):
        candidate.append(x)
        
    # Verify validity
    for i in range(len(candidate)):
        for j in range(i+1, len(candidate)):
            if lcm(candidate[i], candidate[j]) <= 256:
                # Fallback to greedy if manual construction fails (should not happen)
                return greedy_construct()
                
    return candidate

def greedy_construct():
    selected = []
    for x in range(2, 257):
        if all(lcm(x, y) > 256 for y in selected):
            selected.append(x)
    return selected
