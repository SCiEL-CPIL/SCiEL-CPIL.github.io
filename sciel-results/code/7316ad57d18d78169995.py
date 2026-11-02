import numpy as np

N = 50

def construct():
    # Greedy Sidon set construction
    S = []
    diffs = set()
    for i in range(N):
        # Check if adding i creates a duplicate difference
        valid = True
        for j in S:
            d = abs(i - j)
            if d in diffs:
                valid = False
                break
            diffs.add(d)
        if valid:
            S.append(i)
            
    # Create the sequence
    h = [0.0] * N
    for i in S:
        h[i] = 1.0
        
    return h
