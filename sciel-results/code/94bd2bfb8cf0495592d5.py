import time

N = 16
NUM_EDGES = N * (N - 1) // 2

def construct():
    # Deadline guard
    start_time = time.time()
    
    # Generate colors
    colors = [0] * NUM_EDGES
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            # Color is min(i, j) which is i since i < j
            colors[idx] = i
            idx += 1
            
    return colors
