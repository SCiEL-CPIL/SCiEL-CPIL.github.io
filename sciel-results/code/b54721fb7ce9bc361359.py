import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Precompute all 4-term arithmetic progressions
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
    
    total_aps = len(aps)
    
    # Precompute neighbors for each index
    # neighbors[i] is a list of tuples (j, k, l) where j,k,l are the other 3 indices in an AP containing i
    neighbors = [[] for _ in range(N)]
    for i1, i2, i3, i4 in aps:
        neighbors[i1].append((i2, i3, i4))
        neighbors[i2].append((i1, i3, i4))
        neighbors[i3].append((i1, i2, i4))
        neighbors[i4].append((i1, i2, i3))
        
    # Convert to numpy arrays for faster access
    neighbors_np = []
    for i in range(N):
        if neighbors[i]:
            neighbors_np.append(np.array(neighbors[i], dtype=np.int32))
        else:
            neighbors_np.append(np.empty((0, 3), dtype=np.int32))

    def count_mono(colors):
        count = 0
        for i1, i2, i3, i4 in aps:
            if colors[i1] == colors[i2] == colors[i3] == colors[i4]:
                count += 1
        return count

    best_score = float('inf')
    best_coloring = None

    while time.time() < deadline - 5.0:  # Leave 5s for finalization
        # Initialize random coloring
        colors = np.random.randint(0, 2, size=N)
        score = count_mono(colors)
        
        if score < best_score:
            best_score = score
            best_coloring = colors.copy()
        
        # Simulated Annealing
        T = 2.0
        alpha = 0.9995
        inner_deadline = time.time() + 5.0  # Run SA for ~5 seconds per restart
        
        while time.time() < inner_deadline and T > 0.01:
            i = random.randint(0, N - 1)
            old_c = colors[i]
            new_c = 1 - old_c
            
            others = neighbors_np[i]
            if len(others) == 0:
                delta = 0
            else:
                cols = colors[others]
                c0, c1, c2 = cols[:, 0], cols[:, 1], cols[:, 2]
                all_eq = (c0 == c1) & (c1 == c2)
                val = c0
                was_mono = all_eq & (val == old_c)
                becomes_mono = all_eq & (val == new_c)
                delta = np.sum(becomes_mono) - np.sum(was_mono)
                
            if delta < 0 or random.random() < np.exp(-delta / T):
                colors[i] = new_c
                score += delta
                
            T *= alpha

    return best_coloring.tolist()
# EVOLVE-BLOCK-END
