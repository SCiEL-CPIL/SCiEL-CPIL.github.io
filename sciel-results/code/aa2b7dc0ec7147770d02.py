import time
import random
import copy

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # 115 seconds to leave margin
    
    # 1. Seed: Greenwood-Gleason colouring of K16 (Clebsch) doubled to n=32
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
        
    def get_seed_graph(n):
        graph = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if i // 16 == j // 16:
                    c = log[(i % 16) ^ (j % 16)] % 3
                else:
                    c = 3
                graph[i][j] = c
                graph[j][i] = c
        return graph

    current_n = 32
    # We will use a fixed size array for graph to avoid resizing overhead, 
    # but only indices < current_n are active.
    MAX_N = 62
    graph = [[0]*MAX_N for _ in range(MAX_N)]
    
    # Initialize with seed
    seed_graph = get_seed_graph(32)
    for i in range(32):
        for j in range(32):
            graph[i][j] = seed_graph[i][j]
            
    best_n = 32
    best_graph = copy.deepcopy(graph)
    
    def count_triangles(n):
        """Count monochromatic triangles in graph[0..n-1]"""
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                c = graph[i][j]
                for k in range(j+1, n):
                    if graph[i][k] == c and graph[j][k] == c:
                        count += 1
        return count

    def delta_triangle_count(i, j, new_color, n):
        """
        Calculate change in triangle count if we change edge (i,j) to new_color.
        We only need to consider triangles involving (i,j).
        Triangles are (i, j, k) for k != i, j.
        """
        old_color = graph[i][j]
        if old_color == new_color:
            return 0
        
        delta = 0
        for k in range(n):
            if k == i or k == j:
                continue
            
            # Triangle (i, j, k)
            # Old state: colors (old_color, graph[i][k], graph[j][k])
            # New state: colors (new_color, graph[i][k], graph[j][k])
            
            c_ik = graph[i][k]
            c_jk = graph[j][k]
            
            old_mono = 1 if (old_color == c_ik == c_jk) else 0
            new_mono = 1 if (new_color == c_ik == c_jk) else 0
            
            delta += (new_mono - old_mono)
            
        return delta

    def shuffle_valid_graph(n, steps=1000):
        """
        Perform a random walk on valid colorings of K_n.
        Only accept moves that keep triangle count 0.
        """
        # Current count is 0
        for _ in range(steps):
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_color = graph[i][j]
            # Try a different color
            new_color = random.randint(0, 3)
            if new_color == old_color:
                continue
            
            d = delta_triangle_count(i, j, new_color, n)
            if d == 0:
                # Accept
                graph[i][j] = new_color
                graph[j][i] = new_color
            # else reject

    def try_extend_with_sa(n, time_budget=1.0):
        """
        Try to extend K_n to K_{n+1} using Simulated Annealing.
        Returns True if successful.
        """
        new_n = n + 1
        if new_n > MAX_N:
            return False
            
        # Save current state
        old_graph = copy.deepcopy(graph)
        
        # Add new vertex n
        for i in range(n):
            c = random.randint(0, 3)
            graph[i][new_n] = c
            graph[new_n][i] = c
        graph[new_n][new_n] = 0
        
        # Initial energy
        current_energy = count_triangles(new_n)
        if current_energy == 0:
            return True # Lucky!
            
        # Simulated Annealing
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.999
        start_time_sa = time.time()
        
        # We want to minimize energy.
        # Move: pick random edge (i,j) in 0..new_n-1, change color.
        
        while time.time() - start_time_sa < time_budget:
            if current_energy == 0:
                return True
                
            i = random.randint(0, new_n-1)
            j = random.randint(0, new_n-1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
                
            old_color = graph[i][j]
            new_color = random.randint(0, 3)
            if new_color == old_color:
                continue
            
            d = delta_triangle_count(i, j, new_color, new_n)
            
            if d < 0 or random.random() < pow(2.71828, -d / T):
                graph[i][j] = new_color
                graph[j][i] = new_color
                current_energy += d
                
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        # Check final state
        if count_triangles(new_n) == 0:
            return True
        else:
            # Rollback
            graph[:] = old_graph
            return False

    attempts = 0
    max_attempts = 10000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # 1. Try to extend
        if try_extend_with_sa(current_n, time_budget=1.0):
            current_n += 1
            best_n = current_n
            best_graph = copy.deepcopy(graph)
            # If we reached a high n, maybe reduce time budget for extension to allow more shuffles?
            # Or just continue.
            
        else:
            # 2. Failed to extend. Shuffle the current valid graph to explore other valid colorings.
            # This helps escape local minima in the extension search space.
            shuffle_valid_graph(current_n, steps=500)
            
            # Check if time is running out
            if time.time() > deadline - 5.0:
                break

    # Finalize
    # Ensure we return the best valid state
    # The current graph is valid (we only accept valid shuffles and successful extensions).
    # But we might have rolled back the last failed extension, so graph is valid K_current_n.
    # best_graph is a copy of the best state found.
    
    # Reconstruct flat list from best_graph
    colors = []
    for i in range(best_n):
        for j in range(i+1, best_n):
            colors.append(best_graph[i][j])
            
    return {'n': best_n, 'colors': colors}

# EVOLVE-BLOCK-END
