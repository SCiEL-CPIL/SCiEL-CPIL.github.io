"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    n = 256
    deadline = time.time() + 110  # Leave 10s buffer for 120s limit
    
    # Precompute all 4-APs: a, a+d, a+2d, a+3d with d >= 1, all in [0, n-1]
    # (We use 0-indexed internally, convert at the end)
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(ap_list)
    
    # Precompute for each position, which AP indices contain it
    # And for each AP, the 4 positions
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for pos in ap:
            pos_aps[pos].append(idx)
    
    # Count monochromatic 4-APs
    def count_mono(colors):
        count = 0
        for ap in ap_list:
            if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                count += 1
        return count
    
    # Local search: try flipping each position, accept if it reduces mono count
    def local_search(colors):
        improved = True
        while improved:
            improved = False
            # Compute current mono count
            mono_count = count_mono(colors)
            # Try each position
            for pos in range(n):
                old_color = colors[pos]
                new_color = 1 - old_color
                colors[pos] = new_color
                new_count = count_mono(colors)
                if new_count < mono_count:
                    mono_count = new_count
                    improved = True
                else:
                    colors[pos] = old_color
        return colors
    
    # Simulated annealing-style approach with multiple restarts
    best_colors = None
    best_count = float('inf')
    
    # Initial solutions to try
    initial_solutions = []
    
    # 1. Thue-Morse sequence (0-indexed)
    def thue_morse(n):
        seq = [0]
        for _ in range(n.bit_length() - 1):
            seq = seq + [1 - x for x in seq]
        return seq[:n]
    
    initial_solutions.append(thue_morse(n))
    
    # 2. Random initializations
    for _ in range(10):
        initial_solutions.append([random.randint(0, 1) for _ in range(n)])
    
    # 3. Alternating blocks of various sizes
    for block_size in [2, 3, 4, 5, 7, 8]:
        initial_solutions.append([(i // block_size) % 2 for i in range(n)])
    
    # 4. Walsh-Hadamard based
    def walsh_hadamard(n):
        # Simple Walsh-Hadamard sequence
        seq = [0]
        for _ in range(n.bit_length() - 1):
            seq = seq + seq
            # Actually let's use a proper Walsh function
        # Use the first row of Walsh-Hadamard matrix
        result = []
        for i in range(n):
            # Count set bits in i
            result.append(bin(i).count('1') % 2)
        return result
    
    initial_solutions.append(walsh_hadamard(n))
    
    # Run local search on each initial solution
    for init_colors in initial_solutions:
        if time.time() > deadline - 10:
            break
        
        colors = init_colors[:]
        colors = local_search(colors)
        mono_count = count_mono(colors)
        
        if mono_count < best_count:
            best_count = mono_count
            best_colors = colors[:]
    
    # Try simulated annealing from the best solution found
    if best_colors is not None and time.time() < deadline - 20:
        colors = best_colors[:]
        current_count = best_count
        temperature = 5.0
        cooling = 0.999
        
        for step in range(100000):
            if time.time() > deadline - 10:
                break
            
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            new_color = 1 - old_color
            colors[pos] = new_color
            new_count = count_mono(colors)
            
            delta = new_count - current_count
            if delta < 0 or random.random() < pow(2.718281828, -delta / max(temperature, 0.001)):
                current_count = new_count
                if current_count < best_count:
                    best_count = current_count
                    best_colors = colors[:]
            else:
                colors[pos] = old_color
            
            temperature *= cooling
    
    # Return 1-indexed colors (the task says integers 1..256)
    # But the seed returns colors for i in range(256), which is 0-indexed internally
    # The task says "256 colours in {0,1} for the integers 1..256"
    # The seed returns [(i // 2) % 2 for i in range(256)] which is for positions 0..255
    # So we return 256 values, one for each integer 1..256 (but indexed 0..255 in the list)
    
    if best_colors is None:
        # Fallback
        return [(i // 2) % 2 for i in range(n)]
    
    return best_colors
# EVOLVE-BLOCK-END
