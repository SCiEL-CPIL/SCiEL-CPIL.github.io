import random
import time

def construct():
    n = 256
    max_time = 115  # Leave buffer for overhead
    
    def count_mono_4ap(colors):
        count = 0
        # d ranges from 1 to 85 (since a+3d <= 256 => 3d <= 255 => d <= 85)
        for d in range(1, 86):
            # a ranges from 1 to 256 - 3d
            limit = n - 3 * d
            for a in range(1, limit + 1):
                # Indices are 0-based in Python list
                if colors[a - 1] == colors[a + d - 1] == colors[a + 2 * d - 1] == colors[a + 3 * d - 1]:
                    count += 1
        return count

    best_count = float('inf')
    best_colors = None
    
    start_time = time.time()
    
    # Try a few restarts with random initializations
    # Also try a structured initialization
    initializations = []
    
    # 1. Random
    initializations.append([random.randint(0, 1) for _ in range(n)])
    # 2. Alternating
    initializations.append([i % 2 for i in range(n)])
    # 3. Blocks of 2
    initializations.append([(i // 2) % 2 for i in range(n)])
    # 4. Modular 3
    initializations.append([i % 3 for i in range(n)])
    # 5. Modular 5
    initializations.append([i % 5 for i in range(n)])
    
    for init in initializations:
        colors = init.copy()
        current_count = count_mono_4ap(colors)
        
        # Local search
        improved = True
        while improved and (time.time() - start_time) < max_time:
            improved = False
            # Randomize order of indices to break symmetry
            indices = list(range(n))
            random.shuffle(indices)
            
            for i in indices:
                if (time.time() - start_time) > max_time:
                    break
                
                # Flip bit i
                colors[i] = 1 - colors[i]
                new_count = count_mono_4ap(colors)
                
                if new_count < current_count:
                    current_count = new_count
                    improved = True
                else:
                    # Revert
                    colors[i] = 1 - colors[i]
        
        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()
            
        if (time.time() - start_time) > max_time:
            break
            
    if best_colors is None:
        # Fallback to seed behavior if something went wrong
        best_colors = [(i // 2) % 2 for i in range(n)]
        
    return best_colors
