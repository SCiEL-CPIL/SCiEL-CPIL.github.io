import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def count_distinct(colors):
    return len(set(colors))

def check_rainbow_c4(colors):
    """Check if there's any rainbow 4-cycle. Returns True if invalid (has rainbow)."""
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a
                    c1 = [colors[edge_idx(a,b)], colors[edge_idx(b,c)], 
                          colors[edge_idx(c,d)], colors[edge_idx(d,a)]]
                    if len(set(c1)) == 4:
                        return True
                    # Cycle 2: a-b-d-c-a
                    c2 = [colors[edge_idx(a,b)], colors[edge_idx(b,d)], 
                          colors[edge_idx(d,c)], colors[edge_idx(c,a)]]
                    if len(set(c2)) == 4:
                        return True
                    # Cycle 3: a-c-b-d-a
                    c3 = [colors[edge_idx(a,c)], colors[edge_idx(c,b)], 
                          colors[edge_idx(b,d)], colors[edge_idx(d,a)]]
                    if len(set(c3)) == 4:
                        return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Start with the min(i,j) coloring which is known valid
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            colors[edge_idx(i, j)] = i
    
    best_colors = colors[:]
    best_count = count_distinct(best_colors)
    
    # Try to add more colors by recoloring edges
    # Process edges in a strategic order
    # Edges incident to high-index vertices have more flexibility
    
    # First, try the known good modifications
    colors[edge_idx(13, 15)] = 15
    colors[edge_idx(14, 15)] = 16
    colors[edge_idx(12, 15)] = 17
    
    if not check_rainbow_c4(colors):
        best_colors = colors[:]
        best_count = count_distinct(best_colors)
    
    # Now try to add more colors greedily
    # For each edge, try assigning a new color and check validity
    # Use a fast local check: only check 4-cycles containing the modified edge
    
    def can_recolor_fast(i, j, new_color, current_colors):
        """Check if recoloring edge (i,j) to new_color creates any rainbow 4-cycle.
        Only check 4-cycles that contain edge (i,j)."""
        old_color = current_colors[edge_idx(i, j)]
        if new_color == old_color:
            return True
        
        # Temporarily apply the change
        e = edge_idx(i, j)
        current_colors[e] = new_color
        
        # Check all 4-cycles containing edge (i,j)
        # A 4-cycle containing edge (i,j) has form: i-j-x-y-i or i-j-y-x-i
        # where x,y are distinct vertices not equal to i or j
        for x in range(N):
            if x == i or x == j:
                continue
            for y in range(N):
                if y == i or y == j or y == x:
                    continue
                # Cycle: i-j-x-y-i
                c1 = [current_colors[edge_idx(i,j)], current_colors[edge_idx(j,x)],
                      current_colors[edge_idx(x,y)], current_colors[edge_idx(y,i)]]
                if len(set(c1)) == 4:
                    current_colors[e] = old_color
                    return False
                # Cycle: i-j-y-x-i
                c2 = [current_colors[edge_idx(i,j)], current_colors[edge_idx(j,y)],
                      current_colors[edge_idx(y,x)], current_colors[edge_idx(x,i)]]
                if len(set(c2)) == 4:
                    current_colors[e] = old_color
                    return False
        
        current_colors[e] = old_color
        return True
    
    # Try to add new colors starting from 18
    used_colors = set(best_colors)
    next_color = 18
    
    # Order edges: try edges that are "more constrained" first
    # Edges with higher min(i,j) tend to have fewer conflicts
    edges_list = [(i, j) for i in range(N) for j in range(i + 1, N)]
    edges_list.sort(key=lambda x: -min(x))
    
    for i, j in edges_list:
        if time.time() > deadline:
            break
        
        e = edge_idx(i, j)
        old_color = best_colors[e]
        
        # Try to assign a new unique color
        if next_color < 120:
            test_colors = best_colors[:]
            if can_recolor_fast(i, j, next_color, test_colors):
                best_colors[e] = next_color
                used_colors.add(next_color)
                next_color += 1
                best_count = count_distinct(best_colors)
    
    # Try more aggressive search: random restarts with local search
    if time.time() < deadline - 10.0:
        for restart in range(50):
            if time.time() > deadline - 5.0:
                break
            
            # Start from best solution and make random changes
            trial = best_colors[:]
            
            # Try random recolorings
            for _ in range(200):
                if time.time() > deadline - 5.0:
                    break
                
                # Pick a random edge
                i = random.randint(0, N - 1)
                j = random.randint(i + 1, N - 1)
                e = edge_idx(i, j)
                old_color = trial[e]
                
                # Try a random new color
                new_color = random.randint(0, 119)
                if new_color == old_color:
                    continue
                
                test_trial = trial[:]
                if can_recolor_fast(i, j, new_color, test_trial):
                    trial[e] = new_color
                    if count_distinct(trial) > best_count:
                        best_colors = trial[:]
                        best_count = count_distinct(best_colors)
    
    # Final verification
    if check_rainbow_c4(best_colors):
        # Fallback to known valid solution
        best_colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                best_colors[edge_idx(i, j)] = i
        best_colors[edge_idx(13, 15)] = 15
        best_colors[edge_idx(14, 15)] = 16
        best_colors[edge_idx(12, 15)] = 17
    
    return best_colors
