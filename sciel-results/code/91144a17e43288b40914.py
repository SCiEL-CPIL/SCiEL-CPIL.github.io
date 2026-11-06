import time
import itertools
import random

N = 16
NUM_EDGES = N * (N - 1) // 2
DEADLINE = 50  # safety margin before 60s limit

def construct():
    start_time = time.time()
    
    # Build edge index mapping
    edge_index = {}
    edge_list = []
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_index[(i, j)] = idx
            edge_index[(j, i)] = idx
            edge_list.append((i, j))
            idx += 1
    
    # Enumerate all 4-sets and precompute cycles
    all_4sets = list(itertools.combinations(range(N), 4))
    
    def get_cycles(a, b, c, d):
        e1 = (edge_index[(a,b)], edge_index[(b,c)], edge_index[(c,d)], edge_index[(d,a)])
        e2 = (edge_index[(a,b)], edge_index[(b,d)], edge_index[(d,c)], edge_index[(c,a)])
        e3 = (edge_index[(a,c)], edge_index[(c,b)], edge_index[(b,d)], edge_index[(d,a)])
        return (e1, e2, e3)
    
    # For each edge, store the list of (4set_index, cycle_index) pairs
    edge_to_4sets = [[] for _ in range(NUM_EDGES)]
    cycles_by_4set = []
    for s_idx, (a, b, c, d) in enumerate(all_4sets):
        cycles = get_cycles(a, b, c, d)
        cycles_by_4set.append(cycles)
        for c_idx, cycle in enumerate(cycles):
            for e_idx in cycle:
                edge_to_4sets[e_idx].append((s_idx, c_idx))
    
    def is_valid_cycle(colors, cycle):
        c0 = colors[cycle[0]]
        c1 = colors[cycle[1]]
        c2 = colors[cycle[2]]
        c3 = colors[cycle[3]]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
        return True
    
    def check_edge_validity(colors, e_idx):
        for (s_idx, c_idx) in edge_to_4sets[e_idx]:
            cycle = cycles_by_4set[s_idx][c_idx]
            if not is_valid_cycle(colors, cycle):
                return False
        return True
    
    def full_validation(colors):
        for s_idx in range(len(all_4sets)):
            for c_idx in range(3):
                cycle = cycles_by_4set[s_idx][c_idx]
                if not is_valid_cycle(colors, cycle):
                    return False
        return True
    
    # Initialize with the known valid coloring: color(i,j) = i for i<j
    colors = [0] * NUM_EDGES
    for idx, (i, j) in enumerate(edge_list):
        colors[idx] = i
    
    if not full_validation(colors):
        return [0] * NUM_EDGES
    
    best_colors = colors[:]
    best_num_colors = len(set(colors))
    
    # Strategy: try to add new colors by recoloring edges
    # Prioritize edges with fewer constraints
    edge_constraint_counts = [(len(edge_to_4sets[e]), e) for e in range(NUM_EDGES)]
    edge_constraint_counts.sort()
    sorted_edges = [e for _, e in edge_constraint_counts]
    
    # Try to add colors one at a time
    for new_color in range(15, 120):
        if time.time() - start_time > DEADLINE:
            break
        
        found = False
        for e_idx in sorted_edges:
            if time.time() - start_time > DEADLINE:
                break
            
            old_color = colors[e_idx]
            colors[e_idx] = new_color
            
            if check_edge_validity(colors, e_idx):
                found = True
                break
            
            colors[e_idx] = old_color
        
        if not found:
            # Try reverse order
            for e_idx in reversed(sorted_edges):
                if time.time() - start_time > DEADLINE:
                    break
                
                old_color = colors[e_idx]
                colors[e_idx] = new_color
                
                if check_edge_validity(colors, e_idx):
                    found = True
                    break
                
                colors[e_idx] = old_color
        
        if found:
            current_num = len(set(colors))
            if current_num > best_num_colors:
                best_num_colors = current_num
                best_colors = colors[:]
    
    # Try randomized local search to find better solutions
    if time.time() - start_time < DEADLINE - 5:
        random.seed(42)
        for _ in range(50):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Start from best known solution
            current = best_colors[:]
            current_num = len(set(current))
            
            # Try random swaps and recolorings
            for _ in range(100):
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                # Pick a random edge to recolor
                e_idx = random.randint(0, NUM_EDGES - 1)
                old_color = current[e_idx]
                
                # Try a random new color
                new_color = random.randint(0, 119)
                if new_color == old_color:
                    continue
                
                current[e_idx] = new_color
                
                if check_edge_validity(current, e_idx):
                    new_num = len(set(current))
                    if new_num > current_num:
                        current_num = new_num
                        if current_num > best_num_colors:
                            best_num_colors = current_num
                            best_colors = current[:]
                else:
                    current[e_idx] = old_color
            
            # Try to add new colors systematically
            used_colors = set(current)
            for c in range(120):
                if c in used_colors:
                    continue
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                for e_idx in sorted_edges:
                    if time.time() - start_time > DEADLINE - 5:
                        break
                    
                    old_color = current[e_idx]
                    current[e_idx] = c
                    
                    if check_edge_validity(current, e_idx):
                        new_num = len(set(current))
                        if new_num > current_num:
                            current_num = new_num
                            if current_num > best_num_colors:
                                best_num_colors = current_num
                                best_colors = current[:]
                        break
                    else:
                        current[e_idx] = old_color
    
    # Final validation
    if full_validation(best_colors):
        return best_colors
    else:
        return colors
