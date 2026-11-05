import random
import time
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def build_four_cycles():
    cycles = []
    for a, b, c, d in itertools.combinations(range(N), 4):
        # Cycle a-b-c-d-a
        e1 = edge_index(a, b)
        e2 = edge_index(b, c)
        e3 = edge_index(c, d)
        e4 = edge_index(d, a)
        cycles.append((e1, e2, e3, e4))
        
        # Cycle a-b-d-c-a
        e1 = edge_index(a, b)
        e2 = edge_index(b, d)
        e3 = edge_index(d, c)
        e4 = edge_index(c, a)
        cycles.append((e1, e2, e3, e4))
        
        # Cycle a-c-b-d-a
        e1 = edge_index(a, c)
        e2 = edge_index(c, b)
        e3 = edge_index(b, d)
        e4 = edge_index(d, a)
        cycles.append((e1, e2, e3, e4))
    return cycles

CYCLES = build_four_cycles()

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Pre-compute edge to cycles mapping
    edge_cycles = [[] for _ in range(NUM_EDGES)]
    for idx, cycle in enumerate(CYCLES):
        for e in cycle:
            edge_cycles[e].append(idx)
            
    def get_conflict_status(cycle_idx, colors):
        e1, e2, e3, e4 = CYCLES[cycle_idx]
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        return (c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4)
        
    best_colors = None
    best_score = -1
    
    # Try multiple restarts
    for _ in range(10):
        if time.time() > deadline:
            break
            
        # Initialize with random colors from a small palette
        colors = [random.randint(0, 15) for _ in range(NUM_EDGES)]
        
        # Calculate initial conflict status
        cycle_is_rainbow = [get_conflict_status(i, colors) for i in range(len(CYCLES))]
        current_conflicts = sum(cycle_is_rainbow)
        
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < deadline and no_improve < max_no_improve:
            e = random.randint(0, NUM_EDGES - 1)
            old_color = colors[e]
            
            # Choose new color
            # Bias towards existing colors to maintain structure, or new colors to increase diversity
            if random.random() < 0.8:
                new_color = random.choice(colors)
            else:
                new_color = random.randint(0, 119)
                
            if new_color == old_color:
                continue
                
            # Calculate delta in conflicts
            delta = 0
            for cyc_idx in edge_cycles[e]:
                old_state = cycle_is_rainbow[cyc_idx]
                colors[e] = new_color
                new_state = get_conflict_status(cyc_idx, colors)
                colors[e] = old_color
                if new_state and not old_state:
                    delta += 1
                elif old_state and not new_state:
                    delta -= 1
                    
            new_total_conflicts = current_conflicts + delta
            
            # Calculate change in distinct colors
            old_distinct = len(set(colors))
            colors[e] = new_color
            new_distinct = len(set(colors))
            colors[e] = old_color
            
            # Acceptance Criteria
            accept = False
            if new_total_conflicts < current_conflicts:
                accept = True
            elif new_total_conflicts == current_conflicts:
                if new_distinct > old_distinct:
                    accept = True
                elif new_distinct == old_distinct:
                    # Random walk to escape local optima
                    accept = random.random() < 0.5
            # If conflicts increase, reject
            
            if accept:
                colors[e] = new_color
                for cyc_idx in edge_cycles[e]:
                    cycle_is_rainbow[cyc_idx] = get_conflict_status(cyc_idx, colors)
                current_conflicts = new_total_conflicts
                no_improve = 0
            else:
                no_improve += 1
                
        # If valid, update best
        if current_conflicts == 0:
            distinct = len(set(colors))
            if distinct > best_score:
                best_score = distinct
                best_colors = colors[:]
                
    if best_colors is None:
        # Fallback: All zeros is valid
        return [0] * NUM_EDGES
        
    return best_colors
# EVOLVE-BLOCK-END
