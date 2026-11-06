import random
import math

def construct():
    n = 20
    
    # Precompute cycles
    unique_cycles_set = set()
    cycles_by_edge = {}
    
    for u in range(n):
        for v in range(n):
            if v == u: continue
            for w in range(n):
                if w == u or w == v: continue
                for x in range(n):
                    if x == u or x == v or x == w: continue
                    e1 = (min(u,v), max(u,v))
                    e2 = (min(v,w), max(v,w))
                    e3 = (min(w,x), max(w,x))
                    e4 = (min(x,u), max(x,u))
                    cycle_edges = frozenset([e1, e2, e3, e4])
                    if cycle_edges not in unique_cycles_set:
                        unique_cycles_set.add(cycle_edges)
                        for e in cycle_edges:
                            if e not in cycles_by_edge:
                                cycles_by_edge[e] = []
                            cycles_by_edge[e].append(cycle_edges)

    all_cycles = list(unique_cycles_set)
    
    # State
    edge_colors = {}
    bad_cycles_set = set()
    
    def check_cycle(cycle_edges):
        colors = []
        for e in cycle_edges:
            if e not in edge_colors:
                return False
            colors.append(edge_colors[e])
        return len(set(colors)) < 4

    # Initialize
    random.seed(123)
    a_part = list(range(10))
    b_part = list(range(10, 20))
    for u in a_part:
        for v in b_part:
            edge_colors[(u, v)] = random.randint(0, 19)
            
    for cycle in all_cycles:
        if check_cycle(cycle):
            bad_cycles_set.add(cycle)
            
    current_edges = len(edge_colors)
    best_edges = current_edges if not bad_cycles_set else 0
    best_state = dict(edge_colors) if not bad_cycles_set else {}
    
    T = 1.0
    cooling = 0.999
    steps = 30000
    
    for i in range(steps):
        move_type = random.random()
        modified_edge = None
        
        if move_type < 0.4:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            key = (min(u,v), max(u,v))
            if key in edge_colors: continue
            c = random.randint(0, 19)
            modified_edge = key
            edge_colors[key] = c
            current_edges += 1
        elif move_type < 0.7:
            if not edge_colors: continue
            keys = list(edge_colors.keys())
            key = random.choice(keys)
            c = random.randint(0, 19)
            if c == edge_colors[key]: continue
            modified_edge = key
            edge_colors[key] = c
        else:
            if not edge_colors: continue
            keys = list(edge_colors.keys())
            key = random.choice(keys)
            modified_edge = key
            del edge_colors[key]
            current_edges -= 1
            
        if modified_edge is None:
            continue
            
        # Update bad_cycles_set
        if modified_edge in cycles_by_edge:
            for cycle in cycles_by_edge[modified_edge]:
                # If it was bad, remove it tentatively
                if cycle in bad_cycles_set:
                    bad_cycles_set.remove(cycle)
                # Check if it is bad now
                if check_cycle(cycle):
                    bad_cycles_set.add(cycle)
                    
        # Calculate score
        old_score = 0 # We don't store old score explicitly, we compute delta
        # Actually, we can just compute new score and compare to best?
        # No, SA needs to accept worse states.
        # We need the score of the PREVIOUS state.
        # Let's store previous score.
        
    # This requires storing prev_score.
    
    return [[u, v, c] for (u, v), c in best_state.items()]
