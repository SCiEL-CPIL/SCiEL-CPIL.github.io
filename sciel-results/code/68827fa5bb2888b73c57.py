"""Dense 4-cycle-rainbow edge-coloured graph on 20 vertices.
Uses simulated annealing with algebraic initial colorings to escape local optima."""
import time
import random
import math
from itertools import combinations

N = 20
DEADLINE_MARGIN = 8.0

def _precompute_cycles():
    """Precompute all 4-cycles and map each edge to the cycles it belongs to."""
    cycles = []
    edge_to_cycles = {e: [] for e in combinations(range(N), 2)}
    
    for a, b, c, d in combinations(range(N), 4):
        perms = [
            ((a,b), (b,c), (c,d), (d,a)),
            ((a,b), (b,d), (d,c), (c,a)),
            ((a,c), (c,b), (b,d), (d,a))
        ]
        for edges in perms:
            norm_edges = tuple(sorted((min(u,v), max(u,v)) for u,v in edges))
            cyc_id = len(cycles)
            cycles.append(norm_edges)
            for e in norm_edges:
                edge_to_cycles[e].append(cyc_id)
    return cycles, edge_to_cycles

def _count_bad_cycles(edges_set, colors, cycles, edge_to_cycles):
    """Count bad 4-cycles (non-rainbow) in the current graph."""
    bad_count = 0
    for cyc_id, cyc_edges in enumerate(cycles):
        if all(e in edges_set for e in cyc_edges):
            cols = [colors[e] for e in cyc_edges]
            if len(set(cols)) < 4:
                bad_count += 1
    return bad_count

def _try_add_edge(u, v, edges_set, colors, cycles, edge_to_cycles):
    """Try to add edge (u,v) with the best valid color. Returns (success, color)."""
    e = (min(u, v), max(u, v))
    if e in edges_set:
        return False, -1
    
    # Find colors that don't create a bad cycle
    for c in range(N):
        # Check all 4-cycles containing e
        valid = True
        for cyc_id in edge_to_cycles[e]:
            cyc_edges = cycles[cyc_id]
            others = [ed for ed in cyc_edges if ed != e]
            if all(ed in edges_set for ed in others):
                cols = [colors[ed] for ed in others]
                if c in cols or len(set(cols)) < 3:
                    valid = False
                    break
        if valid:
            return True, c
    return False, -1

def _simulated_annealing(start_edges, start_colors, cycles, edge_to_cycles, time_limit):
    """Simulated annealing to maximize edges while keeping all 4-cycles rainbow."""
    t_start = time.time()
    deadline = t_start + time_limit
    
    edges_set = set(start_edges)
    colors = dict(start_colors)
    current_bad = _count_bad_cycles(edges_set, colors, cycles, edge_to_cycles)
    current_edges = len(edges_set)
    
    best_edges = current_edges
    best_edges_set = set(edges_set)
    best_colors = dict(colors)
    
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    
    all_edges = list(combinations(range(N), 2))
    
    while time.time() < deadline and T > T_min:
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.4 and len(edges_set) > 0:
            # Remove a random edge
            e = random.choice(tuple(edges_set))
            edges_set.discard(e)
            colors.pop(e, None)
            new_bad = _count_bad_cycles(edges_set, colors, cycles, edge_to_cycles)
            new_edges = len(edges_set)
            delta_energy = new_bad - current_bad
            delta_edges = new_edges - current_edges
            
            # Accept if better, or with probability exp(-delta/T)
            if delta_energy < 0 or delta_edges > 0 or random.random() < math.exp(-delta_energy / T):
                current_bad = new_bad
                current_edges = new_edges
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_edges_set = set(edges_set)
                    best_colors = dict(colors)
            else:
                # Revert
                edges_set.add(e)
                colors[e] = start_colors.get(e, random.randint(0, N-1))
        
        elif move_type < 0.7:
            # Recolor a random existing edge
            if not edges_set:
                continue
            e = random.choice(tuple(edges_set))
            old_c = colors[e]
            new_c = random.randint(0, N-1)
            if new_c == old_c:
                continue
            colors[e] = new_c
            new_bad = _count_bad_cycles(edges_set, colors, cycles, edge_to_cycles)
            delta_energy = new_bad - current_bad
            
            if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                current_bad = new_bad
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_edges_set = set(edges_set)
                    best_colors = dict(colors)
            else:
                colors[e] = old_c
        
        else:
            # Try to add a missing edge
            missing = [e for e in all_edges if e not in edges_set]
            if not missing:
                continue
            e = random.choice(missing)
            u, v = e
            success, c = _try_add_edge(u, v, edges_set, colors, cycles, edge_to_cycles)
            if success:
                edges_set.add(e)
                colors[e] = c
                new_bad = _count_bad_cycles(edges_set, colors, cycles, edge_to_cycles)
                new_edges = len(edges_set)
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_edges_set = set(edges_set)
                    best_colors = dict(colors)
                current_bad = new_bad
                current_edges = new_edges
        
        T *= cooling
    
    return best_edges, best_edges_set, best_colors

def construct():
    t0 = time.time()
    deadline = 150.0 - DEADLINE_MARGIN
    
    cycles, edge_to_cycles = _precompute_cycles()
    all_edges = list(combinations(range(N), 2))
    
    best_edges_count = 0
    best_result = []
    
    # Generate diverse starting colorings
    def make_coloring(scheme, seed=0):
        """Create a coloring based on a scheme."""
        coloring = {}
        for u, v in all_edges:
            if scheme == 'sum':
                coloring[(u, v)] = (u + v) % N
            elif scheme == 'diff':
                coloring[(u, v)] = (u - v) % N
            elif scheme == 'prod':
                coloring[(u, v)] = (u * v) % N
            elif scheme == 'lin1':
                coloring[(u, v)] = (u + 2 * v) % N
            elif scheme == 'lin2':
                coloring[(u, v)] = (2 * u + v) % N
            elif scheme == 'lin3':
                coloring[(u, v)] = (3 * u + v) % N
            elif scheme == 'lin4':
                coloring[(u, v)] = (u + 3 * v) % N
            elif scheme == 'random':
                rng = random.Random(seed)
                coloring[(u, v)] = rng.randint(0, N-1)
            else:
                coloring[(u, v)] = (u + v + seed) % N
        return coloring
    
    # Try multiple schemes with simulated annealing
    schemes = ['sum', 'diff', 'prod', 'lin1', 'lin2', 'lin3', 'lin4']
    random_seeds = [42, 123, 456, 789, 101112]
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Pick a scheme or random seed
        if attempts <= len(schemes):
            scheme = schemes[attempts - 1]
            coloring = make_coloring(scheme)
        else:
            scheme = 'random'
            seed = random_seeds[(attempts - len(schemes) - 1) % len(random_seeds)]
            coloring = make_coloring(scheme, seed)
        
        # Start with all edges and this coloring
        start_edges = set(all_edges)
        start_colors = coloring
        
        # Run simulated annealing with time allocation
        time_left = deadline - time.time()
        if time_left < 5.0:
            break
        
        # Allocate more time to early attempts, less to later ones
        sa_time = min(time_left * 0.3, max(2.0, time_left / max(1, 15 - attempts)))
        
        count, edges_set, colors = _simulated_annealing(
            start_edges, start_colors, cycles, edge_to_cycles, sa_time
        )
        
        if count > best_edges_count:
            best_edges_count = count
            best_result = [[u, v, colors[(u, v)]] for (u, v) in sorted(edges_set)]
        
        # If we've found a good solution, do a final greedy add pass
        if best_edges_count >= 80 and time.time() < deadline - 3.0:
            # Try to add back missing edges greedily
            current_edges = set()
            current_colors = {}
            for u, v, c in best_result:
                e = (min(u, v), max(u, v))
                current_edges.add(e)
                current_colors[e] = c
            
            missing = [e for e in all_edges if e not in current_edges]
            random.shuffle(missing)
            
            added_any = True
            while added_any and time.time() < deadline - 2.0:
                added_any = False
                for e in missing[:]:
                    u, v = e
                    success, c = _try_add_edge(u, v, current_edges, current_colors, cycles, edge_to_cycles)
                    if success:
                        current_edges.add(e)
                        current_colors[e] = c
                        missing.remove(e)
                        added_any = True
            
            if len(current_edges) > best_edges_count:
                best_edges_count = len(current_edges)
                best_result = [[u, v, current_colors[(u, v)]] for (u, v) in sorted(current_edges)]
    
    return best_result if best_result else [[0, v, 0] for v in range(1, N)]

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} edges")
