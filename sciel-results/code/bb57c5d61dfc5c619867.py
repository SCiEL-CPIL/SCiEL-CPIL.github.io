import time
import random
from itertools import combinations

def check_not_2colorable(n, edges):
    """Returns True if hypergraph is NOT 2-colorable (no proper 2-coloring exists)."""
    if not edges:
        return False
    
    m = len(edges)
    
    # For small n, use bitmask enumeration with early termination
    if n <= 16:
        edge_masks = []
        for e in edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        # Check all 2^n colorings
        for mask in range(1 << n):
            valid = True
            for em in edge_masks:
                if (mask & em) == 0 or (mask & em) == em:
                    valid = False
                    break
            if valid:
                return False  # Found a valid 2-coloring
        return True  # No valid 2-coloring found
    
    # For larger n, use backtracking with unit propagation
    # Build adjacency: for each vertex, list of edges containing it
    vert_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vert_edges[v].append(i)
    
    # Degree ordering
    degrees = [len(vert_edges[v]) for v in range(n)]
    
    colors = [-1] * n
    edge_state = [[0, 0] for _ in range(m)]  # [count_color0, count_color1]
    
    def propagate():
        """Unit propagation. Returns False on conflict."""
        changed = True
        while changed:
            changed = False
            for i in range(m):
                c0, c1 = edge_state[i]
                if c0 >= 5 or c1 >= 5:
                    return False
                uncolored = []
                for v in edges[i]:
                    if colors[v] == -1:
                        uncolored.append(v)
                if len(uncolored) == 1:
                    if c0 == 4:
                        colors[uncolored[0]] = 1
                        edge_state[i][1] += 1
                        changed = True
                    elif c1 == 4:
                        colors[uncolored[0]] = 0
                        edge_state[i][0] += 1
                        changed = True
        return True
    
    def backtrack():
        # Find uncolored vertex with highest degree among uncolored
        best_v = -1
        best_deg = -1
        for v in range(n):
            if colors[v] == -1 and degrees[v] > best_deg:
                best_deg = degrees[v]
                best_v = v
        
        if best_v == -1:
            # All colored - check if valid
            for i in range(m):
                if edge_state[i][0] == 5 or edge_state[i][1] == 5:
                    return False
            return True  # Valid coloring found
        
        for c in [0, 1]:
            colors[best_v] = c
            # Update edge states
            saved = []
            for i in vert_edges[best_v]:
                saved.append(edge_state[i][:])
                if c == 0:
                    edge_state[i][0] += 1
                else:
                    edge_state[i][1] += 1
            
            if propagate():
                if backtrack():
                    return True
            
            # Undo
            for i, s in zip(vert_edges[best_v], saved):
                edge_state[i] = s
            colors[best_v] = -1
        
        return False
    
    # Initial propagation
    if not propagate():
        return True  # Conflict means not 2-colorable
    
    # If backtracking finds a valid coloring, it IS 2-colorable
    if backtrack():
        return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_edges = None
    best_n = 0
    best_m = float('inf')
    
    rng = random.Random(42)
    
    # Strategy: Try n from 8 to 14, starting from complete hypergraph and removing edges
    # For each n, run multiple restarts with different random seeds
    
    for n in [8, 9, 10, 11, 12, 13, 14]:
        if time.time() > deadline - 5:
            break
        
        # Complete hypergraph on n vertices
        all_edges = list(combinations(range(n), 5))
        cur_edges = [list(e) for e in all_edges]
        cur_m = len(cur_edges)
        
        # Verify it's not 2-colorable (it should be for n >= 8)
        if not check_not_2colorable(n, cur_edges):
            continue
        
        if cur_m < best_m:
            best_m = cur_m
            best_edges = cur_edges[:]
            best_n = n
        
        # Local search: try to remove edges
        all_possible = set(all_edges)
        current_set = set(tuple(e) for e in cur_edges)
        
        # Multiple restarts for this n
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            
            # Reset to complete
            cur_edges = [list(e) for e in all_edges]
            cur_m = len(cur_edges)
            current_set = set(all_edges)
            
            # Temperature-based search
            temp = 1.0
            cooling = 0.998
            min_temp = 0.05
            
            while time.time() < deadline - 5 and temp > min_temp:
                if cur_m == 0:
                    break
                
                # Try removing a random edge
                idx = rng.randint(0, cur_m - 1)
                e_remove = cur_edges[idx]
                
                # Temporarily remove
                new_edges = cur_edges[:idx] + cur_edges[idx+1:]
                
                if check_not_2colorable(n, new_edges):
                    # Accept removal
                    cur_edges = new_edges
                    cur_m -= 1
                    current_set.discard(tuple(e_remove))
                    
                    if cur_m < best_m:
                        best_m = cur_m
                        best_edges = cur_edges[:]
                        best_n = n
                else:
                    # Try a swap: remove one, add another
                    available = list(all_possible - current_set)
                    if available:
                        e_add = rng.choice(available)
                        new_edges_swap = cur_edges[:idx] + cur_edges[idx+1:] + [list(e_add)]
                        
                        if check_not_2colorable(n, new_edges_swap):
                            # Accept swap (neutral move)
                            cur_edges = new_edges_swap
                            current_set.discard(tuple(e_remove))
                            current_set.add(e_add)
                    
                    # With some probability, accept a "worse" move (add an edge)
                    if rng.random() < temp * 0.1:
                        available = list(all_possible - current_set)
                        if available:
                            e_add = rng.choice(available)
                            new_edges_add = cur_edges + [list(e_add)]
                            if check_not_2colorable(n, new_edges_add):
                                cur_edges = new_edges_add
                                cur_m += 1
                                current_set.add(e_add)
                
                temp *= cooling
        
        # After local search, try to add edges to enable more removals
        # (sometimes adding an edge allows removing 2+ others)
        if time.time() < deadline - 5 and best_edges is not None and best_n == n:
            # Try to improve best solution
            trial_edges = best_edges[:]
            trial_m = len(trial_edges)
            trial_set = set(tuple(e) for e in trial_edges)
            
            for _ in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Try removing a random edge from trial
                if trial_m == 0:
                    break
                idx = rng.randint(0, trial_m - 1)
                e_rem = trial_edges[idx]
                new_trial = trial_edges[:idx] + trial_edges[idx+1:]
                
                if check_not_2colorable(n, new_trial):
                    trial_edges = new_trial
                    trial_m -= 1
                    trial_set.discard(tuple(e_rem))
                    if trial_m < best_m:
                        best_m = trial_m
                        best_edges = trial_edges[:]
                        best_n = n
                else:
                    # Try swap
                    avail = list(all_possible - trial_set)
                    if avail:
                        e_add = rng.choice(avail)
                        new_trial = trial_edges[:idx] + trial_edges[idx+1:] + [list(e_add)]
                        if check_not_2colorable(n, new_trial):
                            trial_edges = new_trial
                            trial_set.discard(tuple(e_rem))
                            trial_set.add(e_add)
    
    # Final verification
    if best_edges is not None:
        if check_not_2colorable(best_n, best_edges):
            return {'n': best_n, 'edges': best_edges}
    
    # Fallback: complete hypergraph on 9 vertices
    fallback = [list(e) for e in combinations(range(9), 5)]
    return {'n': 9, 'edges': fallback}


if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
    print("Verifying...")
    assert check_not_2colorable(res['n'], res['edges']), "Result is 2-colorable!"
    print("Valid.")
