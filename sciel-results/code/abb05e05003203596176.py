import time
import random
from itertools import combinations
import sys
import numpy as np

def is_2_colorable_fast(n, edges):
    """
    Check if a 5-uniform hypergraph is 2-colorable.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    if n <= 14:
        # Bitset union approach
        full_mask = (1 << (1 << n)) - 1
        covered = 0
        
        for e in edges:
            e_mask = 0
            for v in e:
                e_mask |= (1 << v)
                
            free_mask = ((1 << n) - 1) ^ e_mask
            
            sub = free_mask
            while True:
                m1 = sub
                m2 = sub | e_mask
                
                covered |= (1 << m1)
                covered |= (1 << m2)
                
                if sub == 0:
                    break
                sub = (sub - 1) & free_mask
                
            if covered == full_mask:
                return False
                
        return True
    else:
        # Backtracking for n > 14
        # (Keep the existing backtracking code from the original program)
        color = [-1] * n
        c0 = [0] * len(edges)
        c1 = [0] * len(edges)
        
        vertex_edges = [[] for _ in range(n)]
        edge_masks = []
        for e in edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
            for v in e:
                vertex_edges[v].append(len(edge_masks)-1)
                
        def propagate():
            changed = True
            while changed:
                changed = False
                for i in range(len(edges)):
                    if c0[i] == 5 or c1[i] == 5:
                        return False
                    if c0[i] == 4 and c1[i] == 0:
                        for v in range(n):
                            if color[v] == -1 and (edge_masks[i] & (1 << v)):
                                color[v] = 1
                                c1[i] += 1
                                for other_e in vertex_edges[v]:
                                    c1[other_e] += 1
                                changed = True
                                if c1[i] == 5:
                                    return False
                                break
                    elif c1[i] == 4 and c0[i] == 0:
                        for v in range(n):
                            if color[v] == -1 and (edge_masks[i] & (1 << v)):
                                color[v] = 0
                                c0[i] += 1
                                for other_e in vertex_edges[v]:
                                    c0[other_e] += 1
                                changed = True
                                if c0[i] == 5:
                                    return False
                                break
            return True
        
        def backtrack():
            best_v = -1
            best_edge = -1
            best_unassigned = 6
            best_constraint = 0
            
            for i in range(len(edges)):
                if c0[i] == 5 or c1[i] == 5:
                    return False
                unassigned = 5 - c0[i] - c1[i]
                constraint = max(c0[i], c1[i])
                if unassigned < best_unassigned or (unassigned == best_unassigned and constraint > best_constraint):
                    best_unassigned = unassigned
                    best_constraint = constraint
                    best_edge = i
                    if unassigned == 0:
                        return False
            
            if best_edge == -1:
                return True
            
            candidates = []
            for v in range(n):
                if color[v] == -1 and (edge_masks[best_edge] & (1 << v)):
                    candidates.append(v)
            
            if not candidates:
                return True
            
            for v in candidates:
                saved_c0 = c0[:]
                saved_c1 = c1[:]
                color[v] = 0
                ok = True
                for e_idx in vertex_edges[v]:
                    c0[e_idx] += 1
                    if c0[e_idx] == 5:
                        ok = False
                        break
                if ok and propagate():
                    if backtrack():
                        return True
                color[v] = -1
                c0[:] = saved_c0
                c1[:] = saved_c1
                
                saved_c0 = c0[:]
                saved_c1 = c1[:]
                color[v] = 1
                ok = True
                for e_idx in vertex_edges[v]:
                    c1[e_idx] += 1
                    if c1[e_idx] == 5:
                        ok = False
                        break
                if ok and propagate():
                    if backtrack():
                        return True
                color[v] = -1
                c0[:] = saved_c0
                c1[:] = saved_c1
            
            return False
        
        return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_edges = None
    best_n = 9
    best_m = 10**9
    
    for n in [11, 12, 10, 9, 13, 14]:
        if time.time() > deadline - 10:
            break
            
        all_possible = list(combinations(range(n), 5))
        
        if is_2_colorable_fast(n, all_possible):
            continue
            
        # Phase 1: Greedy Removal
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
                
            curr = [list(e) for e in all_possible]
            edge_indices = list(range(len(curr)))
            random.shuffle(edge_indices)
            
            removed_any = True
            while removed_any and time.time() < deadline - 20:
                removed_any = False
                for i in edge_indices:
                    if time.time() > deadline - 20:
                        break
                    if i >= len(curr):
                        continue
                    
                    new_edges = curr[:i] + curr[i+1:]
                    if not is_2_colorable_fast(n, new_edges):
                        curr = new_edges
                        edge_indices = list(range(len(curr)))
                        random.shuffle(edge_indices)
                        removed_any = True
                        break
                        
            if len(curr) < best_m:
                best_m = len(curr)
                best_edges = curr[:]
                best_n = n
                
        # Phase 2: Local Search (SA)
        if best_edges and time.time() < deadline - 5:
            curr_edges = best_edges[:]
            curr_n = best_n
            all_edges_set = set(tuple(e) for e in all_possible)
            curr_set = set(tuple(e) for e in curr_edges)
            
            T = 10.0
            cooling = 0.95
            steps = 0
            max_steps = 10000
            
            while steps < max_steps and time.time() < deadline - 2:
                steps += 1
                move_type = random.random()
                new_set = set(curr_set)
                
                if move_type < 0.5:
                    if not curr_set: continue
                    e_to_remove = random.choice(list(curr_set))
                    new_set.remove(e_to_remove)
                elif move_type < 0.8:
                    available = list(all_edges_set - curr_set)
                    if not available: continue
                    e_to_add = random.choice(available)
                    new_set.add(e_to_add)
                else:
                    if not curr_set: continue
                    e_to_remove = random.choice(list(curr_set))
                    available = list(all_edges_set - curr_set)
                    if not available: continue
                    e_to_add = random.choice(available)
                    new_set.remove(e_to_remove)
                    new_set.add(e_to_add)
                    
                new_edges_list = [list(e) for e in new_set]
                is_valid_new = not is_2_colorable_fast(curr_n, new_edges_list)
                
                if is_valid_new:
                    delta = len(new_set) - len(curr_set)
                    if delta < 0:
                        curr_set = new_set
                        if len(curr_set) < best_m:
                            best_m = len(curr_set)
                            best_edges = [list(e) for e in curr_set]
                            best_n = curr_n
                    else:
                        if random.random() < np.exp(-delta / T):
                            curr_set = new_set
                else:
                    pass
                    
                T *= cooling
                
        if best_m <= 51:
            break

    if best_edges is None:
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_n = 9
        
    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
