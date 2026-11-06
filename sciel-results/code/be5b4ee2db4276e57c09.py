import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    def solve_coloring(n, edges):
        """
        Returns a list of colors (0/1) for vertices 0..n-1 if 2-colorable.
        Returns None if not 2-colorable.
        """
        if time.time() > deadline:
            return None  # Timeout, assume not colorable? Or raise? 
                          # For construction, if we assume not colorable, we might return invalid.
                          # Better to return a dummy coloring or handle carefully.
                          # Let's return None and let the caller handle.
        
        m = len(edges)
        if m == 0:
            return [0] * n
        
        # Precompute edge masks
        edge_masks = []
        for e in edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        # Brute force for small n
        if n <= 14:
            for mask in range(1 << (n - 1)):
                full_mask = mask
                is_colorable = True
                for em in edge_masks:
                    if (full_mask & em) == 0 or (full_mask & em) == em:
                        is_colorable = False
                        break
                if is_colorable:
                    # Reconstruct coloring
                    coloring = [0] * n
                    for v in range(1, n):
                        if (full_mask >> v) & 1:
                            coloring[v] = 1
                    return coloring
            return None

        # Backtracking for larger n
        assignment = [-1] * n
        vert_edges = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                vert_edges[v].append(i)
                
        def propagate(assignment, edge_masks, n, m):
            changed = True
            while changed:
                changed = False
                for i in range(m):
                    em = edge_masks[i]
                    zeros = 0
                    ones = 0
                    unassigned = 0
                    temp = em
                    while temp:
                        lsb = temp & -temp
                        v = lsb.bit_length() - 1
                        temp ^= lsb
                        a = assignment[v]
                        if a == 0: zeros += 1
                        elif a == 1: ones += 1
                        else: unassigned += 1
                    
                    if zeros > 0 and ones > 0: continue
                    if zeros == 0 and ones == 0: continue
                    
                    if zeros > 0 and ones == 0:
                        if unassigned == 0: return False
                        if unassigned == 1:
                            temp = em
                            while temp:
                                lsb = temp & -temp
                                v = lsb.bit_length() - 1
                                temp ^= lsb
                                if assignment[v] == -1:
                                    assignment[v] = 1
                                    changed = True
                                    break
                    elif ones > 0 and zeros == 0:
                        if unassigned == 0: return False
                        if unassigned == 1:
                            temp = em
                            while temp:
                                lsb = temp & -temp
                                v = lsb.bit_length() - 1
                                temp ^= lsb
                                if assignment[v] == -1:
                                    assignment[v] = 0
                                    changed = True
                                    break
            return True

        def backtrack(assignment, edge_masks, n, m):
            if time.time() > deadline:
                return None
            
            unassigned = [v for v in range(n) if assignment[v] == -1]
            if not unassigned:
                # Verify
                full_mask = 0
                for v in range(n):
                    if assignment[v] == 1:
                        full_mask |= (1 << v)
                for em in edge_masks:
                    if (full_mask & em) == 0 or (full_mask & em) == em:
                        return None
                return assignment.copy()
            
            best_v = max(unassigned, key=lambda v: len(vert_edges[v]))
            
            for color in [0, 1]:
                new_assignment = assignment.copy()
                new_assignment[best_v] = color
                if propagate(new_assignment, edge_masks, n, m):
                    res = backtrack(new_assignment, edge_masks, n, m)
                    if res is not None:
                        return res
            return None

        if propagate(assignment, edge_masks, n, m):
            return backtrack(assignment, edge_masks, n, m)
        return None

    best_solution = None
    best_m = 200
    
    # Iterate M from 29 upwards
    for m_try in range(29, 51):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Try different N
        for n_try in range(9, 16):
            if time.time() > deadline - 5:
                break
            
            all_edges = list(itertools.combinations(range(n_try), 5))
            if len(all_edges) < m_try:
                continue
                
            # Greedy Break with Coloring Guidance
            for restart in range(10):
                if time.time() > deadline - 5:
                    break
                
                random.shuffle(all_edges)
                start_size = max(5, m_try // 2)
                current_edges = all_edges[:start_size]
                current_set = set(current_edges)
                
                while len(current_edges) < m_try:
                    coloring = solve_coloring(n_try, current_edges)
                    
                    if coloring is None:
                        # Not 2-colorable
                        if len(current_edges) < best_m:
                            best_m = len(current_edges)
                            best_solution = (n_try, [list(e) for e in current_edges])
                            found = True
                        break
                    
                    # Find edges that are monochromatic in this coloring
                    mono_edges = []
                    for e in all_edges:
                        if e in current_set:
                            continue
                        colors = [coloring[v] for v in e]
                        if len(set(colors)) == 1:
                            mono_edges.append(e)
                            
                    if not mono_edges:
                        # No mono edges available, pick random
                        available = [e for e in all_edges if e not in current_set]
                        if not available: break
                        new_edge = random.choice(available)
                    else:
                        new_edge = random.choice(mono_edges)
                        
                    current_edges.append(new_edge)
                    current_set.add(new_edge)
                    
                if found:
                    break
            
            if found:
                break
        
        if found:
            break

    if best_solution:
        n, edges = best_solution
        return {'n': n, 'edges': edges}
        
    # Fallback
    n = 10
    edges = [list(e) for e in itertools.combinations(range(n), 5)][:51]
    return {'n': n, 'edges': edges}
