import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Fast backtracking with unit propagation for 2-colorability of 5-uniform hypergraph.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
        
    m = len(edges)
    vertex_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_to_edges[v].append(i)
            
    colors = [-1] * n
    # edge_counts[i] = [count_0, count_1, count_unknown]
    edge_counts = [[0, 0, 5] for _ in range(m)]
    
    def propagate(queue):
        while queue:
            v, c = queue.pop()
            if colors[v] != -1:
                if colors[v] != c:
                    return False
                else:
                    continue
            
            colors[v] = c
            
            for e_idx in vertex_to_edges[v]:
                edge_counts[e_idx][2] -= 1
                edge_counts[e_idx][c] += 1
                
                c0 = edge_counts[e_idx][0]
                c1 = edge_counts[e_idx][1]
                cU = edge_counts[e_idx][2]
                
                if c0 == 5 or c1 == 5:
                    return False
                
                if cU == 1:
                    e = edges[e_idx]
                    unknown_v = -1
                    for u in e:
                        if colors[u] == -1:
                            unknown_v = u
                            break
                    if unknown_v == -1:
                        continue
                    
                    if c0 == 4:
                        if colors[unknown_v] == 0:
                            return False
                        if colors[unknown_v] == -1:
                            queue.append((unknown_v, 1))
                    elif c1 == 4:
                        if colors[unknown_v] == 1:
                            return False
                        if colors[unknown_v] == -1:
                            queue.append((unknown_v, 0))
        return True

    def backtrack():
        # Variable ordering: Most Constrained First
        best_v = -1
        best_score = -1
        
        for v in range(n):
            if colors[v] != -1:
                continue
            score = 0
            for e_idx in vertex_to_edges[v]:
                c0 = edge_counts[e_idx][0]
                c1 = edge_counts[e_idx][1]
                cU = edge_counts[e_idx][2]
                if cU == 1:
                    if c0 == 4 or c1 == 4:
                        score += 10
                    else:
                        score += 1
                elif cU == 2:
                    score += 0.5
            if score > best_score:
                best_score = score
                best_v = v
                
        if best_v == -1:
            return True
            
        v = best_v
        colors_choice = [0, 1]
        random.shuffle(colors_choice)
        
        for c in colors_choice:
            old_colors = colors.copy()
            old_edge_counts = [list(ec) for ec in edge_counts]
            
            if propagate([(v, c)]):
                if backtrack():
                    return True
            
            colors[:] = old_colors
            edge_counts[:] = old_edge_counts
            
        return False

    if propagate([]):
        return backtrack()
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_result = None
    best_edge_count = 200
    
    # Try n=11 first
    for n in [11, 12, 10]:
        if time.time() > deadline - 20.0:
            break
            
        all_subsets = []
        max_size = n // 2
        for size in range(0, max_size + 1):
            for s in combinations(range(n), size):
                all_subsets.append(s)
                
        if not all_subsets:
            continue
            
        num_subsets = len(all_subsets)
        
        subsets_as_masks = []
        for s in all_subsets:
            mask = 0
            for v in s:
                mask |= (1 << v)
            subsets_as_masks.append(mask)
            
        all_edges = list(combinations(range(n), 5))
        num_edges = len(all_edges)
        
        edge_masks = []
        for e in all_edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
            
        # Precompute coverage: edge_covers[e_idx] = list of subset indices covered by e_idx
        edge_covers = [[] for _ in range(num_edges)]
        
        for s_idx, s_mask in enumerate(subsets_as_masks):
            for e_idx, e_mask in enumerate(edge_masks):
                if (e_mask & ~s_mask) == 0 or (e_mask & s_mask) == 0:
                    edge_covers[e_idx].append(s_idx)
                    
        # Greedy Set Cover
        uncovered = set(range(num_subsets))
        chosen_edges = set() # Use set for fast membership
        
        while uncovered and time.time() < deadline - 20.0:
            best_e = -1
            best_count = -1
            
            for e_idx in range(num_edges):
                if e_idx in chosen_edges:
                    continue
                count = 0
                for s_idx in edge_covers[e_idx]:
                    if s_idx in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_e = e_idx
                    
            if best_e == -1 or best_count == 0:
                break
                
            chosen_edges.add(best_e)
            for s_idx in edge_covers[best_e]:
                uncovered.discard(s_idx)
                
        if not uncovered:
            current_edges = list(chosen_edges)
            current_edge_list = [list(all_edges[e]) for e in current_edges]
            
            # Verify
            if not is_2_colorable(n, current_edge_list):
                if len(current_edges) < best_edge_count:
                    best_edge_count = len(current_edges)
                    best_result = {'n': n, 'edges': current_edge_list}
                    
                # Local Search: Randomized k-Swap / Removal
                # We represent the cover by the set of edge indices.
                # We need to check if a set of edges is a valid cover.
                # To speed up, we maintain a count of how many times each subset is covered.
                
                cover_counts = [0] * num_subsets
                for e_idx in chosen_edges:
                    for s_idx in edge_covers[e_idx]:
                        cover_counts[s_idx] += 1
                        
                improved = True
                while improved and time.time() < deadline - 10.0:
                    improved = False
                    
                    # Try to remove a random edge
                    if chosen_edges:
                        e_out = random.choice(list(chosen_edges))
                        # Check if e_out is redundant
                        # i.e., all s in edge_covers[e_out] have cover_counts[s] > 1
                        is_redundant = True
                        for s_idx in edge_covers[e_out]:
                            if cover_counts[s_idx] == 1:
                                is_redundant = False
                                break
                                
                        if is_redundant:
                            # Remove it
                            chosen_edges.remove(e_out)
                            for s_idx in edge_covers[e_out]:
                                cover_counts[s_idx] -= 1
                            improved = True
                            if len(chosen_edges) < best_edge_count:
                                best_edge_count = len(chosen_edges)
                                best_result = {'n': n, 'edges': [list(all_edges[e]) for e in chosen_edges]}
                            continue # Restart loop with new state
                        
                    # Try to swap
                    # Pick e_out randomly
                    if chosen_edges:
                        e_out = random.choice(list(chosen_edges))
                        # Find subsets that become uncovered if e_out is removed
                        critical_subsets = []
                        for s_idx in edge_covers[e_out]:
                            if cover_counts[s_idx] == 1:
                                critical_subsets.append(s_idx)
                            
                        if critical_subsets:
                            # We need an e_in that covers ALL critical_subsets
                            # And e_in is not in chosen_edges
                            # To speed up, iterate through edges not in chosen_edges
                            # and check if it covers critical_subsets
                            found_swap = False
                            # Randomize order of candidate edges?
                            candidates = [e for e in range(num_edges) if e not in chosen_edges]
                            random.shuffle(candidates)
                            
                            for e_in in candidates:
                                # Check if e_in covers all critical_subsets
                                # We can precompute a mask or just check membership
                                # edge_covers[e_in] is a list.
                                # Check if set(critical_subsets) is subset of set(edge_covers[e_in])
                                # This is O(|critical|).
                                cov_in = set(edge_covers[e_in])
                                if all(s in cov_in for s in critical_subsets):
                                    # Perform swap
                                    # Remove e_out
                                    chosen_edges.remove(e_out)
                                    for s_idx in edge_covers[e_out]:
                                        cover_counts[s_idx] -= 1
                                    
                                    # Add e_in
                                    chosen_edges.add(e_in)
                                    for s_idx in edge_covers[e_in]:
                                        cover_counts[s_idx] += 1
                                        
                                    found_swap = True
                                    improved = True # Count as improvement to allow more steps?
                                    # Actually, swap keeps size same. We only update best if size decreases.
                                    # But it might enable future removals.
                                    break
                            
                            if found_swap:
                                continue

    if best_result is not None:
        return best_result
        
    # Fallback
    n_fb = 9
    all_edges_fb = list(combinations(range(n_fb), 5))
    return {'n': n_fb, 'edges': [list(e) for e in all_edges_fb]}
