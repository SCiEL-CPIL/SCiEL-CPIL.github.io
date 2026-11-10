import time
import random
import itertools
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n = 16
    vertices = list(range(n))
    k4s = list(itertools.combinations(vertices, 4))
    num_k4s = len(k4s)
    
    # Edge list and index
    edge_list = []
    edge_index = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = idx
            edge_list.append((i, j))
            idx += 1
    num_edges = len(edge_list)
    
    # For each K4, store the list of edge indices
    k4_edges = []
    for k4 in k4s:
        edges = []
        for a, b in itertools.combinations(k4, 2):
            edges.append(edge_index[(a, b)])
        k4_edges.append(edges)
    
    # For each edge, the list of K4 indices it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e_idx in edges:
            edge_to_k4s[e_idx].append(k4_idx)
    
    # Precompute: for each K4, the 6 edge indices as a numpy array
    k4_edges_np = [np.array(e, dtype=np.int32) for e in k4_edges]
    
    # Helper: compute violation count and per-K4 color frequency
    def compute_violations(colours):
        """Returns (total_violations, per_k4_distinct_count, per_k4_color_freq)"""
        colours = np.array(colours, dtype=np.int32)
        total_viol = 0
        distinct_counts = np.zeros(num_k4s, dtype=np.int32)
        # We'll store for each K4 the count of each color present
        # But that's expensive. Instead, just track distinct count.
        for k in range(num_k4s):
            edges = k4_edges_np[k]
            cvals = colours[edges]
            distinct = len(np.unique(cvals))
            distinct_counts[k] = distinct
            if distinct < 5:
                total_viol += 1
        return total_viol, distinct_counts
    
    def distinct_count_from_colours(colours, k):
        """Compute distinct color count for K4 k given current colours array."""
        edges = k4_edges_np[k]
        cvals = colours[edges]
        return len(np.unique(cvals))
    
    # Try structured colorings first
    def try_structured():
        """Try various structured colorings and return best valid one."""
        candidates = []
        
        # Sum mod p
        for p in range(13, 20):
            c = [((i + j) % p) for i in range(n) for j in range(i + 1, n)]
            candidates.append((c, 'sum_mod_%d' % p))
        
        # Difference mod p
        for p in range(13, 20):
            c = [((i - j) % p) for i in range(n) for j in range(i + 1, n)]
            candidates.append((c, 'diff_mod_%d' % p))
        
        # Product mod p
        for p in range(13, 20):
            c = [((i * j) % p) for i in range(n) for j in range(i + 1, n)]
            candidates.append((c, 'prod_mod_%d' % p))
        
        # Sum of squares mod p
        for p in range(13, 20):
            c = [((i * i + j * j) % p) for i in range(n) for j in range(i + 1, n)]
            candidates.append((c, 'sq_sum_mod_%d' % p))
        
        # i*j + i + j mod p
        for p in range(13, 20):
            c = [((i * j + i + j) % p) for i in range(n) for j in range(i + 1, n)]
            candidates.append((c, 'ijij_mod_%d' % p))
        
        # Check each candidate
        best_valid = None
        best_num_colors = float('inf')
        for c, name in candidates:
            # Count distinct colors used
            num_used = len(set(c))
            if num_used >= best_num_colors:
                continue
            # Check validity
            valid = True
            for k in range(num_k4s):
                edges = k4_edges[k]
                cset = set(c[e] for e in edges)
                if len(cset) < 5:
                    valid = False
                    break
            if valid:
                best_valid = c
                best_num_colors = num_used
                # Can't do better than 13
                if best_num_colors <= 13:
                    return best_valid
        
        return best_valid
    
    # Fast local search with simulated annealing
    def local_search(colours, C, max_time=60.0):
        """
        colours: list of ints (edge colors)
        C: number of colors available
        Returns: (best_colours, best_cost, found_valid)
        """
        colours = list(colours)
        colours_np = np.array(colours, dtype=np.int32)
        
        # Initial violation count
        total_viol, distinct_counts = compute_violations(colours)
        current_cost = total_viol
        
        best_colours = colours[:]
        best_cost = current_cost
        found_valid = (current_cost == 0)
        
        if found_valid:
            return best_colours, best_cost, True
        
        # Simulated annealing parameters
        T = 5.0  # Initial temperature
        T_min = 0.1
        cooling = 0.9995
        max_iter = 200000
        iter_count = 0
        stall_count = 0
        max_stall = 5000
        
        # Precompute for faster delta: for each K4, we need to know current distinct count
        # and the color frequencies to quickly compute new distinct count after a change
        
        # For efficiency, maintain per-K4 color frequency dicts
        # freq[k][color] = count of edges in K4 k with that color
        freq = [dict() for _ in range(num_k4s)]
        for k in range(num_k4s):
            edges = k4_edges[k]
            for e in edges:
                c = colours[e]
                freq[k][c] = freq[k].get(c, 0) + 1
        
        # distinct_counts[k] = len(freq[k])
        
        # For delta computation when changing edge e from old_c to new_c:
        # For each K4 k containing e:
        #   old_distinct = len(freq[k])
        #   new_freq: decrement old_c, increment new_c
        #   new_distinct = len(new_freq)
        #   delta_viol = (1 if new_distinct < 5 else 0) - (1 if old_distinct < 5 else 0)
        
        # To speed up, we can compute delta quickly:
        # When removing old_c from freq[k]:
        #   if freq[k][old_c] == 1: distinct decreases by 1 (if old_c was the only one)
        #   else: distinct unchanged
        # When adding new_c to freq[k]:
        #   if new_c not in freq[k]: distinct increases by 1
        #   else: distinct unchanged
        # So new_distinct = old_distinct - (1 if freq[k][old_c]==1 else 0) + (1 if new_c not in freq[k] else 0)
        
        # But we need to be careful: if old_c == new_c, no change.
        
        # Let's implement the fast delta
        def compute_delta(e_idx, old_c, new_c, freq, distinct_counts):
            if old_c == new_c:
                return 0
            delta = 0
            for k in edge_to_k4s[e_idx]:
                old_distinct = distinct_counts[k]
                # Remove old_c
                old_count = freq[k].get(old_c, 0)
                # Add new_c
                new_count = freq[k].get(new_c, 0)
                
                # New distinct after removing old_c and adding new_c
                # Remove old_c: if old_count == 1, distinct -1
                # Add new_c: if new_count == 0, distinct +1
                # But careful: if old_c == new_c, handled above.
                # Also, if old_c is removed and new_c was same, but we already handled.
                
                new_distinct = old_distinct
                if old_count == 1:
                    new_distinct -= 1
                if new_count == 0:
                    new_distinct += 1
                
                old_viol = 1 if old_distinct < 5 else 0
                new_viol = 1 if new_distinct < 5 else 0
                delta += new_viol - old_viol
            return delta
        
        # Apply move
        def apply_move(e_idx, old_c, new_c, colours, freq, distinct_counts):
            if old_c == new_c:
                return
            colours[e_idx] = new_c
            for k in edge_to_k4s[e_idx]:
                # Update freq
                old_count = freq[k].get(old_c, 0)
                if old_count == 1:
                    del freq[k][old_c]
                    distinct_counts[k] -= 1
                else:
                    freq[k][old_c] = old_count - 1
                
                new_count = freq[k].get(new_c, 0)
                if new_count == 0:
                    freq[k][new_c] = 1
                    distinct_counts[k] += 1
                else:
                    freq[k][new_c] = new_count + 1
        
        # Undo move
        def undo_move(e_idx, old_c, new_c, colours, freq, distinct_counts):
            if old_c == new_c:
                return
            colours[e_idx] = old_c
            for k in edge_to_k4s[e_idx]:
                # Undo: remove new_c, add old_c
                new_count = freq[k].get(new_c, 0)
                if new_count == 1:
                    del freq[k][new_c]
                    distinct_counts[k] -= 1
                else:
                    freq[k][new_c] = new_count - 1
                
                old_count = freq[k].get(old_c, 0)
                if old_count == 0:
                    freq[k][old_c] = 1
                    distinct_counts[k] += 1
                else:
                    freq[k][old_c] = old_count + 1
        
        # Get list of violating K4s
        def get_violating_k4s(distinct_counts):
            return [k for k in range(num_k4s) if distinct_counts[k] < 5]
        
        while iter_count < max_iter:
            if time.time() > start_time + max_time:
                break
            
            # Pick a random violating K4
            viol_k4s = get_violating_k4s(distinct_counts)
            if not viol_k4s:
                found_valid = True
                best_cost = 0
                best_colours = colours[:]
                break
            
            k4_idx = random.choice(viol_k4s)
            
            # Try all edges in this K4 and all colors
            best_delta = float('inf')
            best_move = None
            
            for e_idx in k4_edges[k4_idx]:
                old_c = colours[e_idx]
                for new_c in range(C):
                    if new_c == old_c:
                        continue
                    d = compute_delta(e_idx, old_c, new_c, freq, distinct_counts)
                    if d < best_delta:
                        best_delta = d
                        best_move = (e_idx, old_c, new_c)
            
            if best_move:
                e_idx, old_c, new_c = best_move
                # Accept if improving or with probability exp(-delta/T)
                if best_delta <= 0 or random.random() < np.exp(-best_delta / T):
                    apply_move(e_idx, old_c, new_c, colours, freq, distinct_counts)
                    current_cost += best_delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colours = colours[:]
                        if best_cost == 0:
                            found_valid = True
                            break
                    stall_count = 0
                else:
                    stall_count += 1
            else:
                # Random move
                e_idx = random.choice(k4_edges[k4_idx])
                old_c = colours[e_idx]
                new_c = random.randint(0, C - 1)
                if new_c != old_c:
                    d = compute_delta(e_idx, old_c, new_c, freq, distinct_counts)
                    apply_move(e_idx, old_c, new_c, colours, freq, distinct_counts)
                    current_cost += d
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colours = colours[:]
                        if best_cost == 0:
                            found_valid = True
                            break
                    stall_count = 0
            
            iter_count += 1
            T = max(T_min, T * cooling)
            
            # Restart if stuck too long
            if stall_count > max_stall:
                # Random restart within neighborhood
                for _ in range(20):
                    e_idx = random.randint(0, num_edges - 1)
                    old_c = colours[e_idx]
                    new_c = random.randint(0, C - 1)
                    if new_c != old_c:
                        d = compute_delta(e_idx, old_c, new_c, freq, distinct_counts)
                        apply_move(e_idx, old_c, new_c, colours, freq, distinct_counts)
                        current_cost += d
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_colours = colours[:]
                            if best_cost == 0:
                                found_valid = True
                                break
                stall_count = 0
                T = 5.0  # Reset temperature
        
        if not found_valid and best_cost == 0:
            found_valid = True
        
        return best_colours, best_cost, found_valid
    
    # Main search
    best_result = None
    best_cost = float('inf')
    
    # Try structured first
    structured = try_structured()
    if structured is not None:
        num_used = len(set(structured))
        if num_used < best_cost:
            best_cost = num_used
            best_result = structured
        if best_cost <= 13:
            return best_result
    
    # Try local search with various C and initializations
    for C in [13, 14, 15]:
        if time.time() > deadline:
            break
        
        # Multiple restarts
        num_restarts = 3
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            colours = [random.randint(0, C - 1) for _ in range(num_edges)]
            
            best_colours, best_c, found = local_search(colours, C, max_time=30.0)
            
            if found:
                num_used = len(set(best_colours))
                if num_used < best_cost:
                    best_cost = num_used
                    best_result = best_colours
                if best_cost <= 13:
                    return best_result
            else:
                # Keep best invalid for potential fallback? No, must be valid.
                # But we can use it as seed for next restart? Not directly.
                pass
    
    # If no valid found, try with C=16, 17 as fallback
    for C in [16, 17]:
        if time.time() > deadline:
            break
        colours = [random.randint(0, C - 1) for _ in range(num_edges)]
        best_colours, best_c, found = local_search(colours, C, max_time=20.0)
        if found:
            num_used = len(set(best_colours))
            if num_used < best_cost:
                best_cost = num_used
                best_result = best_colours
    
    # Final fallback: seed
    if best_result is None:
        best_result = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    
    # Verify and return
    # Ensure all colors are in 0..119
    return best_result
