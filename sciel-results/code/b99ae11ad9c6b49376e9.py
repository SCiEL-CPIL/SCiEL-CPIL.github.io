import random
import time
import itertools

def construct():
    N = 16
    # Pairs (i, j) with i < j
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # All K4s
    k4s = list(itertools.combinations(range(N), 4))
    # Map each edge to the indices of K4s it belongs to
    edge_to_k4s = {e: [] for e in edges}
    for idx, k4 in enumerate(k4s):
        # 6 edges in a K4
        for i in range(4):
            for j in range(i + 1, 4):
                e = (k4[i], k4[j])
                edge_to_k4s[e].append(idx)
                
    def get_distinct_colors(k4, colors):
        # k4 is a tuple of 4 vertices
        # Get the 6 edges
        c_set = set()
        for i in range(4):
            for j in range(i + 1, 4):
                e = (k4[i], k4[j])
                c_set.add(colors[edge_index[e]])
        return len(c_set)

    def count_violations(colors):
        viol_count = 0
        viol_list = []
        for idx, k4 in enumerate(k4s):
            if get_distinct_colors(k4, colors) < 5:
                viol_count += 1
                viol_list.append(idx)
        return viol_count, viol_list

    def evaluate_move(colors, edge, new_color):
        # Returns the change in violation count for K4s containing 'edge'
        # We need to compare current distinct counts with new distinct counts
        # For each K4 containing 'edge', we need to recalculate distinct colors.
        # This is O(91 * 6) per move evaluation.
        # We try K colors, so O(91 * 6 * K).
        # This is feasible.
        
        old_viol = 0
        new_viol = 0
        affected_k4s = edge_to_k4s[edge]
        
        # We need the current colors of the other 5 edges in each K4
        # And the current color of 'edge'
        old_edge_color = colors[edge_index[edge]]
        
        for k4_idx in affected_k4s:
            k4 = k4s[k4_idx]
            
            # Calculate current distinct colors
            c_set_old = set()
            for i in range(4):
                for j in range(i + 1, 4):
                    e = (k4[i], k4[j])
                    c_set_old.add(colors[edge_index[e]])
            old_viol += 1 if len(c_set_old) < 5 else 0
            
            # Calculate new distinct colors
            # We can optimize by removing old color and adding new color
            # But set operations are small (size <= 6), so just rebuild or update
            c_set_new = set()
            for i in range(4):
                for j in range(i + 1, 4):
                    e = (k4[i], k4[j])
                    if e == edge:
                        c_set_new.add(new_color)
                    else:
                        c_set_new.add(colors[edge_index[e]])
            new_viol += 1 if len(c_set_new) < 5 else 0
            
        return new_viol - old_viol

    # Search strategy
    # Try K=13, then 14, then 15.
    # Time limit 150s total.
    # Allocate 40s for 13, 40s for 14, 40s for 15.
    # If 13 fails, try 14. If 14 fails, try 15.
    # Return the best (lowest K) valid coloring found.
    
    best_result = None
    best_K = 17 # Seed
    
    # If we find a valid coloring with K < best_K, update best_result
    
    for K in [13, 14, 15]:
        start_time = time.time()
        time_limit = 40 # seconds
        
        # Initialize
        colors = [random.randint(0, K - 1) for _ in range(120)]
        
        # Check initial
        v_count, v_list = count_violations(colors)
        
        if v_count == 0:
            best_result = colors[:]
            best_K = K
            break
        
        while time.time() - start_time < time_limit:
            if v_count == 0:
                best_result = colors[:]
                best_K = K
                break
                
            # Pick a random violation
            # If v_list is empty, v_count should be 0, but just in case
            if not v_list:
                v_count, v_list = count_violations(colors)
                if v_count == 0:
                    best_result = colors[:]
                    best_K = K
                    break
            
            k4_idx = random.choice(v_list)
            k4 = k4s[k4_idx]
            
            # Pick a random edge in this K4
            edges_in_k4 = []
            for i in range(4):
                for j in range(i + 1, 4):
                    edges_in_k4.append((k4[i], k4[j]))
            
            edge = random.choice(edges_in_k4)
            edge_idx = edge_index[edge]
            
            # Try all K colors
            best_delta = 1000
            best_new_color = None
            
            for c in range(K):
                if c == colors[edge_idx]:
                    continue
                delta = evaluate_move(colors, edge, c)
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = c
            
            if best_new_color is not None:
                # Make the move
                colors[edge_idx] = best_new_color
                # Update violation count approximately?
                # We only know the delta for affected K4s.
                # We need to update v_count and v_list.
                # Recalculating all is O(1820 * 6) = 10920 ops.
                # Doing this every step might be slow if steps are frequent.
                # But 10k ops is very fast in Python.
                v_count, v_list = count_violations(colors)
            else:
                # No improving move found for this edge.
                # Try a different edge in the same K4? Or random restart?
                # Let's just pick a random edge and a random color to escape local minima
                edge = random.choice(edges_in_k4)
                edge_idx = edge_index[edge]
                colors[edge_idx] = random.randint(0, K - 1)
                v_count, v_list = count_violations(colors)
                
    if best_result is None:
        # Fallback to seed if no improvement found (should not happen as 15 is easy)
        # But let's try one more random search for 15 if we didn't get there?
        # The loop above covers 15.
        # If best_K is still 17, we return the seed.
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    else:
        return best_result
