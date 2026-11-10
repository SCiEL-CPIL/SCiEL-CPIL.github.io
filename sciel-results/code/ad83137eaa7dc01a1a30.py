import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit

    n = 16
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))

    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx

    # Precompute all K4s as lists of 6 edge indices
    k4s = []
    k4_vertices = []
    for combo in itertools.combinations(range(n), 4):
        a, b, c, d = combo
        e1 = edge_idx[(a, b)]
        e2 = edge_idx[(a, c)]
        e3 = edge_idx[(a, d)]
        e4 = edge_idx[(b, c)]
        e5 = edge_idx[(b, d)]
        e6 = edge_idx[(c, d)]
        k4s.append((e1, e2, e3, e4, e5, e6))
        k4_vertices.append(combo)

    k4_edges_np = np.array(k4s, dtype=np.int32)  # shape (1820, 6)
    num_k4s = len(k4s)
    
    # Map edge index to list of K4 indices
    edge_to_k4s = [[] for _ in range(120)]
    for k4_idx, k4 in enumerate(k4s):
        for eidx in k4:
            edge_to_k4s[eidx].append(k4_idx)

    def count_violations_full(colours):
        c_np = np.asarray(colours, dtype=np.int32)
        k4_colors = c_np[k4_edges_np]
        sorted_colors = np.sort(k4_colors, axis=1)
        unique_mask = np.ones((num_k4s, 6), dtype=bool)
        unique_mask[:, 1:] = (sorted_colors[:, 1:] != sorted_colors[:, :-1])
        counts = np.sum(unique_mask, axis=1)
        return int(np.sum(counts < 5))

    def is_valid(colours):
        return count_violations_full(colours) == 0

    # Start with seed (i+j) % 17
    current = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    # Relabel to 0..15
    used_colors = sorted(set(current))
    color_map = {c: i for i, c in enumerate(used_colors)}
    current = [color_map[c] for c in current]
    
    if not is_valid(current):
        for _ in range(100):
            if time.time() > deadline:
                break
            rand = [random.randint(0, 15) for _ in range(120)]
            if is_valid(rand):
                current = rand
                break

    def local_search_fast(colours, K, max_steps=5000):
        """Local search with incremental violation updates."""
        colours = list(colours)
        
        # Initialize violation counts
        # We need to track the number of distinct colors in each K4
        # Or just a boolean 'is_violated'
        # Let's track the set of colors? No, too slow.
        # Let's track the count of unique colors.
        # To update count of unique colors efficiently, we need to know the frequency of each color in the K4.
        # Since K4 has 6 edges, we can just recompute the unique count for the affected K4s.
        # There are at most 105 K4s per edge.
        
        k4_unique_counts = np.zeros(num_k4s, dtype=np.int8)
        
        def update_k4_counts():
            # This is the initial full pass
            for k4_idx in range(num_k4s):
                e1, e2, e3, e4, e5, e6 = k4s[k4_idx]
                c = {colours[e1], colours[e2], colours[e3], colours[e4], colours[e5], colours[e6]}
                k4_unique_counts[k4_idx] = len(c)
            return int(np.sum(k4_unique_counts < 5))

        v = update_k4_counts()
        if v == 0:
            return colours, 0

        # Precompute the list of K4s for each edge for fast access
        # edge_to_k4s is already built

        for step in range(max_steps):
            if time.time() > deadline:
                break
            if v == 0:
                break
            
            # Find violated K4s
            violated = np.where(k4_unique_counts < 5)[0]
            if len(violated) == 0:
                v = 0
                break
            
            # Pick a random violated K4
            k4_idx = violated[random.randint(0, len(violated) - 1)]
            k4_edge_idxs = k4_edges_np[k4_idx]
            
            # Try to change color of one edge in this K4
            best_new_v = v
            best_edge = -1
            best_color = -1
            
            # Shuffle edges to avoid bias
            edge_order = list(k4_edge_idxs)
            random.shuffle(edge_order)
            
            for eidx in edge_order:
                old_c = colours[eidx]
                # Try all colors
                for new_c in range(K):
                    if new_c == old_c:
                        continue
                    
                    colours[eidx] = new_c
                    
                    # Update affected K4s
                    affected_k4s = edge_to_k4s[eidx]
                    delta_v = 0
                    for ak4 in affected_k4s:
                        old_count = k4_unique_counts[ak4]
                        # Recompute unique count for this K4
                        e1, e2, e3, e4, e5, e6 = k4s[ak4]
                        c = {colours[e1], colours[e2], colours[e3], colours[e4], colours[e5], colours[e6]}
                        new_count = len(c)
                        k4_unique_counts[ak4] = new_count
                        
                        if old_count < 5 and new_count >= 5:
                            delta_v -= 1
                        elif old_count >= 5 and new_count < 5:
                            delta_v += 1
                    
                    new_v = v + delta_v
                    
                    if new_v < best_new_v:
                        best_new_v = new_v
                        best_edge = eidx
                        best_color = new_c
                        # We don't break here because we want to find the best move among all edges in this K4?
                        # Or just the first improvement?
                        # Finding the best move among 6 edges * 15 colors = 90 updates is cheap.
                    
                    colours[eidx] = old_c
                    # Revert k4_unique_counts? 
                    # This is expensive. 
                    # Better approach: save the old counts for affected K4s, revert them.
                    # Or just recompute them.
                    # Given 105 K4s, recomputing is O(105 * 6) which is fast.
                    # But we are doing this inside a loop of 90 trials. 90 * 105 * 6 ~ 56k ops.
                    # This might be too slow.
                    
                    # Alternative: Just use the full count_violations but less often?
                    # No, the incremental update is key.
                    # Let's optimize the revert.
                    # We stored old_count. We can just set it back.
                    for ak4 in affected_k4s:
                        # We need the old count. We didn't store it in a list, we just used it for delta.
                        # Let's restructure to store old counts.
                        pass 

                    # This revert logic is messy. Let's simplify.
                    # Just calculate new_v from scratch for the affected K4s?
                    # No, we need the global v.
                    
                    # Let's try a different approach:
                    # 1. Save old counts for affected K4s.
                    # 2. Update counts.
                    # 3. Calculate delta.
                    # 4. Restore old counts.
                    
                    # Let's refactor the inner loop to be cleaner.
                    pass

            # The above implementation is getting complex and potentially slow due to the revert logic.
            # Let's stick to a simpler but faster full-update if we can batch it?
            # No, 1820 K4s is small. 
            # Let's benchmark: 
            # Python loop over 1820 K4s, each taking ~1 microsecond? 
            # 1820 * 1us = 1.8ms.
            # If we do 1000 steps, that's 1.8 seconds.
            # We have 140 seconds. So we can do ~70,000 steps.
            # The previous code did full updates. 
            # Why was it slow? 
            # `count_violations` uses NumPy. 
            # `np.sort` on (1820, 6) array.
            # This is very fast in C.
            # So the bottleneck might not be the violation count itself, but the Python overhead of the search loop?
            # Or maybe `count_violations` is not that fast?
            
            # Let's check the speed of `count_violations`.
            # If it's 1ms, then 1000 steps is 1s.
            # The previous code had `local_search` with `max_steps=5000`.
            # If each step takes 1ms, that's 5s.
            # Then it tries 30 restarts. 30 * 5s = 150s. Timeout.
            
            # So we need to either:
            # 1. Make `count_violations` faster.
            # 2. Reduce the number of steps/restarts.
            # 3. Use incremental updates correctly.
            
            # Let's try to optimize `count_violations` using a different NumPy trick or just accept the cost and reduce steps.
            # Or, use the incremental update but implement the revert carefully.
            
            # Let's rewrite local_search with proper incremental updates.
            pass

        return colours, v

    # Since the incremental update implementation is tricky to get right quickly, 
    # let's try to optimize the existing full-update approach by reducing the search space.
    
    def local_search(colours, K, max_steps=5000):
        colours = list(colours)
        v = count_violations_full(colours)
        if v == 0:
            return colours, 0
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if v == 0:
                break
            
            c_np = np.asarray(colours, dtype=np.int32)
            k4_colors = c_np[k4_edges_np]
            sorted_colors = np.sort(k4_colors, axis=1)
            unique_mask = np.ones((num_k4s, 6), dtype=bool)
            unique_mask[:, 1:] = (sorted_colors[:, 1:] != sorted_colors[:, :-1])
            counts = np.sum(unique_mask, axis=1)
            violated = np.where(counts < 5)[0]
            
            if len(violated) == 0:
                v = 0
                break
            
            k4_idx = violated[random.randint(0, len(violated) - 1)]
            k4_edge_idxs = k4_edges_np[k4_idx]
            
            best_new_v = v
            best_edge = -1
            best_color = -1
            
            # Try to improve by changing one edge in this K4
            for eidx in k4_edge_idxs:
                old_c = colours[eidx]
                for new_c in range(K):
                    if new_c == old_c:
                        continue
                    colours[eidx] = new_c
                    new_v = count_violations_full(colours)
                    if new_v < best_new_v:
                        best_new_v = new_v
                        best_edge = eidx
                        best_color = new_c
                    colours[eidx] = old_c
            
            if best_edge >= 0:
                colours[best_edge] = best_color
                v = best_new_v
            else:
                # Random perturbation
                eidx = random.randint(0, 119)
                colours[eidx] = random.randint(0, K - 1)
                v = count_violations_full(colours)
                
        return colours, v

    # Use the faster local search
    for target_max_colors in range(16, 12, -1):
        if time.time() > deadline:
            break

        if len(set(current)) <= target_max_colors:
            continue

        current_set = set(current)
        colors_list = sorted(current_set)

        if len(colors_list) <= target_max_colors:
            continue

        merged = False
        for i in range(len(colors_list)):
            for j in range(i + 1, len(colors_list)):
                if time.time() > deadline:
                    break
                c1 = colors_list[i]
                c2 = colors_list[j]
                new_colours = [c2 if c == c1 else c for c in current]
                if is_valid(new_colours):
                    current = new_colours
                    merged = True
                    break
            if merged:
                break

        if not merged:
            K = target_max_colors
            mapping = {}
            next_id = 0
            init_colours = []
            for c in current:
                if c not in mapping:
                    if next_id >= K:
                        break
                    mapping[c] = next_id
                    next_id += 1
                init_colours.append(mapping[c])

            if len(init_colours) < 120:
                continue

            if is_valid(init_colours):
                current = init_colours
                continue

            # Local search on init_colours
            best_colours, best_violations = local_search(init_colours, K)
            if best_violations == 0:
                current = best_colours
                continue

            # Random restarts with local search
            success = False
            for restart in range(30):
                if time.time() > deadline:
                    break
                rand_colours = [random.randint(0, K - 1) for _ in range(120)]
                rand_colours, v = local_search(rand_colours, K)
                if v == 0:
                    current = rand_colours
                    success = True
                    break

            if not success:
                break

    if not is_valid(current):
        seed = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
        if is_valid(seed):
            current = seed

    return current
