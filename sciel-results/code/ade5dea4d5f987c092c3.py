import random
import time
import itertools

def construct():
    start_time = time.time()
    time_limit = 110  # Leave 10s margin
    
    # Precompute K4s
    # Vertices 0..15
    # Edges (i,j) with i<j
    # Map edge to index
    edge_indices = {}
    idx = 0
    for i in range(16):
        for j in range(i+1, 16):
            edge_indices[(i,j)] = idx
            idx += 1
    
    k4s = []
    for quad in itertools.combinations(range(16), 4):
        a,b,c,d = quad
        edges = [
            edge_indices[(a,b)],
            edge_indices[(a,c)],
            edge_indices[(a,d)],
            edge_indices[(b,c)],
            edge_indices[(b,d)],
            edge_indices[(c,d)]
        ]
        k4s.append(edges)
        
    def count_bad_k4s(colors):
        bad_count = 0
        for edges in k4s:
            c0 = colors[edges[0]]
            c1 = colors[edges[1]]
            c2 = colors[edges[2]]
            c3 = colors[edges[3]]
            c4 = colors[edges[4]]
            c5 = colors[edges[5]]
            # Check if number of distinct colors < 5
            # Using a set is slow, use manual check for 6 items
            # Or just len(set(...))
            if len({c0, c1, c2, c3, c4, c5}) < 5:
                bad_count += 1
        return bad_count

    def try_find(k):
        # Random init
        colors = [random.randint(0, k-1) for _ in range(120)]
        bad_count = count_bad_k4s(colors)
        if bad_count == 0:
            return colors
        
        best_colors = colors[:]
        best_bad = bad_count
        
        # Simulated Annealing
        T = 10.0
        cool_rate = 0.995
        steps = 0
        max_steps = 500000
        
        while time.time() - start_time < time_limit - 5 and steps < max_steps:
            steps += 1
            # Pick a random edge
            e_idx = random.randint(0, 119)
            old_color = colors[e_idx]
            new_color = random.randint(0, k-1)
            if new_color == old_color:
                continue
            
            colors[e_idx] = new_color
            new_bad = count_bad_k4s(colors)
            
            delta = new_bad - bad_count
            if delta < 0 or (delta > 0 and random.random() < pow(2.718, -delta / T)):
                bad_count = new_bad
                if bad_count < best_bad:
                    best_bad = bad_count
                    best_colors = colors[:]
                    if best_bad == 0:
                        return best_colors
            else:
                colors[e_idx] = old_color
            
            T *= cool_rate
            
        if best_bad == 0:
            return best_colors
        return None

    # Try k from 13 to 17
    for k in range(13, 18):
        if time.time() - start_time > time_limit - 10:
            break
        # Give each k a time budget
        time_budget = time_limit - 10 - (time.time() - start_time)
        # Simple heuristic: try harder for smaller k?
        # Just run try_find which has its own time check against global limit
        # But try_find uses start_time from outer scope? No, it captures it.
        # Let's modify try_find to accept a deadline
        
        def try_find_deadline(k, deadline):
            colors = [random.randint(0, k-1) for _ in range(120)]
            bad_count = count_bad_k4s(colors)
            if bad_count == 0:
                return colors
            
            best_colors = colors[:]
            best_bad = bad_count
            
            T = 10.0
            cool_rate = 0.995
            steps = 0
            max_steps = 500000
            
            while time.time() < deadline and steps < max_steps:
                steps += 1
                e_idx = random.randint(0, 119)
                old_color = colors[e_idx]
                new_color = random.randint(0, k-1)
                if new_color == old_color:
                    continue
                
                colors[e_idx] = new_color
                new_bad = count_bad_k4s(colors)
                
                delta = new_bad - bad_count
                if delta < 0 or (delta > 0 and random.random() < pow(2.718, -delta / T)):
                    bad_count = new_bad
                    if bad_count < best_bad:
                        best_bad = bad_count
                        best_colors = colors[:]
                        if best_bad == 0:
                            return best_colors
                else:
                    colors[e_idx] = old_color
                
                T *= cool_rate
                
            if best_bad == 0:
                return best_colors
            return None

        deadline = time.time() + 15 # 15s per k
        if time.time() + 15 > time_limit:
            deadline = time_limit - 1
            
        result = try_find_deadline(k, deadline)
        if result is not None:
            return result
            
    # Fallback to seed if no better found
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
