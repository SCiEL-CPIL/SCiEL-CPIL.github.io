import time
import random
import math

def construct():
    deadline = time.time() + 135
    random.seed(42)
    
    # Seed: n=32 Greenwood-Gleason
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    
    colors_32 = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors_32.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    
    def colors_to_adj(colors, n):
        adj = [[-1]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                adj[i][j] = adj[j][i] = colors[idx]
                idx += 1
        return adj
    
    def adj_to_colors(adj, n):
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(adj[i][j])
        return colors
        
    def count_mono_triangles(adj, n):
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                c = adj[i][j]
                for k in range(j+1, n):
                    if adj[i][k] == c and adj[j][k] == c:
                        count += 1
        return count

    def try_add_vertex_sa(adj, n, time_limit=5.0):
        """
        Try to add vertex n to the existing coloring of n vertices.
        Uses multi-start local search + simulated annealing to find a valid assignment.
        """
        start_time = time.time()
        
        def get_energy(assign):
            energy = 0
            for u in range(n):
                cu = assign[u]
                for v in range(u + 1, n):
                    if assign[v] == cu and adj[u][v] == cu:
                        energy += 1
            return energy

        def greedy_init():
            """Greedy: assign each vertex the color that minimizes immediate conflicts."""
            assign = [0] * n
            for u in range(n):
                best_c = 0
                best_conf = 0
                for c in range(4):
                    conf = 0
                    for v in range(u):
                        if assign[v] == c and adj[u][v] == c:
                            conf += 1
                    if conf < best_conf:
                        best_conf = conf
                        best_c = c
                assign[u] = best_c
            return assign

        def local_search(assign, energy, time_limit):
            """Steepest-descent local search until no improving move or time limit."""
            t_start = time.time()
            improved = True
            while improved and time.time() - t_start < time_limit:
                improved = False
                best_delta = 0
                best_u = -1
                best_c = -1
                for u in range(n):
                    old_c = assign[u]
                    for new_c in range(4):
                        if new_c == old_c:
                            continue
                        delta = 0
                        for v in range(n):
                            if v == u:
                                continue
                            cv = assign[v]
                            if old_c == cv and adj[u][v] == old_c:
                                delta -= 1
                            if new_c == cv and adj[u][v] == new_c:
                                delta += 1
                        if delta < best_delta:
                            best_delta = delta
                            best_u = u
                            best_c = new_c
                if best_u >= 0:
                    assign[best_u] = best_c
                    energy += best_delta
                    improved = True
                    if energy == 0:
                        break
            return assign, energy

        best_energy = float('inf')
        best_assignment = None
        
        # Multiple restarts with different initializations
        for restart in range(3):
            if time.time() - start_time > time_limit:
                break
            
            if restart == 0:
                assignment = greedy_init()
            elif restart == 1:
                assignment = [random.randint(0, 3) for _ in range(n)]
            else:
                # Perturb the greedy solution
                assignment = greedy_init()
                for _ in range(n // 4):
                    u = random.randint(0, n - 1)
                    assignment[u] = random.choice([0, 1, 2, 3])
            
            current_energy = get_energy(assignment)
            
            if current_energy == 0:
                return assignment
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_assignment = assignment[:]
            
            # Local search first
            remaining = time_limit - (time.time() - start_time)
            ls_time = min(0.3, remaining * 0.3)
            if ls_time > 0.05:
                assignment, current_energy = local_search(assignment, current_energy, ls_time)
                if current_energy == 0:
                    return assignment
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_assignment = assignment[:]
            
            # Simulated annealing
            remaining = time_limit - (time.time() - start_time)
            if remaining < 0.1:
                continue
            
            T = max(1.0, current_energy * 0.5)
            T_min = 0.01
            cooling = 0.99
            
            while time.time() - start_time < time_limit:
                if T < T_min:
                    break
                
                u = random.randint(0, n - 1)
                old_color = assignment[u]
                new_color = random.choice([0, 1, 2, 3])
                if new_color == old_color:
                    continue
                
                delta = 0
                for v in range(n):
                    if v == u:
                        continue
                    cv = assignment[v]
                    if old_color == cv and adj[u][v] == old_color:
                        delta -= 1
                    if new_color == cv and adj[u][v] == new_color:
                        delta += 1
                
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    assignment[u] = new_color
                    current_energy += delta
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_assignment = assignment[:]
                        if best_energy == 0:
                            return best_assignment
                
                T *= cooling
        
        return best_assignment if best_energy == 0 else None

    # Start
    adj = colors_to_adj(colors_32, 32)
    best_n = 32
    best_colors = colors_32
    
    # Try to expand to 50
    for target_n in range(33, 51):
        if time.time() > deadline - 10:
            break
            
        n = len(adj)
        # We are adding vertex n (0-indexed, so it becomes the n-th vertex in a list of n+1)
        # The existing vertices are 0..n-1
        
        found = False
        # Try multiple times with different random seeds for SA
        remaining_budget = deadline - time.time()
        if remaining_budget < 5:
            break
        max_attempts = min(8, max(1, int(remaining_budget / 3)))
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            random.seed(42 + target_n * 100 + attempt * 7)
            per_attempt_time = min(2.5, max(1.0, (deadline - 10 - time.time()) / (max_attempts - attempt)))
            assignment = try_add_vertex_sa(adj, n, time_limit=per_attempt_time)
            if assignment is not None:
                # Success! Add vertex
                new_n = n + 1
                new_adj = [row[:] for row in adj]
                for _ in range(new_n):
                    new_adj.append([-1] * new_n)
                for i in range(n):
                    new_adj[i][n] = new_adj[n][i] = assignment[i]
                new_adj[n][n] = -1 # Self-loop, doesn't matter
                
                adj = new_adj
                best_n = new_n
                best_colors = adj_to_colors(adj, best_n)
                found = True
                break
                
        if not found:
            # If we can't add the next vertex, we might try to repair the current one
            # But for simplicity, we stop if we can't extend.
            # However, maybe a different ordering or repair could help?
            # For now, break.
            break
            
    adj_final = colors_to_adj(best_colors, best_n)
    if count_mono_triangles(adj_final, best_n) > 0:
        return {'n': 32, 'colors': colors_32}
    
    return {'n': best_n, 'colors': best_colors}
