"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110
    random.seed(42)
    
    def extract_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    def count_k4(adj, n):
        """Count number of K4 cliques using bitmask."""
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            for l in range(k+1, n):
                                if adj[i][l] and adj[j][l] and adj[k][l]:
                                    count += 1
        return count

    def count_indep6(adj, n):
        """Count number of independent 6-sets using backtracking."""
        count = 0
        
        def backtrack(remaining, chosen_size):
            nonlocal count
            if chosen_size == 6:
                count += 1
                return
            if len(remaining) < 6 - chosen_size:
                return
            for idx, v in enumerate(remaining):
                new_remaining = []
                for u in remaining[idx+1:]:
                    if not adj[v][u]:
                        new_remaining.append(u)
                backtrack(new_remaining, chosen_size + 1)
        
        backtrack(list(range(n)), 0)
        return count

    def is_valid(adj, n):
        return count_k4(adj, n) == 0 and count_indep6(adj, n) == 0

    def energy(adj, n):
        """Energy = #K4 + #indep6. Lower is better; 0 means valid."""
        k4 = count_k4(adj, n)
        if k4 > 0:
            return k4  # Short-circuit: K4 is the bigger problem
        indep = count_indep6(adj, n)
        return indep

    def flip_edge(adj, n, u, v):
        adj[u][v] = adj[v][u] = not adj[u][v]

    def simulated_annealing(adj, n, start_temp, end_temp, steps, deadline):
        """Simulated annealing to minimize energy."""
        current_energy = energy(adj, n)
        best_adj = [row[:] for row in adj]
        best_energy = current_energy
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Pick random edge to flip
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            old_val = adj[u][v]
            flip_edge(adj, n, u, v)
            new_energy = energy(adj, n)
            
            delta = new_energy - current_energy
            temp = start_temp * (end_temp / start_temp) ** (step / max(steps, 1))
            
            if delta <= 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_adj = [row[:] for row in adj]
            else:
                flip_edge(adj, n, u, v)
        
        return best_adj, best_energy

    def greedy_repair(adj, n, max_steps, deadline):
        """Greedy repair: repeatedly fix the worst violation."""
        for _ in range(max_steps):
            if time.time() > deadline:
                break
            
            # Find a K4
            k4_found = False
            for i in range(n):
                if k4_found:
                    break
                for j in range(i+1, n):
                    if k4_found:
                        break
                    if adj[i][j]:
                        for k in range(j+1, n):
                            if k4_found:
                                break
                            if adj[i][k] and adj[j][k]:
                                for l in range(k+1, n):
                                    if adj[i][l] and adj[j][l] and adj[k][l]:
                                        # Found K4: remove one edge
                                        # Remove edge that's in the most K4s
                                        pairs = [(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)]
                                        # Simple heuristic: remove first pair
                                        u, v = pairs[0]
                                        flip_edge(adj, n, u, v)
                                        k4_found = True
                                        break
            if k4_found:
                continue
            
            # Find an independent 6-set
            indep_found = False
            def find_one_indep6():
                def bt(remaining, chosen):
                    if len(chosen) == 6:
                        return chosen
                    if len(remaining) < 6 - len(chosen):
                        return None
                    for idx, v in enumerate(remaining):
                        new_rem = [u for u in remaining[idx+1:] if not adj[v][u]]
                        res = bt(new_rem, chosen + [v])
                        if res is not None:
                            return res
                    return None
                return bt(list(range(n)), [])
            
            indep6 = find_one_indep6()
            if indep6:
                # Add an edge within the independent set
                # Try to add edge that breaks most independent sets
                # Simple: add edge between two random vertices in the set
                u, v = random.sample(indep6, 2)
                flip_edge(adj, n, u, v)
                indep_found = True
            if indep_found:
                continue
            
            # No violations found
            break
        
        return adj

    # ---- Start with known constructions ----
    
    # Paley graph on 17 vertices
    squares_17 = {x*x % 17 for x in range(1, 17)}
    adj17 = [[False]*17 for _ in range(17)]
    for i in range(17):
        for j in range(i+1, 17):
            if (j-i) % 17 in squares_17:
                adj17[i][j] = adj17[j][i] = True
    best_result = {'n': 17, 'edges': extract_edges(adj17, 17)}
    
    # Try product graph on 35 vertices
    n35 = 35
    D5 = {1, 4}
    D7 = {1, 2, 4}
    adj35 = [[False]*n35 for _ in range(n35)]
    for i in range(n35):
        a1, b1 = divmod(i, 7)
        for j in range(i+1, n35):
            a2, b2 = divmod(j, 7)
            da = (a2 - a1) % 5
            db = (b2 - b1) % 7
            if (da in D5 and db in D7) or (da not in D5 and db not in D7):
                adj35[i][j] = adj35[j][i] = True
    
    # Check if valid
    if is_valid(adj35, n35):
        best_result = {'n': n35, 'edges': extract_edges(adj35, n35)}
        return best_result
    
    # ---- Structural change: Use simulated annealing with proper energy landscape ----
    # Try multiple target sizes, using SA to find valid graphs
    
    for target_n in [35, 34, 33, 32, 31, 30]:
        if time.time() > deadline - 10:
            break
        
        found_valid = False
        
        # Multiple random restarts with SA
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            if found_valid:
                break
            
            # Initialize with different strategies
            if restart == 0:
                # Start from product graph
                adj = [row[:] for row in adj35] if target_n == 35 else None
                if target_n != 35:
                    adj = [[False]*target_n for _ in range(target_n)]
                    for i in range(target_n):
                        for j in range(i+1, target_n):
                            if random.random() < 0.45:
                                adj[i][j] = adj[j][i] = True
            elif restart == 1:
                # Start from sparse random
                adj = [[False]*target_n for _ in range(target_n)]
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        if random.random() < 0.35:
                            adj[i][j] = adj[j][i] = True
            elif restart == 2:
                # Start from dense random
                adj = [[False]*target_n for _ in range(target_n)]
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        if random.random() < 0.55:
                            adj[i][j] = adj[j][i] = True
            else:
                # Random
                adj = [[False]*target_n for _ in range(target_n)]
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        if random.random() < 0.45:
                            adj[i][j] = adj[j][i] = True
            
            # First do greedy repair to get close to valid
            adj = greedy_repair(adj, target_n, 500, deadline - 10)
            e = energy(adj, target_n)
            
            if e == 0:
                found_valid = True
                best_result = {'n': target_n, 'edges': extract_edges(adj, target_n)}
                continue
            
            # Then use simulated annealing to escape local minima
            start_temp = max(1.0, e * 0.5)
            end_temp = 0.01
            sa_steps = 3000
            adj, best_e = simulated_annealing(adj, target_n, start_temp, end_temp, sa_steps, deadline - 10)
            
            if best_e == 0:
                found_valid = True
                best_result = {'n': target_n, 'edges': extract_edges(adj, target_n)}
        
        if found_valid:
            break
    
    return best_result
# EVOLVE-BLOCK-END
