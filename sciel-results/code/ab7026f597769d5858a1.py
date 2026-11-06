import time
import random

def construct():
    start_time = time.time()
    # Leave 10 seconds margin for the evaluator's own checks/overhead
    deadline = start_time + 140.0 

    def find_K4(n, adj):
        """
        Finds a K4 in the graph. Returns a list of 4 vertices or None.
        adj is a 2D list of booleans.
        """
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i][j]:
                    continue
                # Find common neighbors of i and j with index > j
                common = []
                for k in range(j + 1, n):
                    if adj[i][k] and adj[j][k]:
                        common.append(k)
                
                # If there are at least 2 common neighbors, check if they are connected
                if len(common) >= 2:
                    for idx_k in range(len(common)):
                        k = common[idx_k]
                        for idx_l in range(idx_k + 1, len(common)):
                            l = common[idx_l]
                            if adj[k][l]:
                                return [i, j, k, l]
        return None

    def find_indep_6(n, adj):
        """
        Finds an independent set of size 6. Returns a list of 6 vertices or None.
        Uses backtracking with pruning.
        """
        def backtrack(candidates, current_set):
            if len(current_set) == 6:
                return current_set[:]
            
            # Pruning: not enough candidates left
            if len(candidates) < 6 - len(current_set):
                return None
            
            for i in range(len(candidates)):
                v = candidates[i]
                
                # Check if v is non-adjacent to all in current_set
                is_indep = True
                for u in current_set:
                    if adj[v][u]:
                        is_indep = False
                        break
                if not is_indep:
                    continue
                
                # Recurse: new candidates are those after i that are non-adjacent to v
                new_candidates = []
                for j in range(i + 1, len(candidates)):
                    w = candidates[j]
                    if not adj[v][w]:
                        new_candidates.append(w)
                
                res = backtrack(new_candidates, current_set + [v])
                if res:
                    return res
            return None
        
        return backtrack(list(range(n)), [])

    def local_search_with_kick(n, max_time):
        """
        Local search with random kicks to escape local minima.
        """
        adj = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i][j] = adj[j][i] = True
        
        current_time = time.time()
        steps = 0
        max_steps = 50000
        kick_interval = 2000 # Kick every 2000 steps if no progress? 
        # Or just kick if we loop too long without finding a solution.
        # Since we don't track "progress" easily (energy is binary valid/invalid),
        # we kick periodically.
        
        while current_time < max_time and steps < max_steps:
            # Check for K4
            k4 = find_K4(n, adj)
            if k4:
                edge_idx = random.randrange(6)
                pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
                u, v = pairs[edge_idx]
                adj[k4[u]][k4[v]] = False
                adj[k4[v]][k4[u]] = False
                steps += 1
                current_time = time.time()
                continue
            
            # Check for Independent Set of 6
            indep6 = find_indep_6(n, adj)
            if indep6:
                edge_idx = random.randrange(15)
                pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)]
                u, v = pairs[edge_idx]
                adj[indep6[u]][indep6[v]] = True
                adj[indep6[v]][indep6[u]] = True
                steps += 1
                current_time = time.time()
                continue
            
            # If neither K4 nor Indep 6 is found, the graph is valid
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i][j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
            # Kick logic: if we are here, it means we didn't find a violation?
            # No, the structure is:
            # if k4: fix, continue
            # if indep: fix, continue
            # else: return valid
            
            # So we never reach the kick logic if we are in a loop of violations.
            # The loop only breaks if we return valid or time out.
            # This means if we are stuck in a local minimum where violations exist 
            # but our random repair keeps creating new violations or cycling, 
            # we just keep going until timeout.
            # To escape, we should add a kick step BEFORE checking violations, 
            # or if we detect cycling.
            
        return None

    # Better Kick Logic:
    def local_search_with_kick_v2(n, max_time):
        adj = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i][j] = adj[j][i] = True
        
        current_time = time.time()
        steps = 0
        max_steps = 50000
        no_improvement_steps = 0
        last_valid_check = 0
        kick_threshold = 500
        
        while current_time < max_time and steps < max_steps:
            # Check for K4
            k4 = find_K4(n, adj)
            if k4:
                # Remove edge
                edge_idx = random.randrange(6)
                pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
                u, v = pairs[edge_idx]
                adj[k4[u]][k4[v]] = False
                adj[k4[v]][k4[u]] = False
                no_improvement_steps += 1
                steps += 1
                current_time = time.time()
                
                # Kick if stuck
                if no_improvement_steps > kick_threshold:
                    # Randomly flip 5% of edges
                    num_flips = int(0.05 * n * (n-1) / 2)
                    for _ in range(num_flips):
                        u = random.randrange(n)
                        v = random.randrange(n)
                        if u != v:
                            adj[u][v] = not adj[u][v]
                            adj[v][u] = not adj[v][u]
                    no_improvement_steps = 0
                continue
            
            # Check for Independent Set of 6
            indep6 = find_indep_6(n, adj)
            if indep6:
                # Add edge
                edge_idx = random.randrange(15)
                pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)]
                u, v = pairs[edge_idx]
                adj[indep6[u]][indep6[v]] = True
                adj[indep6[v]][indep6[u]] = True
                no_improvement_steps += 1
                steps += 1
                current_time = time.time()
                
                # Kick if stuck
                if no_improvement_steps > kick_threshold:
                    num_flips = int(0.05 * n * (n-1) / 2)
                    for _ in range(num_flips):
                        u = random.randrange(n)
                        v = random.randrange(n)
                        if u != v:
                            adj[u][v] = not adj[u][v]
                            adj[v][u] = not adj[v][u]
                    no_improvement_steps = 0
                continue
            
            # Valid graph found
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i][j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
        return None

    # Try to construct graphs for n = 35 down to 17
    best_result = None
    
    for n in range(35, 16, -1):
        if time.time() > deadline:
            break
            
        max_restarts = 50 if n >= 33 else 20
        
        for r in range(max_restarts):
            if time.time() > deadline:
                break
            
            result = local_search_with_kick_v2(n, deadline)
            if result:
                best_result = result
                break
        
        if best_result and best_result['n'] == n:
            return best_result

    # Fallback: Return the seed graph (Paley 17)
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
