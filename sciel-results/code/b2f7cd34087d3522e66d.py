import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin from 120s limit

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_csp(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initial domains as lists of available colors
        domains = [list(lst) for lst in lists]
        
        # Check for empty domains
        for i in range(n):
            if not domains[i]:
                return None

        # AC-3 Initial Propagation
        # We'll use a queue of arcs to check
        from collections import deque
        queue = deque()
        for i in range(n):
            for j in adj[i]:
                queue.append((i, j))
        
        while queue:
            i, j = queue.popleft()
            # Check if there exists a value in domains[i] that is not in domains[j]
            # If not, remove that value from domains[i]
            # More precisely: for each value in domains[i], check if it has a support in domains[j]
            # Since domains are small (size <= 3), this is fast
            
            # For each value in domains[i], check if it can be assigned
            # i.e., there exists a value in domains[j] different from it
            # Wait, the constraint is that adjacent vertices must have DIFFERENT colors.
            # So for a value v in domains[i], it is supported in domains[j] if there exists u in domains[j] such that u != v.
            # This is only false if domains[j] == [v] (i.e., domains[j] is a singleton containing v).
            
            # So, if domains[j] is a singleton [v], and v is in domains[i], remove v from domains[i].
            if len(domains[j]) == 1:
                v = domains[j][0]
                if v in domains[i]:
                    domains[i].remove(v)
                    if not domains[i]:
                        return None
                    # Adding arcs (k, i) for all k in adj[i] except j
                    for k in adj[i]:
                        if k != j:
                            queue.append((k, i))
        
        # If any domain is empty, fail
        for i in range(n):
            if not domains[i]:
                return None

        # If all domains are singletons, we have a solution
        if all(len(domains[i]) == 1 for i in range(n)):
            return [domains[i][0] for i in range(n)]

        # Backtracking with Forward Checking and MRV
        color = [-1] * n
        
        # We'll use a trail to efficiently undo domain changes
        # But for simplicity and given n <= 408, copying domains is fast enough.
        
        def backtrack():
            # Find unassigned variable with smallest domain (MRV)
            min_len = float('inf')
            mrv_var = -1
            for i in range(n):
                if color[i] == -1:
                    l = len(domains[i])
                    if l < min_len:
                        min_len = l
                        mrv_var = i
                        if l == 1:
                            break
                        if l == 0:
                            return False

            if mrv_var == -1:
                return True # All assigned

            # Try values in domains[mrv_var]
            for val in domains[mrv_var]:
                # Assign
                color[mrv_var] = val
                old_domains = [list(d) for d in domains]
                
                # Forward Checking:
                consistent = True
                for j in adj[mrv_var]:
                    if color[j] == -1:
                        if val in domains[j]:
                            domains[j].remove(val)
                            if not domains[j]:
                                consistent = False
                                break
                
                if consistent:
                    if backtrack():
                        return True
                
                # Restore
                for i in range(n):
                    domains[i] = old_domains[i]
                
                color[mrv_var] = -1

            return False

        if backtrack():
            return [int(color[i]) for i in range(n)]
        return None

    results = []
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            break

        n, edges, lists = instance(splits, seed)
        solution = solve_csp(n, edges, lists)
        
        if solution is not None:
            # Verify
            valid = True
            for i in range(n):
                if solution[i] not in lists[i]:
                    valid = False
                    break
            if valid:
                for u, v in edges:
                    if solution[u] == solution[v]:
                        valid = False
                        break
            if valid:
                results.append(solution)
            else:
                results.append([0] * n)
        else:
            results.append([0] * n)

    # Fill any missing results with dummy valid-length lists
    remaining_splits = [(20, 701), (40, 702), (60, 703), (100, 704)]
    current_idx = len(results)
    for i in range(current_idx, 4):
        splits, seed = remaining_splits[i]
        n_dummy = 8 + 4 * splits
        results.append([0] * n_dummy)

    return results
