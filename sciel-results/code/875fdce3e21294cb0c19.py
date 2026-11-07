import time
import random
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for 150s limit
    
    n = 20
    trees = instance_trees(n, 611)
    
    # Precompute tree edge lists
    tree_edges = [trees[i] for i in range(n)]
    
    # Precompute degree of each vertex in each tree
    tree_deg = []
    for i in range(n):
        k = i + 1
        deg = [0] * k
        for u, v in tree_edges[i]:
            deg[u] += 1
            deg[v] += 1
        tree_deg.append(deg)
    
    # All possible edges in K20
    all_edges_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges_set.add((i, j))
    
    # Find a good ordering: most constrained first (higher max degree, larger size)
    def tree_priority(t_idx):
        k = t_idx + 1
        max_deg = max(tree_deg[t_idx]) if k > 1 else 0
        return (-max_deg, -k, t_idx)
    
    order = sorted(range(n), key=tree_priority)
    
    def find_embedding(t_idx, used_edges, host_map, adj, deadline_check):
        """
        DFS to find embedding of tree t_idx into available edges.
        host_map: dict {tree_vertex: host_vertex} for vertices already placed
        Returns True if found, modifies host_map in place.
        """
        k = t_idx + 1
        if k == 1:
            return True
        
        if deadline_check():
            return False
        
        # Find unplaced tree vertices
        unplaced = [v for v in range(k) if v not in host_map]
        if not unplaced:
            return True
        
        # Pick the most constrained unplaced vertex
        best_v = None
        best_key = (-1, -1)
        for v in unplaced:
            placed_neighbors = [host_map[nb] for nb in adj[v] if nb in host_map]
            constraint = len(placed_neighbors)
            deg = tree_deg[t_idx][v]
            key = (constraint, deg)
            if key > best_key:
                best_key = key
                best_v = v
        
        if best_v is None:
            best_v = unplaced[0]
        
        # Determine candidate host vertices for best_v
        placed_neighbors = [host_map[nb] for nb in adj[best_v] if nb in host_map]
        if not placed_neighbors:
            candidates = list(range(n))
        else:
            candidates = set(range(n))
            for ph in placed_neighbors:
                valid = set()
                for v in range(n):
                    if v == ph:
                        continue
                    e = (min(ph, v), max(ph, v))
                    if e not in used_edges:
                        valid.add(v)
                candidates &= valid
        
        # Filter out vertices already used by this tree
        used_in_tree = set(host_map.values())
        candidates = [v for v in candidates if v not in used_in_tree]
        
        if not candidates:
            return False
        
        # Order candidates: prefer vertices with fewer used edges (more flexibility)
        def candidate_score(hv):
            used_count = 0
            for other in range(n):
                if other == hv:
                    continue
                e = (min(hv, other), max(hv, other))
                if e in used_edges:
                    used_count += 1
            return used_count
        
        candidates.sort(key=candidate_score)
        
        for hv in candidates:
            host_map[best_v] = hv
            if find_embedding(t_idx, used_edges, host_map, adj, deadline_check):
                return True
            del host_map[best_v]
        
        return False
    
    def verify_solution(sol):
        """Verify that the solution is valid and compute its value."""
        used = set()
        total = 0
        for i in range(n):
            if sol[i] is not None:
                k = i + 1
                if len(sol[i]) != k:
                    return None
                if len(set(sol[i])) != k:
                    return None
                for v in sol[i]:
                    if not (0 <= v < n):
                        return None
                    if not isinstance(v, int):
                        return None
                for u, v in tree_edges[i]:
                    hu, hv = sol[i][u], sol[i][v]
                    if hu == hv:
                        return None
                    e = (min(hu, hv), max(hu, hv))
                    if e in used:
                        return None
                    used.add(e)
                    total += 1
        return total
    
    def try_full_backtrack(order, deadline_check):
        """
        Attempt to pack all trees in given order using backtracking.
        Returns solution list or None.
        """
        used_edges = set()
        solution = [None] * n
        tree_used_edges = [set() for _ in range(n)]
        
        def backtrack(idx):
            if idx == len(order):
                return True
            
            if deadline_check():
                return False
            
            t_idx = order[idx]
            k = t_idx + 1
            
            if k == 1:
                # T1: assign a random vertex
                solution[t_idx] = [random.randint(0, n - 1)]
                if backtrack(idx + 1):
                    return True
                solution[t_idx] = None
                return False
            
            adj = []
            for u, v in tree_edges[t_idx]:
                adj.append((u, v))
            
            # Build adjacency list for this tree
            tree_adj = [[] for _ in range(k)]
            for u, v in tree_edges[t_idx]:
                tree_adj[u].append(v)
                tree_adj[v].append(u)
            
            # Try different random roots
            for attempt in range(50):
                if deadline_check():
                    return False
                
                root = random.randint(0, k - 1)
                root_host = random.randint(0, n - 1)
                host_map = {root: root_host}
                
                if find_embedding(t_idx, used_edges, host_map, tree_adj, deadline_check):
                    emb = [host_map[i] for i in range(k)]
                    solution[t_idx] = emb
                    for u, v in tree_edges[t_idx]:
                        hu, hv = emb[u], emb[v]
                        e = (min(hu, hv), max(hu, hv))
                        used_edges.add(e)
                        tree_used_edges[t_idx].add(e)
                    
                    if backtrack(idx + 1):
                        return True
                    
                    # Backtrack: undo this tree
                    for e in tree_used_edges[t_idx]:
                        used_edges.discard(e)
                    tree_used_edges[t_idx] = set()
                    solution[t_idx] = None
            
            return False
        
        if backtrack(0):
            return solution
        return None
    
    def deadline_check():
        return time.time() > deadline
    
    best_solution = None
    best_value = 0
    
    # Try full backtracking with the prioritized order
    random.seed(42)
    sol = try_full_backtrack(order, deadline_check)
    if sol is not None:
        val = verify_solution(sol)
        if val is not None:
            best_solution = sol
            best_value = val
            if val == 190:
                return sol
    
    # If full backtracking didn't complete, try greedy with restarts
    def try_greedy(order, deadline_check):
        used_edges = set()
        solution = [None] * n
        
        for t_idx in order:
            k = t_idx + 1
            if k == 1:
                solution[t_idx] = [random.randint(0, n - 1)]
                continue
            
            # Build adjacency list for this tree
            tree_adj = [[] for _ in range(k)]
            for u, v in tree_edges[t_idx]:
                tree_adj[u].append(v)
                tree_adj[v].append(u)
            
            found = False
            
            for attempt in range(30):
                if deadline_check():
                    break
                root = random.randint(0, k - 1)
                root_host = random.randint(0, n - 1)
                host_map = {root: root_host}
                if find_embedding(t_idx, used_edges, host_map, tree_adj, deadline_check):
                    emb = [host_map[i] for i in range(k)]
                    solution[t_idx] = emb
                    valid = True
                    for u, v in tree_edges[t_idx]:
                        hu, hv = emb[u], emb[v]
                        e = (min(hu, hv), max(hu, hv))
                        if e in used_edges:
                            solution[t_idx] = None
                            valid = False
                            break
                        used_edges.add(e)
                    if valid:
                        found = True
                        break
            
            if not found:
                solution[t_idx] = None
        
        val = verify_solution(solution)
        if val is None:
            return solution, 0
        return solution, val
    
    # Do greedy restarts with different random seeds
    for restart in range(5000):
        if time.time() > deadline:
            break
        
        random.seed(restart)
        sol, val = try_greedy(order, deadline_check)
        if val > best_value:
            best_value = val
            best_solution = sol
        if val == 190:
            return sol
    
    if best_solution is not None:
        # Final verification
        val = verify_solution(best_solution)
        if val is not None:
            return best_solution
    
    # Fallback: return a minimal valid solution
    fallback = [None] * n
    fallback[0] = [0]  # T1: single vertex
    return fallback

if __name__ == "__main__":
    pass
