# EVOLVE-BLOCK-START
import itertools
import time

def construct():
    n = 16
    num_edges = n * (n - 1) // 2
    edge_list = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_map = {e: idx for idx, e in enumerate(edge_list)}
    
    k4s = list(itertools.combinations(range(n), 4))
    k4_edges = []
    for quad in k4s:
        edges_in_quad = [edge_map[(u, v)] for u, v in itertools.combinations(quad, 2)]
        k4_edges.append(edges_in_quad)
        
    k4s_per_edge = [[] for _ in range(num_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            k4s_per_edge[e].append(k4_idx)

    def solve(k, time_limit):
        assignment = [None] * num_edges
        # Symmetry breaking: 
        # Edge 0 (0,1) -> 0
        # Edge 1 (0,2) -> 1 (if k>1)
        # Edge 2 (0,3) -> 2 (if k>2) ... this might be too strong or wrong.
        # Let's just fix Edge 0 to 0.
        
        def check_k4(k4_idx):
            edges = k4_edges[k4_idx]
            cols = set()
            unassigned = 0
            for e in edges:
                c = assignment[e]
                if c is None:
                    unassigned += 1
                else:
                    cols.add(c)
            if len(cols) + unassigned < 5:
                return False
            if unassigned == 0 and len(cols) < 5:
                return False
            return True

        def backtrack(idx, start_time):
            if time.time() - start_time > time_limit:
                return False
            if idx == num_edges:
                return True
                
            e = edge_list[idx]
            
            # Heuristic: Try colours in a specific order?
            # Maybe try colours that are "new" to the K4s first?
            
            for c in range(k):
                assignment[e] = c
                valid = True
                for k4_idx in k4s_per_edge[e]:
                    if not check_k4(k4_idx):
                        valid = False
                        break
                if valid:
                    if backtrack(idx + 1, start_time):
                        return True
                assignment[e] = None
            return False

        # Apply symmetry breaking
        assignment[edge_list[0]] = 0
        if not check_k4_for_edge(edge_list[0]): # Need helper
            pass
            
        # To keep it simple, just start backtrack at 1 if we fixed 0?
        # Or let backtrack handle it.
        
        start_time = time.time()
        if backtrack(0, start_time):
            return assignment[:]
        return None

    # Helper for symmetry breaking check
    def check_k4_for_edge(e):
        for k4_idx in k4s_per_edge[e]:
            edges = k4_edges[k4_idx]
            cols = set()
            unassigned = 0
            for ee in edges:
                c = assignment[ee]
                if c is None:
                    unassigned += 1
                else:
                    cols.add(c)
            if len(cols) + unassigned < 5:
                return False
            if unassigned == 0 and len(cols) < 5:
                return False
        return True

    # Try 15 first to ensure we have a result
    sol_15 = solve(15, 20)
    if sol_15:
        return sol_15
        
    sol_14 = solve(14, 20)
    if sol_14:
        return sol_14
        
    sol_13 = solve(13, 20)
    if sol_13:
        return sol_13
        
    # Fallback
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
# EVOLVE-BLOCK-END
