import random
import time
import math

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    n = 31
    target_edges = 91
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def count_c4(neighbors):
        """
        Count 4-cycles in graph defined by neighbors list of sets.
        Sum C(common_neighbors(u,v), 2) for all pairs (u,v), divide by 2.
        """
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = len(neighbors[u].intersection(neighbors[v]))
                if common >= 2:
                    total += common * (common - 1) // 2
        return total // 2

    def add_edge_if_no_c4(neighbors, edge):
        """
        Check if adding edge creates a C4.
        Returns True if added, False otherwise.
        """
        u, v = edge
        if v in neighbors[u]:
            return False # Already exists
        
        # A C4 is created if u and v have at least 2 common neighbors
        common = neighbors[u].intersection(neighbors[v])
        if len(common) >= 2:
            return False
        
        # Safe to add
        neighbors[u].add(v)
        neighbors[v].add(u)
        return True

    def generate_initial_graph():
        """
        Generate a graph with 91 edges.
        Strategy: Build a maximal C4-free graph (greedy), then add one random edge.
        """
        neighbors = [set() for _ in range(n)]
        edges = set()
        
        # Greedy C4-free construction
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        for e in shuffled_edges:
            if add_edge_if_no_c4(neighbors, e):
                edges.add(e)
            if len(edges) >= target_edges:
                break
                
        # If we didn't reach 91 (shouldn't happen if ex(31,C4) >= 90), add random edges
        if len(edges) < target_edges:
            remaining = [e for e in all_edges if e not in edges]
            random.shuffle(remaining)
            for e in remaining:
                if len(edges) >= target_edges:
                    break
                u, v = e
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.add(e)
                
        # If we have more than 91 (unlikely with greedy stop, but just in case), remove random
        if len(edges) > target_edges:
            to_remove = len(edges) - target_edges
            edge_list = list(edges)
            random.shuffle(edge_list)
            for _ in range(to_remove):
                e = edge_list.pop()
                edges.discard(e)
                u, v = e
                neighbors[u].discard(v)
                neighbors[v].discard(u)
                
        return neighbors, edges

    def local_search(neighbors_init, edges_init, time_limit):
        """
        Run Simulated Annealing to minimize C4 count.
        """
        neighbors = [set(s) for s in neighbors_init]
        edges = set(edges_init)
        
        current_score = count_c4(neighbors)
        best_score = current_score
        best_neighbors = [set(s) for s in neighbors]
        best_edges = set(edges)
        
        T = 1.0
        cooling = 0.999
        min_T = 0.001
        
        while time.time() - start_time < time_limit and T > min_T:
            # Pick a random edge to remove
            if not edges:
                break
            e_rm = random.choice(list(edges))
            u_rm, v_rm = e_rm
            
            # Pick a random non-edge to add
            # To be efficient, we can pick a random pair and check if it's an edge
            # Or maintain a list of non-edges. For n=31, 465 total edges, 91 in graph.
            # 374 non-edges. Picking random pair is fast enough.
            while True:
                u_add = random.randint(0, n - 1)
                v_add = random.randint(0, n - 1)
                if u_add == v_add:
                    continue
                if u_add > v_add:
                    u_add, v_add = v_add, u_add
                e_add = (u_add, v_add)
                if e_add not in edges:
                    break
                # If same as e_rm, skip (though we removed it, so it's not in edges set yet? 
                # No, we haven't modified edges set yet. So if e_add == e_rm, it IS in edges. 
                # So this check works.)
                
            # Perform swap
            edges.discard(e_rm)
            neighbors[u_rm].discard(v_rm)
            neighbors[v_rm].discard(u_rm)
            
            edges.add(e_add)
            neighbors[u_add].add(v_add)
            neighbors[v_add].add(u_add)
            
            new_score = count_c4(neighbors)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_neighbors = [set(s) for s in neighbors]
                    best_edges = set(edges)
            else:
                # Revert
                edges.discard(e_add)
                neighbors[u_add].discard(v_add)
                neighbors[v_add].discard(u_add)
                
                edges.add(e_rm)
                neighbors[u_rm].add(v_rm)
                neighbors[v_rm].add(u_rm)
            
            T *= cooling
            
        return best_score, best_neighbors, best_edges

    best_overall_score = float('inf')
    best_overall_edges = None
    
    # Run several restarts
    while time.time() - start_time < deadline - 2:
        init_neighbors, init_edges = generate_initial_graph()
        score, neighbors, edges = local_search(init_neighbors, init_edges, deadline - 2)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_edges = edges
            if best_overall_score == 1:
                break

    if best_overall_edges is None:
        # Fallback
        init_neighbors, init_edges = generate_initial_graph()
        best_overall_edges = init_edges

    return [[u, v] for u, v in best_overall_edges]
