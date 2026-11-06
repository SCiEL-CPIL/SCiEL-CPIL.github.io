import time
import random
import sys

def construct():
    """
    Constructs a simple graph on 20 vertices with edge colours 0..19.
    Maximizes the number of edges while ensuring every 4-cycle is rainbow.
    
    Strategy:
    1. Build a dense C4-free graph using randomized greedy construction.
    2. Use Simulated Annealing to refine and improve edge count.
    3. Since the graph is C4-free, any edge colouring is valid (no 4-cycles to check).
    4. Return edges all coloured 0 (or use varied colours for aesthetics).
    """
    n = 20
    deadline = time.time() + 115  # safety margin before 120s limit
    
    def is_c4_free(adj):
        """Check if graph is C4-free: no two vertices share more than 1 common neighbor."""
        for u in range(n):
            for v in range(u + 1, n):
                if len(adj[u] & adj[v]) > 1:
                    return False
        return True
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph C4-free."""
        # 1. u and v must have at most 1 common neighbor currently
        if len(adj[u] & adj[v]) > 1:
            return False
        
        # 2. For each neighbor x of v: v becomes common neighbor of u and x
        #    So N(u) and N(x) must be disjoint (before adding)
        for x in adj[v]:
            if x == u:
                continue
            if adj[u] & adj[x]:  # non-empty intersection
                return False
        
        # 3. For each neighbor x of u: u becomes common neighbor of v and x
        #    So N(v) and N(x) must be disjoint
        for x in adj[u]:
            if x == v:
                continue
            if adj[v] & adj[x]:  # non-empty intersection
                return False
        
        return True
    
    def try_greedy(seed, max_time=None):
        """Randomized greedy construction."""
        random.seed(seed)
        adj = [set() for _ in range(n)]
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if max_time and time.time() > max_time:
                break
            if can_add_edge(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
        
        return adj
    
    def count_edges(adj):
        """Count edges in adjacency list."""
        return sum(len(adj[u]) for u in range(n)) // 2
    
    def simulated_annealing(start_adj, time_budget):
        """Refine a C4-free graph using simulated annealing."""
        current_adj = [set(s) for s in start_adj]
        current_count = count_edges(current_adj)
        best_adj = [set(s) for s in current_adj]
        best_count = current_count
        
        temperature = 1.0
        cooling_rate = 0.999
        end_time = time.time() + time_budget
        
        while time.time() < end_time:
            if temperature < 0.01:
                break
            
            # Generate a random move: remove one edge and try to add a different one
            # Or: swap an edge (remove one, add another)
            edges = [(u, v) for u in range(n) for v in current_adj[u] if u < v]
            if not edges:
                break
            
            # Randomly remove an edge
            u_remove, v_remove = random.choice(edges)
            new_adj = [set(s) for s in current_adj]
            new_adj[u_remove].discard(v_remove)
            new_adj[v_remove].discard(u_remove)
            
            # Try to add a random non-edge
            all_possible = [(u, v) for u in range(n) for v in range(u + 1, n)]
            non_edges = [(u, v) for u, v in all_possible if v not in new_adj[u]]
            
            if non_edges:
                u_add, v_add = random.choice(non_edges)
                if can_add_edge(new_adj, u_add, v_add):
                    new_adj[u_add].add(v_add)
                    new_adj[v_add].add(u_add)
            
            new_count = count_edges(new_adj)
            
            # Acceptance criterion
            if new_count > current_count:
                current_adj = new_adj
                current_count = new_count
                if current_count > best_count:
                    best_count = current_count
                    best_adj = [set(s) for s in current_adj]
            elif new_count == current_count:
                # Accept equal moves with probability based on temperature
                if random.random() < 0.5:
                    current_adj = new_adj
            else:
                # Accept worse moves with probability exp(-(delta)/T)
                delta = current_count - new_count
                if random.random() < 0.5 ** delta:  # simplified acceptance
                    current_adj = new_adj
            
            temperature *= cooling_rate
        
        return best_adj, best_count
    
    def local_search(adj, time_budget):
        """Iterative local search: try to add any missing edge that fits."""
        current_adj = [set(s) for s in adj]
        current_count = count_edges(current_adj)
        best_adj = [set(s) for s in current_adj]
        best_count = current_count
        
        end_time = time.time() + time_budget
        improved = True
        
        while improved and time.time() < end_time:
            improved = False
            all_possible = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(all_possible)
            
            for u, v in all_possible:
                if time.time() > end_time:
                    break
                if v not in current_adj[u]:  # non-edge
                    if can_add_edge(current_adj, u, v):
                        current_adj[u].add(v)
                        current_adj[v].add(u)
                        current_count += 1
                        improved = True
                        if current_count > best_count:
                            best_count = current_count
                            best_adj = [set(s) for s in current_adj]
        
        return best_adj, best_count
    
    # Main construction
    best_adj = None
    best_count = 0
    
    # Phase 1: Multiple greedy restarts
    num_restarts = 20
    time_per_restart = 2.0
    
    for i in range(num_restarts):
        if time.time() > deadline - 30:  # leave time for SA
            break
        adj = try_greedy(i, deadline - 30)
        count = count_edges(adj)
        if count > best_count:
            best_count = count
            best_adj = adj
    
    # Phase 2: Simulated annealing on best greedy result
    if best_adj and time.time() < deadline - 15:
        sa_budget = min(60, deadline - 15 - time.time())
        sa_adj, sa_count = simulated_annealing(best_adj, sa_budget)
        if sa_count > best_count:
            best_count = sa_count
            best_adj = sa_adj
    
    # Phase 3: Local search to squeeze out more edges
    if best_adj and time.time() < deadline - 5:
        ls_budget = min(20, deadline - 5 - time.time())
        ls_adj, ls_count = local_search(best_adj, ls_budget)
        if ls_count > best_count:
            best_count = ls_count
            best_adj = ls_adj
    
    # Final verification
    if best_adj is None or not is_c4_free(best_adj):
        # Fallback: star graph (always valid)
        return [[0, v, 0] for v in range(1, n)]
    
    # Extract edges and assign colours
    edges = [(u, v) for u in range(n) for v in best_adj[u] if u < v]
    
    # Assign colours: since graph is C4-free, any colouring works.
    # Use varied colours for aesthetic purposes, but it's not required.
    coloured_edges = []
    for idx, (u, v) in enumerate(edges):
        c = idx % 20  # cycle through colours 0-19
        coloured_edges.append([u, v, c])
    
    return coloured_edges

# For testing
if __name__ == "__main__":
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    print(f"Time: {elapsed:.2f}s, Edges: {len(result)}")
    
    # Quick validity check
    n = 20
    adj = [set() for _ in range(n)]
    for u, v, c in result:
        adj[u].add(v)
        adj[v].add(u)
    valid = True
    for u in range(n):
        for v in range(u + 1, n):
            if len(adj[u] & adj[v]) > 1:
                valid = False
                break
    print(f"Valid C4-free: {valid}")
