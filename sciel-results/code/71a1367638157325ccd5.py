import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def count_violations(adj):
        """Count the number of triples (i,j,k) with > 2 common neighbors."""
        v_count = 0
        for i in range(n):
            ai = adj[i]
            for j in range(i+1, n):
                aj = adj[j]
                common_ij = ai & aj
                if common_ij.bit_count() < 3:
                    continue
                for k in range(j+1, n):
                    if (common_ij & adj[k]).bit_count() > 2:
                        v_count += 1
        return v_count

    def get_edge_count(adj):
        return sum(x.bit_count() for x in adj) // 2

    def repair(adj):
        """Remove edges until the graph is K3,3-free."""
        # Safety limit to prevent infinite loops in case of bug
        max_iter = 5000
        while count_violations(adj) > 0 and max_iter > 0:
            max_iter -= 1
            found = False
            # Find a violating triple
            for i in range(n):
                ai = adj[i]
                for j in range(i+1, n):
                    aj = adj[j]
                    common_ij = ai & aj
                    for k in range(j+1, n):
                        if (common_ij & adj[k]).bit_count() > 2:
                            # Violation at i, j, k
                            C = common_ij & adj[k]
                            # Get list of common neighbors
                            c_list = [b for b in range(n) if C & (1 << b)]
                            if not c_list:
                                continue
                            c = random.choice(c_list)
                            # Remove edge between c and one of i, j, k
                            # Removing any edge incident to c and one of the triple vertices
                            # will reduce the common neighbor count for this triple.
                            # It cannot increase violations elsewhere.
                            v_to_remove = random.choice([i, j, k])
                            adj[v_to_remove] &= ~(1 << c)
                            adj[c] &= ~(1 << v_to_remove)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
        return adj

    def simulated_annealing(adj, max_time):
        """Run Simulated Annealing to maximize edges while keeping graph K3,3-free."""
        sa_start = time.time()
        T = 1.0
        P = 10.0  # Penalty for violations
        
        current_adj = adj[:]
        current_edges = get_edge_count(current_adj)
        current_violations = count_violations(current_adj)
        
        best_adj = current_adj[:]
        best_edges = current_edges
        best_violations = current_violations
        
        # If initial is invalid, repair it first
        if current_violations > 0:
            current_adj = repair(current_adj)
            current_edges = get_edge_count(current_adj)
            current_violations = count_violations(current_adj)
            best_adj = current_adj[:]
            best_edges = current_edges
            best_violations = current_violations
            
        while time.time() - sa_start < max_time:
            # Annealing schedule
            T = max(0.01, T * 0.995)
            # Decrease penalty over time to allow more edges, but keep it high enough to discourage violations
            # Actually, we only accept valid states for the "best" tracking, 
            # but SA can wander through invalid states if penalty is low.
            # Let's keep P constant or slowly decreasing.
            P = max(0.5, P * 0.999)
            
            # Propose a move: Add a random missing edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if current_adj[u] & (1 << v):
                # Try removing an edge instead? 
                # Usually adding is the goal. Removing is only for escape.
                # Let's stick to adding for simplicity, as repair handles removal.
                continue
            
            # Tentatively add
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            
            new_edges = current_edges + 1
            new_violations = count_violations(current_adj)
            
            delta_edges = 1
            delta_violations = new_violations - current_violations
            delta_score = delta_edges - P * delta_violations
            
            # Acceptance criterion
            if delta_score > 0:
                accept = True
            else:
                # Metropolis criterion
                # Avoid overflow
                if delta_score < -50:
                    accept = False
                else:
                    accept = random.random() < np.exp(delta_score / T)
            
            if accept:
                current_edges = new_edges
                current_violations = new_violations
                
                # Update best if valid
                if new_violations == 0 and new_edges > best_edges:
                    best_edges = new_edges
                    best_adj = current_adj[:]
                    best_violations = 0
            else:
                # Revert
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                
        return best_adj, best_edges

    # Main search loop
    best_adj = [0] * n
    best_edges = 0
    
    # Run multiple restarts
    num_restarts = 20
    time_per_restart = (deadline - start_time - 5) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize random graph
        adj = [0] * n
        density = random.uniform(0.6, 0.8)
        for u, v in all_edges:
            if random.random() < density:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Repair to valid
        adj = repair(adj)
        edges = get_edge_count(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
        # Run SA
        remaining_time = deadline - time.time() - 2
        sa_time = max(0.1, remaining_time / (num_restarts - r))
        adj, edges = simulated_annealing(adj, sa_time)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
    # Final polish: try to add any remaining edges greedily to best_adj
    if time.time() < deadline - 1:
        # Simple greedy fill on best_adj
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (best_adj[i] & (1 << j))]
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline - 1:
                break
            best_adj[u] |= (1 << v)
            best_adj[v] |= (1 << u)
            if count_violations(best_adj) > 0:
                # Revert
                best_adj[u] &= ~(1 << v)
                best_adj[v] &= ~(1 << u)
            else:
                best_edges += 1

    # Convert to list of lists
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
                
    return result

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
