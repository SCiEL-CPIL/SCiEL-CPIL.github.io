import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 20
    
    def is_valid_addition(adj, u, v):
        """
        Checks if adding edge (u, v) violates the K3,3-free condition.
        Condition: For any triple T, |N(T)| <= 2.
        """
        if adj[u] & (1 << v):
            return False
            
        for a in range(n):
            if a == u or a == v:
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                
                if (adj[a] & (1 << v)) and (adj[b] & (1 << v)):
                    count = (adj[u] & adj[a] & adj[b]).bit_count()
                    if count > 1:
                        return False
                        
                if (adj[a] & (1 << u)) and (adj[b] & (1 << u)):
                    count = (adj[v] & adj[a] & adj[b]).bit_count()
                    if count > 1:
                        return False
                        
        return True

    def count_edges(adj):
        return sum(x.bit_count() for x in adj) // 2

    def find_violation(adj):
        """
        Find a K3,3 violation: three vertices with three common neighbors.
        Returns (triple, common_neighbors) or None.
        """
        for i in range(n):
            for j in range(i+1, n):
                common_ij = adj[i] & adj[j]
                # For each pair (k, l) where k,l are in common_ij, check if k and l share 3+ common neighbors with i,j
                # Actually, we need: find 3 vertices a,b,c such that |N(a)∩N(b)∩N(c)| >= 3
                # Equivalent: find 3 neighbors x,y,z of some triple
                pass
        
        # Direct check: for each triple (i,j,k), check if they have 3+ common neighbors
        for i in range(n):
            for j in range(i+1, n):
                common_ij = adj[i] & adj[j]
                if common_ij.bit_count() < 3:
                    continue
                for k in range(j+1, n):
                    common_ijk = common_ij & adj[k]
                    if common_ijk.bit_count() >= 3:
                        # Found violation: triple (i,j,k) has 3+ common neighbors
                        # The common neighbors are in common_ijk
                        return (i, j, k), common_ijk
        return None

    def repair_graph(adj):
        """
        Iteratively remove edges to eliminate all K3,3 violations.
        """
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            result = find_violation(adj)
            if result is None:
                break
            (i, j, k), common_neighbors = result
            # We need to remove an edge to break this violation.
            # The violation is that vertices i,j,k have 3+ common neighbors.
            # To break it, we can remove an edge from one of {i,j,k} to one of the common neighbors.
            # Strategy: remove the edge that participates in the most violations.
            # For simplicity, pick a random common neighbor and remove edge (i, cn)
            cn_list = [v for v in range(n) if common_neighbors & (1 << v)]
            random.shuffle(cn_list)
            cn = cn_list[0]
            # Remove edge (i, cn)
            adj[i] &= ~(1 << cn)
            adj[cn] &= ~(1 << i)
            changed = True
            
        return adj

    def local_search(adj, max_steps=500):
        current_adj = adj[:]
        current_edges = count_edges(current_adj)
        
        # Phase 1: Try to add edges in random order
        pairs = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                break
            if is_valid_addition(current_adj, u, v):
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges += 1
                
        # Phase 2: Edge swap local search
        improved = True
        swap_iterations = 0
        max_swap_iterations = 200
        while improved and swap_iterations < max_swap_iterations:
            if time.time() > deadline:
                break
            improved = False
            swap_iterations += 1
            
            existing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if current_adj[i] & (1 << j)]
            missing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
            random.shuffle(existing_edges)
            random.shuffle(missing_edges)
            
            for u, v in existing_edges:
                if time.time() > deadline:
                    break
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                
                added_count = 0
                for a, b in missing_edges:
                    if time.time() > deadline:
                        break
                    if (a == u or a == v or b == u or b == v):
                        continue
                    if is_valid_addition(current_adj, a, b):
                        current_adj[a] |= (1 << b)
                        current_adj[b] |= (1 << a)
                        added_count += 1
                        if added_count >= 2:
                            break
                
                if added_count >= 1:
                    current_edges += added_count - 1
                    improved = True
                    missing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
                else:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    
        # Phase 3: Final pass
        pairs = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                break
            if is_valid_addition(current_adj, u, v):
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges += 1
                
        return current_adj, current_edges

    best_adj = [0] * n
    best_edges = 0
    
    num_restarts = 500
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.seed(i)
        
        # Probabilistic method: start with dense random graph
        p = 0.35
        adj = [0] * n
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Repair: remove edges to eliminate K3,3 violations
        adj = repair_graph(adj)
        
        edges = count_edges(adj)
        
        # Local search to add back edges
        adj, edges = local_search(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
    # Additional polish
    if time.time() < deadline - 5:
        adj, edges = local_search(best_adj)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj
    
    if time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            adj, edges = local_search(best_adj)
            if edges > best_edges:
                best_edges = edges
                best_adj = adj
            else:
                break
            
    # Convert bitmask to list of edges
    result_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result_edges.append([i, j])
                
    return result_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
