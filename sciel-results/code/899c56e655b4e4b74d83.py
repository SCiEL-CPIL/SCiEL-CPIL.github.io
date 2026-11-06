import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = 20
    # Precompute all triples of vertices
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))
    
    # Total possible edges
    total_pairs = n * (n - 1) // 2
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def is_valid(adj):
        """
        adj: list of integers representing adjacency bitmasks.
        Returns True if graph is K3,3-free.
        A K3,3 exists if there are 3 vertices {a,b,c} with at least 3 common neighbors.
        """
        for i, j, k in triples:
            common = adj[i] & adj[j] & adj[k]
            # If there are at least 3 common neighbors, we can form K3,3
            # by picking any 3 from common.
            if common.bit_count() >= 3:
                return False
        return True

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def get_violating_triple(adj):
        """
        Returns a triple (i, j, k) such that common neighbors >= 3, or None.
        Also returns the set of common neighbors as a bitmask.
        """
        for i, j, k in triples:
            common = adj[i] & adj[j] & adj[k]
            if common.bit_count() >= 3:
                return (i, j, k, common)
        return None

    def repair(adj):
        """
        Removes edges until the graph is valid.
        Uses a heuristic: remove an edge incident to one of the vertices in the violating triple
        and one of the common neighbors.
        """
        max_iter = 100
        while max_iter > 0:
            max_iter -= 1
            v = get_violating_triple(adj)
            if v is None:
                break
            i, j, k, common = v
            # We need to break the K3,3 formed by {i,j,k} and any 3 nodes in 'common'.
            # Let's pick a node m from common and remove edge (i, m).
            # To be more effective, we can pick the edge that appears in the most violations,
            # but that's expensive. Random choice from the conflict is usually sufficient.
            
            # Get list of nodes in common
            candidates = []
            mask = common
            while mask:
                # Get lowest set bit
                lsb = mask & -mask
                m = lsb.bit_length() - 1
                candidates.append(m)
                mask ^= lsb
            
            if not candidates:
                break
                
            m = random.choice(candidates)
            # Remove edge (i, m)
            if adj[i] & (1 << m):
                adj[i] &= ~(1 << m)
                adj[m] &= ~(1 << i)
            else:
                # Should not happen, but safety
                break
                
        return adj

    def try_add_edge(adj, u, v):
        """
        Try to add edge (u, v). Returns True if added and valid.
        """
        if u == v:
            return False
        if adj[u] & (1 << v):
            return True  # Already there
        
        # Add edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        if is_valid(adj):
            return True
        else:
            # Revert
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            return False

    def greedy_fill(adj):
        """
        Try to add all possible missing edges.
        """
        changed = True
        while changed:
            changed = False
            for u, v in all_pairs:
                if not (adj[u] & (1 << v)):
                    if try_add_edge(adj, u, v):
                        changed = True
        return adj

    def local_search(adj, time_limit):
        """
        Perform local search: try edge swaps.
        If an edge addition fails, try removing a random edge and adding a different one.
        """
        current_edges = count_edges(adj)
        best_adj = [x for x in adj]
        best_edges = current_edges
        
        no_improve_count = 0
        max_iterations = 5000
        iter_count = 0
        
        while iter_count < max_iterations:
            if time.time() > time_limit:
                break
            iter_count += 1
            
            # Try to add a random non-edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            if adj[u] & (1 << v):
                continue
            
            # Try adding it
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if is_valid(adj):
                current_edges += 1
                no_improve_count = 0
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = [x for x in adj]
            else:
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                no_improve_count += 1
                
                # If stuck for a while, try a swap: remove a random edge, then try to add
                if no_improve_count > 20:
                    # Pick a random existing edge to remove
                    removed_edge = None
                    for _ in range(10): # Try to find an edge
                        ru = random.randint(0, n - 1)
                        if adj[ru] == 0:
                            continue
                        neighbors = []
                        mask = adj[ru]
                        while mask:
                            lsb = mask & -mask
                            rv = lsb.bit_length() - 1
                            neighbors.append(rv)
                            mask ^= lsb
                        if neighbors:
                            rv = random.choice(neighbors)
                            # Remove edge (ru, rv)
                            adj[ru] &= ~(1 << rv)
                            adj[rv] &= ~(1 << ru)
                            current_edges -= 1
                            removed_edge = (ru, rv)
                            break
                    
                    if removed_edge:
                        # Now try to add a random non-edge
                        added = False
                        for _ in range(10):
                            au = random.randint(0, n - 1)
                            av = random.randint(0, n - 1)
                            if au == av:
                                continue
                            if adj[au] & (1 << av):
                                continue
                            
                            adj[au] |= (1 << av)
                            adj[av] |= (1 << au)
                            if is_valid(adj):
                                current_edges += 1
                                added = True
                                if current_edges > best_edges:
                                    best_edges = current_edges
                                    best_adj = [x for x in adj]
                                break
                            else:
                                adj[au] &= ~(1 << av)
                                adj[av] &= ~(1 << au)
                        
                        if not added:
                            # Put the removed edge back if we didn't improve
                            ru, rv = removed_edge
                            adj[ru] |= (1 << rv)
                            adj[rv] |= (1 << ru)
                            current_edges += 1
                    
                    no_improve_count = 0
                    
        return best_adj, best_edges

    best_overall_edges = 0
    best_overall_adj = None
    
    # Try multiple random starts
    # We can use different initialization strategies
    for seed in range(100):
        if time.time() > deadline - 5.0:
            break
            
        random.seed(seed)
        
        # Initialization: Random graph with p=0.5, then repair
        # Or start with a bipartite graph
        # Let's mix: 50% random, 50% bipartite-like
        if seed % 2 == 0:
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        else:
            # Bipartite initialization: split into two sets of 10
            adj = [0] * n
            for i in range(10):
                for j in range(10, 20):
                    if random.random() < 0.8:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        
        # Repair
        adj = repair(adj)
        
        # Greedy fill
        adj = greedy_fill(adj)
        
        # Local search
        # Allocate remaining time roughly evenly, but cap per iteration
        # We have ~140s total. 100 seeds -> 1.4s per seed roughly?
        # But some seeds might be faster.
        # Let's just run local search until time runs out for this seed, or a small limit.
        # Actually, it's better to do a fixed number of iterations or a time slice.
        # Let's do a time slice of 1.0s for local search per seed if we have many seeds.
        # Or just let it run until global deadline.
        
        adj, edges = local_search(adj, deadline - 2.0)
        
        if edges > best_overall_edges:
            best_overall_edges = edges
            best_overall_adj = [x for x in adj]
            
        # If we found a very good solution, we can spend more time polishing it
        # But for now, we just keep the best.

    # Final polish on the best solution
    if best_overall_adj:
        best_overall_adj, best_overall_edges = local_search(best_overall_adj, deadline - 1.0)
        # One more greedy pass
        best_overall_adj = greedy_fill(best_overall_adj)
        best_overall_edges = count_edges(best_overall_adj)

    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_overall_adj[i] & (1 << j):
                edges.append([i, j])
                
    return edges
