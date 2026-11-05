import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 120.0
    n = 31
    target_edges = 90  # We build a C4-free graph with 90 edges first
    
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def count_c4(neighbors):
        """Count 4-cycles in graph defined by neighbors list of sets."""
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

    best_c4_count = float('inf')
    best_edges = None

    # We want to find a C4-free graph with exactly 90 edges.
    # Then add 1 edge that creates the minimum number of C4s.
    
    attempts = 0
    while time.time() - start_time < deadline - 5:
        attempts += 1
        
        # 1. Generate a maximal C4-free graph
        random.shuffle(all_edges)
        neighbors = [set() for _ in range(n)]
        edges = set()
        
        for e in all_edges:
            if add_edge_if_no_c4(neighbors, e):
                edges.add(e)
        
        # Check number of edges
        if len(edges) < target_edges:
            continue
            
        # 2. If we have more than 90 edges, remove random ones until we have exactly 90
        # Removing edges from a C4-free graph keeps it C4-free
        if len(edges) > target_edges:
            # Remove random edges
            edge_list = list(edges)
            random.shuffle(edge_list)
            to_remove = len(edges) - target_edges
            for _ in range(to_remove):
                e = edge_list.pop()
                edges.discard(e)
                u, v = e
                neighbors[u].discard(v)
                neighbors[v].discard(u)
                
        # Now we have a C4-free graph with exactly 90 edges
        # 3. Find the non-edge that creates the fewest C4s when added
        current_non_edges = []
        for e in all_edges:
            if e not in edges:
                current_non_edges.append(e)
                
        if not current_non_edges:
            continue
            
        best_add_edge = None
        min_c4_created = float('inf')
        
        for e in current_non_edges:
            u, v = e
            common = neighbors[u].intersection(neighbors[v])
            # Number of C4s created is C(common, 2)
            c4s = len(common) * (len(common) - 1) // 2
            if c4s < min_c4_created:
                min_c4_created = c4s
                best_add_edge = e
                if min_c4_created == 1:
                    break # Can't do better than 1
            
        if best_add_edge is None:
            continue
            
        # Add the best edge
        u, v = best_add_edge
        neighbors[u].add(v)
        neighbors[v].add(u)
        edges.add(best_add_edge)
        
        # Verify count
        final_c4 = count_c4(neighbors)
        
        if final_c4 < best_c4_count:
            best_c4_count = final_c4
            best_edges = set(edges)
            
            if best_c4_count == 1:
                # Found optimal solution (since ex(n,C4)=90, min C4s for 91 edges is >= 1)
                break

    # If we didn't find a good one, return the best we have or a fallback
    if best_edges is None:
        # Fallback: random 91 edges
        random.shuffle(all_edges)
        best_edges = set(all_edges[:91])
        
    return [[u, v] for u, v in best_edges]
