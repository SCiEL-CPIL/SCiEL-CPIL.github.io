import random
import time
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable, False otherwise.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return True
    
    m = len(edges)
    
    # Precompute edge indices for each vertex
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    # For very small n, brute force is faster
    if n <= 12:
        # Bitmask brute force
        # Represent color of vertex v as bit v in mask
        # Edge is monochromatic if (mask & edge_mask) == 0 or (mask & edge_mask) == edge_mask
        edge_masks = [0] * m
        for i, e in enumerate(edges):
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks[i] = mask
        
        for mask in range(1 << n):
            valid = True
            for em in edge_masks:
                # Check if em is subset of mask or disjoint from mask
                if (mask & em) == 0 or (mask & em) == em:
                    valid = False
                    break
            if valid:
                return True
        return False

    # For larger n, use backtracking
    colors = [-1] * n
    edge_counts = [[0, 0] for _ in range(m)]
    
    # Vertex ordering: sort by degree descending
    degrees = [len(vertex_edges[v]) for v in range(n)]
    vertex_order = sorted(range(n), key=lambda v: degrees[v], reverse=True)
    
    def propagate(start_idx=0):
        """
        Perform unit propagation.
        Returns:
        - True if no conflict and propagation complete
        - False if conflict
        - (False, forced_vertex, forced_color) if a new assignment is forced? 
        Actually, let's just do a loop until stable.
        """
        changed = True
        while changed:
            changed = False
            for i in range(m):
                c0, c1 = edge_counts[i]
                if c0 == 5 or c1 == 5:
                    return False
                if c0 == 4 and c1 == 0:
                    # Force the uncolored vertex to 1
                    for v in edges[i]:
                        if colors[v] == -1:
                            colors[v] = 1
                            # Update counts for edges containing v
                            for ei in vertex_edges[v]:
                                edge_counts[ei][1] += 1
                            changed = True
                            break
                elif c1 == 4 and c0 == 0:
                    # Force the uncolored vertex to 0
                    for v in edges[i]:
                        if colors[v] == -1:
                            colors[v] = 0
                            for ei in vertex_edges[v]:
                                edge_counts[ei][0] += 1
                            changed = True
                            break
        return True

    def backtrack():
        # Check for conflicts first (should be handled by propagate, but safe)
        for i in range(m):
            if edge_counts[i][0] == 5 or edge_counts[i][1] == 5:
                return False
        
        # Find unassigned vertex
        target_vertex = -1
        for v in vertex_order:
            if colors[v] == -1:
                target_vertex = v
                break
        
        if target_vertex == -1:
            return True
        
        # Try color 0
        colors[target_vertex] = 0
        affected_edges = vertex_edges[target_vertex]
        saved_counts = []
        for i in affected_edges:
            saved_counts.append((edge_counts[i][0], edge_counts[i][1]))
            edge_counts[i][0] += 1
        
        if propagate():
            if backtrack():
                return True
        
        # Undo
        colors[target_vertex] = -1
        for i, (sc0, sc1) in zip(affected_edges, saved_counts):
            edge_counts[i][0] = sc0
            edge_counts[i][1] = sc1
            
        # Try color 1
        colors[target_vertex] = 1
        saved_counts = []
        for i in affected_edges:
            saved_counts.append((edge_counts[i][0], edge_counts[i][1]))
            edge_counts[i][1] += 1
            
        if propagate():
            if backtrack():
                return True
                
        # Undo
        colors[target_vertex] = -1
        for i, (sc0, sc1) in zip(affected_edges, saved_counts):
            edge_counts[i][0] = sc0
            edge_counts[i][1] = sc1
            
        return False

    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for 120s limit
    
    best_edges = None
    best_n = 0
    best_m = 201
    
    # Candidate n values
    # Smaller n has fewer possible edges, easier to search
    # But must be large enough to support a non-2-colorable hypergraph with few edges
    # m(5) lower bound is 29.
    # n=9: C(9,5)=126.
    # n=10: C(10,5)=252.
    # n=11: C(11,5)=462.
    # n=12: C(12,5)=792.
    
    for n_try in [10, 11, 12, 13]:
        if time.time() > deadline - 10:
            break
            
        all_possible_edges = list(combinations(range(n_try), 5))
        all_possible_edges = [list(e) for e in all_possible_edges]
        num_possible = len(all_possible_edges)
        
        # Simulated Annealing / Random Walk
        # Start with a random set of edges
        # We want to find a non-2-colorable set with minimal size
        
        # Initialize state
        # Start with m=40
        current_m = 40
        current_indices = random.sample(range(num_possible), current_m)
        current_edges = [all_possible_edges[i] for i in current_indices]
        
        # Check initial state
        is_non_2c = not is_2_colorable(n_try, current_edges)
        
        if is_non_2c and current_m < best_m:
            best_m = current_m
            best_n = n_try
            best_edges = current_edges[:]
        
        temp = 1.0
        cooling_rate = 0.995
        
        while time.time() < deadline - 5:
            # Choose a move: add or remove an edge
            # If non-2-colorable, prefer removal
            # If 2-colorable, prefer addition
            
            if is_non_2c:
                # Try to remove
                if current_m > 25: # Don't go below known lower bound
                    idx_to_remove = random.choice(range(current_m))
                    new_indices = current_indices[:idx_to_remove] + current_indices[idx_to_remove+1:]
                    new_edges = [all_possible_edges[i] for i in new_indices]
                    
                    new_is_non_2c = not is_2_colorable(n_try, new_edges)
                    
                    if new_is_non_2c:
                        # Accept
                        current_indices = new_indices
                        current_edges = new_edges
                        current_m -= 1
                        is_non_2c = True
                        
                        if current_m < best_m:
                            best_m = current_m
                            best_n = n_try
                            best_edges = current_edges[:]
                    else:
                        # Reject (or accept with low prob, but for minimization, reject is better)
                        pass
                else:
                    # Can't remove more, try to add? No, we want min.
                    # Maybe restart or change n
                    pass
            else:
                # Try to add
                # Pick a random edge not in current set
                # To be efficient, pick a random index from all_possible
                # and check if it's already there.
                # For small num_possible, this is fine.
                
                # Optimization: Maintain a set of current indices for O(1) check
                # But list is easier. For num_possible ~ 250-800, linear check is ok.
                
                attempts = 0
                added = False
                while attempts < 20:
                    cand_idx = random.randint(0, num_possible - 1)
                    if cand_idx not in current_indices:
                        new_indices = current_indices + [cand_idx]
                        new_edges = [all_possible_edges[i] for i in new_indices]
                        
                        new_is_non_2c = not is_2_colorable(n_try, new_edges)
                        
                        if new_is_non_2c:
                            current_indices = new_indices
                            current_edges = new_edges
                            current_m += 1
                            is_non_2c = True
                            added = True
                            
                            # Update best if needed (though we are adding, so m increases)
                            # But this new state is a seed for further removal
                            break
                        else:
                            # Still 2-colorable. Accept with probability?
                            # To explore, we might accept even if it doesn't change state type
                            # But we want to reach non-2c state.
                            # Let's just reject and try another edge.
                            pass
                    attempts += 1
                
                if not added:
                    # Failed to add an edge that makes it non-2c?
                    # This shouldn't happen if we add enough edges.
                    # If we are stuck, maybe increase temp or restart
                    pass
            
            temp *= cooling_rate
            if temp < 0.1:
                # Restart with new random seed for this n
                break

    # Fallback: if no good result, return a known valid one or best found
    if best_edges is None:
        # Fallback to a simple construction that is definitely non-2-colorable
        # K9^5 is non-2-colorable.
        n_fb = 9
        edges_fb = [list(e) for e in combinations(range(n_fb), 5)]
        return {'n': n_fb, 'edges': edges_fb}
        
    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
    # Verify
    print("Verifying...")
    assert not is_2_colorable(res['n'], res['edges']), "Result is 2-colorable!"
    print("Valid.")
