import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    def _elapsed():
        return time.time() - start_time

    def is_2_colorable_fast(n, edges):
        """
        Check if hypergraph is 2-colorable using backtracking with unit propagation.
        Returns True if colorable, False if not.
        """
        if not edges:
            return True
        
        # Convert edges to tuples of indices for fast lookup
        # We use a simple backtracking search.
        # State: assignment array of -1 (unassigned), 0, 1.
        
        # Optimization: Order vertices by degree? Or just use indices 0..n-1.
        # For sparse graphs, simple backtracking is often fast enough if we pick the next vertex
        # that is constrained by the most assigned neighbors.
        
        # Let's use a recursive helper.
        # To avoid recursion depth issues and overhead, we can use an explicit stack or
        # just rely on Python's recursion limit (which is usually 1000, and n<=26 so depth <= 26).
        
        assignments = [-1] * n
        edge_adj = [[] for _ in range(n)]
        for idx, e in enumerate(edges):
            for v in e:
                edge_adj[v].append(idx)
        
        # Precompute edge masks?
        # Let's stick to the list of edges and check each edge when a vertex is assigned.
        # Actually, checking all edges every time is O(E).
        # Better: maintain a count of how many vertices in each edge are 0, 1, or unassigned.
        
        count_0 = [0] * len(edges)
        count_1 = [0] * len(edges)
        count_unassigned = [len(e) for e in edges]
        
        # Unit propagation:
        # If count_0 == 5, edge is satisfied (monochromatic 0) -> Wait, we want to AVOID monochromatic.
        # The condition for "valid coloring" is that NO edge is monochromatic.
        # So if an edge becomes monochromatic, the current branch is INVALID.
        # Unit propagation in this context:
        # If an edge has 4 vertices of color 0 and 1 unassigned, the unassigned vertex MUST be 1?
        # No. If it is 0, the edge is mono-0 (invalid). So it MUST be 1.
        # Similarly, if 4 are 1, the last must be 0.
        
        def backtrack(idx):
            # idx is the index of the next vertex to assign?
            # It's better to pick the vertex with the most constraints.
            # But for simplicity and speed in Python, picking the first unassigned vertex is okay if n is small.
            # However, "most constrained" is much faster.
            
            # Find the unassigned vertex with the most assigned neighbors in its edges?
            # Or just iterate 0..n-1 and find the first unassigned.
            # Let's try the "first unassigned" strategy first, it's simpler.
            # If it's too slow, we can switch to MRV (Minimum Remaining Values) or similar.
            
            v = -1
            for i in range(n):
                if assignments[i] == -1:
                    v = i
                    break
            
            if v == -1:
                # All assigned. Check if any edge is monochromatic.
                # If we maintained counts, we can just check if any edge has count_0==5 or count_1==5.
                # But we check for validity at each step.
                # If we reach the end, it means no edge became monochromatic during the process?
                # Yes, because we prune branches where an edge becomes monochromatic.
                return True
            
            # Try color 0
            # Check if assigning 0 to v makes any edge monochromatic.
            valid_0 = True
            valid_1 = True
            
            # We need to update counts for edges containing v.
            # Let's just simulate the assignment and check.
            
            # To make it fast, we check edges containing v.
            for e_idx in edge_adj[v]:
                e = edges[e_idx]
                # Count colors in e excluding v
                c0 = 0
                c1 = 0
                for u in e:
                    if u == v:
                        continue
                    if assignments[u] == 0:
                        c0 += 1
                    elif assignments[u] == 1:
                        c1 += 1
                
                # If we assign 0 to v:
                if c0 == 4: # Then edge becomes 5 zeros -> mono
                    valid_0 = False
                # If we assign 1 to v:
                if c1 == 4: # Then edge becomes 5 ones -> mono
                    valid_1 = False
            
            if valid_0:
                assignments[v] = 0
                if backtrack(-1):
                    return True
                assignments[v] = -1
                
            if valid_1:
                assignments[v] = 1
                if backtrack(-1):
                    return True
                assignments[v] = -1
                
            return False
        
        return backtrack(-1)

    def greedy_set_cover(n):
        """
        Greedy set cover to find a small non-2-colorable hypergraph on n vertices.
        Returns list of edges or None if fails.
        """
        total_colorings = 1 << n
        all_edges = list(combinations(range(n), 5))
        
        # Precompute kill masks
        # kill_mask[i] has bit j set if coloring j is killed by edge i
        kill_masks = []
        for e in all_edges:
            e_mask = 0
            for v in e:
                e_mask |= (1 << v)
            km = 0
            for c in range(total_colorings):
                if (c & e_mask) == 0 or (c & e_mask) == e_mask:
                    km |= (1 << c)
            kill_masks.append(km)
            
        remaining = (1 << total_colorings) - 1
        selected_edges = []
        selected_indices = set()
        
        while remaining:
            if _elapsed() > deadline - 10:
                return None
                
            best_idx = -1
            best_count = 0
            best_new_remaining = remaining
            
            for i, km in enumerate(kill_masks):
                if i in selected_indices:
                    continue
                new_killed = remaining & km
                if new_killed:
                    cnt = new_killed.bit_count()
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = i
                        best_new_remaining = remaining & ~km
                        
            if best_idx == -1:
                return None # Should not happen for n>=9
                
            selected_edges.append(all_edges[best_idx])
            selected_indices.add(best_idx)
            remaining = best_new_remaining
            
        return selected_edges

    def local_search_delete(n, edges):
        """
        Try to remove edges while keeping the hypergraph non-2-colorable.
        """
        current_edges = edges[:]
        changed = True
        while changed and _elapsed() < deadline - 5:
            changed = False
            for i in range(len(current_edges)):
                test_edges = current_edges[:i] + current_edges[i+1:]
                # Check if still non-2-colorable
                if not is_2_colorable_fast(n, test_edges):
                    current_edges = test_edges
                    changed = True
                    break # Restart with new size
        return current_edges

    best_edges = None
    best_n = 0
    best_count = float('inf')

    # Try n from 9 to 13
    # n=9 is the smallest possible for non-2-colorable 5-uniform hypergraph?
    # Yes, as reasoned before.
    for n in range(9, 14):
        if _elapsed() > deadline - 15:
            break
            
        # Get a candidate via greedy
        cand_edges = greedy_set_cover(n)
        if cand_edges is None:
            continue
            
        # Improve via local search
        improved_edges = local_search_delete(n, cand_edges)
        
        if len(improved_edges) < best_count:
            best_count = len(improved_edges)
            best_edges = improved_edges
            best_n = n
            
        # If we find a very good one, we might stop or continue to see if smaller n works?
        # We are iterating n from 9 upwards. Smaller n is better?
        # The score is based on the number of edges.
        # Usually, smaller n allows for fewer edges?
        # Actually, m(5) is the minimum number of edges.
        # It is known that m(5) <= 51.
        # Greedy on n=9 might give a large number of edges.
        # Greedy on n=10, 11 might give fewer?
        # Let's collect the best.
        
    if best_edges is None:
        # Fallback: Complete graph on 9
        n = 9
        best_edges = list(combinations(range(n), 5))
        best_n = n

    # Final verification
    if is_2_colorable_fast(best_n, best_edges):
        # Should not happen
        n = 9
        best_edges = list(combinations(range(n), 5))
        best_n = n

    final_edges = [list(e) for e in best_edges]
    return {'n': best_n, 'edges': final_edges}

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, edges={len(res['edges'])}")
