import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Lower bound from counting: n_rows * k*(k+1)/2 differences must fit in [1, M]
    # Actually, differences are in [1, M]. 
    # Counting bound: M >= n_rows * k*(k+1)/2
    min_scope = n_rows * k * (k + 1) // 2
    
    # Greedy estimate for initial upper bound
    def greedy_estimate():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                x = last + 1
                while True:
                    valid = True
                    for a in row:
                        if (x - a) in used:
                            valid = False
                            break
                    if valid:
                        break
                    x += 1
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
            rows.append(row)
        return max(max(r) for r in rows)

    # We will search for the minimum M such that a clique of size n_rows exists.
    # We start from a reasonable lower bound and go up.
    # However, the graph can be huge. We need to be careful with M.
    # Let's try a few candidate M values.
    
    # For small n_rows, k, the graph is manageable.
    # n_rows=4, k=7 -> 8 elements per row.
    # Number of diffs per row = 28. Total diffs = 112.
    # M=112 is the counting bound. Best known is 134.
    
    # Strategy:
    # 1. Try M from min_scope to min_scope + delta.
    # 2. For each M, generate all valid rows (Sidon rows) with max element <= M.
    # 3. Build conflict graph.
    # 4. Run Max Clique.
    
    # To avoid generating too many candidates, we can limit the number of candidates per M.
    # Or we can use a randomized generation of candidates if the space is too large.
    
    # Let's implement a helper to generate valid rows up to M.
    # A row is valid if all internal differences are unique.
    
    def generate_valid_rows(M, limit_candidates=None):
        rows = []
        # Recursively build rows
        # row[0] is 0
        def backtrack(current_row, last_val):
            if time.time() > deadline - 5:
                return
            if len(current_row) == k + 1:
                rows.append(current_row[:])
                if limit_candidates and len(rows) >= limit_candidates:
                    return
                return
            
            # Next value must be > last_val and <= M
            # Also, the new differences (new_val - x for x in current_row) must be unique
            # and not collide with existing differences in current_row?
            # Wait, the condition is "all differences between two entries of the same row are distinct".
            # So yes, within the row, all diffs must be unique.
            
            used_diffs_in_row = set()
            for i in range(len(current_row)):
                for j in range(i):
                    used_diffs_in_row.add(current_row[i] - current_row[j])
            
            # Optimization: precompute? No, current_row is small (k+1 <= 8 or so).
            
            start_val = last_val + 1
            for val in range(start_val, M + 1):
                # Check if adding val creates duplicate diffs
                valid = True
                for x in current_row:
                    d = val - x
                    if d in used_diffs_in_row:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Also check if new diffs collide with each other?
                # The loop above checks d against used_diffs_in_row.
                # But what about d1 = val - x1 and d2 = val - x2?
                # x1 != x2 implies d1 != d2. So no collision between new diffs.
                
                # Add val
                current_row.append(val)
                backtrack(current_row, val)
                current_row.pop()
                
                if limit_candidates and len(rows) >= limit_candidates:
                    return
        
        backtrack([0], 0)
        return rows

    def build_conflict_graph(rows):
        n = len(rows)
        # diff_sets[i] is a set of differences for row i
        diff_sets = []
        for r in rows:
            ds = set()
            for i in range(len(r)):
                for j in range(i):
                    ds.add(r[i] - r[j])
            diff_sets.append(ds)
        
        # adjacency list: adj[i] is set of j such that row i and row j are compatible
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if diff_sets[i].isdisjoint(diff_sets[j]):
                    adj[i].add(j)
                    adj[j].add(i)
        return adj

    def max_clique(adj, n, target_size, time_limit):
        """
        Finds a clique of size target_size in graph (adj, n).
        Returns the list of vertices in the clique or None.
        Uses Tomita's branch and bound with greedy coloring.
        """
        start = time.time()
        
        # We want to find a clique of size exactly target_size (or at least).
        # Standard MaxClique finds the max size. We can stop if we find one of size target_size.
        
        # Coloring heuristic:
        # We maintain a list of candidate vertices.
        # We color them such that vertices with the same color are not adjacent.
        # The color number gives an upper bound on the clique size from that subset.
        
        # For efficiency, we use a simple greedy coloring on the induced subgraph of candidates.
        
        def color_candidates(candidates):
            # candidates is a list of vertices
            # Returns a list of (vertex, color) sorted by color
            # Greedy coloring:
            # We want to assign colors 1, 2, ... such that no two adjacent vertices have same color.
            # Actually, for MaxClique pruning, we often order vertices by degree or use a specific coloring order.
            # Tomita's algorithm colors the graph such that if a vertex has color c, it can be part of a clique of size at most c.
            
            # Simple approach:
            # Sort candidates by degree in the candidate subgraph (descending) to improve pruning?
            # Or just use the order they appear.
            
            colors = {}
            color_count = 0
            # We'll assign colors greedily.
            # To make the bound tight, we should try to use as few colors as possible?
            # No, the bound is: if we are looking for a clique of size k, and a vertex has color c, 
            # and we need to pick k more vertices from the remaining candidates, 
            # then we need at least k colors available?
            
            # Let's implement a standard greedy coloring that returns the color number for each vertex.
            # We process vertices one by one.
            # For each vertex, assign the smallest color not used by its neighbors in the already colored set.
            
            # However, the standard Tomita algorithm uses a specific ordering.
            # Let's use a simple heuristic: order by degree in the full graph (or candidate subgraph).
            
            # To save time, we'll just use the input order or a random shuffle if we want variety.
            # But for deterministic pruning, a consistent order is better.
            
            # Let's try sorting by degree in the candidate subgraph (descending).
            # This tends to produce better colorings.
            
            # Calculate degrees within candidates
            cand_set = set(candidates)
            degrees = {}
            for v in candidates:
                deg = 0
                for u in adj[v]:
                    if u in cand_set:
                        deg += 1
                degrees[v] = deg
            
            sorted_cands = sorted(candidates, key=lambda x: degrees[x], reverse=True)
            
            # Greedy coloring
            vertex_colors = {}
            color_map = {} # color -> set of vertices
            
            for v in sorted_cands:
                # Find smallest color not used by neighbors
                used_colors = set()
                for u in adj[v]:
                    if u in vertex_colors:
                        used_colors.add(vertex_colors[u])
                
                c = 1
                while c in used_colors:
                    c += 1
                
                vertex_colors[v] = c
                if c not in color_map:
                    color_map[c] = set()
                color_map[c].add(v)
                color_count = max(color_count, c)
            
            # Return list of (vertex, color) sorted by color (ascending)
            # This allows us to prune: if color < required_size, we can't form a clique of required_size from this vertex and its successors.
            result = []
            for v in sorted_cands:
                result.append((v, vertex_colors[v]))
            
            # Sort by color, then by degree (optional)
            result.sort(key=lambda x: (x[1], -degrees[x[0]]))
            return result

        # State: current_clique, candidates_list
        # We use recursion
        # To avoid deep recursion limits, we can use a stack, but Python recursion is fine for small n.
        
        best_clique = None
        
        def expand(clique, candidates):
            nonlocal best_clique
            if time.time() > start + time_limit:
                return False
            
            if len(clique) == target_size:
                best_clique = clique[:]
                return True
            
            if not candidates:
                return False
            
            # Color the candidates
            colored = color_candidates(candidates)
            
            # Process in reverse order (highest color first)
            # Actually, Tomita's algorithm processes from the end of the sorted list (highest color) to the beginning.
            # Why? Because if a vertex has a low color, it can't be part of a large clique.
            # We want to explore vertices with high color potential first.
            
            for i in range(len(colored) - 1, -1, -1):
                v, color = colored[i]
                
                # Pruning:
                # The maximum clique size we can form from v and the remaining candidates (colored[i+1:])
                # is bounded by color (if we assume the coloring is valid and covers the subgraph).
                # Actually, the bound is: current_clique_size + color.
                # If current_clique_size + color < target_size, we can prune.
                
                if len(clique) + color < target_size:
                    break # Since colors are non-decreasing in the list, all previous (lower i) have <= color. 
                         # Wait, the list is sorted by color ascending. 
                         # So colored[i] has color c. colored[i-1] has color <= c.
                         # If len(clique) + c < target, then for any j <= i, len(clique) + colored[j][1] <= len(clique) + c < target.
                         # So we can break.
                
                # Include v in clique
                clique.append(v)
                
                # New candidates are those in colored[i] (lower index) that are adjacent to v
                # Note: colored[i] contains vertices with index < i in the sorted list?
                # No, colored is a list. We are iterating i from end to start.
                # The "remaining candidates" after picking v are those in colored[0..i-1] that are adjacent to v.
                
                new_candidates = []
                for j in range(i):
                    u, _ = colored[j]
                    if u in adj[v]:
                        new_candidates.append(u)
                
                if expand(clique, new_candidates):
                    return True
                
                clique.pop()
                
                # Remove v from candidates for the next iteration
                # In Tomita's algorithm, we just move to the next i.
                # The candidates for the next step are effectively colored[0..i-1].
                # But we are passing a list. 
                # The loop structure handles this: next iteration i-1 will consider candidates from colored[0..i-2].
                # Wait, the standard algorithm removes v from the candidate set.
                # My `new_candidates` construction for the recursive call is correct.
                # For the loop continuation, we just decrement i.
                # The "candidates" for the next branch (not taking v) are the ones that were in the set before v was considered, minus v.
                # Since we iterate i downwards, the set of candidates available for the next vertex (at i-1) is implicitly colored[0..i-1].
                # This is correct.
                
            return False
        
        # Initial candidates: all vertices
        # We can order them by degree to help coloring?
        # color_candidates does its own sorting.
        
        # To improve performance, we can shuffle initial vertices if we want randomization,
        # but for MaxClique, a good static order is often better.
        
        # Let's try to find a clique of size target_size.
        # We need to pass a list of all vertices.
        all_vertices = list(range(n))
        
        # Optional: Sort initial vertices by degree to help the first coloring?
        # color_candidates sorts anyway.
        
        expand([], all_vertices)
        
        if best_clique:
            return best_clique
        return None

    # Main Loop: Try increasing M
    # Start from a slightly higher bound to ensure we have enough candidates?
    # If M is too small, there might be no valid rows or no clique.
    
    # Heuristic: Start from greedy estimate?
    # Greedy estimate might be high.
    # Let's start from min_scope + 10 and go up by 2.
    # Or start from greedy estimate - 20?
    
    # Let's check time.
    # If n_rows is small (e.g., 4), we can afford to search a range.
    
    # Estimate scope
    est_scope = greedy_estimate()
    
    # We want to minimize scope.
    # Let's try M from max(min_scope, est_scope - 20) to est_scope + 10.
    # But if est_scope is 134, min_scope is 112.
    # Range 112..144.
    
    # However, generating rows for M=112 might yield very few rows.
    # If no clique is found, increase M.
    
    search_start = max(min_scope, est_scope - 15)
    search_end = est_scope + 5
    
    # If est_scope is not computed yet or failed, fallback to min_scope
    if est_scope < min_scope:
        est_scope = min_scope
        search_start = min_scope
        search_end = min_scope + 20

    best_rows_found = None
    
    # To speed up, we can limit the number of candidates per M.
    # If the graph is too large, MaxClique will be slow.
    # Limit candidates to e.g., 5000.
    
    for M in range(search_start, search_end + 1):
        if time.time() > deadline - 10:
            break
            
        # Generate rows
        # We don't need ALL rows, just enough to form a clique.
        # But we don't know which ones will form the clique.
        # So we generate a random subset or the first N valid rows.
        
        # Generating ALL valid rows for M=134, k=7 might be too many?
        # Number of Sidon sets of size 8 in [1, 134]?
        # It's a subset of all increasing sequences.
        # C(134, 7) is huge. But Sidon constraint is strong.
        
        # Let's generate up to 2000 valid rows randomly?
        # Random generation of valid rows:
        # Pick random increasing sequence, check Sidon.
        # This might be slow.
        
        # Better: Use the backtrack generator but stop after finding `limit` rows.
        # The backtrack generator finds rows in lexicographic order.
        # This might bias towards small numbers.
        
        rows = generate_valid_rows(M, limit_candidates=3000)
        
        if len(rows) < n_rows:
            continue
        
        # Build graph
        adj = build_conflict_graph(rows)
        
        # Run Max Clique
        # Time limit per M: 10s?
        # If we have 30 M values, 10s each is 300s. Too long.
        # We have 140s total.
        # Let's allocate 5s per M.
        
        clique = max_clique(adj, len(rows), n_rows, 5.0)
        
        if clique:
            # Found a valid set of rows
            best_rows_found = [rows[i] for i in clique]
            # Since we are iterating M upwards, the first M that works is the minimum?
            # Not necessarily, because we might have missed a clique in a smaller M due to candidate limit or time limit.
            # But if we found one at M, it's a valid upper bound.
            # We can stop here if we are confident?
            # Or continue to see if a smaller M works? No, we went up.
            # So this is the best we found so far.
            # We can try to verify if a smaller M was possible?
            # No, we started from search_start.
            # If we found a solution at M, it's a valid answer.
            # To ensure we minimize, we should have started from min_scope.
            # If we started from est_scope - 15, we might miss a solution at est_scope - 20.
            
            # Let's break and return this.
            break
            
    if best_rows_found:
        return best_rows_found
    
    # Fallback: Return the greedy solution
    # This ensures we always return a valid result.
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            last = row[-1]
            x = last + 1
            while True:
                valid = True
                for a in row:
                    if (x - a) in used:
                        valid = False
                        break
                if valid:
                    break
                x += 1
            row.append(x)
            for a in row[:-1]:
                used.add(x - a)
        rows.append(row)
    return rows

if __name__ == "__main__":
    # Simple test
    rows = construct(4, 7)
    print("Scope:", max(max(r) for r in rows))
    # Verify
    diffs = set()
    valid = True
    for r in rows:
        if r[0] != 0:
            valid = False
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                diffs.add(d)
    print("Valid:", valid)
