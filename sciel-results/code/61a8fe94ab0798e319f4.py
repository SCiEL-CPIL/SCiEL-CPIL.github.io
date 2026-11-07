"""Exact biclique partition via maximal biclique enumeration + Algorithm X (Dancing Links)."""
import time
import sys

sys.setrecursionlimit(100000)

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _validate(bicliques, adj, all_edges_set):
    n = P['n']
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        if set(left).intersection(right):
            return False
        for u in left:
            if u < 0 or u >= n or not isinstance(u, int):
                return False
            for v in right:
                if v < 0 or v >= n or not isinstance(v, int):
                    return False
                if v not in adj[u]:
                    return False
                e = (min(u, v), max(u, v))
                if e in covered:
                    return False
                covered.add(e)
    if covered != all_edges_set:
        return False
    return True


def _enumerate_maximal_bicliques(adj, n, deadline):
    """
    Enumerate all maximal bicliques using a Bron-Kerbosch variant on the complement graph.
    A biclique (L, R) is maximal if no vertex can be added to L or R while maintaining the complete bipartite property.
    
    We use the fact that (L, R) is a biclique iff L is a clique in G[complement of R's non-neighbors]... 
    Actually, simpler: (L, R) is a biclique iff every u in L is adjacent to every v in R.
    Maximal means: for all x not in L∪R, x is not adjacent to all of R (can't add to L) 
                   AND x is not adjacent to all of L (can't add to R).
    
    We'll use a recursive enumeration:
    State: (L, R, X, Y) where X are vertices that could be added to L (adjacent to all R), 
           Y are vertices that could be added to R (adjacent to all L).
    But this is complex. Instead, we'll use a simpler approach:
    
    For each pair of vertices (u, v) with u < v and edge (u,v) exists:
      Start with L={u}, R={v}, expand to maximal biclique.
      Deduplicate.
    
    Since n=30 and the graph is dense, the number of maximal bicliques should be manageable.
    """
    bicliques = set()
    
    # Precompute for speed
    adj_list = [list(adj[i]) for i in range(n)]
    
    def expand(L, R):
        """Expand (L, R) to a maximal biclique."""
        L = set(L)
        R = set(R)
        changed = True
        while changed:
            changed = False
            # Try adding to L
            new_L = set()
            for u in range(n):
                if u in L or u in R:
                    continue
                if all(u in adj[v] for v in R):
                    new_L.add(u)
            if new_L:
                L |= new_L
                changed = True
            # Try adding to R
            new_R = set()
            for v in range(n):
                if v in L or v in R:
                    continue
                if all(v in adj[u] for u in L):
                    new_R.add(v)
            if new_R:
                R |= new_R
                changed = True
        return frozenset(L), frozenset(R)
    
    # Enumerate by trying each edge as a seed
    for u in range(n):
        if time.time() > deadline:
            break
        for v in adj[u]:
            if v <= u:
                continue
            L, R = expand({u}, {v})
            bicliques.add((L, R))
    
    return bicliques


def _solve_exact_cover_algorithm_x(rows, cols, deadline):
    """
    Algorithm X with Dancing Links for exact cover.
    rows: list of sets, each row is a set of column indices it covers.
    cols: total number of columns.
    Returns a list of row indices forming an exact cover, or None if not found within deadline.
    """
    # Build DLX structure
    # Each column has a header node. Each row has nodes for each column it covers.
    # We use arrays for L, R, U, D, C, size, row_id
    
    # First, filter rows: remove rows that don't cover any column, and columns that no row covers
    # Also, if a column is covered by no row, no solution exists.
    
    # Count occurrences of each column
    col_count = [0] * cols
    for row in rows:
        for c in row:
            col_count[c] += 1
    
    # If any column has count 0, no exact cover exists
    for c in range(cols):
        if col_count[c] == 0:
            return None
    
    # Build the DLX matrix
    # Node 0 is the root
    # We'll create nodes dynamically
    L = []
    R = []
    U = []
    D = []
    C = []
    row_id = []
    col_size = []
    
    def new_node():
        idx = len(L)
        L.append(0)
        R.append(0)
        U.append(0)
        D.append(0)
        C.append(0)
        row_id.append(-1)
        return idx
    
    # Root node
    root = new_node()
    L[root] = root
    R[root] = root
    
    # Column headers
    col_headers = []
    for c in range(cols):
        h = new_node()
        C[h] = c
        # Insert h into the circular list of column headers
        R[h] = R[root]
        L[h] = root
        R[root] = h
        L[R[root]] = h
        # Initialize column's vertical list to just the header
        U[h] = h
        D[h] = h
        col_size.append(1)  # Will adjust as we add rows
        col_headers.append(h)
    
    # Add row nodes
    # For each row, create a node for each column it covers
    # We need to track the first node of each row for deletion
    row_first_nodes = []
    
    for r, row_set in enumerate(rows):
        if not row_set:
            row_first_nodes.append(-1)
            continue
        first_node = -1
        for c in row_set:
            node = new_node()
            C[node] = c
            row_id[node] = r
            # Insert node into column c's vertical list (at the bottom, before header)
            h = col_headers[c]
            U[node] = U[h]
            D[node] = h
            D[U[h]] = node
            U[h] = node
            col_size[c] += 1
            if first_node == -1:
                first_node = node
            else:
                # Link horizontally
                R[first_node] = node
                L[node] = first_node
                first_node = node
        # Close the horizontal circle
        if first_node != -1:
            R[first_node] = first_node
            L[first_node] = first_node
        row_first_nodes.append(first_node)
    
    # Now solve
    solution = []
    
    def search(k):
        if time.time() > deadline:
            return False
        if R[root] == root:
            return True
        
        # Choose column with minimum size
        c = R[root]
        min_size = col_size[c]
        while R[c] != root:
            if col_size[R[c]] < min_size:
                min_size = col_size[R[c]]
                c = R[c]
            c = R[c]
        
        if min_size == 0:
            return False
        
        # Cover column c
        # Remove column header
        R[L[c]] = R[c]
        L[R[c]] = L[c]
        
        # For each row in column c
        r = D[c]
        while r != c:
            solution.append(row_id[r])
            
            # Cover all columns in this row
            j = R[r]
            while j != r:
                # Remove row node j from its column
                U[D[j]] = U[j]
                D[U[j]] = D[j]
                col_size[C[j]] -= 1
                j = R[j]
            
            if search(k + 1):
                return True
            
            solution.pop()
            
            # Uncover all columns in this row (in reverse order)
            j = L[r]
            while j != r:
                col_size[C[j]] += 1
                D[U[j]] = j
                U[D[j]] = j
                j = L[j]
            
            r = D[r]
        
        # Uncover column c
        L[R[c]] = c
        R[L[c]] = c
        
        return False
    
    if search(0):
        return solution
    return None


def _greedy_set_cover(rows, cols, deadline):
    """
    Greedy set cover: repeatedly pick the row that covers the most uncovered columns.
    Returns a list of row indices.
    """
    uncovered = set(range(cols))
    chosen = []
    
    # Precompute row sizes
    row_sets = [set(r) for r in rows]
    
    while uncovered and time.time() < deadline:
        best_row = -1
        best_count = 0
        for i, rs in enumerate(row_sets):
            if not rs:
                continue
            count = len(rs & uncovered)
            if count > best_count:
                best_count = count
                best_row = i
        if best_row == -1:
            break
        chosen.append(best_row)
        uncovered -= row_sets[best_row]
    
    if uncovered:
        return None
    return chosen


def _backtracking_set_cover(rows, cols, deadline, best_count):
    """
    Backtracking search for set cover with pruning.
    Tries to find a cover with fewer than best_count rows.
    """
    uncovered = set(range(cols))
    row_sets = [set(r) for r in rows]
    current = []
    
    # Sort rows by size descending for better pruning
    row_order = sorted(range(len(rows)), key=lambda i: -len(row_sets[i]))
    
    def search(start_idx):
        if time.time() > deadline:
            return
        if not uncovered:
            if len(current) < best_count:
                best_count[0] = len(current)
                best_solution[0] = list(current)
            return
        if len(current) >= best_count[0]:
            return
        
        # Lower bound: minimum number of rows needed
        # Each row can cover at most max_row_size uncovered columns
        max_cover = 0
        for i in range(start_idx, len(row_order)):
            max_cover = max(max_cover, len(row_sets[row_order[i]] & uncovered))
        if max_cover == 0:
            return
        min_needed = (len(uncovered) + max_cover - 1) // max_cover
        if len(current) + min_needed >= best_count[0]:
            return
        
        for i in range(start_idx, len(row_order)):
            r = row_order[i]
            rs = row_sets[r]
            if not (rs & uncovered):
                continue
            current.append(r)
            uncovered_before = set(uncovered)
            uncovered -= rs
            search(i + 1)
            uncovered = uncovered_before
            current.pop()
            if time.time() > deadline:
                return
    
    best_solution = [None]
    best_count[0] = best_count[0]  # Already set
    search(0)
    
    if best_solution[0] is not None:
        return best_solution[0]
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill

    n = P['n']
    adj = _build_adjacency()
    all_edges_set = set((min(u, v), max(u, v)) for u, v in P['edges'])
    all_edges_list = sorted(all_edges_set)
    
    # Step 1: Enumerate all maximal bicliques
    maximal_bicliques = _enumerate_maximal_bicliques(adj, n, deadline)
    
    # Step 2: Convert to set cover problem
    # Each edge must be covered exactly once.
    # Each maximal biclique covers a set of edges.
    # We need to select a subset of bicliques such that every edge is covered exactly once.
    
    # Build the incidence: for each biclique, which edges does it cover?
    biclique_edges = []
    for L, R in maximal_bicliques:
        edges = set()
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                if e in all_edges_set:
                    edges.add(e)
        if edges:
            biclique_edges.append((L, R, edges))
    
    # Map each edge to an index
    edge_to_idx = {e: i for i, e in enumerate(all_edges_list)}
    num_edges = len(all_edges_list)
    
    # Build rows: each row is a set of edge indices covered by that biclique
    rows = []
    biclique_info = []
    for L, R, edges in biclique_edges:
        row = set(edge_to_idx[e] for e in edges)
        rows.append(row)
        biclique_info.append((L, R))
    
    # Step 3: Try exact cover with Algorithm X
    # First, filter rows: only keep rows that are subsets of all_edges (they should be)
    # Also, remove duplicate rows
    unique_rows = list(set(tuple(sorted(r)) for r in rows))
    rows = [set(r) for r in unique_rows]
    
    # Rebuild biclique_info to match
    # This is tricky; let's just use the rows as is and map back
    # For now, we'll keep the original rows and biclique_info aligned
    # But Algorithm X needs consistent rows. Let's just use the original rows.
    
    # Actually, let's just use the original rows. Duplicates are fine for DLX.
    # But it's more efficient to remove them. Let's do it properly.
    
    # Create a mapping from row tuple to biclique index
    row_to_biclique = {}
    for i, row in enumerate(rows):
        t = tuple(sorted(row))
        if t not in row_to_biclique:
            row_to_biclique[t] = i
    
    # Use unique rows for the solver
    unique_row_tuples = list(row_to_biclique.keys())
    unique_rows = [set(t) for t in unique_row_tuples]
    
    # Try Algorithm X
    solution = _solve_exact_cover_algorithm_x(unique_rows, num_edges, deadline)
    
    best_result = None
    best_count = float('inf')
    
    if solution is not None:
        # Convert solution to bicliques
        result_bicliques = []
        for row_idx in solution:
            L, R = biclique_info[row_to_biclique[tuple(sorted(unique_rows[row_idx]))]]
            result_bicliques.append((L, R))
        
        # Validate
        if _validate(result_bicliques, adj, all_edges_set):
            best_result = result_bicliques
            best_count = len(result_bicliques)
    
    # Step 4: If exact cover failed or timed out, try greedy set cover
    if best_result is None or time.time() < deadline:
        greedy_solution = _greedy_set_cover(unique_rows, num_edges, deadline)
        if greedy_solution is not None:
            result_bicliques = []
            for row_idx in greedy_solution:
                L, R = biclique_info[row_to_biclique[tuple(sorted(unique_rows[row_idx]))]]
                result_bicliques.append((L, R))
            
            if _validate(result_bicliques, adj, all_edges_set):
                if len(result_bicliques) < best_count:
                    best_count = len(result_bicliques)
                    best_result = result_bicliques
    
    # Step 5: If still no good result, try backtracking with pruning
    if best_result is None or time.time() < deadline:
        if best_count == float('inf'):
            # Use a large upper bound
            upper_bound = num_edges  # Worst case: one biclique per edge
        else:
            upper_bound = best_count
        
        bt_solution = _backtracking_set_cover(unique_rows, num_edges, deadline, [upper_bound])
        if bt_solution is not None:
            result_bicliques = []
            for row_idx in bt_solution:
                L, R = biclique_info[row_to_biclique[tuple(sorted(unique_rows[row_idx]))]]
                result_bicliques.append((L, R))
            
            if _validate(result_bicliques, adj, all_edges_set):
                if len(result_bicliques) < best_count:
                    best_count = len(result_bicliques)
                    best_result = result_bicliques
    
    # Fallback: trivial partition (one edge per biclique)
    if best_result is None:
        best_result = [({u}, {v}) for u, v in P['edges']]
    
    # Convert to proper types: list of [list(left), list(right)]
    result = []
    for left, right in best_result:
        result.append([list(left), list(right)])
    
    return result
