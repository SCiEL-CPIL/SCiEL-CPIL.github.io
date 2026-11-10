"""Rewritten program: exact cochromatic number via Algorithm X (Dancing Links) on maximal cliques/IS."""
# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

import time

def _bron_kerbosch_maximal(adj, n):
    """Find all maximal cliques using Bron-Kerbosch with pivot."""
    results = []
    def bk(R, P_set, X):
        if not P_set and not X:
            results.append(frozenset(R))
            return
        # Pivot: choose u from P_set ∪ X with max |P_set ∩ N(u)|
        candidates = list(P_set | X)
        if not candidates:
            return
        # Pick pivot with max degree into P_set
        best_u = max(candidates, key=lambda u: len(P_set & adj[u]))
        # Iterate over P_set \ N(best_u)
        for v in list(P_set - adj[best_u]):
            bk(R | {v}, P_set & adj[v], X & adj[v])
            P_set.remove(v)
            X.add(v)
    bk(frozenset(), frozenset(range(n)), frozenset())
    return results

def _maximal_independent_sets(adj, n):
    """Find all maximal independent sets = maximal cliques of complement."""
    comp_adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u != v and v not in adj[u]:
                comp_adj[u].add(v)
    return _bron_kerbosch_maximal(comp_adj, n)

def _is_clique(vertices, adj):
    vlist = list(vertices)
    for i in range(len(vlist)):
        for j in range(i + 1, len(vlist)):
            if vlist[j] not in adj[vlist[i]]:
                return False
    return True

def _is_independent(vertices, adj):
    vlist = list(vertices)
    for i in range(len(vlist)):
        for j in range(i + 1, len(vlist)):
            if vlist[j] in adj[vlist[i]]:
                return False
    return True

def _is_valid_part(vertices, adj):
    if not vertices:
        return False
    return _is_clique(vertices, adj) or _is_independent(vertices, adj)

class DancingLinks:
    """Dancing Links (Algorithm X) implementation."""
    def __init__(self, n_cols, rows):
        """
        n_cols: number of columns (vertices to cover)
        rows: list of (row_id, set_of_cols)
        """
        self.n_cols = n_cols
        self.header = list(range(n_cols + 1))  # 0..n_cols, 0 is header
        self.size = [0] * (n_cols + 1)
        
        # Node structure: each node is [left, right, up, down, col, row_id]
        # We use arrays for speed
        self.L = []
        self.R = []
        self.U = []
        self.D = []
        self.C = []
        self.row_id = []
        
        # Initialize header row
        for j in range(n_cols + 1):
            self.L.append(j)
            self.R.append(j)
            self.U.append(j)
            self.D.append(j)
            self.C.append(j)
            self.row_id.append(-1)
        
        # Link header
        for j in range(n_cols + 1):
            self.L[j] = (j - 1) % (n_cols + 1)
            self.R[j] = (j + 1) % (n_cols + 1)
        
        # Add rows
        self.rows_data = rows
        self.row_to_cols = {}
        for row_id, cols in rows:
            if not cols:
                continue
            self.row_to_cols[row_id] = cols
            first = -1
            for c in sorted(cols):
                # Create node
                node = len(self.L)
                self.L.append(0)
                self.R.append(0)
                self.U.append(0)
                self.D.append(0)
                self.C.append(c)
                self.row_id.append(row_id)
                
                # Insert at bottom of column c
                self.D[node] = self.D[c]
                self.U[node] = self.U[self.D[c]]
                self.D[self.D[c]] = node
                self.U[self.U[node]] = node
                self.size[c] += 1
                
                if first == -1:
                    first = node
                else:
                    self.L[node] = self.R[first]
                    self.R[self.L[node]] = node
                    self.R[first] = node
                    self.L[node] = self.R[first]
        
        self.solution = []
        self.node_to_row = {}
        for row_id, cols in rows:
            if not cols:
                continue
            for c in cols:
                pass  # We'll track via row_id array
        
    def _cover(self, j):
        self.R[self.L[j]] = self.R[j]
        self.L[self.R[j]] = self.L[j]
        i = self.D[j]
        while i != j:
            self.R[self.L[i]] = self.R[i]
            self.L[self.R[i]] = self.L[i]
            j2 = self.D[i]
            while j2 != i:
                self.U[self.D[j2]] = self.U[j2]
                self.D[self.U[j2]] = self.D[j2]
                j2 = self.D[j2]
            i = j2
    
    def _uncover(self, j):
        i = self.U[j]
        while i != j:
            j2 = self.U[i]
            while j2 != i:
                self.D[self.U[j2]] = j2
                self.U[self.D[j2]] = j2
                j2 = self.U[j2]
            i = j2
        self.R[self.L[j]] = j
        self.L[self.R[j]] = j
    
    def search(self, depth=0, deadline=None):
        if deadline is not None and time.time() > deadline:
            return None
        
        if self.R[0] == 0:
            return list(self.solution)
        
        # Choose column with minimum size
        j = self.R[0]
        min_size = self.size[j]
        while self.R[j] != 0:
            if self.size[self.R[j]] < min_size:
                min_size = self.size[self.R[j]]
                j = self.R[j]
            j = self.R[j]
        
        self._cover(j)
        r = self.D[j]
        while r != j:
            if deadline is not None and time.time() > deadline:
                self._uncover(j)
                return None
            
            self.solution.append(self.row_id[r])
            
            # Cover columns in this row
            j2 = self.R[r]
            while j2 != r:
                self._cover(self.C[j2])
                j2 = self.R[j2]
            
            result = self.search(depth + 1, deadline)
            if result is not None:
                return result
            
            self.solution.pop()
            
            # Uncover columns in this row (reverse order)
            j2 = self.L[r]
            while j2 != r:
                self._uncover(self.C[j2])
                j2 = self.L[j2]
            
            r = self.D[r]
        
        self._uncover(j)
        return None

def _solve_exact(n, adj, deadline):
    """Solve exact cochromatic number using Algorithm X on maximal cliques/IS."""
    # Find all maximal cliques
    max_cliques = _bron_kerbosch_maximal(adj, n)
    # Find all maximal independent sets
    max_indep = _maximal_independent_sets(adj, n)
    
    # Combine all candidate parts (maximal cliques and maximal independent sets)
    all_parts = []
    seen = set()
    for part in max_cliques:
        key = frozenset(part)
        if key not in seen:
            seen.add(key)
            all_parts.append(key)
    for part in max_indep:
        key = frozenset(part)
        if key not in seen:
            seen.add(key)
            all_parts.append(key)
    
    # Also add individual vertices as fallback (they're both cliques and independent sets)
    for v in range(n):
        key = frozenset([v])
        if key not in seen:
            seen.add(key)
            all_parts.append(key)
    
    # Build rows: each row is (row_id, set of vertices it covers)
    rows = []
    for i, part in enumerate(all_parts):
        rows.append((i, set(part)))
    
    # Sort rows by size descending for better pruning
    rows.sort(key=lambda x: -len(x[1]))
    
    # Rebuild with new IDs
    rows_new = []
    for new_id, (old_id, cols) in enumerate(rows):
        rows_new.append((new_id, cols))
    
    dl = DancingLinks(n, rows_new)
    result = dl.search(deadline=deadline)
    
    if result is None:
        return None
    
    # Reconstruct partition
    parts = []
    for row_id in result:
        parts.append(sorted(rows_new[row_id][1]))
    
    return parts

def _greedy_cochromatic(n, adj):
    """Greedy heuristic."""
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            trial = set(part) | {v}
            if _is_valid_part(trial, adj):
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return parts

def construct():
    start = time.time()
    deadline = start + 80  # Safety margin within 90s limit
    
    n = P['n']
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Try exact method
    result = _solve_exact(n, adj, deadline)
    if result is not None:
        return [list(part) for part in result]
    
    # Fallback: greedy
    return _greedy_cochromatic(n, adj)
# EVOLVE-BLOCK-END
