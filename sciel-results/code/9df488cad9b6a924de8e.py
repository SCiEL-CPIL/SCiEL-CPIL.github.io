import time
import random
from itertools import combinations

def _precompute_k32():
    """Precompute the known valid 32-vertex 4-coloring (seed construction)."""
    def mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_map = {}
    x = 1
    for e in range(15):
        log_map[x] = e
        x = mul(x, 2)

    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(log_map[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return colors

# Precompute at module load
_K32_COLORS = _precompute_k32()

def _edge_idx(i, j, n):
    """Get the index of edge (i,j) in the flat colors list, assuming i<j and n is current vertex count."""
    if i > j:
        i, j = j, i
    # Number of edges before vertex i
    prev = i * (i + 1) // 2
    # In vertex i's row, j is at position (j - i - 1)
    return prev + (j - i - 1)

def _validate_fast(colors, n):
    """Fast validation using bitsets or sets. Returns True if valid."""
    if len(colors) != n * (n - 1) // 2:
        return False
    
    adj = [set() for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c < 0 or c > 3:
                return False
            adj[c].add((i, j))
            idx += 1

    for c in range(4):
        adj_set = adj[c]
        # For each vertex, check neighbors
        adj_list = [[] for _ in range(n)]
        for (u, v) in adj_set:
            adj_list[u].append(v)
            adj_list[v].append(u)
        
        for u in range(n):
            nbrs = adj_list[u]
            for i in range(len(nbrs)):
                for j in range(i + 1, len(nbrs)):
                    a, b = nbrs[i], nbrs[j]
                    if a > b:
                        a, b = b, a
                    if (a, b) in adj_set:
                        return False
    return True

def _add_vertex_fast(colors, n):
    """
    Try to add one vertex to the coloring of size n.
    Returns new_colors (length (n+1)*n//2) or None.
    Uses backtracking with constraint propagation (forward checking).
    """
    # Precompute for each color c, and each pair of existing vertices (u,v),
    # if edge(u,v) is color c.
    # For the new vertex w, let x_i be the color of edge (i, w).
    # Constraint: For each c, the set S_c = {i | x_i = c} must be an independent set in G_c.
    # i.e., for any u,v in S_c, edge(u,v) must NOT be color c.
    
    # To speed up, we can precompute for each vertex i, and color c,
    # the set of vertices j such that edge(i,j) is color c.
    # If x_i = c, then x_j cannot be c for any j in forbidden[i][c].
    
    # forbidden[i][c] is a bitset of vertices j that cannot have x_j = c if x_i = c.
    # Note: j != i. Also, if i and j are connected by color c, then they can't both be c.
    
    # Let's build a structure to quickly check if a partial assignment is valid.
    # We will use a simple backtracking.
    
    # Precompute conflicts
    # conflict[i][c] = bitmask of vertices j such that if x_i=c, x_j cannot be c.
    # Since n <= 50, we can use Python integers as bitmasks.
    conflicts = [[0]*4 for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            idx = _edge_idx(i, j, n)
            c = colors[idx]
            conflicts[i][c] |= (1 << j)
            conflicts[j][c] |= (1 << i)
    
    x = [-1] * n
    
    def is_consistent(pos):
        # Check if assignment up to pos is consistent
        # We check for each color c, if there are two vertices i < j <= pos with x_i=x_j=c and conflict
        # Actually, we can check during assignment.
        # Let's just check the last assigned vertex pos against previous ones.
        c = x[pos]
        if c == -1: return True
        # Check if any previous vertex i with x_i = c has conflict with pos
        # We can maintain a count or just check.
        # For efficiency, let's check:
        for i in range(pos):
            if x[i] == c:
                if conflicts[pos][c] & (1 << i):
                    return False
        return True

    def backtrack(pos):
        if pos == n:
            return True
        
        # Try colors 0..3
        for c in range(4):
            # Check if c is allowed for pos
            allowed = True
            for i in range(pos):
                if x[i] == c:
                    if conflicts[pos][c] & (1 << i):
                        allowed = False
                        break
            if allowed:
                x[pos] = c
                if backtrack(pos + 1):
                    return True
                x[pos] = -1
        return False

    # Simple backtracking might be slow. Let's add some heuristic.
    # Try colors in random order? Or use MRV (Minimum Remaining Values)?
    # MRV is better.
    
    # Let's implement a slightly smarter backtracking.
    # We'll keep the same simple structure but maybe shuffle colors?
    # Actually, for n=32, the density is high, so constraints are tight.
    
    # Let's try a few random initializations if simple fails?
    # No, backtracking with forward checking (as implemented above) is exact.
    # The issue is speed.
    # For n=32, branching factor is small due to constraints.
    
    # Let's add a time limit to the search.
    start = time.time()
    limit = 1.0 # 1 second per vertex extension
    
    # We need to make the backtracking interruptible.
    # Python doesn't have easy interruption.
    # We can check time every few steps.
    
    # Let's rewrite backtrack to check time.
    def backtrack_timed(pos, time_left):
        if time.time() - start > limit:
            return False, None
            
        if pos == n:
            return True, x[:]
        
        # Heuristic: Try colors that are most constrained?
        # For simplicity, just 0..3.
        for c in range(4):
            allowed = True
            for i in range(pos):
                if x[i] == c:
                    if conflicts[pos][c] & (1 << i):
                        allowed = False
                        break
            if allowed:
                x[pos] = c
                res, sol = backtrack_timed(pos + 1, time_left)
                if res:
                    return res, sol
                x[pos] = -1
                if time.time() - start > limit:
                    return False, None
        return False, None

    res, sol = backtrack_timed(0, limit)
    if res:
        return sol
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit

    best_n = 32
    best_colors = _K32_COLORS

    # Try to extend the 32-vertex coloring
    current_n = 32
    current_colors = list(_K32_COLORS)
    
    # We will try to add vertices one by one.
    # If we fail, we stop.
    # This is a greedy approach to building a larger graph.
    
    for target_n in range(33, 51):
        if time.time() > deadline - 5:
            break
        
        # Try to add vertex
        new_assign = _add_vertex_fast(current_colors, current_n)
        if new_assign is not None:
            # Append new edges
            # Edges are (i, current_n) for i in range(current_n)
            # In lex order, these come after all existing edges.
            # Existing edges count: current_n*(current_n-1)//2
            # New edges: (0, current_n), (1, current_n), ..., (current_n-1, current_n)
            # This matches the order of new_assign.
            current_colors.extend(new_assign)
            current_n += 1
            best_n = current_n
            best_colors = list(current_colors)
        else:
            # Failed to extend further
            break

    # Final validation
    if time.time() < deadline - 1:
        if not _validate_fast(best_colors, best_n):
            best_n = 32
            best_colors = _K32_COLORS

    return {'n': best_n, 'colors': best_colors}
