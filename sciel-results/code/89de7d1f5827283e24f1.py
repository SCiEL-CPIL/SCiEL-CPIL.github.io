import time
import random
from itertools import combinations

def _has_k222_fast(adj, n):
    """
    Check if graph with adjacency bitmask list adj on n vertices contains K2,2,2.
    Returns True if found.
    """
    for a in range(n):
        for b in range(a+1, n):
            if not (adj[a] >> b) & 1:
                continue
            for c in range(b+1, n):
                if not (adj[a] >> c) & 1 or not (adj[b] >> c) & 1:
                    continue
                for d in range(c+1, n):
                    if not (adj[a] >> d) & 1 or not (adj[b] >> d) & 1:
                        continue
                    cn_mask = adj[a] & adj[b] & adj[c] & adj[d]
                    cn_mask &= ~(1 << a)
                    cn_mask &= ~(1 << b)
                    cn_mask &= ~(1 << c)
                    cn_mask &= ~(1 << d)
                    if cn_mask.bit_count() >= 2:
                        return True
    return False

def _count_k222(adj, n):
    """Count K2,2,2 subgraphs in graph adj."""
    count = 0
    for a in range(n):
        for b in range(a+1, n):
            if not (adj[a] >> b) & 1:
                continue
            for c in range(b+1, n):
                if not (adj[a] >> c) & 1 or not (adj[b] >> c) & 1:
                    continue
                for d in range(c+1, n):
                    if not (adj[a] >> d) & 1 or not (adj[b] >> d) & 1:
                        continue
                    cn_mask = adj[a] & adj[b] & adj[c] & adj[d]
                    cn_mask &= ~(1 << a)
                    cn_mask &= ~(1 << b)
                    cn_mask &= ~(1 << c)
                    cn_mask &= ~(1 << d)
                    if cn_mask.bit_count() >= 2:
                        count += 1
    return count

def _is_valid_coloring(adj, n):
    """Check if neither color class contains K2,2,2."""
    if _has_k222_fast(adj, n):
        return False
    comp = [0] * n
    full = (1 << n) - 1
    for i in range(n):
        comp[i] = full ^ adj[i] ^ (1 << i)
    if _has_k222_fast(comp, n):
        return False
    return True

def _build_paley29():
    """Build Paley graph on 29 vertices."""
    q = 29
    squares = set()
    for x in range(1, q):
        squares.add(x * x % q)
    
    n = q
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            diff = (j - i) % q
            if diff in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _backtrack_30(base_adj, deadline):
    """
    Backtracking search for valid 30-vertex coloring extending Paley-29.
    Tries to assign colors to edges (29, i) for i in 0..28.
    """
    n_base = 29
    n_target = 30
    
    # Precompute base adjacency
    adj = [0] * n_target
    for i in range(n_base):
        for j in range(i+1, n_base):
            if (base_adj[i] >> j) & 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # We'll build the new vertex's edges incrementally
    new_edges = [0] * n_base  # 0 or 1 for edge (29, i)
    
    # Precompute for each pair (i,j) in base: whether they form a "critical" configuration
    # A pair (i,j) is critical if adding vertex 29 with certain edges could create K2,2,2
    
    def check_partial(new_edges_so_far):
        """Check if current partial assignment is consistent (no K2,2,2 yet)."""
        # Build current adj
        cur_adj = [0] * n_target
        for i in range(n_base):
            for j in range(i+1, n_base):
                if (base_adj[i] >> j) & 1:
                    cur_adj[i] |= (1 << j)
                    cur_adj[j] |= (1 << i)
        
        # Add edges from new vertex
        for i in range(n_base):
            if i < len(new_edges_so_far) and new_edges_so_far[i] == 1:
                cur_adj[29] |= (1 << i)
                cur_adj[i] |= (1 << 29)
        
        # Check color 1
        if _has_k222_fast(cur_adj, n_target):
            return False
        
        # Check color 0
        comp = [0] * n_target
        full = (1 << n_target) - 1
        for i in range(n_target):
            comp[i] = full ^ cur_adj[i] ^ (1 << i)
        if _has_k222_fast(comp, n_target):
            return False
        
        return True
    
    # More efficient: incremental check
    # Precompute neighbor sets for base graph
    def incremental_check(new_edges):
        """
        Check if adding new vertex with given edges creates K2,2,2 in either color.
        Only check K2,2,2 that involve the new vertex (vertex 29).
        """
        # For K2,2,2 involving vertex 29:
        # Vertex 29 is in one of the three parts.
        # Case 1: 29 is in part {a,b} (with another vertex a)
        #   Then we need: 29-a edge, and two other parts {c,d} and {e,f}
        #   such that all cross edges exist.
        #   
        # Let's enumerate all possible K2,2,2 that include vertex 29.
        
        # Build adjacency for new vertex
        adj29 = 0
        for i in range(n_base):
            if new_edges[i] == 1:
                adj29 |= (1 << i)
        
        # Case 1: 29 is in part 1 (with vertex a)
        # Need: edge(29,a), and parts {c,d} and {e,f} from remaining vertices
        # such that:
        #   - All edges between {29,a} and {c,d} exist (in same color)
        #   - All edges between {29,a} and {e,f} exist
        #   - All edges between {c,d} and {e,f} exist
        #   - Edges within {c,d} and {e,f} are NOT required
        
        # For color 1:
        # Edge(29,a) must be 1, so a must be in adj29
        # For each a in adj29:
        #   For each pair (c,d) from remaining:
        #     Check if 29-c, 29-d, a-c, a-d are all 1
        #     If yes, find common neighbors of {29,a,c,d} in color 1
        #     If >= 2, we have K2,2,2
        
        # This is getting complex. Let's just do the full check but optimized.
        
        # Build full adj
        cur_adj = [0] * n_target
        for i in range(n_base):
            for j in range(i+1, n_base):
                if (base_adj[i] >> j) & 1:
                    cur_adj[i] |= (1 << j)
                    cur_adj[j] |= (1 << i)
        
        for i in range(n_base):
            if new_edges[i] == 1:
                cur_adj[29] |= (1 << i)
                cur_adj[i] |= (1 << 29)
        
        # Check color 1 for K2,2,2 involving vertex 29
        # Only check tuples where 29 is one of the 6 vertices
        for a in range(n_target):
            for b in range(a+1, n_target):
                if not (cur_adj[a] >> b) & 1:
                    continue
                # Skip if neither a nor b is 29
                if a != 29 and b != 29:
                    continue
                for c in range(b+1, n_target):
                    if not (cur_adj[a] >> c) & 1 or not (cur_adj[b] >> c) & 1:
                        continue
                    if c != 29 and not (a == 29 or b == 29):
                        # c is not 29, and a,b are not 29, so 29 not in {a,b,c}
                        # But we need 29 in the K2,2,2, so skip
                        # Actually, 29 could be in {d} or in common neighbors
                        # Let's not skip, just check all
                        pass
                    for d in range(c+1, n_target):
                        if not (cur_adj[a] >> d) & 1 or not (cur_adj[b] >> d) & 1:
                            continue
                        cn_mask = cur_adj[a] & cur_adj[b] & cur_adj[c] & cur_adj[d]
                        cn_mask &= ~(1 << a)
                        cn_mask &= ~(1 << b)
                        cn_mask &= ~(1 << c)
                        cn_mask &= ~(1 << d)
                        if cn_mask.bit_count() >= 2:
                            return False  # Found K2,2,2 in color 1
        
        # Check color 0
        comp = [0] * n_target
        full = (1 << n_target) - 1
        for i in range(n_target):
            comp[i] = full ^ cur_adj[i] ^ (1 << i)
        
        for a in range(n_target):
            for b in range(a+1, n_target):
                if not (comp[a] >> b) & 1:
                    continue
                if a != 29 and b != 29:
                    continue
                for c in range(b+1, n_target):
                    if not (comp[a] >> c) & 1 or not (comp[b] >> c) & 1:
                        continue
                    for d in range(c+1, n_target):
                        if not (comp[a] >> d) & 1 or not (comp[b] >> d) & 1:
                            continue
                        cn_mask = comp[a] & comp[b] & comp[c] & comp[d]
                        cn_mask &= ~(1 << a)
                        cn_mask &= ~(1 << b)
                        cn_mask &= ~(1 << c)
                        cn_mask &= ~(1 << d)
                        if cn_mask.bit_count() >= 2:
                            return False  # Found K2,2,2 in color 0
        
        return True
    
    # Backtracking
    def backtrack(idx):
        if time.time() > deadline:
            return None
        if idx == n_base:
            return new_edges[:]
        
        # Try color 1 first, then 0
        for color in [1, 0]:
            new_edges[idx] = color
            if incremental_check(new_edges):
                result = backtrack(idx + 1)
                if result is not None:
                    return result
            new_edges[idx] = 0
        
        return None
    
    return backtrack(0)

def _local_search_30(base_adj, deadline):
    """
    Local search for valid 30-vertex coloring extending Paley-29.
    """
    n_base = 29
    n_target = 30
    
    def count_violations(new_edges):
        """Count K2,2,2 violations in both colors."""
        adj = [0] * n_target
        for i in range(n_base):
            for j in range(i+1, n_base):
                if (base_adj[i] >> j) & 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        for i in range(n_base):
            if new_edges[i] == 1:
                adj[29] |= (1 << i)
                adj[i] |= (1 << 29)
        
        count = _count_k222(adj, n_target)
        
        comp = [0] * n_target
        full = (1 << n_target) - 1
        for i in range(n_target):
            comp[i] = full ^ adj[i] ^ (1 << i)
        count += _count_k222(comp, n_target)
        
        return count
    
    best_x = None
    best_count = float('inf')
    random.seed(42)
    
    for restart in range(100):
        if time.time() > deadline:
            break
        
        x = [random.randint(0, 1) for _ in range(n_base)]
        current_count = count_violations(x)
        
        if current_count == 0:
            return x
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n_base):
                if time.time() > deadline:
                    break
                old_val = x[i]
                x[i] = 1 - old_val
                new_count = count_violations(x)
                if new_count < current_count:
                    current_count = new_count
                    improved = True
                    if current_count == 0:
                        return x
                else:
                    x[i] = old_val
        
        if current_count < best_count:
            best_count = current_count
            best_x = x[:]
    
    return best_x

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Build Paley-29 base
    base_adj = _build_paley29()
    
    # First, verify base is valid
    if not _is_valid_coloring(base_adj, 29):
        # Fallback: random search for n=29
        random.seed(42)
        for _ in range(1000):
            if time.time() > deadline:
                break
            adj = [0] * 29
            for i in range(29):
                for j in range(i+1, 29):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            if _is_valid_coloring(adj, 29):
                base_adj = adj
                break
    
    # Try backtracking for n=30
    result = _backtrack_30(base_adj, deadline)
    if result is not None:
        # Build final coloring
        n = 30
        adj = [0] * n
        for i in range(29):
            for j in range(i+1, 29):
                if (base_adj[i] >> j) & 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        for i in range(29):
            if result[i] == 1:
                adj[29] |= (1 << i)
                adj[i] |= (1 << 29)
        
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(1 if (adj[i] >> j) & 1 else 0)
        return {'n': n, 'colors': colors}
    
    # Try local search for n=30
    result = _local_search_30(base_adj, deadline)
    if result is not None:
        # Verify
        adj = [0] * 30
        for i in range(29):
            for j in range(i+1, 29):
                if (base_adj[i] >> j) & 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        for i in range(29):
            if result[i] == 1:
                adj[29] |= (1 << i)
                adj[i] |= (1 << 29)
        
        if _is_valid_coloring(adj, 30):
            colors = []
            for i in range(30):
                for j in range(i+1, 30):
                    colors.append(1 if (adj[i] >> j) & 1 else 0)
            return {'n': 30, 'colors': colors}
    
    # Fallback: return n=29
    colors = []
    for i in range(29):
        for j in range(i+1, 29):
            colors.append(1 if (base_adj[i] >> j) & 1 else 0)
    return {'n': 29, 'colors': colors}
