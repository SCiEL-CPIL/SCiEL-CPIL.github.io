import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_bitmasks(edges, adj):
        m = len(edges)
        if m == 0:
            return []
        
        # adj is dict of vertex -> set of neighbors
        # conflict[i] is bitset of edges j that conflict with i
        conflict = [0] * m
        
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
                    continue
                if (u2 in adj.get(u1, set()) or v2 in adj.get(u1, set()) or
                    u2 in adj.get(v1, set()) or v2 in adj.get(v1, set())):
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
        
        return conflict

    def find_max_independent_set(conflict, candidates, time_limit):
        """
        Find a large independent set in the subgraph induced by 'candidates'.
        Uses Branch and Bound with bitsets.
        """
        if candidates == 0:
            return 0
            
        best_set = 0
        best_size = 0
        m = len(conflict)
        
        # Pre-compute degree in candidates for heuristic?
        # We will compute on the fly.
        
        # Heuristic: Greedy Min-Degree to get a good initial lower bound
        # This helps pruning.
        greedy_set = 0
        c = candidates
        while c:
            # Pick vertex with min degree in c
            min_deg = float('inf')
            best_v = -1
            temp_c = c
            while temp_c:
                lsb = temp_c & -temp_c
                v = lsb.bit_length() - 1
                deg = bin(c & conflict[v]).count('1')
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                temp_c ^= lsb
            
            if best_v == -1: break
            greedy_set |= (1 << best_v)
            c &= ~conflict[best_v]
            c &= ~(1 << best_v)
            
        best_size = bin(greedy_set).count('1')
        best_set = greedy_set
        
        if best_size == 0:
            return 0
            
        # B&B
        # P: candidates to process
        # X: vertices in current independent set
        # size: current size of X
        
        def dfs(P, X, size):
            nonlocal best_set, best_size
            if time.time() > time_limit:
                return
                
            if P == 0:
                if size > best_size:
                    best_size = size
                    best_set = X
                return
            
            # Upper bound: size + popcount(P)
            if size + bin(P).count('1') <= best_size:
                return
                
            # Select vertex v from P
            # Heuristic: Max degree in P (to prune more)
            max_deg = -1
            v = -1
            temp_P = P
            while temp_P:
                lsb = temp_P & -temp_P
                candidate_v = lsb.bit_length() - 1
                deg = bin(P & conflict[candidate_v]).count('1')
                if deg > max_deg:
                    max_deg = deg
                    v = candidate_v
                temp_P ^= lsb
                
            if v == -1:
                return
                
            # Branch 1: Include v
            # New P: P without v and without neighbors of v
            new_P = P & ~conflict[v] & ~(1 << v)
            dfs(new_P, X | (1 << v), size + 1)
            
            # Branch 2: Exclude v
            new_P = P & ~(1 << v)
            dfs(new_P, X, size)

        dfs(candidates, 0, 0)
        
        return best_set

    def solve_instance(n, d, seed):
        edges = instance(n, d, seed)
        m = len(edges)
        
        if m == 0:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        conflict = build_conflict_bitmasks(edges, adj)
        
        color = [-1] * m
        remaining = (1 << m) - 1
        color_id = 0
        
        # Time budget per instance
        inst_deadline = start_time + 45.0 # ~45s per instance
        
        while remaining:
            if time.time() > inst_deadline:
                # Fallback: Greedy coloring for remaining
                # Just assign unique colors to be safe? No, try greedy.
                # Let's do a fast greedy for remaining
                c_rem = remaining
                while c_rem:
                    lsb = c_rem & -c_rem
                    idx = lsb.bit_length() - 1
                    if color[idx] == -1:
                        # Find available color
                        used = set()
                        nb = conflict[idx] & remaining
                        while nb:
                            lsb_nb = nb & -nb
                            j = lsb_nb.bit_length() - 1
                            if color[j] != -1:
                                used.add(color[j])
                            nb ^= lsb_nb
                        c = 0
                        while c in used:
                            c += 1
                        color[idx] = c
                        if c > color_id:
                            color_id = c
                    c_rem ^= lsb
                break
            
            # Find large independent set
            # Limit time for B&B to 1-2 seconds
            time_limit_for_bnb = min(inst_deadline - 5, time.time() + 1.0)
            iso_set = find_max_independent_set(conflict, remaining, time_limit_for_bnb)
            
            if iso_set == 0:
                # Should not happen if remaining != 0
                # Pick one edge
                lsb = remaining & -remaining
                iso_set = lsb
                
            # Assign color
            c_set = iso_set
            while c_set:
                lsb = c_set & -c_set
                idx = lsb.bit_length() - 1
                color[idx] = color_id
                c_set ^= lsb
                
            remaining &= ~iso_set
            color_id += 1
            
        # Polish: Try to remove the last color
        # This is a local search step
        # We try to recolor edges of the last color class into previous colors
        if color_id > 0:
            last_color = color_id - 1
            last_edges = [i for i in range(m) if color[i] == last_color]
            
            # Try to move each edge to an earlier color
            # This is a constraint satisfaction problem.
            # Simple heuristic: try each edge, find if it fits in any previous color.
            # If not, try swapping?
            
            # Let's do a simple pass:
            # For each edge in last_edges:
            #   Check if it can take color c < last_color
            #   If yes, move it.
            #   If not, leave it.
            
            # This might not remove the color if multiple edges are stuck.
            # A better approach:
            # Check if the subgraph induced by last_edges can be colored with colors < last_color.
            # This is equivalent to checking if last_edges can be merged.
            
            # Given time, let's just do a greedy merge attempt.
            # If we can move all edges of last_color to other colors, we reduce color_id.
            
            trial_color = color[:]
            success = True
            for idx in last_edges:
                # Find a valid color < last_color
                nb = conflict[idx]
                used_colors = set()
                # Check neighbors
                # We only care about neighbors that have color < last_color
                # Actually, we can use any color < last_color.
                
                # Get colors of neighbors
                # This is O(m) per edge.
                
                valid_color = None
                for c in range(last_color):
                    # Check if c is available for idx
                    # c is available if no neighbor of idx has color c
                    # AND c < last_color
                    # We need to check neighbors in the *current* trial_color state?
                    # No, if we move multiple edges, conflicts change.
                    # This simple greedy might fail.
                    
                    # Let's just check against the *original* colors for simplicity?
                    # No, that's unsafe.
                    
                    # Let's skip complex merging in the polish if time is tight.
                    # The B&B approach should already minimize colors.
                    pass
                
            # A safer polish:
            # Run local search on the final coloring to reduce colors.
            # This is what the previous program did.
            # Let's add a quick local search here.
            
            # We'll use the local search from the previous program, but limited.
            # Actually, the B&B iterative removal is usually quite good.
            # Let's just return the result.
            
        return color

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline - 10:
            break
        
        n, d, seed = spec
        out.append(solve_instance(n, d, seed))
    
    while len(out) < 3:
        out.append([])
    
    return out
