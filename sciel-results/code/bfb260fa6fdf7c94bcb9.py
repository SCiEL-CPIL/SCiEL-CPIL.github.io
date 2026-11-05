"""Strong edge colourings via Maximum Induced Matching extraction + local refinement."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict, deque
from math import exp

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

    def build_conflict_graph(edges, n):
        m = len(edges)
        vertex_to_edges = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            vertex_to_edges[u].append(i)
            vertex_to_edges[v].append(i)
        
        conflict_adj = [set() for _ in range(m)]
        
        for u in range(n):
            inc_edges = vertex_to_edges[u]
            for i in range(len(inc_edges)):
                for j in range(i + 1, len(inc_edges)):
                    e_i = inc_edges[i]
                    e_j = inc_edges[j]
                    conflict_adj[e_i].add(e_j)
                    conflict_adj[e_j].add(e_i)
        
        vertex_adj = [set() for _ in range(n)]
        for u, v in edges:
            vertex_adj[u].add(v)
            vertex_adj[v].add(u)
        
        for i, (u, v) in enumerate(edges):
            for nu in vertex_adj[u]:
                if nu == v:
                    continue
                for j in vertex_to_edges[nu]:
                    if j != i:
                        conflict_adj[i].add(j)
                        conflict_adj[j].add(i)
            for nv in vertex_adj[v]:
                if nv == u:
                    continue
                for j in vertex_to_edges[nv]:
                    if j != i:
                        conflict_adj[i].add(j)
                        conflict_adj[j].add(i)
                        
        return conflict_adj

    def find_mis_greedy(conflict_adj, candidates, rng):
        """
        Find a Maximum Independent Set (Induced Matching) in the subgraph induced by candidates.
        Uses a greedy heuristic: repeatedly pick a vertex with minimum degree in the remaining graph.
        """
        if not candidates:
            return []
        
        remaining = set(candidates)
        mis = []
        
        while remaining:
            min_deg = float('inf')
            chosen = -1
            for v in remaining:
                deg = len(conflict_adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    chosen = v
            
            mis.append(chosen)
            remaining.discard(chosen)
            for nb in conflict_adj[chosen]:
                if nb in remaining:
                    remaining.discard(nb)
        
        return mis

    def max_is_bnb(adj, candidates, time_limit, current_time_check):
        """
        Branch-and-bound for Maximum Independent Set using bitsets.
        adj: list of sets (conflict adjacency)
        candidates: set of vertices
        Returns the maximum independent set found within time limit.
        """
        if not candidates:
            return []
        
        # Convert to sorted list for bitset operations
        verts = sorted(candidates)
        n = len(verts)
        if n == 0:
            return []
        
        # Build bitset representation
        # Map vertex index to position in verts
        vert_to_pos = {v: i for i, v in enumerate(verts)}
        
        # Build adjacency as bitsets (Python integers as bitsets)
        adj_bits = [0] * n
        for i, v in enumerate(verts):
            for nb in adj[v]:
                if nb in vert_to_pos:
                    adj_bits[i] |= (1 << vert_to_pos[nb])
        
        # Initial candidate set as bitset
        candidates_bits = (1 << n) - 1
        
        best_set = []
        best_size = 0
        
        # Upper bound: greedy coloring of the induced subgraph
        def upper_bound(bits):
            """Compute upper bound using greedy coloring."""
            if bits == 0:
                return 0
            # Simple upper bound: count vertices
            return bits.bit_count()
        
        # More refined upper bound using greedy coloring
        def greedy_color_ub(bits, adj_bits):
            """
            Upper bound via greedy coloring of the conflict subgraph.
            The number of colors needed >= size of max independent set complement... 
            Actually, for MIS, we want to bound the size.
            Use: partition into cliques, each clique contributes at most 1 to MIS.
            Greedy clique cover gives upper bound.
            """
            remaining = bits
            clique_count = 0
            while remaining:
                # Find a maximal clique greedily
                clique = 0
                r = remaining
                while r:
                    # Pick vertex with max degree in remaining
                    best_v = -1
                    best_deg = -1
                    for i in range(n):
                        if (r >> i) & 1:
                            deg = (adj_bits[i] & r).bit_count()
                            if deg > best_deg:
                                best_deg = deg
                                best_v = i
                    if best_v == -1:
                        break
                    clique |= (1 << best_v)
                    r &= ~(1 << best_v)
                    r &= ~(adj_bits[best_v])
                remaining &= ~clique
                clique_count += 1
            return clique_count
        
        # Branch and bound
        def bnb(bits, current_set, depth):
            nonlocal best_set, best_size
            
            if time.time() > time_limit:
                return
            
            if bits == 0:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                return
            
            # Upper bound
            ub = greedy_color_ub(bits, adj_bits)
            if len(current_set) + ub <= best_size:
                return
            
            # Choose vertex to branch on
            # Pick vertex with max degree in remaining
            best_v = -1
            best_deg = -1
            for i in range(n):
                if (bits >> i) & 1:
                    deg = (adj_bits[i] & bits).bit_count()
                    if deg > best_deg:
                        best_deg = deg
                        best_v = i
            
            if best_v == -1:
                return
            
            # Branch 1: include best_v
            new_bits = bits & ~(1 << best_v)
            new_bits &= ~(adj_bits[best_v])
            current_set.append(verts[best_v])
            bnb(new_bits, current_set, depth + 1)
            current_set.pop()
            
            # Branch 2: exclude best_v
            bnb(bits & ~(1 << best_v), current_set, depth + 1)
        
        bnb(candidates_bits, [], 0)
        
        if not best_set and candidates:
            # Fallback to greedy if B&B found nothing (shouldn't happen)
            return find_mis_greedy(adj, candidates, random.Random(42))
        
        return best_set

    def solve_instance(edges, n, d, seed, time_limit):
        m = len(edges)
        if m == 0:
            return []
        
        conflict_adj = build_conflict_graph(edges, n)
        
        best_colours = None
        best_count = float('inf')
        
        rng = random.Random(seed)
        
        max_restarts = 30
        
        for restart in range(max_restarts):
            if time.time() > time_limit:
                break
            
            colours = [0] * m
            used_colors = 0
            remaining = set(range(m))
            
            # 1. Greedy MIM Packing with B&B for each color class
            while remaining:
                if time.time() > time_limit:
                    break
                # Use B&B for larger sets, greedy for small sets
                if len(remaining) > 30:
                    matching = max_is_bnb(conflict_adj, remaining, min(time.time() + 2.0, time_limit), None)
                else:
                    matching = find_mis_greedy(conflict_adj, remaining, rng)
                
                if not matching:
                    # Fallback: pick one edge
                    matching = [next(iter(remaining))]
                
                # Assign color
                for e in matching:
                    colours[e] = used_colors
                
                # Remove from remaining
                remaining.difference_update(matching)
                used_colors += 1
            
            current_count = used_colors
            
            # 2. Local search with swap moves
            improved = True
            while improved and time.time() < time_limit:
                improved = False
                max_c = max(colours) if m > 0 else -1
                
                # Try to remove max_c by reassigning its edges
                edges_with_max = [i for i in range(m) if colours[i] == max_c]
                
                old_colours = colours[:]
                success = True
                rng.shuffle(edges_with_max)
                
                for e in edges_with_max:
                    blocked = set()
                    for nb in conflict_adj[e]:
                        if colours[nb] != max_c:
                            if colours[nb] < max_c:
                                blocked.add(colours[nb])
                    
                    valid = False
                    for c in range(max_c):
                        if c not in blocked:
                            colours[e] = c
                            valid = True
                            break
                    
                    if not valid:
                        success = False
                        break
                
                if success:
                    improved = True
                    current_count -= 1
                else:
                    colours = old_colours
            
            # 3. Swap-based local search
            for _ in range(200):
                if time.time() > time_limit:
                    break
                
                # Try to improve by swapping colors of two edges
                # Pick a random edge and try to swap its color with another edge
                e1 = rng.randint(0, m - 1)
                c1 = colours[e1]
                
                # Find an edge e2 with a different color
                candidates = [i for i in range(m) if colours[i] != c1]
                if not candidates:
                    break
                
                e2 = rng.choice(candidates)
                c2 = colours[e2]
                
                # Check if swapping c1 and c2 is valid
                # e1 gets c2: check no neighbor of e1 has c2
                # e2 gets c1: check no neighbor of e2 has c1
                valid = True
                for nb in conflict_adj[e1]:
                    if colours[nb] == c2:
                        valid = False
                        break
                if valid:
                    for nb in conflict_adj[e2]:
                        if colours[nb] == c1:
                            valid = False
                            break
                
                if valid:
                    # Do the swap
                    colours[e1] = c2
                    colours[e2] = c1
                    # This doesn't change the count, but may enable further improvements
                    # Try to remove a color after swap
                    max_c = max(colours)
                    edges_with_max = [i for i in range(m) if colours[i] == max_c]
                    
                    old_colours = colours[:]
                    success = True
                    rng.shuffle(edges_with_max)
                    
                    for e in edges_with_max:
                        blocked = set()
                        for nb in conflict_adj[e]:
                            if colours[nb] != max_c:
                                if colours[nb] < max_c:
                                    blocked.add(colours[nb])
                        
                        valid2 = False
                        for c in range(max_c):
                            if c not in blocked:
                                colours[e] = c
                                valid2 = True
                                break
                        
                        if not valid2:
                            success = False
                            break
                    
                    if success:
                        current_count -= 1
                        improved = True
                    else:
                        colours = old_colours
            
            if current_count < best_count:
                best_count = current_count
                best_colours = colours[:]
            
            # Early stop if we hit the bound
            lb = 5 * d * d // 4
            if current_count <= lb:
                break
        
        if best_colours is None:
            # Fallback: simple greedy coloring
            colours = [0] * m
            coloured_set = set()
            for idx in range(m):
                blocked = set()
                for nb in conflict_adj[idx]:
                    if nb in coloured_set:
                        blocked.add(colours[nb])
                c = 0
                while c in blocked:
                    c += 1
                colours[idx] = c
                coloured_set.add(idx)
            best_colours = colours
        
        return best_colours

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        
        inst_start = time.time()
        inst_limit = min(inst_start + 45.0, deadline)
        
        colours = solve_instance(edges, n, d, seed, inst_limit)
        results.append(colours)
        
        if time.time() > deadline:
            break

    return results
# EVOLVE-BLOCK-END
