"""Strong edge colourings via maximum independent set extraction on conflict graph."""
import time
from random import Random

def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill
    
    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)
    
    def solve_instance(edges):
        n_edges = len(edges)
        if n_edges == 0:
            return []
        
        # Build adjacency for the graph
        max_vertex = max(max(u, v) for u, v in edges)
        adj = [set() for _ in range(max_vertex + 1)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Precompute touched vertices for each edge
        # Edge i = (u, v): touched = {u, v} ∪ N(u) ∪ N(v)
        touched = []
        for u, v in edges:
            t = set()
            t.add(u)
            t.add(v)
            t.update(adj[u])
            t.update(adj[v])
            touched.append(t)
        
        # Build conflict bitsets: edge i conflicts with edge j if
        # some vertex of edge j is in touched[i] (covers sharing vertex or being joined)
        conflicts = [0] * n_edges
        for i in range(n_edges):
            ti = touched[i]
            for j in range(n_edges):
                a, b = edges[j]
                if a in ti or b in ti:
                    conflicts[i] |= (1 << j)
        
        # Remove self-conflicts
        for i in range(n_edges):
            conflicts[i] &= ~(1 << i)
        
        # Greedy independent set for initial lower bound
        def greedy_is(candidates):
            result = 0
            remaining_bits = candidates
            while remaining_bits:
                best_v = -1
                best_deg = n_edges + 1
                temp = remaining_bits
                while temp:
                    v = (temp & -temp).bit_length() - 1
                    deg = (conflicts[v] & remaining_bits).bit_count()
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                    temp &= temp - 1
                result |= (1 << best_v)
                remaining_bits &= ~(conflicts[best_v] | (1 << best_v))
            return result, result.bit_count()
        
        all_edges = (1 << n_edges) - 1
        greedy_result, greedy_size = greedy_is(all_edges)
        best_size = greedy_size
        best_set = greedy_result
        
        # Branch and bound
        def branch_and_bound(candidates, current_set, current_size):
            nonlocal best_size, best_set, deadline
            
            if time.time() > deadline:
                return
            
            if candidates == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set
                return
            
            # Upper bound pruning
            if current_size + candidates.bit_count() <= best_size:
                return
            
            # Pick vertex with maximum degree in candidate subgraph
            v = -1
            max_deg = -1
            temp = candidates
            while temp:
                vi = (temp & -temp).bit_length() - 1
                deg = (conflicts[vi] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = vi
                temp &= temp - 1
            
            # Branch 1: include v
            new_candidates = candidates & ~(conflicts[v] | (1 << v))
            branch_and_bound(new_candidates, current_set | (1 << v), current_size + 1)
            
            # Branch 2: exclude v
            branch_and_bound(candidates & ~(1 << v), current_set, current_size)
        
        branch_and_bound(all_edges, 0, 0)
        
        # Extract independent set as list of edge indices
        independent_set = []
        for i in range(n_edges):
            if (best_set >> i) & 1:
                independent_set.append(i)
        
        return independent_set
    
    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        n_edges = len(edges)
        
        if n_edges == 0:
            out.append([])
            continue
        
        # Build adjacency
        max_vertex = max(max(u, v) for u, v in edges)
        adj = [set() for _ in range(max_vertex + 1)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Precompute touched for all edges
        touched = []
        for u, v in edges:
            t = set()
            t.add(u)
            t.add(v)
            t.update(adj[u])
            t.update(adj[v])
            touched.append(t)
        
        # Repeatedly extract maximum independent sets
        color = {}
        remaining = set(range(n_edges))
        color_count = 0
        
        while remaining:
            if time.time() > deadline:
                # Fallback: assign each remaining edge a new color
                for i in remaining:
                    color[i] = color_count
                    color_count += 1
                break
            
            # Get list of remaining edge indices
            rem_list = sorted(remaining)
            n_rem = len(rem_list)
            
            # Build conflict bitsets for the subgraph induced by remaining edges
            # Map rem index to original index
            rem_to_orig = {i: rem_list[i] for i in range(n_rem)}
            
            conflicts_rem = [0] * n_rem
            for i in range(n_rem):
                ti = touched[rem_list[i]]
                for j in range(n_rem):
                    a, b = edges[rem_list[j]]
                    if a in ti or b in ti:
                        conflicts_rem[i] |= (1 << j)
            
            for i in range(n_rem):
                conflicts_rem[i] &= ~(1 << i)
            
            # Greedy initial solution
            def greedy_is(candidates):
                result = 0
                remaining_bits = candidates
                while remaining_bits:
                    best_v = -1
                    best_deg = n_rem + 1
                    temp = remaining_bits
                    while temp:
                        v = (temp & -temp).bit_length() - 1
                        deg = (conflicts_rem[v] & remaining_bits).bit_count()
                        if deg < best_deg:
                            best_deg = deg
                            best_v = v
                        temp &= temp - 1
                    result |= (1 << best_v)
                    remaining_bits &= ~(conflicts_rem[best_v] | (1 << best_v))
                return result, result.bit_count()
            
            all_rem = (1 << n_rem) - 1
            greedy_result, greedy_size = greedy_is(all_rem)
            best_size = greedy_size
            best_set = greedy_result
            
            # Branch and bound
            def branch_and_bound(candidates, current_set, current_size):
                nonlocal best_size, best_set, deadline
                
                if time.time() > deadline:
                    return
                
                if candidates == 0:
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_set
                    return
                
                if current_size + candidates.bit_count() <= best_size:
                    return
                
                v = -1
                max_deg = -1
                temp = candidates
                while temp:
                    vi = (temp & -temp).bit_length() - 1
                    deg = (conflicts_rem[vi] & candidates).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        v = vi
                    temp &= temp - 1
                
                new_candidates = candidates & ~(conflicts_rem[v] | (1 << v))
                branch_and_bound(new_candidates, current_set | (1 << v), current_size + 1)
                
                branch_and_bound(candidates & ~(1 << v), current_set, current_size)
            
            branch_and_bound(all_rem, 0, 0)
            
            # Assign this color to all edges in the independent set
            for i in range(n_rem):
                if (best_set >> i) & 1:
                    orig_idx = rem_list[i]
                    color[orig_idx] = color_count
                    remaining.discard(orig_idx)
            
            color_count += 1
        
        # Build the output list in sorted edge order
        result = [color[i] for i in range(n_edges)]
        out.append(result)
    
    return out
