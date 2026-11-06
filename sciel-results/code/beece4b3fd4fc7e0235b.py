"""Strong edge colouring: conflict-graph greedy with multiple orderings + local search."""
import time
import random
from collections import defaultdict


def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0  # safety margin under the 150 s kill

    def gen_instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict(edges, adj):
        """Build the strong edge-colouring conflict graph as a list of sets (conflict[i] = set of edge indices conflicting with edge i)."""
        m = len(edges)
        # Map vertex to list of incident edge indices
        vert_to_edges = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            vert_to_edges[u].append(i)
            vert_to_edges[v].append(i)
        
        conflict = [set() for _ in range(m)]
        
        for i, (u, v) in enumerate(edges):
            # Edges sharing a vertex
            for w in (u, v):
                for j in vert_to_edges[w]:
                    if j != i:
                        conflict[i].add(j)
                        conflict[j].add(i)
            # Edges joined by an edge (distance-2 edges)
            for w in (u, v):
                for x in adj[w]:
                    for j in vert_to_edges[x]:
                        if j != i:
                            conflict[i].add(j)
                            conflict[j].add(i)
        
        return conflict

    def greedy_first_fit(order, conflict, m):
        """Assign colours using first-fit in the given order."""
        colour = [-1] * m
        colour_used = []  # For each colour, set of edges with that colour
        # For efficiency, track for each edge which colours are forbidden
        # Actually, let's just check conflicts as we go
        # Better: maintain for each colour a set of edges, and for each edge check if any conflicting edge has that colour
        
        # Simpler approach: for each edge in order, find the smallest colour not used by any neighbor
        # To speed up, we can precompute for each edge the set of conflicting edges
        # Then for each edge, collect all colours used by its neighbors
        
        colour_assignment = [-1] * m
        # next_color is a list where next_color[c] = set of edges with colour c
        # We'll use a list of sets
        colour_sets = []
        
        for edge_idx in order:
            # Find all colours used by neighbors
            forbidden = set()
            for nb in conflict[edge_idx]:
                c = colour_assignment[nb]
                if c >= 0:
                    forbidden.add(c)
            
            # Find smallest available colour
            c = 0
            while c in forbidden:
                c += 1
            
            colour_assignment[edge_idx] = c
            if c >= len(colour_sets):
                colour_sets.append(set())
            colour_sets[c].add(edge_idx)
        
        return colour_assignment, len(colour_sets)

    def local_search(colour_assignment, conflict, m, max_iter=5000, seed=42):
        """Improve colouring by trying to reduce number of colours."""
        rng = random.Random(seed)
        num_colours = max(colour_assignment) + 1 if m > 0 else 0
        
        # Build colour sets
        colour_sets = defaultdict(set)
        for i in range(m):
            colour_sets[colour_assignment[i]].add(i)
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() > DEADLINE - 5:
                break
            
            # Try to remove the highest colour
            if num_colours <= 1:
                break
            target_colour = num_colours - 1
            edges_in_colour = list(colour_sets[target_colour])
            if not edges_in_colour:
                num_colours -= 1
                continue
            
            rng.shuffle(edges_in_colour)
            
            for edge_idx in edges_in_colour:
                # Try to move this edge to a lower colour
                forbidden = set()
                for nb in conflict[edge_idx]:
                    c = colour_assignment[nb]
                    if c >= 0 and c != target_colour:
                        forbidden.add(c)
                
                # Find a colour lower than target_colour that is not forbidden
                new_colour = None
                for c in range(target_colour):
                    if c not in forbidden:
                        new_colour = c
                        break
                
                if new_colour is not None:
                    # Move edge to new_colour
                    colour_sets[target_colour].discard(edge_idx)
                    colour_sets[new_colour].add(edge_idx)
                    colour_assignment[edge_idx] = new_colour
                    improved = True
                    break
            
            if not improved:
                # Try all edges in the highest colour
                for edge_idx in list(colour_sets[target_colour]):
                    forbidden = set()
                    for nb in conflict[edge_idx]:
                        c = colour_assignment[nb]
                        if c >= 0 and c != target_colour:
                            forbidden.add(c)
                    
                    new_colour = None
                    for c in range(target_colour):
                        if c not in forbidden:
                            new_colour = c
                            break
                    
                    if new_colour is not None:
                        colour_sets[target_colour].discard(edge_idx)
                        colour_sets[new_colour].add(edge_idx)
                        colour_assignment[edge_idx] = new_colour
                        improved = True
                        break
            
            if improved:
                # Check if target colour is now empty
                if not colour_sets[target_colour]:
                    num_colours -= 1
                # Rebuild colour_sets for consistency
                colour_sets = defaultdict(set)
                for i in range(m):
                    colour_sets[colour_assignment[i]].add(i)
                # Update num_colours
                if m > 0:
                    num_colours = max(colour_assignment) + 1
                else:
                    num_colours = 0
        
        return colour_assignment, num_colours

    def solve_instance(edges):
        """Solve one instance: return list of colours in sorted edge order."""
        m = len(edges)
        if m == 0:
            return []
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Build conflict graph
        conflict = build_conflict(edges, adj)
        
        # Try multiple orderings
        best_colour = None
        best_num = m  # worst case: all different
        
        # Ordering 1: natural order
        order1 = list(range(m))
        col1, num1 = greedy_first_fit(order1, conflict, m)
        col1, num1 = local_search(col1, conflict, m, max_iter=3000, seed=1)
        if num1 < best_num:
            best_num = num1
            best_colour = col1[:]
        
        if time.time() > DEADLINE - 10:
            return best_colour
        
        # Ordering 2: by degree (higher degree first)
        deg = [len(conflict[i]) for i in range(m)]
        order2 = sorted(range(m), key=lambda i: -deg[i])
        col2, num2 = greedy_first_fit(order2, conflict, m)
        col2, num2 = local_search(col2, conflict, m, max_iter=3000, seed=2)
        if num2 < best_num:
            best_num = num2
            best_colour = col2[:]
        
        if time.time() > DEADLINE - 10:
            return best_colour
        
        # Ordering 3: random
        order3 = list(range(m))
        random.Random(42).shuffle(order3)
        col3, num3 = greedy_first_fit(order3, conflict, m)
        col3, num3 = local_search(col3, conflict, m, max_iter=3000, seed=3)
        if num3 < best_num:
            best_num = num3
            best_colour = col3[:]
        
        if time.time() > DEADLINE - 10:
            return best_colour
        
        # Try more random orderings if time permits
        for trial in range(5):
            if time.time() > DEADLINE - 10:
                break
            order = list(range(m))
            random.Random(100 + trial).shuffle(order)
            col, num = greedy_first_fit(order, conflict, m)
            col, num = local_search(col, conflict, m, max_iter=2000, seed=100 + trial)
            if num < best_num:
                best_num = num
                best_colour = col[:]
        
        return best_colour

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > DEADLINE - 5:
            # Not enough time, return empty (shouldn't happen)
            out.append([])
            continue
        edges = gen_instance(*spec)
        colours = solve_instance(edges)
        out.append(colours)
    
    return out
