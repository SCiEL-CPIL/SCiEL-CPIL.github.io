import time
import sys
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 115.0 # Safety margin

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(edges_set, n):
        # Represent adjacency as a list of integers (bitmasks)
        # adj_mask[i] is a bitmask where bit j is 1 if (i,j) is an edge
        adj_mask = [0] * n
        for u, v in edges_set:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # Track which edges are still uncovered
        # We can track this by maintaining the current adjacency masks
        # But since we remove edges, we need to update adj_mask.
        # Actually, it's easier to maintain a set of remaining edges or a bitmask of remaining edges.
        # For n=64, total edges ~2000. A set of tuples is fine, but bitmask operations on vertices are faster.
        # Let's maintain a set of remaining edges for the "uncovered" check, 
        # but use bitmasks for the clique finding logic on the *remaining* graph structure.
        
        # Wait, if we remove edges, the graph structure changes.
        # We need to find a clique in the *current* graph.
        # The current graph is defined by the remaining edges.
        
        remaining_edges = set(edges_set)
        cliques = []
        
        # Helper to check if a set of vertices (bitmask) forms a clique in remaining_edges
        # This is expensive if done naively.
        # Better: Maintain adj_mask dynamically?
        # If we remove an edge (u,v), we clear bit v in adj_mask[u] and bit u in adj_mask[v].
        # This makes adjacency checks O(1).
        
        # Let's use the dynamic adj_mask approach.
        # adj_mask[i] reflects only the edges currently in remaining_edges.
        
        # Function to remove an edge
        def remove_edge(u, v):
            if u == v: return
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            remaining_edges.discard((min(u,v), max(u,v)))

        # Bron-Kerbosch with pivot to find a maximum clique
        # R: current clique (bitmask)
        # P: candidates (bitmask)
        # X: excluded (bitmask)
        # Returns the largest clique found in this branch? 
        # Standard BK finds all maximal cliques. We want the largest.
        # We can modify it to keep track of the global best.
        
        best_clique_mask = 0
        best_clique_size = 0
        
        # To speed up, we can use a heuristic for pivot selection
        # and maybe limit the search if time is tight?
        # For n=64, BK is fast enough if the graph is not too complex.
        # Dense graphs have few maximal cliques usually? No, dense graphs have many cliques.
        # But we only need the *largest* one.
        # We can use a branch-and-bound: if |R| + |P| <= best_clique_size, prune.
        
        def bron_kerbosch(R, P, X):
            nonlocal best_clique_mask, best_clique_size
            
            # Pruning
            if not P and not X:
                # Maximal clique found
                size = bin(R).count('1')
                if size > best_clique_size:
                    best_clique_size = size
                    best_clique_mask = R
                return
            
            # If potential max size is not better, prune
            if bin(R).count('1') + bin(P).count('1') <= best_clique_size:
                return
                
            # Pivot selection
            # We want u in P | X that maximizes |P & ~N(u)|
            # To minimize branching, we iterate over candidates in P \ N(u)
            
            # Find pivot
            pivot = -1
            max_uncovered = -1
            # Iterate over P | X
            # For speed, we can just pick a random vertex from P or X, or the one with max degree in P?
            # Standard pivot: u that maximizes |P \ N(u)|.
            # Let's iterate through bits in P | X
            
            PX = P | X
            # To find the best pivot, we'd need to check all.
            # Heuristic: Pick u from P with max degree in P?
            # Actually, picking u with max degree in P usually minimizes P \ N(u).
            
            # Let's just pick the first vertex in P for simplicity? 
            # No, that's bad.
            # Let's pick the vertex in P with the highest degree in P.
            
            if P:
                # Find vertex in P with max degree in P
                max_deg = -1
                pivot = -1
                temp_P = P
                while temp_P:
                    # Extract lowest set bit
                    lsb = temp_P & -temp_P
                    v = lsb.bit_length() - 1
                    temp_P ^= lsb
                    
                    # Count neighbors of v in P
                    # Neighbors of v in P = P & adj_mask[v]
                    deg = bin(P & adj_mask[v]).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        pivot = v
            else:
                # P is empty, so we should have returned if X was empty too
                # If P is empty and X is not, it's a maximal clique
                if not X:
                    size = bin(R).count('1')
                    if size > best_clique_size:
                        best_clique_size = size
                        best_clique_mask = R
                return

            if pivot != -1:
                # Candidates to branch on: P \ N(pivot)
                candidates = P & ~adj_mask[pivot]
            else:
                candidates = P
            
            while candidates:
                # Pick a vertex from candidates
                lsb = candidates & -candidates
                v = lsb.bit_length() - 1
                candidates ^= lsb
                
                # Recurse
                # New R = R | {v}
                # New P = P & N(v)
                # New X = X & N(v)
                
                new_R = R | lsb
                new_P = P & adj_mask[v]
                new_X = X & adj_mask[v]
                
                bron_kerbosch(new_R, new_P, new_X)
                
                # Move v from P to X
                P &= ~lsb
                X |= lsb

        while remaining_edges:
            if time.time() > DEADLINE:
                # Fallback: add all remaining as edges
                for u, v in remaining_edges:
                    cliques.append([u, v])
                break

            # Find a large clique in the current graph
            # Start with all vertices as candidates
            # We can restrict candidates to vertices that have at least one remaining edge
            # Actually, just use all vertices. Isolated vertices won't form cliques > 1.
            
            # Reset best clique tracker
            best_clique_mask = 0
            best_clique_size = 0
            
            # Initial P: all vertices that have degree > 0 in remaining graph?
            # Or just all vertices.
            # Let's use all vertices.
            all_vertices_mask = (1 << n) - 1
            
            # Optimization: If the graph is very dense, the largest clique is likely large.
            # We can seed P with a subset?
            # Let's just run BK on all vertices.
            
            # To speed up BK, we can sort vertices by degree?
            # BK doesn't strictly need sorted vertices, but pivot selection benefits.
            
            bron_kerbosch(0, all_vertices_mask, 0)
            
            if best_clique_size < 2:
                # No clique of size >= 2 found?
                # This implies no edges? But we check remaining_edges.
                # Maybe the only edges are isolated? No, an edge is a clique of size 2.
                # If best_clique_size is 1, it means no edges?
                # If there is an edge (u,v), then {u,v} is a clique.
                # So best_clique_size should be at least 2 if remaining_edges is not empty.
                # If it's 1, something is wrong or the graph is empty.
                # Let's force pick an edge if BK fails to find size 2.
                if remaining_edges:
                    u, v = next(iter(remaining_edges))
                    clique_vertices = [u, v]
                else:
                    break
            else:
                # Extract vertices from best_clique_mask
                clique_vertices = []
                temp_mask = best_clique_mask
                while temp_mask:
                    lsb = temp_mask & -temp_mask
                    v = lsb.bit_length() - 1
                    clique_vertices.append(v)
                    temp_mask ^= lsb
            
            # Add clique to result
            cliques.append(clique_vertices)
            
            # Remove all edges in this clique from the graph
            m = len(clique_vertices)
            for i in range(m):
                for j in range(i+1, m):
                    a = clique_vertices[i]
                    b = clique_vertices[j]
                    remove_edge(a, b)
        
        return cliques

    # Instance 1
    n1, p1, s1 = 32, 0.6, 601
    e1 = random_graph(n1, p1, s1)
    c1 = solve_instance(e1, n1)
    
    # Instance 2
    n2, p2, s2 = 64, 0.55, 602
    e2 = random_graph(n2, p2, s2)
    c2 = solve_instance(e2, n2)
    
    return [c1, c2]
