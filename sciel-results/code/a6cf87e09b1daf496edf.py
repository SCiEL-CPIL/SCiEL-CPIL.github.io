"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from random import Random

def construct():
    t0 = time.time()
    TOTAL_DEADLINE = t0 + 145.0  # safety margin before 150s kill

    def gen_edges(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def greedy_clique_partition(n, edges_set, deadline):
        """
        Greedy partitioning based on edge coverage efficiency.
        """
        if not edges_set:
            return []
        
        rem_adj = [set() for _ in range(n)]
        for u, v in edges_set:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        remaining_edges = set(edges_set)
        cliques = []
        
        # Pre-compute initial degrees for priority? Not strictly necessary if we check all.
        # We will pick candidates from all active vertices.
        
        while remaining_edges and time.time() < deadline - 2.0:
            active_vertices = [u for u in range(n) if rem_adj[u]]
            if not active_vertices:
                break
            
            best_clique = None
            best_score = -1
            best_size = 0
            
            # Generate candidate cliques via greedy expansion from each active vertex
            # To save time, we might limit the number of starting vertices if n is large,
            # but for n=64, checking all is fine.
            for start_v in active_vertices:
                # Greedy expansion
                C = [start_v]
                P = set(rem_adj[start_v]) # Potential vertices to add
                
                while P:
                    # Pick u in P that maximizes common neighbors in P
                    # To speed up, we can just pick the one with highest degree in P
                    # or highest original degree. Let's use degree in P (common neighbors).
                    best_u = None
                    max_common = -1
                    # Optimization: If P is small, check all. If large, maybe sample?
                    # For n=64, P is at most 63. Checking all is fast.
                    for u in P:
                        common = len(P & rem_adj[u])
                        if common > max_common:
                            max_common = common
                            best_u = u
                    if best_u is None:
                        break
                    C.append(best_u)
                    P &= rem_adj[best_u]
                
                # Evaluate this clique
                size = len(C)
                if size < 2:
                    continue
                    
                # Count how many edges of this clique are in remaining_edges
                # Since C is a clique in rem_adj, all edges in C are in remaining_edges
                # Wait, rem_adj is updated as we remove edges.
                # So if C is a clique in rem_adj, all pairs in C are in remaining_edges.
                # So edge_count = size * (size - 1) / 2
                edge_count = size * (size - 1) // 2
                score = edge_count / size
                
                if score > best_score or (score == best_score and size > best_size):
                    best_score = score
                    best_size = size
                    best_clique = C
            
            if best_clique is None:
                # Fallback: pick any edge
                u, v = next(iter(remaining_edges))
                cliques.append([u, v])
                remaining_edges.discard((u, v))
                rem_adj[u].discard(v)
                rem_adj[v].discard(u)
            else:
                cliques.append(best_clique)
                # Remove all edges in best_clique from remaining
                for i in range(len(best_clique)):
                    for j in range(i + 1, len(best_clique)):
                        a, b = best_clique[i], best_clique[j]
                        e = (a, b) if a < b else (b, a)
                        remaining_edges.discard(e)
                        rem_adj[a].discard(b)
                        rem_adj[b].discard(a)
        
        # Add any leftover edges as 2-cliques
        for e in remaining_edges:
            cliques.append([e[0], e[1]])
            
        return cliques

    def is_valid(partition, edges_set, G_adj):
        seen = set()
        for C in partition:
            if len(C) < 2:
                return False
            s = set(C)
            if len(s) != len(C):
                return False
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    a, b = C[i], C[j]
                    if b not in G_adj[a]:
                        return False
                    e = (a, b) if a < b else (b, a)
                    if e in seen:
                        return False
                    seen.add(e)
        return seen == edges_set

    def merge_pass(cliques, G_adj, deadline):
        """
        Repeatedly try to merge pairs of cliques if their union is a clique.
        """
        changed = True
        while changed and time.time() < deadline - 0.5:
            changed = False
            T = len(cliques)
            # Sort cliques by size descending to help merging larger ones?
            # Or just iterate.
            for i in range(T):
                if changed:
                    break
                Ci = set(cliques[i])
                for j in range(i + 1, T):
                    Cj = set(cliques[j])
                    # Quick check: if they don't share vertices, union is not a clique
                    # unless they are disjoint and we are just checking if union is a clique.
                    # Union of two disjoint cliques is a clique iff all edges between them exist.
                    union = Ci | Cj
                    ok = True
                    # Check if union is a clique
                    # We can check if for all u in Ci, u is connected to all v in Cj
                    # And Ci, Cj are already cliques.
                    for u in Ci:
                        for v in Cj:
                            if v not in G_adj[u]:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    
                    # If ok, merge
                    cliques[i] = list(union)
                    del cliques[j]
                    changed = True
                    break
        return cliques

    def solve(n, p, seed, budget):
        edges = gen_edges(n, p, seed)
        edges_set = set(edges)
        if not edges:
            return []
        
        G_adj = [set() for _ in range(n)]
        for u, v in edges:
            G_adj[u].add(v)
            G_adj[v].add(u)

        # Allocate 80% of budget to greedy, 20% to merge
        greedy_deadline = time.time() + (budget - time.time()) * 0.8
        merge_deadline = budget
        
        partition = greedy_clique_partition(n, edges_set, greedy_deadline)
        
        # Validate
        if not is_valid(partition, edges_set, G_adj):
            # Fallback to trivial partition if something went wrong
            return [[u, v] for u, v in edges]
            
        partition = merge_pass(partition, G_adj, merge_deadline)
        
        return partition

    result = []
    # Time allocation:
    # n=32 is small. Give it 20s.
    # n=64 is large. Give it the rest.
    t1 = t0 + 20.0
    result.append(solve(32, 0.6, 601, t1))
    
    if time.time() > TOTAL_DEADLINE - 2.0:
        edges = gen_edges(64, 0.55, 602)
        result.append([[u, v] for (u, v) in edges])
    else:
        result.append(solve(64, 0.55, 602, TOTAL_DEADLINE - 1.0))
        
    return result
# EVOLVE-BLOCK-END
