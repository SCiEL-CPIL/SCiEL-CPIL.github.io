"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = set()
        # Edges in K
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Identify K and I
        K = set(range(k))
        I = set(range(k, n))
        
        # Neighbors in K for each v in I
        neighbors = {v: set() for v in I}
        for v in I:
            for u in K:
                if (u, v) in edges or (v, u) in edges:
                    neighbors[v].add(u)
        
        # Initialize Mixed Cliques
        # Each is a set of vertices in K
        mixed_cliques = []
        for v in I:
            if neighbors[v]:
                mixed_cliques.append((v, set(neighbors[v])))
        
        # Resolve conflicts in internal edges of K
        # An internal edge (u, w) is covered by clique (v, S) if u, w in S
        # We need to ensure each internal edge is covered at most once.
        
        # To make it efficient, we track which internal edges are already covered
        covered_internal_edges = set()
        
        # We process cliques. If a conflict is found, we split.
        # This is a heuristic.
        # A better way:
        # For each v in I, we want to partition N(v) into sets S_1, ..., S_m
        # such that for any pair (u, w) in any S_j, the edge (u, w) is not covered by any other S'_j' for any v'.
        
        # Let's use a greedy splitting approach.
        # We maintain a list of active mixed cliques.
        active_cliques = []
        for v in I:
            if not neighbors[v]:
                continue
            
            # Start with full neighborhood
            current_set = set(neighbors[v])
            
            # Check for conflicts with existing active cliques
            # Conflict if there is an edge (u, w) in current_set that is already covered.
            # Actually, we need to check if ANY edge induced by current_set is already covered.
            
            # We might need to split current_set.
            # Simple strategy:
            # 1. Check if current_set induces any edge in covered_internal_edges.
            # 2. If yes, remove one vertex involved in such an edge from current_set and put it in a new set?
            #    Or remove the edge from coverage? No, we can't un-cover.
            #    We must split so that no two vertices from a conflicting pair are in the same set.
            
            # Let's refine:
            # We build the sets for v incrementally.
            remaining = set(neighbors[v])
            new_sets_for_v = []
            
            while remaining:
                # Try to find the largest subset S of remaining such that
                # no edge induced by S is in covered_internal_edges
                # AND no edge induced by S conflicts with future? No, just current covered.
                
                # Greedy: Pick a vertex, add others if compatible.
                # Compatibility: For all u, w in S, (u, w) not in covered_internal_edges.
                
                # Start with arbitrary vertex
                start = next(iter(remaining))
                S = {start}
                remaining.remove(start)
                
                # Try to add others
                for u in list(remaining):
                    # Check if u can be added to S
                    # u can be added if for all w in S, (u, w) is NOT in covered_internal_edges
                    # Note: (u, w) is an edge in K, so it exists in the graph.
                    # We just check if it's already covered.
                    can_add = True
                    for w in S:
                        if (min(u, w), max(u, w)) in covered_internal_edges:
                            can_add = False
                            break
                    if can_add:
                        S.add(u)
                        remaining.remove(u)
                
                new_sets_for_v.append(S)
                
                # Mark edges in S as covered
                for u, w in combinations(S, 2):
                    covered_internal_edges.add((min(u, w), max(u, w)))
            
            # Add these sets to active cliques
            for S in new_sets_for_v:
                if S:
                    active_cliques.append((v, S))
        
        # Now, construct the list of cliques
        cliques = []
        
        # Add Mixed Cliques
        for v, S in active_cliques:
            clique_verts = list(S) + [v]
            # Ensure at least 2 vertices
            if len(clique_verts) >= 2:
                cliques.append(clique_verts)
        
        # Identify remaining internal edges in K
        remaining_edges = set()
        for u in K:
            for w in K:
                if u < w:
                    if (u, w) not in covered_internal_edges:
                        remaining_edges.add((u, w))
        
        # Cover remaining edges with Pure K cliques
        # Greedy Clique Cover
        rem_edges = set(remaining_edges)
        while rem_edges:
            # Find a maximal clique in the graph of remaining edges
            # Vertices are K. Edge exists if in rem_edges.
            # Since K is small, we can just pick an edge and expand.
            # Pick arbitrary edge (u, v)
            u, v = next(iter(rem_edges))
            S = {u, v}
            rem_edges.discard((u, v))
            
            # Try to add vertices
            changed = True
            while changed:
                changed = False
                for w in K:
                    if w not in S:
                        # Check if w connects to all in S
                        ok = True
                        for x in S:
                            if (min(w, x), max(w, x)) not in rem_edges:
                                ok = False
                                break
                        if ok:
                            S.add(w)
                            # Remove edges from rem_edges
                            for x in list(S):
                                if x != w:
                                    rem_edges.discard((min(w, x), max(w, x)))
                            changed = True
            
            cliques.append(list(S))
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback to seed behavior if time is up
            # This should not happen
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
        
    return results

# EVOLVE-BLOCK-END
