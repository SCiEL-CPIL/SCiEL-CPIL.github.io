"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
from itertools import combinations

N = 40

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    # Precompute potential edges (u, v) with u < v
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    edge_set = set(all_edges)
    
    # Target cycle lengths: 3 to 40
    target_lengths = list(range(3, N + 1))
    
    # 1. Theoretical Lower Bound Construction
    # A graph with N edges is unicyclic. If the single cycle is of length N (Hamiltonian),
    # it contains cycles of length N and 2 (not allowed) and N-1 (by removing an edge? No, removing an edge from a cycle makes a path).
    # Actually, a unicyclic graph with a Hamiltonian cycle is just the cycle C_N.
    # Does C_N contain a cycle of length L for all 3 <= L <= N? No.
    # It only contains the cycle of length N.
    # So N edges is not enough.
    
    # What is the minimum number of edges for a pancyclic graph?
    # It is known that m >= N + floor(N/2) - 2? No.
    # Let's look at the "Fan" construction again.
    # The seed has 2N - 3 edges.
    # Can we do better?
    
    # Consider a graph formed by a Hamiltonian path 0-1-...-N-1 plus chords.
    # If we add chords (0, 2), (0, 4), ..., we cover even lengths?
    # No, we need ALL lengths.
    
    # Let's try a different constructive approach: "Ladder" or "Grid".
    # A 2x20 grid has 40 vertices.
    # Edges: 2*19 (horizontal) + 20 (vertical) = 58 edges.
    # Is a 2x20 grid pancyclic?
    # It contains cycles of length 4, 6, 8, ..., 40 (rectangles).
    # Does it contain odd cycles? No, it's bipartite.
    # So we need to add edges to break bipartiteness or use a different structure.
    
    # Add one edge to make it non-bipartite?
    # If we add an edge between two vertices of the same color, we create an odd cycle.
    # But will we get ALL lengths?
    
    # Let's try to construct a graph with ~N + N/2 edges.
    # Strategy:
    # 1. Start with a Hamiltonian cycle (N edges).
    # 2. Add chords to cover all other lengths.
    # Each chord (i, j) creates cycles of length (j-i+1) and (N-(j-i)+1).
    # If we add chords (0, k) for k=2..N-2, we cover lengths 3..N-1.
    # This is the seed.
    
    # Can we share chords?
    # If we add chord (i, j) and (j, k), do we get new cycles?
    # Yes, but it's complex to count.
    
    # Let's try a heuristic search to find a graph with < 77 edges.
    # We will use a "greedy removal" strategy starting from the seed.
    # But first, we need a fast way to check if a graph is pancyclic.
    
    # Fast Pancyclic Check:
    # For a given graph, we need to find a cycle of length L for each L in 3..N.
    # This is equivalent to finding a cycle of length L in a graph with 40 vertices.
    # For L close to N, we can look for a Hamiltonian cycle in a subgraph? No.
    # For small L, we can use DFS.
    
    # Since N=40 is small, we can use a bitset-based DP for Hamiltonian cycle?
    # 2^40 is too large.
    
    # However, for L < N, we can use a simpler check.
    # Actually, if the graph is "dense" enough, it's likely pancyclic.
    
    # Let's try to improve the seed by removing edges that are "redundant".
    # An edge is redundant if its removal does not destroy any cycle of any length.
    
    # To check this, we can precompute the set of cycles for each length in the seed.
    # If a cycle of length L uses edge e, and there is another cycle of length L that does NOT use e, then e might be redundant for length L.
    # If e is redundant for ALL lengths, it can be removed.
    
    # This is a strong condition.
    
    # Let's implement a local search:
    # 1. Start with seed.
    # 2. For each edge, try to remove it.
    # 3. Check if the new graph is pancyclic.
    # 4. If yes, keep the removal.
    # 5. Repeat until no edge can be removed.
    
    # To speed up the check, we can cache the "witness" cycles.
    # If we have a set of witness cycles (one for each length), and removing an edge breaks none of them, then the graph is still pancyclic.
    # This is a sufficient condition, not necessary. But it's fast.
    
    def get_seed_graph():
        edges = set([(0, i) for i in range(1, N)] + [(i, i+1) for i in range(1, N-1)])
        # Witness cycles for seed:
        # L=3: [0, 1, 2, 0] -> edges (0,1), (1,2), (2,0)
        # L=k: [0, 1, ..., k-1, 0] -> edges (0,1)..(k-2, k-1), (k-1, 0)
        # L=N: [0, 1, ..., N-1, 0] -> edges (0,1)..(N-2, N-1), (N-1, 0)
        
        # Note: The seed construction in the current code returns:
        # edges: (0,i) for i=1..39 and (i,i+1) for i=1..38.
        # Wait, (0,39) is in the first set. (38,39) is in the second.
        # The cycle for L=N is 0-1-...-39-0.
        # Edges: (0,1), (1,2), ..., (38,39), (39,0).
        # All these edges are in the set.
        
        # Cycle for L=k (3<=k<=39):
        # 0-1-...-(k-1)-0.
        # Edges: (0,1), (1,2), ..., (k-2, k-1), (k-1, 0).
        # All these edges are in the set.
        
        witnesses = {}
        for L in range(3, N+1):
            if L == N:
                cyc = list(range(N))
            else:
                cyc = list(range(L))
            # Store the edges of this cycle
            edges_in_cyc = set()
            for i in range(len(cyc)):
                u = cyc[i]
                v = cyc[(i+1)%len(cyc)]
                edges_in_cyc.add((min(u,v), max(u,v)))
            witnesses[L] = edges_in_cyc
            
        return edges, witnesses

    edges, witnesses = get_seed_graph()
    
    # Local Search: Try to remove edges
    # We will iterate until no improvement is found.
    improved = True
    while improved:
        improved = False
        # Shuffle edges to break ties
        current_edges_list = list(edges)
        random.shuffle(current_edges_list)
        
        for e in current_edges_list:
            if time.time() > deadline:
                break
            if e not in edges:
                continue
                
            new_edges = edges - {e}
            
            # Check if the witness cycles are still intact
            # If all witness cycles are intact, the graph is still pancyclic.
            valid = True
            for L, cyc_edges in witnesses.items():
                if not cyc_edges.issubset(new_edges):
                    valid = False
                    break
            
            if valid:
                edges = new_edges
                improved = True
    
    # If we improved, we have a valid graph with fewer edges.
    # We need to return the cycles.
    # The witness cycles are still valid because we only removed edges that didn't break them.
    
    final_edges = list(edges)
    final_cycles = []
    for L in range(3, N+1):
        # Reconstruct the cycle from the witness edges?
        # Or just use the original witness cycles since they are still present.
        if L == N:
            cyc = list(range(N))
        else:
            cyc = list(range(L))
        final_cycles.append(cyc)
        
    # Verification (optional, but good for debugging)
    # Check if all cycles are valid in the final graph
    edge_set_final = set(final_edges)
    for cyc in final_cycles:
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i+1)%len(cyc)]
            if (min(u,v), max(u,v)) not in edge_set_final:
                # This should not happen if our logic is correct
                pass

    return {'edges': final_edges, 'cycles': final_cycles}
# EVOLVE-BLOCK-END
