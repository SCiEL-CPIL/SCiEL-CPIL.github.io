"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import random
import time

def construct():
    n = 20
    max_colors = 20
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {(i, j): idx for idx, (i, j) in enumerate(all_edges)}
    E = len(all_edges)
    
    # Precompute conflict graph:
    # conflict[e_idx] is a list of e_idx' such that e_idx and e_idx' share a 4-cycle.
    # Two edges share a 4-cycle if they are in a cycle of length 4.
    # This happens if:
    # 1. They are adjacent in the 4-cycle (share a vertex in G)
    # 2. They are opposite in the 4-cycle (disjoint in G)
    
    # Let's find all 4-cycles first.
    # A 4-cycle is (u, v, w, x) with u<v<w<x? No, just unique sets of 4 vertices that form a cycle.
    # Vertices u, v, w, x. Edges (u,v), (v,w), (w,x), (x,u).
    # We need to check if these edges exist in the complete graph (they do).
    # So any 4 distinct vertices form a 4-cycle?
    # In a simple graph, a 4-cycle is a subgraph isomorphic to C4.
    # The vertices must be distinct.
    # The edges must exist. Since we are constructing a subgraph of K20, edges exist if we include them.
    # But the constraint applies to the *final* graph.
    # The conflict relationship depends on the *structure* of the graph.
    # If we include edges (u,v) and (w,x), do they conflict?
    # They conflict if there is a 4-cycle containing both.
    # A 4-cycle containing (u,v) and (w,x) must use vertices u,v,w,x.
    # The cycle must be u-v-?-?-u.
    # If (u,v) and (w,x) are opposite, the cycle is u-v-w-x-u (requires edges v-w, w-x, x-u) or u-v-x-w-u (requires edges v-x, w-u, x-w? no x-w is same as w-x).
    # So (u,v) and (w,x) conflict if:
    #   (v,w) and (x,u) are in the graph, OR
    #   (v,x) and (w,u) are in the graph.
    # If (u,v) and (v,w) are adjacent, they share vertex v.
    # They conflict if there is a 4-cycle u-v-w-x-u.
    # This requires edges (w,x) and (x,u) to be in the graph.
    # So (u,v) and (v,w) conflict if there exists x != u,v,w such that (w,x) and (x,u) are in the graph.
    
    # This dependency is dynamic! The conflict graph changes as we add edges.
    # "If we include edges A and B, do they form a 4-cycle?" depends on whether other edges C and D are also included.
    # So we cannot precompute a static conflict graph for ALL pairs of edges in K20.
    # We must check the constraint dynamically.
    
    # However, we can observe:
    # Two edges $e_1, e_2$ MUST have different colors IF they are part of a 4-cycle in the *current* partial graph.
    # If they are not yet part of a 4-cycle, they *might* be later.
    # If we assign them the same color, and later complete a 4-cycle containing both, we fail.
    # So we should treat them as conflicting *potentially*.
    # But if we treat all pairs that *can* form a 4-cycle as conflicting, we might be too restrictive?
    # Actually, if $e_1, e_2$ can form a 4-cycle with some other edges, and we assign them the same color, we block any future completion of that 4-cycle.
    # This is a valid strategy: "Proactive Constraint".
    # If $e_1, e_2$ are in a "potential 4-cycle" (i.e., there exist vertices/edges such that adding them completes a C4), then $c(e_1) \neq c(e_2)$ is a safe constraint to impose.
    
    # Let's define "Potential Conflict":
    # Edges $e_a=(u,v)$ and $e_b=(w,x)$ potentially conflict if:
    # 1. They are disjoint ($\{u,v\} \cap \{w,x\} = \emptyset$) and there exist edges $(v,w)$ and $(x,u)$ OR $(v,x)$ and $(w,u)$ in the graph.
    #    Wait, if those edges are NOT in the graph yet, they might be added later.
    #    If we assume they *might* be added, then $e_a$ and $e_b$ must have different colors.
    #    This is very restrictive. In K20, almost any pair of disjoint edges can be part of a 4-cycle if the other 2 edges are present.
    #    Since we want a dense graph, we will likely have many of those "other" edges.
    #    So effectively, ALL pairs of disjoint edges should have different colors?
    #    And ALL pairs of adjacent edges?
    #    If ALL pairs of edges must have different colors, we can have at most 20 edges (one per color).
    #    But the seed program gets ~4.5 * 20 = 90 edges? No, score 4.5789.
    #    Max edges is 190. Score is value/seed? No, "value/seed" where seed is the baseline score?
    #    "search score is value/seed, so 1.0 equals the seed program".
    #    Current score 4.5789.
    #    If the evaluator measures "raw value", and raw value is number of edges...
    #    Let's check the number of edges in the current program.
    #    The current program likely produces around 80-90 edges.
    #    If the score is 4.5, and it's value/seed, maybe the seed value is ~20?
    #    Or maybe the "value" is not just edge count?
    #    "The evaluator measures the raw value (higher is better)."
    #    "Maximize the number of edges."
    #    So Value = Number of Edges.
    #    If Score = 4.5789, and Score = Value / SeedValue, then Value = 4.5789 * SeedValue.
    #    What is SeedValue? Probably the number of edges in the "initial" or "trivial" solution?
    #    Or maybe the "seed" in the prompt refers to the random seed? No, "score is value/seed".
    #    Let's assume the "Seed Program" produced a specific number of edges, say $E_{seed}$.
    #    And the current program produces $E_{curr}$.
    #    $E_{curr} / E_{seed} = 4.5789$.
    #    If $E_{seed}$ was small (e.g. 20), then $E_{curr} \approx 91$.
    #    If $E_{seed}$ was 100, $E_{curr} \approx 457$ (impossible, max 190).
    #    So $E_{seed}$ is likely small, maybe a trivial coloring with few edges.
    #    Or maybe the "Seed" is a specific reference value.
    #    Regardless, we want to maximize edges.
    
    # Back to the constraint:
    # If we enforce $c(e_a) \neq c(e_b)$ for ALL pairs that *could* form a 4-cycle, we are essentially coloring the "4-cycle hypergraph" edges.
    # In K20, the 4-cycles are very dense.
    # If we enforce distinct colors for all edges in any potential 4-cycle, we might be too strict.
    # Example: $e_1=(0,1), e_2=(2,3)$. Potential cycle $0-1-2-3-0$. Requires edges $(1,2)$ and $(3,0)$.
    # If we do NOT include $(1,2)$ or $(3,0)$, then $e_1, e_2$ do NOT form a 4-cycle.
    # So they CAN have the same color.
    # So the constraint is conditional on the other edges.
    
    # This brings us back to the dynamic check.
    # However, we can use the "Proactive" heuristic:
    # When assigning color $c$ to $e$, we check all 4-cycles that *could* be formed by $e$ and existing edges.
    # If a 4-cycle is "complete" (all 4 edges present), colors must be distinct.
    # If a 4-cycle is "partial", we don't need to enforce distinctness yet, BUT if we assign the same color to two edges in a partial 4-cycle, we *block* the completion of that 4-cycle.
    # Since we want to MAXIMIZE edges, blocking a 4-cycle completion might be bad if we wanted to add those edges.
    # But if we add those edges, we'd have to change colors, which is expensive.
    # So a good heuristic is:
    # 1. Try to add as many edges as possible.
    # 2. When adding an edge, ensure it doesn't violate existing complete 4-cycles.
    # 3. Use a color that minimizes the number of "blocked" future 4-cycles?
    
    # Let's stick to the CSP approach but with dynamic conflict detection.
    # To make it fast:
    # For each edge $e$, maintain a list of "conflicting edges" that are ALREADY ASSIGNED and in a complete 4-cycle with $e$.
    # If $e$ is in a complete 4-cycle with $e'$, then $c(e) \neq c(e')$.
    
    # Algorithm:
    # 1. Start with empty graph.
    # 2. Order edges randomly.
    # 3. For each edge $e$:
    #    a. Determine `forbidden_colors` for $e$.
    #       - For each other edge $e'$ already in the graph:
    #         - Check if $e$ and $e'$ are in a complete 4-cycle in the current graph.
    #         - If yes, add $c(e')$ to `forbidden_colors`.
    #    b. If `forbidden_colors` has size 20, skip $e$ (cannot add).
    #    c. Else, pick a color $c$ not in `forbidden_colors`.
    #       - Heuristic: Pick $c$ that is "least used" among neighbors? Or random.
    #    d. Add $e$ with color $c$.
    
    # Complexity:
    # For each new edge, we check against all existing edges ($O(E)$).
    # For each pair $(e, e')$, checking if they are in a 4-cycle takes $O(N)$ or $O(1)$ with precomputation?
    # Checking if $e=(u,v)$ and $e'=(w,x)$ are in a 4-cycle:
    #   Case 1: Disjoint. Check if $(v,w)$ and $(x,u)$ exist OR $(v,x)$ and $(w,u)$ exist.
    #   Case 2: Adjacent (share vertex). Say $v=w$. Check if there exists $z$ such that $(v,z)$ and $(z,u)$ exist? No, the cycle is $u-v-z-?-u$.
    #   If $e=(u,v)$ and $e'=(v,w)$, the cycle is $u-v-w-x-u$. Requires $(w,x)$ and $(x,u)$.
    #   So we need to check if there exists $x$ such that $(w,x)$ and $(x,u)$ are in the graph.
    #   This is an intersection of neighbor lists.
    
    # This check is $O(\text{degree})$.
    # Total time $O(E^2 \cdot \text{avg\_degree})$. With $E \approx 100$, this is $10000 \times 10 \approx 10^5$ operations. Very fast.
    
    # Let's implement this.

    def solve_once(seed):
        random.seed(seed)
        adj = [[-1]*n for _ in range(n)]
        edge_list = list(range(E))
        random.shuffle(edge_list)
        
        # To speed up "is in 4-cycle" check, we can maintain adjacency sets.
        # adj_set[u] = set of v such that (u,v) is in graph.
        adj_set = [set() for _ in range(n)]
        
        placed_edges = []
        
        for e_idx in edge_list:
            u, v = all_edges[e_idx]
            
            forbidden = set()
            
            # Check against all placed edges
            for pe_idx in placed_edges:
                pu, pv = all_edges[pe_idx]
                
                # Check if (u,v) and (pu,pv) are in a 4-cycle
                # Normalize edges
                # Case 1: Disjoint
                if v != pu and u != pv and u != pu and v != pv:
                    # Check path v-pu-?-u or v-pv-?-u?
                    # Cycle u-v-?-?-u.
                    # If edges are (u,v) and (pu,pv).
                    # Possible cycles:
                    # 1. u-v-pu-pv-u. Requires (v,pu) and (pv,u).
                    # 2. u-v-pv-pu-u. Requires (v,pv) and (pu,u).
                    
                    if (v in adj_set[pu] and u in adj_set[pv]):
                        forbidden.add(adj[v][pu])
                    if (v in adj_set[pv] and u in adj_set[pu]):
                        forbidden.add(adj[v][pv])
                        
                else:
                    # Adjacent edges. They must share a vertex.
                    # Possibilities:
                    # 1. v == pu. Edges (u,v) and (v,pv).
                    #    Cycle u-v-pv-x-u. Requires (pv,x) and (x,u).
                    #    Exists x != u,v,pv such that (pv,x) in graph and (x,u) in graph.
                    # 2. v == pv. Edges (u,v) and (pu,v).
                    #    Cycle u-v-pu-x-u. Requires (pu,x) and (x,u).
                    # 3. u == pu. Edges (u,v) and (u,pv).
                    #    Cycle u-v-?-pv-u. Requires (v,x) and (x,pv).
                    # 4. u == pv. Edges (u,v) and (pu,u).
                    #    Cycle u-v-?-pu-u. Requires (v,x) and (x,pu).
                    
                    if v == pu:
                        # Common vertex v. Other vertices u, pv.
                        # Need x such that (pv, x) and (x, u) exist.
                        # x must be in adj_set[pv] and adj_set[u].
                        # Exclude v (already used), u, pv.
                        common = adj_set[pv] & adj_set[u]
                        # x cannot be v?
                        # Cycle u-v-pv-x-u.
                        # If x=v, cycle is u-v-pv-v-u (not simple).
                        # So x != v.
                        # Also x != u, x != pv.
                        # If there is ANY such x, they conflict.
                        # We just need to know if such x exists.
                        # We don't need the color of the "other" edges, just that the cycle EXISTS.
                        # Wait, the constraint is on the colors of the 4 edges.
                        # If the cycle exists, then c(u,v) != c(v,pv) != c(pv,x) != c(x,u).
                        # So c(u,v) != c(v,pv).
                        # So if the cycle exists, we add c(v,pv) to forbidden.
                        # We don't care about c(pv,x) or c(x,u) for this check, because we are checking conflict between (u,v) and (v,pv).
                        # Actually, we need to ensure c(u,v) is distinct from c(v,pv).
                        # So if cycle exists, forbidden.add(adj[v][pv]).
                        
                        # Check existence of x
                        # x in adj_set[pv] and x in adj_set[u] and x != v and x != u and x != pv
                        # Note: u is in adj_set[u]? No.
                        # We just check intersection.
                        # If intersection is not empty (and doesn't contain only invalid nodes), cycle exists.
                        # Since graph is simple, no self loops.
                        # If u == pv, they are not disjoint, but we handled disjoint.
                        # Here v==pu. u and pv are distinct (else triangle, not 4-cycle).
                        # So u != pv.
                        # If there is any x in adj_set[pv] & adj_set[u] with x != v, cycle exists.
                        # Actually, if x=u, edge (pv,u) and (u,u) no.
                        # If x=pv, edge (pv,pv) no.
                        # So any x in intersection is valid unless x==v.
                        # But v is not in adj_set[u] (simple graph, no loops? u!=v).
                        # v is in adj_set[pv]? No, pv!=v.
                        # So any x in intersection is valid.
                        
                        if adj_set[pv] & adj_set[u]:
                            forbidden.add(adj[v][pv])
                            
                    elif v == pv:
                        # Common vertex v. Edges (u,v) and (pu,v).
                        # Cycle u-v-pu-x-u. Requires (pu,x) and (x,u).
                        # x in adj_set[pu] & adj_set[u].
                        if adj_set[pu] & adj_set[u]:
                            forbidden.add(adj[pu][v])
                            
                    elif u == pu:
                        # Common vertex u. Edges (u,v) and (u,pv).
                        # Cycle u-v-x-pv-u. Requires (v,x) and (x,pv).
                        # x in adj_set[v] & adj_set[pv].
                        if adj_set[v] & adj_set[pv]:
                            forbidden.add(adj[u][pv])
                            
                    elif u == pv:
                        # Common vertex u. Edges (u,v) and (pu,u).
                        # Cycle u-v-x-pu-u. Requires (v,x) and (x,pu).
                        # x in adj_set[v] & adj_set[pu].
                        if adj_set[v] & adj_set[pu]:
                            forbidden.add(adj[pu][u])
            
            if len(forbidden) >= max_colors:
                continue
                
            # Pick a color
            # Heuristic: Pick color that minimizes "future conflicts"?
            # Simple: Random from available.
            available = [c for c in range(max_colors) if c not in forbidden]
            c = random.choice(available)
            
            adj[u][v] = c
            adj[v][u] = c
            adj_set[u].add(v)
            adj_set[v].add(u)
            placed_edges.append(e_idx)
            
        result = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j] != -1:
                    result.append([i, j, adj[i][j]])
        return result

    best_edges = []
    # Run multiple restarts
    # The number of restarts can be increased if time allows.
    # 120s limit. Each run is fast.
    for i in range(500):
        res = solve_once(i)
        if len(res) > len(best_edges):
            best_edges = res
            # Optional: break if we reach a high number?
            # Max possible is 190.
            # If we get close to 190, we can stop.
            if len(res) > 150:
                break
                
    return best_edges
# EVOLVE-BLOCK-END
