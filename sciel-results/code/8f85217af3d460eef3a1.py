<<<<<<< SEARCH
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 16
    total_edges = n * (n - 1) // 2  # 120
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def check_not_3_colorable(adj_mask):
        """Check if graph with adjacency masks is NOT 3-colorable."""
        order = sorted(range(n), key=lambda x: bin(adj_mask[x]).count('1'), reverse=True)
        
        colors = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            
            forbidden = 0
            # Iterate neighbors
            mask = adj_mask[v]
            while mask:
                bit = mask & (-mask)
                nb = bit.bit_length() - 1
                c = colors[nb]
                if c != -1:
                    forbidden |= (1 << c)
                mask ^= bit
            
            available = 7 & ~forbidden
            
            if available == 0:
                return False
            
            while available:
                bit = available & (-available)
                available ^= bit
                c = bit.bit_length() - 1
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
            
            return False
        
        return not backtrack(0)

    def get_clebsch_adj_mask():
        adj_mask = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                diff = u ^ v
                dist = bin(diff).count('1')
                if dist == 1 or dist == 4:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        return adj_mask

    adj_mask = get_clebsch_adj_mask()
    
    if not check_not_3_colorable(adj_mask):
        return []
        
    current_count = sum(bin(adj_mask[i]).count('1') for i in range(n)) // 2
    best_count = current_count
    best_adj_mask = adj_mask[:]
    
    def can_add_edge(u, v, adj_mask):
        if (adj_mask[u] >> v) & 1:
            return False
        common = adj_mask[u] & adj_mask[v]
        if common != 0:
            return False
        return True

    random.seed(42)
    
    # State: adj_mask is known to be non-3-colorable
    # We maintain the invariant that adj_mask is non-3-colorable.
    
    while time.time() < deadline - 2:
        # Phase 1: Greedy Add
        # Since graph is non-3-colorable, any triangle-free addition preserves this.
        # So we just add all possible triangle-free edges.
        added_any = False
        for (u, v) in all_edges:
            if (adj_mask[u] >> v) & 1:
                continue
            if can_add_edge(u, v, adj_mask):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                current_count += 1
                added_any = True
        
        if current_count > best_count:
            best_count = current_count
            best_adj_mask = adj_mask[:]
            
        if not added_any:
            # Phase 2: Local Search / Removal
            # We are at a locally maximal triangle-free graph.
            # Try removing one edge to potentially allow more additions later.
            # We need to ensure the graph remains non-3-colorable after removal.
            
            current_edge_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj_mask[i] >> j) & 1:
                        current_edge_list.append((i, j))
                        
            if not current_edge_list:
                break
                
            random.shuffle(current_edge_list)
            
            improved = False
            for (ru, rv) in current_edge_list:
                if time.time() > deadline:
                    break
                
                # Remove edge
                adj_mask[ru] &= ~(1 << rv)
                adj_mask[rv] &= ~(1 << ru)
                current_count -= 1
                
                # Check if still non-3-colorable
                if check_not_3_colorable(adj_mask):
                    # It is still valid.
                    # Now try to add edges again (Phase 1)
                    # We can just break and let the outer loop handle the add phase.
                    # But we want to be aggressive.
                    # Let's do a quick greedy add to see if we gain.
                    # Actually, just breaking to the outer loop is cleaner and reuses the logic.
                    # However, if we just break, we might not add the edge back immediately.
                    # The outer loop will try to add all possible edges.
                    # This is effectively a "kick".
                    improved = True
                    break
                else:
                    # Revert removal
                    adj_mask[ru] |= (1 << rv)
                    adj_mask[rv] |= (1 << ru)
                    current_count += 1
                    
            if not improved:
                # No single removal led to a valid state?
                # This means the graph is "critically" non-3-colorable?
                # Or we just didn't find a good removal.
                # Try 2-edge removals? Or just restart?
                # For now, if no improvement, we might be stuck.
                # We can try to restart from best_adj_mask with a different seed?
                # Or just break to save time?
                # Let's try to force a change by removing 2 edges?
                # For simplicity, if we can't improve, we stop or shuffle?
                # Since we are maximizing, if we can't find a better path, we stay.
                # But we might be in a local optimum of the "add" phase.
                # If no removal works, we are likely at the global optimum for this seed.
                break

    final_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj_mask[u] >> v) & 1:
                final_edges.append([u, v])
                
    return final_edges
=======
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 16
    total_edges = n * (n - 1) // 2  # 120
    
    # List of all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)
    
    # Build conflict graph for maximum clique
    # Two edges conflict if they share a vertex or form a triangle
    # Conflict graph has num_edges nodes
    # adjacency_mask[i] is a bitmask of edges that conflict with edge i
    conflict_mask = [0] * num_edges
    
    for i in range(num_edges):
        u1, v1 = all_edges[i]
        for j in range(i + 1, num_edges):
            u2, v2 = all_edges[j]
            # Check if they share a vertex
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
            else:
                # Check if they form a triangle
                # Edges (u1, v1) and (u2, v2) form a triangle if 
                # {u1, v1} and {u2, v2} share exactly one vertex? No, they share 0 vertices here.
                # Wait, if they don't share a vertex, they can't form a triangle directly with each other.
                # A triangle is formed by 3 edges.
                # The condition "triangle-free" means no 3 edges form a cycle of length 3.
                # If we pick a set of edges S, it is triangle-free if there is no triangle in S.
                # Two edges e1, e2 are "compatible" (can be in the same triangle-free graph) if they don't form a triangle with any other edge?
                # No, the constraint is on the whole set.
                # However, if e1 and e2 share a vertex, they are part of a potential triangle with any edge connecting their other endpoints.
                # If e1=(u,v) and e2=(w,x), they are disjoint. They don't form a triangle together.
                # So, do they conflict?
                # In the "maximum clique" formulation for triangle-free graphs:
                # We want to find a maximum set of edges such that no 3 form a triangle.
                # This is NOT simply a clique in a graph where edges conflict if they form a triangle.
                # Because a triangle involves 3 edges.
                # The "conflict" relation is not pairwise in the simple sense of "these two edges cannot coexist".
                # Two disjoint edges CAN coexist. They only conflict if there exists a third edge that completes the triangle.
                # So the standard maximum clique reduction doesn't apply directly to "triangle-free" because the constraint is ternary (hypergraph independent set).
                
                # Let's re-read the directive: "Model the problem as a maximum clique search in a custom conflict graph where nodes represent potential edges of K16, and two nodes are adjacent in the conflict graph if the corresponding edges share a vertex or form a triangle"
                # If two edges share a vertex, they are adjacent in conflict graph?
                # If they are adjacent in conflict graph, they CANNOT be in the same clique.
                # If we cannot have two edges sharing a vertex in the same clique, then the clique is a matching!
                # A matching is always triangle-free. But we want the DENSEST triangle-free graph, which is not a matching.
                # So the directive's suggested conflict definition ("share a vertex") would restrict us to matchings, which is suboptimal (max matching size is 8, max triangle-free edges is much higher, e.g. Clebsch has 80).
                
                # Let's look closer. "or form a triangle".
                # Maybe the conflict graph is defined such that we are looking for an INDEPENDENT SET in a hypergraph?
                # Or maybe the directive implies a different structure.
                
                # Let's stick to the previous approach but optimize it, or use a different structural change.
                # The previous approach (greedy + local search) got score 2.0.
                # The directive asks for a STRUCTURAL change.
                
                # Alternative: Use a SAT solver or ILP?
                # We have 120 variables.
                # Constraints:
                # 1. Triangle-free: For every triangle (u,v,w), not all 3 edges are present.
                #    Sum(x_uv, x_vw, x_wu) <= 2
                # 2. Not 3-colorable: This is hard to encode directly as linear constraints.
                #    However, we can check 3-colorability separately.
                
                # Since we need to maximize edges subject to triangle-free AND not-3-colorable:
                # We can iterate over candidate edge counts from high to low?
                # Or use a heuristic search.
                
                # Let's try a different structural approach:
                # Start with a known non-3-colorable triangle-free graph (Clebsch).
                # Use a more sophisticated local search: Simulated Annealing or Tabu Search.
                # Moves: Add edge, Remove edge, Swap edge (remove one, add another).
                # Objective: Maximize edges.
                # Constraint: Must remain triangle-free and non-3-colorable.
                
                # But checking 3-colorability is expensive.
                # We can use the fact that if a graph is non-3-colorable, removing edges might make it 3-colorable.
                # Adding edges to a non-3-colorable graph keeps it non-3-colorable.
                # So, we can safely add any triangle-free edge.
                # The only risk is when we REMOVE an edge: it might become 3-colorable.
                
                # Strategy:
                # 1. Start with Clebsch graph.
                # 2. Greedily add all possible triangle-free edges.
                # 3. Local Search:
                #    - Try removing an edge. If still non-3-colorable, keep removal (temporarily) and try to add more.
                #    - If not non-3-colorable, revert.
                #    - To escape local optima, allow "bad" moves (making it 3-colorable) with penalty? No, invalid.
                #    - Instead, use a "kick" mechanism: remove k edges, check if non-3-colorable, if so, greedily add back.
                
                # The current code does 1-edge removal. Let's try 2-edge or 3-edge removals (kicks).
                # Also, improve the 3-colorability check speed.
                
                pass # Placeholder for logic rewrite
>>>>>>> REPLACE

The directive suggests a maximum clique approach, but as analyzed, the conflict graph definition for "triangle-free" is not a simple pairwise conflict graph suitable for standard Maximum Clique algorithms (which find maximal sets of pairwise compatible nodes). Triangle-freeness is a property of triples of edges.

However, we can interpret "conflict" differently or use a different structural change. A robust structural change is to use **Simulated Annealing** or a more aggressive **Local Search with Kicks** (removing multiple edges) to escape local optima in the space of triangle-free non-3-colorable graphs. The current program stops if a single edge removal doesn't work. We can extend this to try removing 2 or 3 edges simultaneously.

Let's implement a version that:
1. Uses a faster 3-colorability check (bitmask-based, pre-computed neighbor lists).
2. Implements a local search loop that tries:
   - Greedy add.
   - If stuck, try removing 1 edge, then 2 edges, then 3 edges (randomly selected) to see if we can enter a "basin" that allows more edges to be added.
3. Keeps the best valid graph found.

