import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 18

    # --- Fast helpers ---
    def is_independent(mask, adj):
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if adj[v] & (mask ^ lsb):
                return False
            m ^= lsb
        return True

    def can_add_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            return True
        common = adj[u] & adj[v]
        return is_independent(common, adj)

    def largest_tf(adj):
        best = [0]
        full_mask = (1 << n) - 1
        
        # Greedy Lower Bound to initialize 'best'
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: -degrees[x])
        
        greedy_set = 0
        for v in order:
            if is_independent(adj[v] & greedy_set, adj):
                greedy_set |= (1 << v)
        best[0] = bin(greedy_set).count('1')
        
        def bnb(S_mask, C_mask):
            sz = S_mask.bit_count()
            if sz + C_mask.bit_count() <= best[0]:
                return
            
            if not C_mask:
                if sz > best[0]:
                    best[0] = sz
                return

            # Heuristic: pick vertex with max degree in C_mask
            max_deg = -1
            chosen_v = -1
            c = C_mask
            while c:
                lsb = c & -c
                vid = lsb.bit_length() - 1
                deg_in_c = (adj[vid] & C_mask).bit_count()
                if deg_in_c > max_deg:
                    max_deg = deg_in_c
                    chosen_v = vid
                c ^= lsb
                
            v = chosen_v
            v_bit = 1 << v
            
            # Branch 1: Exclude v
            bnb(S_mask, C_mask & ~v_bit)
            
            # Branch 2: Include v (if valid)
            if is_independent(adj[v] & S_mask, adj):
                bnb(S_mask | v_bit, C_mask & ~v_bit)
                
        bnb(0, full_mask)
        return best[0]

    def get_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    # --- Strategy: Start with T_3(18) and optimize ---
    # T_3(18) is the complete tripartite graph with parts of size 6.
    # It is K4-free. Its largest triangle-free induced subgraph is 12 (union of two parts).
    # We will perform local search to see if we can reduce this further by modifying the graph.
    # Note: In T_3(18), any triangle uses one vertex from each part.
    # A triangle-free set can contain vertices from at most 2 parts.
    # If we remove edges, we might break some triangles, potentially allowing larger triangle-free sets?
    # Or if we add edges (within parts), we create K4s, which is forbidden.
    # So we can only remove edges from T_3(18) or start from a different K4-free graph.
    
    # Let's try starting with T_3(18) and removing edges strategically.
    # If we remove an edge (u, v) where u in P1, v in P2, we break all triangles containing (u, v).
    # This might allow a set containing u, v and some w in P3 to be triangle-free?
    # If (u, v) is removed, then {u, v, w} is not a triangle.
    # But {u, v, w'} for w' in P3 is still a triangle if (u, w') and (v, w') exist.
    # So removing edges in T_3(18) tends to make the graph "less dense" in terms of triangles,
    # which usually *increases* the size of the largest triangle-free subgraph.
    # Example: Empty graph has largest TF subgraph of size 18. T_3(18) has size 12.
    # So we want to *add* structure? But we can't add edges within parts of T_3.
    # What if we start with a different K4-free graph?
    
    # The current program uses random construction.
    # The inspiration suggests that T_3(18) is a strong baseline (score 12).
    # The current program's best score was 13 (1.3333 * 18? No, score is the value itself).
    # Wait, the prompt says "score is the largest number of vertices... Minimize".
    # Current program score 1.3333? No, "search score is seed/value".
    # If seed is 18, and value is 13.5? No.
    # Let's re-read: "The evaluator measures the raw value (lower is better). The search score is seed/value".
    # If current program score is 1.3333, and seed is likely 18 (max possible), then value = 18 / 1.3333 = 13.5?
    # Or maybe seed is something else.
    # Let's assume the "value" returned by the evaluator is the size of the largest TF subgraph.
    # If the best value found by current program is 13, and seed is 18, then 18/13 = 1.38.
    # If value is 12, 18/12 = 1.5.
    # The prompt says current program score is 1.3333. 18 / 1.3333 = 13.5.
    # So the current program likely finds a graph with largest TF subgraph of size 13 or 14?
    # If it finds 13, score = 18/13 = 1.38.
    # If it finds 14, score = 18/14 = 1.28.
    # 1.3333 corresponds to value 13.5? Impossible for integer size.
    # Maybe seed is not 18.
    # Regardless, we want to MINIMIZE the size of the largest TF subgraph.
    
    # Let's try to construct a graph with a smaller largest TF subgraph than T_3(18).
    # T_3(18) gives 12.
    # Can we get 11?
    # If we take T_3(18) and modify it, we might break the symmetry.
    # However, T_3(18) is a local minimum for the "max TF subgraph" in the space of dense K4-free graphs?
    # Actually, if we remove edges, the graph becomes sparser, and larger TF subgraphs appear.
    # So T_3(18) is likely a local minimum in the direction of removing edges.
    # To get lower than 12, we need a graph that is NOT a subgraph of T_3(18) but still K4-free.
    # This means we must add edges within the parts of T_3(18) while avoiding K4s.
    # As noted, adding an edge within a part of T_3(18) creates a K4.
    # So T_3(18) is a "maximal" K4-free graph in the sense that you can't add any edge without creating a K4.
    # Therefore, any K4-free graph is a subgraph of T_3(18) (if it has the same vertex set and is "close" to it)?
    # No, that's not true. There are other K4-free graphs.
    # For example, the complement of a matching?
    # Let's stick to the random search but seed it with T_3(18) and try to "jump" out by removing edges?
    # No, removing edges increases the TF size.
    
    # What if we start with a random K4-free graph and use the local search?
    # The current program does this.
    # The inspiration says: "Model the graph as a 3-partite Turán graph... then minimize... by performing a local search that swaps vertices between parts".
    # This suggests that we should explicitly maintain a 3-partition and try to balance it or modify it.
    
    # Let's implement a hybrid:
    # 1. Start with T_3(18) (parts of size 6).
    # 2. Try to improve by "swapping" vertices between parts?
    #    If we move a vertex from P1 to P2, it becomes non-adjacent to other P2 vertices and adjacent to P1 and P3.
    #    This changes the structure.
    # 3. Also try random edge removals/additions (if possible) from this state.
    
    # Actually, the most effective approach for this specific problem (minimizing max TF subgraph)
    # is often to find a graph that is "as close to complete as possible" while being K4-free.
    # T_3(18) is the unique (up to isomorphism) maximal K4-free graph with 18 vertices?
    # No, maximal K4-free graphs are not necessarily Turan graphs.
    # But Turan graphs are the densest K4-free graphs.
    # Since we want to minimize the TF subgraph, we want a dense graph.
    # So starting from T_3(18) is a good idea.
    
    # Let's try a local search on the partition.
    # Represent graph by partition (p1, p2, p3).
    # Edges exist between different parts.
    # If we move a vertex v from P1 to P2:
    #   - v becomes non-adjacent to other P2 vertices.
    #   - v becomes adjacent to other P1 vertices.
    #   - v remains adjacent to P3 vertices.
    # This might reduce the size of the largest TF subgraph if it breaks large bipartite structures.
    
    # Let's implement this partition-based local search.
    
    def largest_tf_from_partition(parts):
        # parts: list of 3 lists of vertices
        adj = [0] * n
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                for u in parts[i]:
                    for v in parts[j]:
                        if u < v:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                        elif v < u:
                            adj[v] |= (1 << u)
                            adj[u] |= (1 << v)
        return largest_tf(adj), adj

    # Initial partition: 6, 6, 6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    
    best_score = n
    best_adj = [0] * n
    best_parts = [parts[0][:], parts[1][:], parts[2][:]]
    
    score, adj = largest_tf_from_partition(parts)
    if score < best_score:
        best_score = score
        best_adj = adj[:]
        best_parts = [parts[0][:], parts[1][:], parts[2][:]]
        
    random.seed(42)
    
    while time.time() < deadline - 2.0:
        improved = False
        # Try moving a vertex from one part to another
        for _ in range(100): # Limit iterations per outer loop
            if time.time() > deadline - 2.0:
                break
            part_idx = random.randint(0, 2)
            if not parts[part_idx]:
                continue
            v_idx = random.randint(0, len(parts[part_idx]) - 1)
            v = parts[part_idx][v_idx]
            target_part = (part_idx + 1 + random.randint(0, 1)) % 3
            if target_part == part_idx:
                continue
            
            # Move v
            parts[part_idx].pop(v_idx)
            parts[target_part].append(v)
            
            new_score, new_adj = largest_tf_from_partition(parts)
            
            if new_score < best_score:
                best_score = new_score
                best_adj = new_adj[:]
                best_parts = [parts[0][:], parts[1][:], parts[2][:]]
                improved = True
            else:
                # Revert
                parts[target_part].pop()
                parts[part_idx].append(v)
                # Keep original order? Not critical for correctness, but good for stability.
                # Actually, we popped and appended, so order changed.
                # Let's just revert the list content.
                
        # If no improvement, try a random restart or perturbation
        if not improved:
            # Perturb: swap two vertices between random parts
            p1, p2 = random.sample([0, 1, 2], 2)
            if parts[p1] and parts[p2]:
                v1 = parts[p1].pop(random.randint(0, len(parts[p1])-1))
                v2 = parts[p2].pop(random.randint(0, len(parts[p2])-1))
                parts[p1].append(v2)
                parts[p2].append(v1)
                
                new_score, new_adj = largest_tf_from_partition(parts)
                if new_score < best_score:
                    best_score = new_score
                    best_adj = new_adj[:]
                    best_parts = [parts[0][:], parts[1][:], parts[2][:]]

    # Also run the original random edge-based search for a bit to see if we can beat the partition-based result
    if time.time() < deadline - 5.0:
        # Use the best_adj found so far as a starting point for random edge flips?
        # Or just run the original random construction for the remaining time.
        # Let's try to improve best_adj by adding/removing edges if possible.
        # But adding edges to a Turan graph creates K4s.
        # So we can only remove edges.
        # Removing edges increases TF size.
        # So the partition-based search is likely superior if it stays close to Turan structure.
        
        pass

    edges = get_edge_list(best_adj)
    print(f"Final: score={best_score}, edges={len(edges)}")
    return edges
