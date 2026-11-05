import random
import time
import sys

def construct():
    n = 20
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We stop at 145s to be safe.
    deadline = start_time + 145.0

    # Precompute all possible edges
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
    m_total = len(all_edges)  # 190

    # Bitmask representation of adjacency
    # adj[u] is an integer where bit v is set if (u, v) is an edge
    def popcount(x):
        return bin(x).count('1')

    def is_valid_add(adj, u, v):
        """
        Check if adding edge (u, v) keeps the graph K2,3-free.
        A graph is K2,3-free if no two vertices have >= 3 common neighbors.
        Adding (u, v) creates new pairs of vertices that might share neighbors.
        
        The condition |N(x) ∩ N(y)| <= 2 must hold for all pairs {x, y}.
        
        When we add (u, v):
        1. The pair {u, v} now shares all neighbors of u and all neighbors of v? 
           No. N(u) and N(v) are updated.
           New N(u) = old N(u) ∪ {v}
           New N(v) = old N(v) ∪ {u}
           
           Common neighbors of u and v after adding:
           (old N(u) ∪ {v}) ∩ (old N(v) ∪ {u})
           = (old N(u) ∩ old N(v)) ∪ (old N(u) ∩ {u}) ∪ ({v} ∩ old N(v)) ∪ ({v} ∩ {u})
           Since u is not in N(u) and v is not in N(v) in a simple graph without self-loops:
           = old N(u) ∩ old N(v)
           So we must check if popcount(adj[u] & adj[v]) <= 2.
           
        2. For any other vertex w, the pairs {u, w} and {v, w} are affected.
           For pair {u, w}:
           New N(u) = old N(u) ∪ {v}
           N(w) is unchanged.
           Common neighbors = (old N(u) ∪ {v}) ∩ N(w)
           = (old N(u) ∩ N(w)) ∪ ({v} ∩ N(w))
           The size is |old N(u) ∩ N(w)| + (1 if v in N(w) else 0).
           We need this to be <= 2.
           Note: v is in N(w) iff w is in N(v).
           So for all w in N(u) (old), we need |old N(v) ∩ N(w)| + 1 <= 2 => |old N(v) ∩ N(w)| <= 1.
           Wait, the condition is for ALL w, not just w in N(u).
           If w is not in N(u), then old N(u) ∩ N(w) doesn't change, but v is added to N(u).
           So common neighbors of u and w become (old N(u) ∩ N(w)) ∪ ({v} if v in N(w) else {}).
           If v is in N(w), the count increases by 1.
           So we need to check: for all w != u, v:
           |old N(u) ∩ N(w)| + (1 if w in N(v) else 0) <= 2
           
           Similarly for pair {v, w}:
           |old N(v) ∩ N(w)| + (1 if w in N(u) else 0) <= 2

        Let's refine the checks:
        Let U = adj[u], V = adj[v].
        
        Check A: Pair (u, v).
        popcount(U & V) <= 2
        
        Check B: Pairs (u, w) for w != u, v.
        New common neighbors = (U ∪ {v}) ∩ N(w) = (U ∩ N(w)) ∪ ({v} ∩ N(w))
        Size = popcount(U & N(w)) + (1 if (V >> w) & 1 else 0)
        We need this <= 2 for all w.
        Since U & N(w) is the old common neighbors, let C_uw = popcount(U & adj[w]).
        If w is in V (i.e., w is a neighbor of v), then new count is C_uw + 1.
        If w is not in V, new count is C_uw.
        So:
        For w in V (w != u, v): C_uw <= 1
        For w not in V (w != u, v): C_uw <= 2
        
        Check C: Pairs (v, w) for w != u, v.
        New common neighbors = (V ∪ {u}) ∩ N(w) = (V ∩ N(w)) ∪ ({u} ∩ N(w))
        Size = popcount(V & N(w)) + (1 if (U >> w) & 1 else 0)
        We need this <= 2 for all w.
        Let C_vw = popcount(V & adj[w]).
        If w is in U (i.e., w is a neighbor of u), then new count is C_vw + 1.
        If w is not in U, new count is C_vw.
        So:
        For w in U (w != u, v): C_vw <= 1
        For w not in U (w != u, v): C_vw <= 2

        Optimization:
        We only need to check w where the count might exceed 2.
        For Check B:
        - If w in V: we need popcount(U & adj[w]) <= 1.
        - If w not in V: we need popcount(U & adj[w]) <= 2.
          But if popcount(U & adj[w]) <= 1, it's automatically <= 2.
          So we only need to verify popcount(U & adj[w]) <= 2 for w not in V?
          Actually, if w is not in V, the count is just C_uw. If C_uw was 2, it stays 2 (valid).
          If C_uw was 3, it was already invalid before adding the edge.
          Since the graph was valid before, C_uw <= 2 for all w.
          So for w not in V, the condition is automatically satisfied?
          Yes, because the graph was K2,3-free before.
          The only risk is if the addition *increases* the count.
          The count increases only if v is a neighbor of w (i.e., w in V).
          So for w in V, the count increases by 1. It was <= 2, so it becomes <= 3.
          We need it to be <= 2, so we must ensure it was <= 1 before.
          So Check B reduces to: For all w in V (w != u, v), popcount(U & adj[w]) <= 1.
          
        Similarly for Check C:
        For all w in U (w != u, v), popcount(V & adj[w]) <= 1.

        So the full check is:
        1. popcount(U & V) <= 2
        2. For all w in V, w != u: popcount(U & adj[w]) <= 1
           (Note: w=u is not in V because no self-loops)
        3. For all w in U, w != v: popcount(V & adj[w]) <= 1
        """
        U = adj[u]
        V = adj[v]
        
        # 1. Check pair (u, v)
        if popcount(U & V) > 2:
            return False
        
        # 2. Check pairs (u, w) for w in V
        # w is a neighbor of v.
        # We need popcount(U & adj[w]) <= 1
        temp = V
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp ^= lsb
            if w == u:
                continue
            # Check if u and w have > 1 common neighbor
            if popcount(U & adj[w]) > 1:
                return False
        
        # 3. Check pairs (v, w) for w in U
        # w is a neighbor of u.
        # We need popcount(V & adj[w]) <= 1
        temp = U
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp ^= lsb
            if w == v:
                continue
            # Check if v and w have > 1 common neighbor
            if popcount(V & adj[w]) > 1:
                return False
        
        return True

    def add_edge(adj, u, v):
        adj[u] |= 1 << v
        adj[v] |= 1 << u

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def edge_count(adj):
        c = 0
        for u in range(n):
            c += popcount(adj[u])
        return c // 2

    def greedy_build(order):
        adj = [0] * n
        count = 0
        for u, v in order:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
                count += 1
        return adj, count

    def local_search(adj, count):
        """
        Iteratively try to add edges and perform 1-1 swaps.
        """
        adj = list(adj)
        improved = True
        while improved:
            improved = False
            # Try adding any missing edge
            for u, v in all_edges:
                if time.time() > deadline:
                    return adj, count
                if not ((adj[u] >> v) & 1):
                    if is_valid_add(adj, u, v):
                        add_edge(adj, u, v)
                        count += 1
                        improved = True
            if time.time() > deadline:
                return adj, count
            
            # Try 1-1 swaps: remove (a, b), add (c, d)
            # To be efficient, we can iterate over present edges to remove
            present_edges = [(u, v) for u, v in all_edges if (adj[u] >> v) & 1]
            for a, b in present_edges:
                if time.time() > deadline:
                    return adj, count
                # Temporarily remove (a, b)
                remove_edge(adj, a, b)
                count -= 1
                
                # Try to add any missing edge
                for c, d in all_edges:
                    if time.time() > deadline:
                        add_edge(adj, a, b)
                        count += 1
                        return adj, count
                    if c == a and d == b:
                        continue
                    if (adj[c] >> d) & 1:
                        continue
                    if is_valid_add(adj, c, d):
                        add_edge(adj, c, d)
                        count += 1
                        # We swapped, count is same, but structure changed.
                        # This might allow further additions.
                        improved = True
                        break
                # Restore (a, b) if we didn't break out with a successful swap that we kept
                # Note: if we added (c, d), we kept it. We removed (a, b).
                # So the state is valid and count is unchanged.
                # If we didn't find a swap, we must restore (a, b).
                if not improved:
                    add_edge(adj, a, b)
                    count += 1
                else:
                    # We accepted the swap. Break out of inner loop.
                    # The outer loop will restart.
                    break
        
        return adj, count

    def simulated_annealing(adj, count, steps=5000):
        adj = list(adj)
        curr_count = count
        best_adj = list(adj)
        best_count = curr_count
        T = 1.0
        cool = 0.9999
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            move_type = random.random()
            
            if move_type < 0.5:
                # Add a random missing edge
                u, v = random.choice(all_edges)
                if not ((adj[u] >> v) & 1):
                    if is_valid_add(adj, u, v):
                        add_edge(adj, u, v)
                        curr_count += 1
                        if curr_count > best_count:
                            best_count = curr_count
                            best_adj = list(adj)
                    else:
                        T *= 0.95
            elif move_type < 0.8:
                # Remove a random present edge
                present = [(u, v) for u, v in all_edges if (adj[u] >> v) & 1]
                if present:
                    u, v = random.choice(present)
                    remove_edge(adj, u, v)
                    curr_count -= 1
            else:
                # Swap: remove one, add another
                present = [(u, v) for u, v in all_edges if (adj[u] >> v) & 1]
                if present:
                    u, v = random.choice(present)
                    c, d = random.choice(all_edges)
                    if ((adj[c] >> d) & 1) or (c == u and d == v) or (c == v and d == u):
                        T *= 0.99
                        continue
                    new_adj = list(adj)
                    remove_edge(new_adj, u, v)
                    if is_valid_add(new_adj, c, d):
                        add_edge(new_adj, c, d)
                        adj = new_adj
                        # Count remains same
                    else:
                        T *= 0.95
            
            T *= cool
            if T < 0.01:
                T = 0.01
        
        return best_adj, best_count

    best_adj = [0] * n
    best_count = 0

    # Strategy 1: Multiple Greedy Builds with different orders
    # Order 1: Random
    for trial in range(20):
        if time.time() > deadline:
            break
        order = list(all_edges)
        random.shuffle(order)
        adj, count = greedy_build(order)
        adj, count = local_search(adj, count)
        if count > best_count:
            best_count = count
            best_adj = list(adj)

    # Strategy 2: Greedy with degree heuristic
    # Prioritize edges that connect low-degree vertices? Or high?
    # Usually, balancing degrees is good.
    for trial in range(10):
        if time.time() > deadline:
            break
        adj = [0] * n
        degrees = [0] * n
        count = 0
        remaining = list(all_edges)
        random.shuffle(remaining)
        
        # Sort dynamically? Too slow.
        # Just use a random order but bias towards low degree?
        # Let's stick to random shuffle for simplicity in this loop, 
        # but we can try a specific heuristic in a separate loop.
        
        # Heuristic: Sort edges by sum of degrees of endpoints (ascending)
        # This tends to balance the graph.
        # We need to re-sort as degrees change.
        # Approximation:
        # 1. Start with empty graph.
        # 2. Pick edge (u,v) with min(deg[u]+deg[v]).
        # 3. Add if valid.
        # 4. Repeat.
        
        adj = [0] * n
        degrees = [0] * n
        count = 0
        valid_edges = list(all_edges)
        
        while valid_edges:
            if time.time() > deadline:
                break
            # Find edge with min degree sum
            min_idx = 0
            min_sum = degrees[valid_edges[0][0]] + degrees[valid_edges[0][1]]
            for i in range(1, len(valid_edges)):
                u, v = valid_edges[i]
                s = degrees[u] + degrees[v]
                if s < min_sum:
                    min_sum = s
                    min_idx = i
            
            u, v = valid_edges[min_idx]
            valid_edges.pop(min_idx)
            
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
                degrees[u] += 1
                degrees[v] += 1
                count += 1
        
        adj, count = local_search(adj, count)
        if count > best_count:
            best_count = count
            best_adj = list(adj)

    # Strategy 3: Simulated Annealing from best solution
    for trial in range(5):
        if time.time() > deadline:
            break
        adj2, count2 = simulated_annealing(best_adj, best_count, steps=3000)
        if count2 > best_count:
            best_count = count2
            best_adj = list(adj2)

    # Final polish
    best_adj, best_count = local_search(best_adj, best_count)

    # Convert to edge list
    result = []
    for u, v in all_edges:
        if (best_adj[u] >> v) & 1:
            result.append([u, v])
    return result

if __name__ == "__main__":
    edges = construct()
    print(f"Edges: {len(edges)}")
