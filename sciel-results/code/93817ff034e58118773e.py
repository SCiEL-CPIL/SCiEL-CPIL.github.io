import time
import random
import numpy as np
from collections import deque

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 120.0
    
    total_space = q ** n
    
    # Precompute neighbors
    neighbors = [[] for _ in range(total_space)]
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    for i in range(total_space):
        neighbors[i].append(i)
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                neighbors[i].append(i + diff * powers[pos])
                
    # Build masks for fast coverage checks
    masks = [0] * total_space
    for i in range(total_space):
        m = 0
        for nb in neighbors[i]:
            m |= (1 << nb)
        masks[i] = m
        
    full_mask = (1 << total_space) - 1

    # Helper: Convert int to tuple
    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    # Helper: Convert tuple to int
    def tuple_to_int(tup):
        val = 0
        for i in range(n):
            val += tup[i] * powers[i]
        return val

    # Helper: Polish a solution (remove redundant, swap)
    def polish(selected):
        if not selected:
            return selected
            
        # Build counts
        counts = [0] * total_space
        for w in selected:
            for nb in neighbors[w]:
                counts[nb] += 1
                
        # Phase 1: Remove redundant
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            indices = list(range(len(selected)))
            random.shuffle(indices)
            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                removable = True
                for nb in neighbors[w]:
                    if counts[nb] == 1:
                        removable = False
                        break
                if removable:
                    for nb in neighbors[w]:
                        counts[nb] -= 1
                    selected.pop(idx)
                    changed = True
            if not changed:
                break

        # Phase 2: Swaps
        max_swaps = 200
        swaps_made = 0
        while swaps_made < max_swaps and time.time() < deadline - 5:
            if not selected:
                break
            idx = random.randint(0, len(selected) - 1)
            w = selected[idx]
            uniques = [nb for nb in neighbors[w] if counts[nb] == 1]
            if not uniques:
                for nb in neighbors[w]:
                    counts[nb] -= 1
                selected.pop(idx)
                continue
                
            found = False
            best_candidate = None
            is_existing = False
            
            # Check existing
            for w2 in selected:
                if w2 == w:
                    continue
                ok = True
                for u in uniques:
                    if u not in neighbors[w2]:
                        ok = False
                        break
                if ok:
                    best_candidate = w2
                    is_existing = True
                    found = True
                    break
            
            if not found:
                # Scan all space (optimized with bitmask if possible, but list is fine for 2187)
                for w2 in range(total_space):
                    if w2 == w:
                        continue
                    ok = True
                    for u in uniques:
                        if u not in neighbors[w2]:
                            ok = False
                            break
                    if ok:
                        best_candidate = w2
                        is_existing = False
                        found = True
                        break
            
            if found:
                for nb in neighbors[w]:
                    counts[nb] -= 1
                selected.pop(idx)
                if not is_existing:
                    selected.append(best_candidate)
                    for nb in neighbors[best_candidate]:
                        counts[nb] += 1
                swaps_made += 1
            else:
                pass
                
        return selected

    # Max Flow / Min Cut Vertex Cover on Bipartite Graph
    # Left: Codewords from a candidate set (e.g., a random linear code or random sample)
    # Right: Words to be covered
    # Edge (L, R) if L covers R.
    # We want min vertex cover.
    # Standard reduction:
    # Source -> L (cap 1)
    # L -> R (cap INF)
    # R -> Sink (cap 1)
    # Min Cut size = Max Flow = Min Vertex Cover Size.
    # The vertex cover is: {L nodes not reachable from Source in residual graph} U {R nodes reachable from Sink in residual graph? No, standard is: L not in S, R in S?}
    # Let's recall:
    # Min cut (S, T). S contains Source.
    # Cover = (Left \ S) U (Right ∩ S).
    # Let's verify.
    # If L in T, it's in (Left \ S).
    # If R in S, it's in (Right ∩ S).
    # Edges crossing cut:
    # Source->L: if L in T, edge crosses. Cost 1. (Corresponds to L in cover)
    # R->Sink: if R in S, edge crosses. Cost 1. (Corresponds to R in cover)
    # L->R: if L in T and R in S, edge crosses. But INF cap prevents this in min cut.
    # So any min cut corresponds to a valid vertex cover.
    
    def solve_vertex_cover(left_nodes, right_nodes):
        # left_nodes: list of ints (codewords)
        # right_nodes: list of ints (words to cover, usually all space)
        
        # If right_nodes is too large, we can't build the graph efficiently in Python without scipy/networkx.
        # But for 2187 nodes, we can use a simple Dinic's algorithm or Edmonds-Karp.
        # Nodes: Source, Sink, Left nodes, Right nodes.
        # Total nodes: 2 + |L| + |R|.
        # If |L| is small (e.g., 200), and |R| is 2187, total ~ 2400 nodes.
        # Edges: |L| + |R| + sum(degrees).
        # This is feasible.
        
        if not left_nodes or not right_nodes:
            return []
            
        # Map original IDs to graph node IDs
        # 0: Source
        # 1: Sink
        # 2 .. 2+|L|-1: Left nodes
        # 2+|L| .. 2+|L|+|R|-1: Right nodes
        
        n_left = len(left_nodes)
        n_right = len(right_nodes)
        n_nodes = 2 + n_left + n_right
        
        # Adjacency list for Dinic
        # graph[u] = list of [v, cap, reverse_index]
        graph = [[] for _ in range(n_nodes)]
        
        def add_edge(u, v, cap):
            graph[u].append([v, cap, len(graph[v])])
            graph[v].append([u, 0, len(graph[u]) - 1])
            
        source = 0
        sink = 1
        
        # Source -> Left
        for i, l_node in enumerate(left_nodes):
            add_edge(source, 2 + i, 1)
            
        # Right -> Sink
        for j, r_node in enumerate(right_nodes):
            add_edge(2 + n_left + j, sink, 1)
            
        # Left -> Right
        # We need to check coverage.
        # For each left node l, for each right node r, if l covers r, add edge.
        # This is O(|L| * |R|) checks.
        # |L| ~ 200, |R| ~ 2187. ~ 400k checks. Fast.
        
        # Optimization: Precompute neighbor sets for right nodes? No, we have masks.
        # But iterating all R for each L is O(|L|*|R|).
        # We can iterate neighbors of L and check if they are in Right.
        # If Right is the full space, then all neighbors are in Right.
        # So for each L, we just add edges to all its neighbors.
        
        # Create a map from right_node_value to right_index
        right_map = {r: j for j, r in enumerate(right_nodes)}
        
        for i, l_node in enumerate(left_nodes):
            for nb in neighbors[l_node]:
                if nb in right_map:
                    j = right_map[nb]
                    add_edge(2 + i, 2 + n_left + j, float('inf'))
                    
        # Dinic's Algorithm
        def dinic():
            flow = 0
            N = n_nodes
            level = [0] * N
            it = [0] * N
            
            def bfs():
                for i in range(N):
                    level[i] = -1
                q = deque([source])
                level[source] = 0
                while q:
                    v = q.popleft()
                    for e in graph[v]:
                        if e[1] > 0 and level[e[0]] < 0:
                            level[e[0]] = level[v] + 1
                            q.append(e[0])
                return level[sink] >= 0
                
            def dfs(v, f):
                if v == sink:
                    return f
                while it[v] < len(graph[v]):
                    e = graph[v][it[v]]
                    if e[1] > 0 and level[v] < level[e[0]]:
                        d = dfs(e[0], min(f, e[1]))
                        if d > 0:
                            e[1] -= d
                            graph[e[0]][e[2]][1] += d
                            return d
                    it[v] += 1
                return 0
                
            while bfs():
                for i in range(N):
                    it[i] = 0
                while True:
                    f = dfs(source, float('inf'))
                    if f == 0:
                        break
                    flow += f
            return flow
            
        # Run Max Flow
        dinic()
        
        # Find Min Cut (Reachable nodes from Source in Residual Graph)
        reachable = [False] * n_nodes
        q = deque([source])
        reachable[source] = True
        while q:
            v = q.popleft()
            for e in graph[v]:
                if e[1] > 0 and not reachable[e[0]]:
                    reachable[e[0]] = True
                    q.append(e[0])
                    
        # Extract Vertex Cover
        # Cover = (Left \ S) U (Right ∩ S)
        cover = []
        for i, l_node in enumerate(left_nodes):
            if not reachable[2 + i]:
                cover.append(l_node)
        for j, r_node in enumerate(right_nodes):
            if reachable[2 + n_left + j]:
                cover.append(r_node)
                
        return cover

    # Strategy:
    # 1. Generate a candidate set of "left nodes". 
    #    A good candidate set is a random linear code or a random sample.
    #    If we use ALL words as left nodes, the min vertex cover is the minimum dominating set? No, it's the minimum vertex cover of the bipartite graph which is equivalent to min dominating set only if the graph is specific? 
    #    Actually, the min vertex cover in the bipartite graph (L=all words, R=all words) is NOT the min dominating set. 
    #    Wait, the domination problem is: select min L such that every R is covered. This is exactly the set cover problem.
    #    Set cover is NOT equivalent to vertex cover in general. 
    #    However, if we fix the set of Left nodes (candidates), we are solving: "From this set of candidates, pick the minimum number to cover all R".
    #    This is a Set Cover problem restricted to a specific universe of sets (the candidates).
    #    Is Set Cover reducible to Vertex Cover?
    #    Yes, if we construct the bipartite graph as above.
    #    BUT, the min vertex cover in this graph gives the min number of nodes (from L and R) to cover all edges.
    #    This is NOT the same as picking min L to cover all R.
    #    The vertex cover allows picking R nodes too!
    #    If we pick an R node, it "covers" all edges incident to it, i.e., it is "covered" by itself? No.
    #    In the reduction for Set Cover to Vertex Cover, we usually add a large penalty to R nodes or constrain the solution.
    #    Actually, the standard reduction for *Minimum Set Cover* to *Minimum Vertex Cover* is not direct because Set Cover is NP-hard and Vertex Cover is NP-hard, but they are different problems.
    #    
    #    Let's re-read the prompt: "Model the problem as a minimum weight vertex cover in a bipartite graph... solved using max-flow/min-cut... to identify the optimal subset of codewords".
    #    
    #    If we set the cost of Left nodes to 1 and Right nodes to a huge penalty (or infinity), then the min vertex cover will only pick Left nodes.
    #    If we set Source->L cap=1 and R->Sink cap=M (large), then min cut will avoid cutting R->Sink if possible.
    #    If M is large enough, the min cut will correspond to a set of L nodes that covers all R nodes.
    #    Specifically, if M > |L|, then it is cheaper to cut Source->L edges than R->Sink edges.
    #    So, set R->Sink capacity to a large value, e.g., total_space + 1.
    #    Then the min cut will consist of some Source->L edges and possibly some R->Sink edges?
    #    No, if M is large, the min cut will prefer to cut Source->L.
    #    The size of the cut is the number of L nodes selected + M * (number of R nodes not covered? No).
    #    Let's trace:
    #    Cut (S, T). Source in S.
    #    Cost = sum(1 for L in T) + sum(M for R in S).
    #    We want to minimize this.
    #    If we pick a set of L nodes C_L, and let S contain Source and all R nodes that are covered by C_L?
    #    If R is covered by C_L, there is an edge L->R with L in T (selected) and R in S. This edge crosses the cut. But its cap is INF. So this is invalid.
    #    So, if L is in T (selected), then ALL R nodes that it covers MUST be in T.
    #    So if we select L, we must put all its neighbors in T.
    #    Then R->Sink edges for R in T do not cross the cut.
    #    R->Sink edges for R in S cross the cut.
    #    So, if we select L, we put its neighbors in T. Any R not covered by selected L must be in S (since Source is in S, and if there's no path from Source to R via selected Ls? No, R is connected to Source only via Ls. If L is in T, edge Source->L crosses. Path Source->L is broken. So R is not reachable from Source via that L. If R is not reachable from any selected L, it stays in S? 
    #    Wait, Reachability in Residual Graph.
    #    In the residual graph, if Source->L is saturated (flow 1), the reverse edge L->Source has cap 1.
    #    If L->R has flow 0 (not used), cap INF.
    #    
    #    Let's stick to the standard result:
    #    Min Vertex Cover in Bipartite Graph = Max Matching.
    #    This solves the problem: "Find min number of vertices (L or R) such that every edge is incident to a selected vertex."
    #    This is NOT the Set Cover problem.
    #    Set Cover: Select min L such that every R is incident to a selected L.
    #    
    #    The prompt suggests this strategy. Is it valid?
    #    If we set the weight of R nodes to be very high, the min vertex cover will consist only of L nodes.
    #    If the min vertex cover consists only of L nodes, then every edge (L, R) must be incident to a selected L.
    #    This means every R is covered by a selected L.
    #    So, if we force the solution to be in L, we solve Set Cover.
    #    To force it, we can set the capacity of R->Sink to be very large (e.g., 1000).
    #    Then the min cut will avoid cutting R->Sink.
    #    So it will only cut Source->L.
    #    The number of Source->L edges cut is the number of selected L nodes.
    #    So this works!
    
    # Revised Solver with Weights
    def solve_set_cover_via_cut(left_nodes, right_nodes, weight_r=1000):
        if not left_nodes or not right_nodes:
            return []
            
        n_left = len(left_nodes)
        n_right = len(right_nodes)
        n_nodes = 2 + n_left + n_right
        graph = [[] for _ in range(n_nodes)]
        
        def add_edge(u, v, cap):
            graph[u].append([v, cap, len(graph[v])])
            graph[v].append([u, 0, len(graph[u]) - 1])
            
        source = 0
        sink = 1
        
        for i in range(n_left):
            add_edge(source, 2 + i, 1)
            
        for j in range(n_right):
            add_edge(2 + n_left + j, sink, weight_r)
            
        right_map = {r: j for j, r in enumerate(right_nodes)}
        for i, l_node in enumerate(left_nodes):
            for nb in neighbors[l_node]:
                if nb in right_map:
                    j = right_map[nb]
                    add_edge(2 + i, 2 + n_left + j, float('inf'))
                    
        # Dinic
        def dinic():
            flow = 0
            N = n_nodes
            level = [0] * N
            it = [0] * N
            
            def bfs():
                for i in range(N):
                    level[i] = -1
                q = deque([source])
                level[source] = 0
                while q:
                    v = q.popleft()
                    for e in graph[v]:
                        if e[1] > 0 and level[e[0]] < 0:
                            level[e[0]] = level[v] + 1
                            q.append(e[0])
                return level[sink] >= 0
                
            def dfs(v, f):
                if v == sink:
                    return f
                while it[v] < len(graph[v]):
                    e = graph[v][it[v]]
                    if e[1] > 0 and level[v] < level[e[0]]:
                        d = dfs(e[0], min(f, e[1]))
                        if d > 0:
                            e[1] -= d
                            graph[e[0]][e[2]][1] += d
                            return d
                    it[v] += 1
                return 0
                
            while bfs():
                for i in range(N):
                    it[i] = 0
                while True:
                    f = dfs(source, float('inf'))
                    if f == 0:
                        break
                    flow += f
            return flow
            
        dinic()
        
        reachable = [False] * n_nodes
        q = deque([source])
        reachable[source] = True
        while q:
            v = q.popleft()
            for e in graph[v]:
                if e[1] > 0 and not reachable[e[0]]:
                    reachable[e[0]] = True
                    q.append(e[0])
                    
        cover = []
        for i in range(n_left):
            if not reachable[2 + i]:
                cover.append(left_nodes[i])
        # We ignore R nodes in the cover because weight_r is high, so they shouldn't be in S (cut).
        # If some R are in S, it means they were not covered, which implies a huge cost.
        # But since we have a valid set cover (all words), the min cut should be small.
        
        return cover

    best_size = total_space
    best_selected = list(range(total_space))
    
    # Strategy:
    # 1. Try a random linear code as the candidate set (Left nodes).
    # 2. Solve Set Cover via Max Flow.
    # 3. Polish.
    # 4. Repeat with different random codes.
    
    # Generate Random Linear Code
    def generate_random_linear_code(n, q, dim):
        # Generate dim x n matrix over GF(q)
        # Basis vectors
        basis = []
        for _ in range(dim):
            vec = [random.randint(0, q-1) for _ in range(n)]
            basis.append(vec)
            
        # Generate all combinations
        code = set()
        # Since dim is small (e.g., 3-5), 3^dim is small.
        for mask in range(q ** dim):
            vec = [0] * n
            m = mask
            for i in range(dim):
                coeff = m % q
                m //= q
                for j in range(n):
                    vec[j] = (vec[j] + coeff * basis[i][j]) % q
            code.add(tuple_to_int(vec))
        return list(code)
    
    # Try several random linear codes
    for dim in range(2, 6):
        if time.time() > deadline - 10:
            break
            
        for trial in range(5):
            if time.time() > deadline - 5:
                break
                
            candidates = generate_random_linear_code(n, q, dim)
            # Ensure candidates are distinct and valid
            if len(candidates) < len(set(candidates)):
                continue
                
            # Solve
            cover = solve_set_cover_via_cut(candidates, list(range(total_space)), weight_r=1000)
            
            # Check validity
            covered = 0
            for w in cover:
                covered |= masks[w]
            if covered == full_mask:
                if len(cover) < best_size:
                    best_size = len(cover)
                    best_selected = cover[:]
                    if time.time() < deadline - 5:
                        polished = polish(best_selected[:])
                        if len(polished) < best_size:
                            best_size = len(polished)
                            best_selected = polished[:]

    # Fallback: If we didn't find a good solution, use a greedy one
    if best_size > 200: # Threshold
        if time.time() < deadline - 10:
            # Simple Greedy
            covered = 0
            selected = []
            remaining = full_mask
            while remaining and time.time() < deadline - 5:
                best_gain = -1
                best_word = -1
                for w in range(total_space):
                    gain = (masks[w] & remaining).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_word = w
                if best_word == -1:
                    break
                selected.append(best_word)
                covered |= masks[best_word]
                remaining = full_mask ^ covered
            if len(selected) < best_size:
                best_size = len(selected)
                best_selected = selected[:]
                if time.time() < deadline - 5:
                    polished = polish(best_selected[:])
                    if len(polished) < best_size:
                        best_size = len(polished)
                        best_selected = polished[:]

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
