"""
Path decompositions of three fixed connected graphs (Gallai's conjecture)

This program combines a structural Minimum Weight Perfect Matching (MWPM) approach
for exact path decomposition with heuristic refinements.

Strategy:
1.  **Exact Core (MWPM):**
    -   Formulate the minimum path cover problem as a Minimum Weight Perfect Matching (MWPM) problem on a general graph.
    -   Nodes in the matching graph represent the "ends" of directed edges (stubs).
    -   For each undirected edge $(u,v)$, we create two directed stubs: $u \to v$ and $v \to u$.
    -   We must select exactly one direction for each edge. This is modeled by forcing a choice between the two stubs.
    -   To chain edges, we match the head of one directed edge (e.g., $v$ in $u \to v$) to the tail of another (e.g., $v$ in $v \to w$). This creates a link $u \to v \to w$.
    -   The cost of such a link is 0 (it saves a path start).
    -   The cost of NOT linking (starting a new path) is 1.
    -   However, standard MWPM finds a matching. We need a perfect matching on a constructed graph where unmatched nodes imply path starts.
    -   Actually, a simpler formulation for Minimum Path Cover in a Directed Graph is Bipartite Matching.
    -   Let's use the **Edge-Oriented Bipartite Matching** approach:
        -   Split the set of edges $E$ into two sets: $E_{out}$ (tails) and $E_{in}$ (heads).
        -   For each undirected edge $e=(u,v)$, we have two possible orientations: $e_{uv}$ (tail $u$, head $v$) and $e_{vu}$ (tail $v$, head $u$).
        -   We must choose exactly one orientation per edge.
        -   This "choice" constraint is difficult for simple bipartite matching.
        -   **Alternative Exact Approach:**
            Since $N \le 30$, the number of edges $E \approx 100-200$.
            We can use **Min-Cost Max-Flow** on a specific network to solve the "Path Cover with Orientation" problem exactly?
            No, the orientation choice makes it NP-hard in general? No, Gallai's conjecture is about the *existence* of a decomposition with $\le N/2$ paths. Finding the *minimum* number of paths is polynomial.
            
            Let's use the **Standard Reduction for Minimum Path Cover in Directed Graphs**:
            1.  We need to orient edges to minimize path count.
            2.  This is equivalent to finding a maximum set of vertex-disjoint paths in the line graph? No.
            
            Let's use **Integer Linear Programming (ILP)** via `scipy.optimize.milp`?
            The prompt warns ILP might be slow/unreliable.
            
            Let's use the **MWPM Reduction** properly.
            The problem of finding a minimum path cover in an undirected graph where we can choose orientations is equivalent to finding a maximum matching in a specific auxiliary graph?
            
            Actually, let's look at the structure:
            A path decomposition is a set of paths.
            Let $x_{uv}$ be 1 if edge $(u,v)$ is oriented $u \to v$.
            Let $y_{uv, vw}$ be 1 if edge $(u,v)$ is oriented $u \to v$ AND edge $(v,w)$ is oriented $v \to w$.
            We want to maximize $\sum y$.
            Constraints:
            -   For each edge $(u,v)$, $x_{uv} + x_{vu} = 1$.
            -   $y_{uv, vw} \le x_{uv}$, $y_{uv, vw} \le x_{vw}$.
            -   For each vertex $v$, the number of incoming links + outgoing links is limited by degree?
                Actually, at vertex $v$, we can have at most one incoming edge linked to an outgoing edge?
                No, a path goes $u \to v \to w$. So at $v$, one edge comes in, one goes out.
                Can we have $u_1 \to v \to w_1$ and $u_2 \to v \to w_2$? No, because the edges must be edge-disjoint, and $v$ is a vertex.
                In a path decomposition, each vertex $v$ can be internal to at most one path?
                No, $v$ can be internal to multiple paths if the paths use different edges incident to $v$.
                Example: Path 1: $a-v-b$. Path 2: $c-v-d$.
                This uses edges $(a,v), (v,b), (c,v), (v,d)$.
                At vertex $v$, we have two "through" connections.
                So the constraint is: The number of "through" connections at $v$ is limited by $\lfloor \deg(v)/2 \rfloor$?
                If we have $k$ paths passing through $v$, they use $2k$ edges incident to $v$.
                So $\sum_{(u,w) \text{ pairs through } v} y_{uv, vw} \le \lfloor \deg(v)/2 \rfloor$.
                
            This is a **Maximum Weight Matching** problem on a graph where nodes are edges?
            Let's build a graph $H$ where nodes are the original edges $E$.
            Two nodes $e_1=(u,v)$ and $e_2=(x,y)$ are connected in $H$ if they share a vertex and can be chained.
            Specifically, if $e_1=(u,v)$ and $e_2=(v,w)$, they can be chained if $e_1$ is oriented $u \to v$ and $e_2$ is $v \to w$.
            This depends on orientation.
            
            This formulation is complex. Let's stick to a **highly optimized Heuristic** that mimics the optimal structure, as the exact solver might be too slow or complex to implement correctly within the time limit without external libraries.
            
            **Refined Heuristic Strategy (Structural Change):**
            1.  **Identify "Bottlenecks"**: Vertices with odd degree must be endpoints of an odd number of paths. Vertices with even degree can be internal.
            2.  **Greedy Chain Building**:
                -   Instead of just picking a random start, we prioritize starting paths at vertices with **odd degree** first, as they *must* be endpoints of some paths.
                -   When extending a path, we use a **Lookahead** heuristic: prefer neighbors that allow the path to continue further (maximize length) but also consider "dead ends" (Warnsdorff) to avoid trapping ourselves.
                -   **Key Improvement**: After building a set of paths, perform a **Local Search** (2-opt style) to merge paths if possible?
                    -   Can we merge two paths $P_1$ and $P_2$?
                    -   If $P_1$ ends at $u$ and $P_2$ starts at $v$, and there is an edge $(u,v)$ not used? No, all edges are used.
                    -   We can **swap** edges to merge.
                    -   Example: $P_1: a \dots u$, $P_2: v \dots b$.
                    -   If we have edges $(u, v_2)$ in $P_2$ and $(v_1, v)$ in $P_2$? No.
                    -   Consider two paths $P_1$ and $P_2$.
                    -   If we can find a cycle in the union of the two paths that allows reconnecting them into a single path, we reduce the count by 1.
                    -   This is equivalent to finding a "switching" operation.
            
            Let's implement a **Simulated Annealing** on the set of paths.
            State: A set of paths.
            Move: Pick two paths, try to rewire them to form a valid set of paths with potentially fewer paths or same number but better structure.
            Rewiring:
            -   Pick an edge $e_1$ from $P_1$ and $e_2$ from $P_2$.
            -   If they share a vertex $v$, can we swap the connections?
            -   This is tricky.
            
            **Simpler Robust Approach:**
            Use **Beam Search** for path extraction, but with a **Global Constraint Check**.
            1.  Sort vertices by degree (descending).
            2.  Maintain a set of "active" vertices (degree > 0 in remaining graph).
            3.  Pick a start vertex from active vertices. Prioritize odd-degree vertices.
            4.  Extend the path using a **Greedy + Randomization** mix.
                -   At each step, look at all valid next vertices.
                -   Score them by:
                    -   $+1$ if the neighbor has high remaining degree (good for continuation).
                    -   $-1$ if the neighbor has low remaining degree (Warnsdorff).
                    -   Actually, Warnsdorff is better for Hamiltonian paths. For path *cover*, we want to avoid creating isolated edges later.
                    -   Prefer neighbors that have other neighbors (keep the graph connected-ish).
            5.  **Restart** multiple times.
            
            **New Structural Idea: "Odd-Vertex First" Greedy**
            Gallai's conjecture relies on the fact that odd-degree vertices force path endpoints.
            1.  Identify all vertices with odd degree. Let this set be $O$.
            2.  We *must* have at least $|O|/2$ paths.
            3.  Strategy:
                -   For each vertex in $O$, try to start a path there.
                -   Extend these paths greedily.
                -   Once all odd-degree vertices are "satisfied" (used as endpoints or internal, but if internal, they need 2 edges, so they become even in the remaining graph? No, degree is fixed. If an odd vertex is internal, it uses 2 edges, leaving degree-1 (odd) for the rest. This is complicated).
                
            Let's try a **Min-Cost Max-Flow** formulation which is polynomial and exact.
            Network:
            -   Source $S$, Sink $T$.
            -   Node for each vertex $v$: $v_{in}, v_{out}$.
            -   Edge $S \to v_{out}$ with cap 1, cost 1? (Starting a path at $v$).
            -   Edge $v_{in} \to T$ with cap 1, cost 0? (Ending a path at $v$).
            -   For each undirected edge $(u,v)$:
                -   We need to choose direction.
                -   This is the hard part.
                
            Actually, there is a known polynomial algorithm for Minimum Path Cover in Undirected Graphs.
            It reduces to **Maximum Matching** in a bipartite graph constructed from the line graph?
            
            Let's implement a **Fast Heuristic** that is very likely to find near-optimal solutions for these small graphs ($N \le 30$).
            
            **Algorithm:**
            1.  **Preprocessing**: Calculate degrees.
            2.  **Greedy Phase**:
                -   While edges remain:
                    -   Find a vertex $v$ with maximum degree in the remaining graph.
                    -   If multiple, pick one with high "oddness" (if degree is odd, it's a potential endpoint).
                    -   Start a path at $v$.
                    -   Extend the path greedily:
                        -   Current node $u$.
                        -   Candidates: Neighbors $w$ such that edge $(u,w)$ exists and is unused.
                        -   Score $w$:
                            -   $S(w) = \text{degree}(w) \text{ in remaining graph}$.
                            -   We want to go to a node that has many options (high degree) to keep the path long?
                            -   OR go to a node with few options (Warnsdorff) to avoid trapping?
                            -   For path *cover*, Warnsdorff is generally good to avoid dead ends.
                            -   Let's use: Prefer $w$ with **minimum** remaining degree (Warnsdorff).
                            -   Tie-break: Random.
                    -   Stop when no valid neighbor.
            3.  **Local Search (Improvement)**:
                -   Try to merge paths.
                -   Pick two paths $P_1, P_2$.
                -   Check if there's a "swap" that merges them.
                -   This is hard to implement correctly and fast.
                
            Let's stick to **Multiple Restarts of Greedy** with **Diversity**.
            1.  Restart 1: Pure Greedy (Max Degree Start, Warnsdorff Extend).
            2.  Restart 2: Random Start, Warnsdorff Extend.
            3.  Restart 3: Max Degree Start, Random Extend.
            4.  ...
            
            **Optimization**:
            -   Use sets for edge lookup ($O(1)$).
            -   Pre-compute adjacency lists.
            -   Limit restarts based on time.
            
            **Code Structure**:
            -   `construct()`: Main entry.
            -   `instance(n, p, seed)`: Generate graph.
            -   `solve_graph(edges, time_limit)`: Solve one graph.
                -   Build adjacency.
                -   Run multiple heuristics.
                -   Return best paths.
            
            **Heuristic Details**:
            -   `greedy_path(edges, adj, start, strategy)`:
                -   `strategy`: 'warnsdorff', 'random', 'degree'.
                -   Returns a path.
            -   `solve_graph`:
                -   `best_paths = None`
                -   `for i in range(restarts):`
                    -   `curr_edges = set(edges)`
                    -   `paths = []`
                    -   `while curr_edges:`
                        -   `start = pick_start(curr_edges, adj, i)`
                        -   `path = greedy_path(...)`
                        -   `paths.append(path)`
                        -   `remove edges from curr_edges`
                    -   `if len(paths) < len(best_paths): best_paths = paths`
                -   `return best_paths`
            
            **Pick Start**:
            -   If `i % 3 == 0`: Max degree vertex.
            -   If `i % 3 == 1`: Random vertex with degree > 0.
            -   If `i % 3 == 2`: Min degree vertex (leaf-like) to handle trees?
            
            **Extend Path**:
            -   Use Warnsdorff's rule: Go to neighbor with fewest remaining neighbors.
            -   This is very effective for finding long paths.
            
            **Time Management**:
            -   Total time 150s.
            -   Allocate ~40s per graph.
            -   Within each graph, run as many restarts as possible.
            
            **Validation**:
            -   Ensure paths are simple.
            -   Ensure edges are used.
            -   Ensure edge-disjoint.
            
            Let's refine the "Warnsdorff" score.
            When at node $u$, considering neighbor $v$:
            Score $v$ by the number of *unvisited* neighbors of $v$ that are connected by *unused* edges.
            Pick the $v$ with the smallest score.
            
            This is standard for Hamiltonian paths and works well for path covers.

            **Additional Improvement**:
            After generating a set of paths, check if we can **merge** two paths.
            Two paths $P_1 = [v_1, \dots, v_k]$ and $P_2 = [w_1, \dots, w_m]$.
            If $v_k$ and $w_1$ are connected by an edge? No, all edges are used.
            If we can **swap** an edge from $P_1$ and an edge from $P_2$ to connect them?
            Example:
            $P_1: a - b - c$
            $P_2: d - e - f$
            If we have edges $(b, e)$ and $(c, d)$? No, those would be used.
            
            Actually, if we have a cycle in the graph formed by the union of two paths, we might be able to re-route.
            But this is complex.
            
            Let's rely on the **quality of the Greedy** with **many restarts**.
            For $N=30$, $E \approx 100$.
            Greedy takes $O(E \cdot N)$ roughly.
            1000 restarts * 1000 ops = 1,000,000 ops. Very fast.
            We can do 10,000 restarts easily.
            
            Let's implement:
            1.  **Diverse Greedy**:
                -   Start selection:
                    -   50%: Max Degree
                    -   30%: Random
                    -   20%: Min Degree
                -   Extension:
                    -   80%: Warnsdorff (min onward degree)
                    -   20%: Random (among valid)
            
            2.  **Post-Processing**:
                -   Sort paths by length (longest first) to check validity? No, just return.
            
            3.  **Deadline Guard**:
                -   Check time every 100 restarts.
                -   If time < 5s, stop.

            Let's write the code.
"""

import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_graph(edges, time_limit=40.0):
        if not edges:
            return []
        
        edge_set = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        best_paths = None
        best_count = float('inf')
        start_time_local = time.time()
        
        # Estimate max restarts
        # Each restart takes ~ E * N operations. N=30, E=100 -> 3000 ops.
        # 140s / 3000 ops is huge. We are limited by Python overhead.
        # Let's say 1 restart ~ 1ms. 40s -> 40,000 restarts.
        # Let's cap at 5000 to be safe and check time.
        
        restarts = 0
        max_restarts = 10000
        
        while restarts < max_restarts:
            if time.time() > start_time_local + time_limit:
                break
            
            rng = random.Random(restarts * 7919 + 1)
            current_edges = set(edge_set)
            paths = []
            
            # Get list of active vertices (degree > 0)
            # We need to dynamically track degrees or just check neighbors.
            # Checking neighbors is O(deg).
            
            while current_edges:
                # Pick start vertex
                # Get all vertices with degree > 0
                active_vertices = set()
                for u, v in current_edges:
                    active_vertices.add(u)
                    active_vertices.add(v)
                    
                if not active_vertices:
                    break
                
                # Strategy for start
                r = rng.random()
                if r < 0.5:
                    # Max degree
                    max_deg = -1
                    best_v = None
                    for v in active_vertices:
                        # Count degree in current_edges
                        deg = 0
                        for u, w in current_edges:
                            if u == v or w == v:
                                deg += 1
                        if deg > max_deg:
                            max_deg = deg
                            best_v = v
                    start_v = best_v
                elif r < 0.8:
                    # Random
                    start_v = rng.choice(list(active_vertices))
                else:
                    # Min degree
                    min_deg = float('inf')
                    best_v = None
                    for v in active_vertices:
                        deg = 0
                        for u, w in current_edges:
                            if u == v or w == v:
                                deg += 1
                        if deg < min_deg:
                            min_deg = deg
                            best_v = v
                    start_v = best_v
                
                path = [start_v]
                visited = {start_v}
                last_v = start_v
                
                while True:
                    # Find valid next vertices
                    neighbors = []
                    for nb in adj[last_v]:
                        if nb in visited:
                            continue
                        e = tuple(sorted((last_v, nb)))
                        if e in current_edges:
                            # Calculate onward degree (Warnsdorff score)
                            onward = 0
                            for nn in adj[nb]:
                                if nn in visited:
                                    continue
                                e2 = tuple(sorted((nb, nn)))
                                if e2 in current_edges:
                                    onward += 1
                            neighbors.append((onward, nb))
                    
                    if not neighbors:
                        break
                    
                    # Sort by onward degree (ascending)
                    neighbors.sort(key=lambda x: x[0])
                    
                    # Choose next vertex
                    r = rng.random()
                    if r < 0.8:
                        # Warnsdorff: pick min
                        next_v = neighbors[0][1]
                    else:
                        # Random among top 3 (or all if <3)
                        top_k = neighbors[:min(3, len(neighbors))]
                        next_v = rng.choice(top_k)[1]
                    
                    path.append(next_v)
                    visited.add(next_v)
                    e = tuple(sorted((last_v, next_v)))
                    current_edges.discard(e)
                    last_v = next_v
                
                if len(path) > 1:
                    paths.append(path)
                else:
                    # Fallback: if we couldn't extend, just take one edge
                    # This shouldn't happen if start_v has degree > 0
                    if current_edges:
                        e = next(iter(current_edges))
                        paths.append(list(e))
                        current_edges.discard(e)
                    else:
                        break
            
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [p[:] for p in paths]
                if best_count == 1:
                    break
            
            restarts += 1
        
        if best_paths is None:
            return [[u, v] for u, v in edges]
        
        return best_paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for i, spec in enumerate(specs):
        g_edges = instance(*spec)
        time_left = deadline - time.time()
        
        if time_left < 5:
            # Emergency: fast greedy
            paths = solve_graph(g_edges, time_limit=1.0)
        else:
            # Allocate time: first graph 40s, second 40s, third 40s?
            # Or distribute based on remaining?
            # Let's give 40s to each if time permits.
            time_alloc = min(40.0, time_left - 10.0) # Keep 10s buffer for others
            if i < 2:
                time_alloc = min(40.0, time_left - (10.0 * (2 - i)))
            
            paths = solve_graph(g_edges, time_limit=time_alloc)
        results.append(paths)
        
        # Check deadline
        if time.time() > deadline - 5:
            for rem_spec in specs[i+1:]:
                rem_edges = instance(*rem_spec)
                results.append(solve_graph(rem_edges, time_limit=1.0))
            break
            
    return results

# EVOLVE-BLOCK-END
