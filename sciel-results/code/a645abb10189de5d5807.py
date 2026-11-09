"""
Strategy: Greedy Set Cover with Lagrangian-style Scoring.

The problem is to partition the edges of a graph into a minimum number of cliques.
This is equivalent to finding a minimum set of cliques such that every edge is in
exactly one clique. Note: "Partition" usually implies disjointness of elements.
Here, the elements are edges. So the cliques must be edge-disjoint.
Wait, the task says: "every edge must lie in exactly one listed clique".
This implies the cliques form a partition of the edge set.
If two cliques share an edge, that edge is in two cliques -> Invalid.
So we need a set of cliques $C_1, ..., C_k$ such that $\bigcup E(C_i) = E(G)$ and $E(C_i) \cap E(C_j) = \emptyset$ for $i \neq j$.

This is exactly the **Minimum Clique Partition** problem (or Edge Clique Cover with disjointness constraint).
Actually, standard Edge Clique Cover allows overlaps. This requires disjointness.
This is equivalent to coloring the line graph $L(G)$ such that each color class is a clique in $L(G)$ (which corresponds to a star or triangle in $G$? No.
A clique in $L(G)$ corresponds to a set of edges in $G$ that are pairwise adjacent.
Pairwise adjacent edges in $G$ form either a star (all share a center) or a triangle.
So a clique in $L(G)$ is a set of edges in $G$ that form a star or a triangle.
Wait, if edges are pairwise adjacent, they must share a common vertex (star) or form a triangle.
So the "cliques" in the original graph sense (vertex sets) are NOT the same as cliques in the line graph sense.
Let's re-read carefully: "Every listed set must be a clique [in the original graph]... and every edge must lie in exactly one listed clique."
So we are partitioning the edges of $G$ into subsets, where each subset is the edge set of a clique in $G$.
Let $C$ be a clique in $G$ with vertex set $V_C$. The edges covered are $E(V_C) = \{ (u,v) \in E(G) \mid u,v \in V_C \}$.
We need a collection of vertex sets $V_1, ..., V_k$ such that:
1. Each $V_i$ induces a clique in $G$.
2. $\bigcup_{i=1}^k E(V_i) = E(G)$.
3. $E(V_i) \cap E(V_j) = \emptyset$ for $i \neq j$.

This is a much harder constraint than simple edge cover.
The current program uses "Greedy Peeling": pick a clique, remove its edges.
This ensures disjointness by construction.

My proposed strategy: **Greedy Set Cover with Priority on "Hard" Edges**.
Instead of just picking a maximal clique, we evaluate potential cliques based on how well they cover edges that are "difficult" to cover.
An edge is "difficult" if it belongs to few large cliques.
We can estimate this by looking at the number of common neighbors of its endpoints.
Edges with high common-neighbor counts are "easy" (part of many triangles/cliques).
Edges with low common-neighbor counts are "hard" (isolated or in sparse parts).
To minimize the *number* of cliques, we want large cliques.
However, if we greedily take the largest clique, we might use up "easy" edges and leave "hard" edges to be covered by small cliques (edges).
A better heuristic is to prioritize covering "hard" edges first, but still form large cliques around them if possible.

Algorithm:
1. Generate a list of candidate cliques. Since n is small (32, 64), we can enumerate all maximal cliques?
   For n=64, maximal cliques can be many, but usually manageable.
   However, enumerating ALL cliques is too expensive.
   Instead, we can use a modified greedy:
   
   Iteratively:
   - Find the edge $e=(u,v)$ in the remaining graph that is "most constrained" (e.g., minimum number of common neighbors in the remaining graph).
   - Find a large clique containing $e$ in the remaining graph.
   - Remove the edges of this clique from the remaining graph.
   - Add the clique to the solution.

   This is similar to the current program but changes the selection criterion from "min edge" or "random" to "most constrained edge".
   The current program uses `min(current_edges)` or shuffled order.
   Using "most constrained" (minimum common neighbors) forces us to deal with sparse areas first, potentially preventing them from being fragmented later.
   Conversely, the directive suggests prioritizing edges with **high** "clique density" (ratio of common neighbors to degree).
   Let's try **High Clique Density** first. Why?
   If we cover high-density edges early with large cliques, we might satisfy the "large clique" requirement for many edges at once.
   But wait, if we take a huge clique, we remove many edges.
   The "Hard" edge argument suggests that if we leave hard edges for later, they might be isolated.
   If we take easy edges first, we form big cliques.
   
   Let's look at the score. Lower is better.
   Current score ~3.8.
   
   Let's try a **Multi-Start Local Search** on the partition itself.
   Start with a greedy partition (e.g., based on high density).
   Then try to improve it by:
   1. Removing a clique $C$ from the partition.
   2. Trying to re-cover its edges with other cliques or merging with neighbors.
   This is complex.

   Alternative: **Simulated Annealing on the Clique Selection Order**.
   We can't easily SA the partition directly due to disjointness constraints.
   
   Let's stick to a robust **Greedy Heuristic with Multiple Criteria**.
   We will run the greedy peeling multiple times with different scoring functions for edge selection:
   1. Min common neighbors (Hard first).
   2. Max common neighbors (Easy first).
   3. Random.
   4. Min degree of endpoints.
   5. Max degree of endpoints.
   
   For each selected edge, we expand to a maximal clique using a secondary heuristic (e.g., pick vertex with max degree in candidate set).
   
   This is essentially what the current program does with "shuffled edges" and "min edge".
   The current program uses `min(current_edges)` for the first pass, then shuffled for restarts.
   It does NOT explicitly use common neighbors for selection.
   
   My change: Explicitly compute common neighbors for each edge in the *remaining* graph and select based on that.
   
   Implementation details:
   - Maintain a set of remaining edges.
   - In each iteration, find the edge $(u,v)$ that minimizes/maximizes the score.
   - Score options:
     - `score_hard`: min(common_neighbors(u,v))
     - `score_easy`: max(common_neighbors(u,v))
     - `score_deg`: min(deg(u) + deg(v))
     - `score_deg_max`: max(deg(u) + deg(v))
   - Expand to maximal clique.
   - Remove edges.
   
   We will run these 4 strategies in parallel (or sequentially) and keep the best result.
   We will also include a "Random" strategy for diversity.
   
   To speed up common neighbor calculation, we can maintain adjacency sets.
   When we remove edges, we update degrees and potentially common neighbor counts?
   Common neighbor count changes when edges are removed.
   Recalculating for all edges every step is $O(E \cdot n)$.
   $E \approx 1000, n=64$. $1000 \cdot 64 = 64000$ ops per step.
   Steps $\approx$ number of cliques $\approx$ 50-100.
   Total ops $\approx$ 6.4M per run. Very fast.
   We can afford 10-20 runs.
"""

import time
import random
from itertools import combinations

_START = time.time()
_LIMIT = 140.0  # safety margin under 150s wall-clock

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _random_edges(n, p, seed):
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _solve_instance(n, p, seed):
    edges = _random_edges(n, p, seed)
    if not edges:
        return []
    
    # Build adjacency sets for fast lookup and common neighbor calculation
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    best_partition = None
    best_count = float('inf')
    
    # Define scoring strategies for edge selection
    # Each strategy is a function that takes the current edge set and adj sets and returns the edge to pick
    # and a flag indicating if we should maximize or minimize the score
    
    # We will implement a generic greedy peeler that takes a "pick_edge" function
    
    def greedy_peel(pick_edge_fn):
        current_edges = set(edges)
        # We need to maintain dynamic adjacency for the remaining graph
        # To avoid copying large structures, we can just use the original adj sets 
        # but filter by current_edges when checking common neighbors.
        # However, for efficiency, we can maintain a "remaining_adj" structure?
        # Given the small size, checking `if (u,v) in current_edges` is fast enough.
        
        partition = []
        
        while current_edges:
            if not _deadline_ok():
                # Fallback: add remaining as edges
                for e in current_edges:
                    partition.append([e[0], e[1]])
                return partition
            
            # Pick edge
            u, v = pick_edge_fn(current_edges, adj, n)
            
            if (u, v) not in current_edges and (v, u) not in current_edges:
                # This shouldn't happen if pick_edge_fn is correct
                continue
            
            # Ensure u < v for consistency
            if u > v:
                u, v = v, u
            
            # Expand to maximal clique
            clique = [u, v]
            
            # Candidates: vertices w such that (u,w) and (v,w) are in current_edges
            candidates = []
            for w in range(n):
                if w == u or w == v:
                    continue
                e_uw = (u, w) if u < w else (w, u)
                e_vw = (v, w) if v < w else (w, v)
                if e_uw in current_edges and e_vw in current_edges:
                    candidates.append(w)
            
            # Greedily add candidates
            # We want to add vertices that keep the clique "dense" or just maximal?
            # Maximal is the goal.
            # To ensure we get a good maximal clique, we pick the candidate with the highest degree in the candidate subgraph.
            
            while candidates:
                if not _deadline_ok():
                    break
                
                best_w = -1
                best_score = -1
                
                # Calculate degree in candidate subgraph
                for w in candidates:
                    score = 0
                    for x in candidates:
                        if w == x:
                            continue
                        e_wx = (w, x) if w < x else (x, w)
                        if e_wx in current_edges:
                            score += 1
                    
                    if score > best_score:
                        best_score = score
                        best_w = w
                        
                if best_w == -1:
                    break
                    
                clique.append(best_w)
                candidates.remove(best_w)
                
                # Filter candidates: must be connected to best_w in current_edges
                new_candidates = []
                for w in candidates:
                    e_wbw = (w, best_w) if w < best_w else (best_w, w)
                    if e_wbw in current_edges:
                        new_candidates.append(w)
                candidates = new_candidates
                
            # Add clique to partition
            partition.append(clique)
            
            # Remove edges
            for a, b in combinations(clique, 2):
                e = (a, b) if a < b else (b, a)
                if e in current_edges:
                    current_edges.remove(e)
                    
        return partition

    # Strategy 1: Pick edge with MINIMUM common neighbors (Hard first)
    def pick_min_common(current_edges, adj, n):
        min_score = float('inf')
        best_e = None
        for u, v in current_edges:
            # Common neighbors in remaining graph
            common = 0
            for w in adj[u]:
                if w != v:
                    e_vw = (v, w) if v < w else (w, v)
                    if e_vw in current_edges:
                        common += 1
            if common < min_score:
                min_score = common
                best_e = (u, v)
        return best_e

    # Strategy 2: Pick edge with MAXIMUM common neighbors (Easy first)
    def pick_max_common(current_edges, adj, n):
        max_score = -1
        best_e = None
        for u, v in current_edges:
            common = 0
            for w in adj[u]:
                if w != v:
                    e_vw = (v, w) if v < w else (w, v)
                    if e_vw in current_edges:
                        common += 1
            if common > max_score:
                max_score = common
                best_e = (u, v)
        return best_e

    # Strategy 3: Pick edge with MINIMUM sum of degrees (Sparse first)
    def pick_min_deg(current_edges, adj, n):
        min_score = float('inf')
        best_e = None
        for u, v in current_edges:
            # Degree in remaining graph
            deg_u = 0
            for w in adj[u]:
                e_uw = (u, w) if u < w else (w, u)
                if e_uw in current_edges:
                    deg_u += 1
            deg_v = 0
            for w in adj[v]:
                e_vw = (v, w) if v < w else (w, v)
                if e_vw in current_edges:
                    deg_v += 1
            score = deg_u + deg_v
            if score < min_score:
                min_score = score
                best_e = (u, v)
        return best_e

    # Strategy 4: Pick edge with MAXIMUM sum of degrees (Dense first)
    def pick_max_deg(current_edges, adj, n):
        max_score = -1
        best_e = None
        for u, v in current_edges:
            deg_u = 0
            for w in adj[u]:
                e_uw = (u, w) if u < w else (w, u)
                if e_uw in current_edges:
                    deg_u += 1
            deg_v = 0
            for w in adj[v]:
                e_vw = (v, w) if v < w else (w, v)
                if e_vw in current_edges:
                    deg_v += 1
            score = deg_u + deg_v
            if score > max_score:
                max_score = score
                best_e = (u, v)
        return best_e

    # Strategy 5: Random
    rng = random.Random(seed)
    def pick_random(current_edges, adj, n):
        return rng.choice(list(current_edges))

    strategies = [pick_min_common, pick_max_common, pick_min_deg, pick_max_deg, pick_random]
    
    # Run each strategy
    for i, strategy in enumerate(strategies):
        if not _deadline_ok():
            break
            
        partition = greedy_peel(strategy)
        count = len(partition)
        
        if count < best_count:
            best_count = count
            best_partition = partition
            
    # If we have time, run a few random restarts of the best performing strategy?
    # Or just run random multiple times.
    # Let's run 5 more random restarts if time permits.
    for i in range(5):
        if not _deadline_ok():
            break
        # Use a different seed for the random strategy
        rng2 = random.Random(seed + i)
        def pick_random_seed(current_edges, adj, n, r=rng2):
            return r.choice(list(current_edges))
        
        partition = greedy_peel(pick_random_seed)
        count = len(partition)
        if count < best_count:
            best_count = count
            best_partition = partition

    if best_partition is None:
        # Fallback
        return [[u, v] for u, v in edges]
        
    return best_partition

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
