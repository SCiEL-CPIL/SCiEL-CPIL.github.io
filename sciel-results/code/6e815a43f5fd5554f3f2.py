"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 15
    # All possible edges in a simple undirected graph on n vertices
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(all_edges)
    
    # Map edge to index for faster lookup
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    idx_to_edge = {i: e for i, e in enumerate(all_edges)}

    def get_degrees_and_adj(edge_mask):
        """
        edge_mask: list of booleans of length m. True if edge is present.
        Returns deg list and adj list of sets.
        """
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for i in range(m):
            if edge_mask[i]:
                u, v = all_edges[i]
                deg[u] += 1
                deg[v] += 1
                adj[u].add(v)
                adj[v].add(u)
        return deg, adj

    def count_violations(edge_mask):
        """
        Returns the number of violating paths s-u-v-t where deg(s) == deg(t).
        A path is defined by a middle edge (u, v) and neighbors s of u (s!=v) and t of v (t!=u).
        """
        deg, adj = get_degrees_and_adj(edge_mask)
        violations = 0
        
        # For each edge (u, v) in the graph
        for i in range(m):
            if not edge_mask[i]:
                continue
            u, v = all_edges[i]
            
            # Get degrees of neighbors of u (excluding v)
            deg_s_counts = {}
            for s in adj[u]:
                if s != v:
                    d = deg[s]
                    deg_s_counts[d] = deg_s_counts.get(d, 0) + 1
            
            if not deg_s_counts:
                continue
                
            # Get degrees of neighbors of v (excluding u) and check for matches
            for t in adj[v]:
                if t != u:
                    dt = deg[t]
                    if dt in deg_s_counts:
                        violations += deg_s_counts[dt]
                        
        return violations

    def evaluate(edge_mask):
        """
        Score function for Simulated Annealing.
        We want to maximize edges while minimizing violations.
        Score = num_edges - lambda * violations
        We want to find a state with 0 violations and max edges.
        To guide the search, we prioritize reducing violations first, then adding edges.
        Let's use a large penalty for violations so that valid states are always better than invalid ones with fewer edges?
        No, we want to find the MAX edges among valid graphs.
        So, fitness = num_edges * BIG_WEIGHT - violations * SMALL_WEIGHT?
        Actually, if violations > 0, we should strongly prefer reducing them.
        Let's define fitness = num_edges - 100 * violations.
        This way, reducing violations by 1 gains 100 points, adding an edge gains 1 point.
        This ensures we search for valid graphs first, then densify.
        """
        num_edges = sum(edge_mask)
        violations = count_violations(edge_mask)
        # High penalty for violations to ensure we find valid structures
        return num_edges - 100 * violations

    def random_initial_state():
        # Start with a moderately dense random graph to be close to the optimum
        # Probability of edge ~ 0.5
        return [random.random() < 0.5 for _ in range(m)]

    def neighbor_state(edge_mask):
        """
        Generate a neighbor state via 1-flip or 2-swap.
        1-flip: Toggle a random edge.
        2-swap: Remove one edge, add another.
        """
        new_mask = edge_mask[:]
        # Decide move type: 70% 1-flip, 30% 2-swap
        if random.random() < 0.7:
            # 1-flip
            i = random.randint(0, m - 1)
            new_mask[i] = not new_mask[i]
        else:
            # 2-swap: pick two distinct edges
            i1 = random.randint(0, m - 1)
            i2 = random.randint(0, m - 1)
            while i2 == i1:
                i2 = random.randint(0, m - 1)
            
            # Try to remove i1 and add i2
            # If i1 is not present, we can't remove it. If i2 is present, we can't add it.
            # For a valid 2-swap move in the context of maximizing edges, 
            # usually we swap an existing edge for a non-existing one.
            if new_mask[i1] and not new_mask[i2]:
                new_mask[i1] = False
                new_mask[i2] = True
            elif not new_mask[i1] and new_mask[i2]:
                new_mask[i1] = True
                new_mask[i2] = False
            else:
                # If both present or both absent, just flip one randomly to ensure change
                # Or retry? Let's just flip i1 for simplicity if swap fails
                new_mask[i1] = not new_mask[i1]
        return new_mask

    def simulated_annealing():
        # Parameters
        T_start = 100.0
        T_end = 1.0
        alpha = 0.999
        max_iter = 20000
        
        # Initialize
        current_mask = random_initial_state()
        current_score = evaluate(current_mask)
        
        best_mask = current_mask[:]
        best_score = current_score
        
        T = T_start
        for i in range(max_iter):
            if time.time() > deadline:
                break
                
            next_mask = neighbor_state(current_mask)
            next_score = evaluate(next_mask)
            
            delta = next_score - current_score
            
            if delta > 0:
                # Accept better solution
                current_mask = next_mask
                current_score = next_score
                if current_score > best_score:
                    best_mask = current_mask[:]
                    best_score = current_score
            else:
                # Accept worse solution with probability exp(delta / T)
                # delta is negative, so exp(negative) < 1
                if T > 0:
                    prob = random.expovariate(1.0) # This is not right.
                    # Correct: accept if random() < exp(delta/T)
                    if random.random() < pow(2.718281828, delta / T):
                        current_mask = next_mask
                        current_score = next_score
                        if current_score > best_score:
                            best_mask = current_mask[:]
                            best_score = current_score
            
            T *= alpha
            if T < T_end:
                break
                
        return best_mask, best_score

    # Run multiple restarts of Simulated Annealing
    best_overall_mask = []
    best_overall_score = -1
    
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        mask, score = simulated_annealing()
        
        # Check if this is a valid solution (0 violations)
        # If it is valid, its score is just num_edges (since penalty is 0)
        # If not valid, score is num_edges - 100*violations, which is likely low.
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_mask = mask[:]
            
        # If we found a valid solution, we can try to polish it
        # But first, let's check if it's actually valid
        violations = count_violations(mask)
        if violations == 0:
            # We have a valid graph. Try to add more edges greedily to maximize count.
            # This is a local improvement step.
            improved = True
            while improved:
                improved = False
                if time.time() > deadline:
                    break
                for i in range(m):
                    if mask[i]:
                        continue
                    test_mask = mask[:]
                    test_mask[i] = True
                    if count_violations(test_mask) == 0:
                        mask[i] = True
                        improved = True
                        if time.time() > deadline:
                            break
                # If we added an edge, update score
                if improved:
                    score = evaluate(mask)
                    if score > best_overall_score:
                        best_overall_score = score
                        best_overall_mask = mask[:]
                        
    # Convert best mask to list of edges
    best_edges = []
    for i in range(m):
        if best_overall_mask[i]:
            best_edges.append(list(all_edges[i]))
            
    # Fallback if no valid graph found (should not happen, star is always valid)
    if not best_edges:
        best_edges = [[0, i] for i in range(1, n)]
        
    # Ensure types are correct
    return best_edges

# EVOLVE-BLOCK-END
