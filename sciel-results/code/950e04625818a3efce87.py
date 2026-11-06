import time
import sys
from itertools import combinations
from random import Random

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We aim to finish well before.
    DEADLINE = start_time + 140.0 

    def random_graph_edges(n, p, seed):
        """
        Generate random graph using the same RNG as the verifier.
        Returns adjacency list as a list of bitmasks.
        """
        rng = Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def expand_clique(rem, start, strategy="max", rng=None):
        """
        Expand a clique starting from 'start' in the graph defined by 'rem' (bitmask adj).
        Returns a list of vertices forming a clique.
        """
        clique = [start]
        # Candidates are neighbors of start in the remaining graph
        cand = rem[start]
        
        while cand:
            # Find the best vertex in cand to add to the clique.
            # Heuristic: Maximize degree within the current candidate set.
            # This is equivalent to finding the vertex with the most common neighbors 
            # with other candidates, which tends to lead to larger cliques.
            
            best_u = -1
            best_score = -1
            second_best_u = -1
            second_best_score = -1
            third_best_u = -1
            third_best_score = -1
            
            temp = cand
            while temp:
                low = temp & (-temp)
                u = low.bit_length() - 1
                # Score: number of neighbors of u that are also in cand
                score = (rem[u] & cand).bit_count()
                
                if score > best_score:
                    third_best_u, third_best_score = second_best_u, second_best_score
                    second_best_u, second_best_score = best_u, best_score
                    best_u, best_score = u, score
                elif score > second_best_score:
                    third_best_u, third_best_score = second_best_u, second_best_score
                    second_best_u, second_best_score = best_u, best_score
                elif score > third_best_score:
                    third_best_u, third_best_score = best_u, best_score
                    
                temp ^= low
            
            if best_u == -1:
                break
                
            # Select vertex based on strategy
            if strategy == "max":
                chosen = best_u
            elif strategy == "random":
                # Pick randomly from top 3 to introduce diversity
                options = []
                if best_u != -1: options.append(best_u)
                if second_best_u != -1 and second_best_u != best_u: options.append(second_best_u)
                if third_best_u != -1 and third_best_u != best_u and third_best_u != second_best_u: options.append(third_best_u)
                if rng and options:
                    chosen = rng.choice(options)
                else:
                    chosen = best_u
            else:
                chosen = best_u
                
            clique.append(chosen)
            # Update candidates: must be neighbors of chosen AND existing candidates
            cand &= rem[chosen]
            
        return clique if len(clique) >= 2 else None

    def solve_instance(n, p, seed, deadline):
        adj = random_graph_edges(n, p, seed)
        # rem[i] contains bits for neighbors j such that edge (i,j) is still uncovered
        rem = [adj[i] for i in range(n)]
        
        # Count remaining edges
        remaining_edges = sum(rem[i].bit_count() for i in range(n)) // 2
        
        if remaining_edges == 0:
            return []
            
        cliques = []
        local_rng = Random(42) # Fixed seed for reproducibility of randomization strategy
        
        while remaining_edges > 0:
            if time.time() > deadline:
                break
                
            # Find vertices with highest degree in the remaining graph
            # We use a simple O(n) scan to find top 2-3 vertices
            max_deg = -1
            v1 = -1
            second_max_deg = -1
            v2 = -1
            
            for i in range(n):
                d = rem[i].bit_count()
                if d > max_deg:
                    v2 = v1
                    second_max_deg = max_deg
                    v1 = i
                    max_deg = d
                elif d > second_max_deg:
                    v2 = i
                    second_max_deg = d
            
            if v1 == -1 or max_deg == 0:
                # Should not happen if remaining_edges > 0
                break
                
            if max_deg == 1:
                # If max degree is 1, the best we can do is a 2-clique (edge)
                low = rem[v1] & (-rem[v1])
                u = low.bit_length() - 1
                cliques.append([v1, u])
                rem[v1] &= ~(1 << u)
                rem[u] &= ~(1 << v1)
                remaining_edges -= 1
                continue
            
            # Generate multiple candidate cliques
            candidates = []
            
            # 1. Greedy from highest degree vertex
            c1 = expand_clique(rem, v1, "max")
            if c1: candidates.append(c1)
            
            # 2. Greedy from second highest degree vertex
            if v2 != -1 and second_max_deg > 0:
                c2 = expand_clique(rem, v2, "max")
                if c2: candidates.append(c2)
                
            # 3. Randomized greedy from highest degree vertex
            c3 = expand_clique(rem, v1, "random", local_rng)
            if c3: candidates.append(c3)
            
            # 4. Randomized greedy from highest degree vertex (another trial)
            c4 = expand_clique(rem, v1, "random", local_rng)
            if c4: candidates.append(c4)
            
            # 5. Greedy from a random neighbor of v1
            if rem[v1]:
                bits = rem[v1].bit_count()
                if bits > 1:
                    r = local_rng.randint(0, bits - 1)
                    temp = rem[v1]
                    for _ in range(r):
                        temp &= temp - 1
                    u = temp.bit_length() - 1
                    c5 = expand_clique(rem, u, "max")
                    if c5: candidates.append(c5)
                    
            if not candidates:
                # Fallback: just take one edge
                low = rem[v1] & (-rem[v1])
                u = low.bit_length() - 1
                cliques.append([v1, u])
                rem[v1] &= ~(1 << u)
                rem[u] &= ~(1 << v1)
                remaining_edges -= 1
                continue
                
            # Select the best candidate (largest size)
            best_clique = max(candidates, key=len)
            m = len(best_clique)
            
            # Add the clique
            cliques.append(best_clique)
            
            # Remove all edges in this clique from the remaining graph
            for i in range(m):
                for j in range(i + 1, m):
                    a, b = best_clique[i], best_clique[j]
                    rem[a] &= ~(1 << b)
                    rem[b] &= ~(1 << a)
                    
            remaining_edges -= m * (m - 1) // 2
            
        # Handle any remaining edges if deadline was hit or loop ended early
        if remaining_edges > 0:
            for i in range(n):
                if time.time() > deadline:
                    break
                temp = rem[i]
                while temp:
                    if time.time() > deadline:
                        break
                    low = temp & (-temp)
                    j = low.bit_length() - 1
                    cliques.append([i, j])
                    rem[i] &= ~(1 << j)
                    rem[j] &= ~(1 << i)
                    temp &= temp - 1
                    rem[i] &= ~(1 << j) # Ensure it's cleared in rem[i] too
                    
        return cliques

    results = []
    # Allocate time for each instance, leaving a small buffer
    time_per_instance = (DEADLINE - time.time()) / 2.0
    
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        t_start = time.time()
        t_limit = t_start + time_per_instance
        # Ensure we don't exceed the global deadline
        t_limit = min(t_limit, DEADLINE)
        
        result = solve_instance(n, p, seed, t_limit)
        results.append(result)
        
    return results
